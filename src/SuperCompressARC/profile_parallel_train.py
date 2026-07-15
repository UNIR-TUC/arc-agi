"""
Low-overhead profiler / metrics harness for parallel_train.py.

Purpose
-------
Measure *where the host is spending time* while parallel_train.py runs, in order
to locate and quantify the CPU bottleneck described in Eje H (§9.11) of
Docs/Architecture/COMPRESS_ARCHITECTURE_30_06.md, and to compare a run *before*
an optimization against a run *after* it.

Design goals (critical)
------------------------
This tool must NEVER worsen the very bottleneck it measures. Therefore it:
  * runs as a single, separate process that only *samples* the OS (psutil),
  * never instruments or imports the training hot loop,
  * samples at a coarse interval (>= 1 s by default),
  * lowers its own scheduling priority so it does not steal CPU from workers,
  * reads GPU stats via a short-lived `rocm-smi`/`nvidia-smi` call (best effort).

What it measures
----------------
  * Wall-clock time of the whole run.
  * System CPU: overall %, per-core %, saturation fraction (time above threshold).
  * Process tree (parent parallel_train.py + spawned worker processes):
      - number of concurrently alive workers over time (effective parallelism),
      - aggregate CPU% and RSS of the tree,
      - per-worker lifetime, mean CPU%, peak RSS  -> "resources per parallel task".
  * Worker completions over time  -> task throughput proxy.
  * GPU utilization and VRAM (best effort, if rocm-smi/nvidia-smi is available).

Outputs
-------
  * <prefix>_timeseries.csv : one row per sample (raw data for plots).
  * <prefix>_summary.json   : aggregated metrics for before/after comparison.

Usage
-----
Run + profile a full training run (pass-through args after `--` go to
parallel_train.py):

    python profile_parallel_train.py --label baseline -- --split training --demo 20

Run again after applying an optimization:

    python profile_parallel_train.py --label optimized -- --split training --demo 20

Compare two summaries (no training is launched):

    python profile_parallel_train.py --compare .profile/baseline_summary.json \
                                                .profile/optimized_summary.json

The comparison prints deltas and FAILS LOUDLY if the CPU bottleneck got worse
(so an "optimization" that actually increases CPU pressure is caught).
"""

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
import time

import psutil


# ── GPU sampling (best effort, vendor-agnostic) ──────────────────────────────

def _detect_gpu_tool():
    """Return ('rocm'|'nvidia'|None, path) for the first available SMI tool."""
    for tool, vendor in (('rocm-smi', 'rocm'), ('nvidia-smi', 'nvidia')):
        path = shutil.which(tool)
        if path:
            return vendor, path
    return None, None


def _sample_gpu(vendor, path):
    """Return (util_percent, mem_used_mb) aggregated over GPUs, or (None, None).

    Kept deliberately short-lived and wrapped in try/except: a flaky SMI call
    must never crash the harness nor block sampling for long.
    """
    try:
        if vendor == 'nvidia':
            out = subprocess.run(
                [path, '--query-gpu=utilization.gpu,memory.used',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=4,
            ).stdout.strip()
            utils, mems = [], []
            for line in out.splitlines():
                u, m = line.split(',')
                utils.append(float(u))
                mems.append(float(m))
            if not utils:
                return None, None
            return max(utils), sum(mems)
        if vendor == 'rocm':
            out = subprocess.run(
                [path, '--showuse', '--showmeminfo', 'vram', '--json'],
                capture_output=True, text=True, timeout=4,
            ).stdout.strip()
            data = json.loads(out) if out.startswith('{') else {}
            utils, mems = [], []
            for card in data.values():
                for key, val in card.items():
                    kl = key.lower()
                    if 'gpu use' in kl or 'gpu_use' in kl:
                        try:
                            utils.append(float(str(val).strip().rstrip('%')))
                        except ValueError:
                            pass
                    if 'vram total used' in kl or ('used' in kl and 'vram' in kl):
                        try:
                            mems.append(float(str(val).strip()) / 1024**2)
                        except ValueError:
                            pass
            return (max(utils) if utils else None,
                    sum(mems) if mems else None)
    except Exception:
        return None, None
    return None, None


# ── Low-priority self so measuring does not steal CPU from workers ────────────

def _lower_own_priority():
    p = psutil.Process()
    try:
        if os.name == 'nt':
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            p.nice(10)
    except Exception:
        pass  # best effort only


# ── Process-tree tracking ────────────────────────────────────────────────────

class TreeTracker:
    """Tracks the parent process and its (recursive) children, priming psutil's
    cpu_percent so subsequent reads are accurate, and records per-worker stats."""

    def __init__(self, root_proc):
        self.root = root_proc
        self._known = {}          # pid -> psutil.Process (primed)
        self.workers = {}         # pid -> {'start','end','cpu_samples','peak_rss','name'}
        self.completions = []     # list of (t_rel, pid) as workers exit

    def _procs(self):
        procs = []
        try:
            procs.append(self.root)
            procs.extend(self.root.children(recursive=True))
        except psutil.Error:
            pass
        return procs

    def refresh(self, t_rel):
        """Discover new procs (prime them) and mark completed workers."""
        live_pids = set()
        for proc in self._procs():
            try:
                pid = proc.pid
                live_pids.add(pid)
                if pid not in self._known:
                    proc.cpu_percent(None)  # prime; first read is 0.0
                    self._known[pid] = proc
                    if pid != self.root.pid:
                        self.workers[pid] = {
                            'start': t_rel, 'end': None, 'cpu_samples': [],
                            'peak_rss': 0.0, 'name': proc.name(),
                        }
            except psutil.Error:
                continue
        # mark workers that disappeared as completed
        for pid, rec in self.workers.items():
            if rec['end'] is None and pid not in live_pids:
                rec['end'] = t_rel
                self.completions.append((t_rel, pid))

    def sample(self):
        """Return aggregate (n_workers, tree_cpu_pct, tree_rss_mb) for this tick."""
        n_workers = 0
        tree_cpu = 0.0
        tree_rss = 0.0
        for pid, proc in self._known.items():
            try:
                cpu = proc.cpu_percent(None)
                rss = proc.memory_info().rss / 1024**2
            except psutil.Error:
                continue
            tree_cpu += cpu
            tree_rss += rss
            if pid in self.workers and self.workers[pid]['end'] is None:
                n_workers += 1
                self.workers[pid]['cpu_samples'].append(cpu)
                self.workers[pid]['peak_rss'] = max(self.workers[pid]['peak_rss'], rss)
        return n_workers, tree_cpu, tree_rss


# ── Main run+profile ─────────────────────────────────────────────────────────

def run_and_profile(args, passthrough):
    os.makedirs(args.outdir, exist_ok=True)
    prefix = os.path.join(args.outdir, args.label)
    csv_path = f'{prefix}_timeseries.csv'
    json_path = f'{prefix}_summary.json'

    n_cores = psutil.cpu_count(logical=True)
    gpu_vendor, gpu_path = _detect_gpu_tool()

    cmd = [sys.executable, '-u', 'parallel_train.py'] + passthrough
    print(f'[profiler] launching: {" ".join(cmd)}')
    print(f'[profiler] cores={n_cores}  gpu={gpu_vendor or "n/a"}  interval={args.interval}s')

    _lower_own_priority()

    # Prime system-wide cpu_percent (first call returns 0.0 / meaningless).
    psutil.cpu_percent(percpu=True)

    child = subprocess.Popen(cmd)
    root = psutil.Process(child.pid)
    tracker = TreeTracker(root)

    t0 = time.time()
    rows = []
    sys_cpu_series = []
    percore_series = []
    workers_series = []
    gpu_util_series = []
    gpu_mem_series = []

    try:
        while child.poll() is None:
            loop_start = time.time()
            t_rel = loop_start - t0

            tracker.refresh(t_rel)
            percore = psutil.cpu_percent(percpu=True)
            sys_cpu = sum(percore) / len(percore) if percore else 0.0
            n_workers, tree_cpu, tree_rss = tracker.sample()
            vmem = psutil.virtual_memory()
            gpu_util, gpu_mem = (_sample_gpu(gpu_vendor, gpu_path)
                                 if gpu_vendor else (None, None))

            rows.append({
                't_rel': round(t_rel, 2),
                'sys_cpu_pct': round(sys_cpu, 1),
                'percore_max_pct': round(max(percore), 1) if percore else 0.0,
                'percore_min_pct': round(min(percore), 1) if percore else 0.0,
                'n_workers': n_workers,
                'tree_cpu_pct': round(tree_cpu, 1),
                'tree_rss_mb': round(tree_rss, 1),
                'host_mem_used_pct': vmem.percent,
                'gpu_util_pct': gpu_util,
                'gpu_mem_mb': round(gpu_mem, 1) if gpu_mem is not None else None,
            })
            sys_cpu_series.append(sys_cpu)
            percore_series.append(percore)
            workers_series.append(n_workers)
            if gpu_util is not None:
                gpu_util_series.append(gpu_util)
            if gpu_mem is not None:
                gpu_mem_series.append(gpu_mem)

            # Sleep the remainder of the interval (keep sampling cost off the host).
            elapsed = time.time() - loop_start
            time.sleep(max(0.0, args.interval - elapsed))
    except KeyboardInterrupt:
        print('\n[profiler] interrupted — terminating training subprocess...')
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()

    wall = time.time() - t0
    return_code = child.poll()

    # ── Write raw time series ────────────────────────────────────────────────
    if rows:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    # ── Aggregate summary ────────────────────────────────────────────────────
    def _pct(series, q):
        if not series:
            return None
        s = sorted(series)
        k = min(len(s) - 1, int(q * (len(s) - 1)))
        return round(s[k], 1)

    sat_threshold = args.saturation_threshold
    sat_fraction = (sum(1 for c in sys_cpu_series if c >= sat_threshold)
                    / len(sys_cpu_series)) if sys_cpu_series else 0.0

    # per-worker records (resources per parallel task)
    worker_records = []
    for pid, rec in tracker.workers.items():
        end = rec['end'] if rec['end'] is not None else wall
        cpu_samples = rec['cpu_samples']
        worker_records.append({
            'pid': pid,
            'name': rec['name'],
            'lifetime_s': round(end - rec['start'], 1),
            'mean_cpu_pct': round(statistics.mean(cpu_samples), 1) if cpu_samples else None,
            'peak_cpu_pct': round(max(cpu_samples), 1) if cpu_samples else None,
            'peak_rss_mb': round(rec['peak_rss'], 1),
        })

    completed = [r for r in worker_records if tracker.workers[r['pid']]['end'] is not None]
    worker_lifetimes = [r['lifetime_s'] for r in completed if r['lifetime_s'] > 0]

    summary = {
        'label': args.label,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'command': cmd,
        'return_code': return_code,
        'wall_time_s': round(wall, 1),
        'n_cores': n_cores,
        'sample_interval_s': args.interval,
        'n_samples': len(rows),
        'cpu': {
            'mean_pct': round(statistics.mean(sys_cpu_series), 1) if sys_cpu_series else None,
            'median_pct': round(statistics.median(sys_cpu_series), 1) if sys_cpu_series else None,
            'p95_pct': _pct(sys_cpu_series, 0.95),
            'max_pct': round(max(sys_cpu_series), 1) if sys_cpu_series else None,
            'saturation_threshold_pct': sat_threshold,
            'saturation_fraction': round(sat_fraction, 3),
        },
        'concurrency': {
            'mean_workers': round(statistics.mean(workers_series), 2) if workers_series else 0,
            'max_workers': max(workers_series) if workers_series else 0,
            'workers_completed': len(completed),
        },
        'throughput': {
            'workers_completed': len(completed),
            'wall_time_s': round(wall, 1),
            'workers_per_hour': round(len(completed) / wall * 3600, 2) if wall > 0 else None,
            'mean_worker_lifetime_s': round(statistics.mean(worker_lifetimes), 1) if worker_lifetimes else None,
        },
        'memory': {
            'tree_rss_max_mb': round(max((r['tree_rss_mb'] for r in rows), default=0.0), 1),
            'host_mem_used_max_pct': max((r['host_mem_used_pct'] for r in rows), default=0.0),
        },
        'gpu': {
            'vendor': gpu_vendor,
            'util_mean_pct': round(statistics.mean(gpu_util_series), 1) if gpu_util_series else None,
            'util_max_pct': round(max(gpu_util_series), 1) if gpu_util_series else None,
            'mem_max_mb': round(max(gpu_mem_series), 1) if gpu_mem_series else None,
        },
        'workers': worker_records,
    }

    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    _print_summary(summary, csv_path, json_path)
    return 0 if return_code == 0 else 1


def _print_summary(s, csv_path, json_path):
    c, cc, tp, g = s['cpu'], s['concurrency'], s['throughput'], s['gpu']
    print('\n' + '=' * 66)
    print(f'  PROFILE SUMMARY — {s["label"]}   (return code {s["return_code"]})')
    print('=' * 66)
    print(f'  Wall time            : {s["wall_time_s"]:.1f} s   ({s["wall_time_s"]/3600:.2f} h)')
    print(f'  Host cores           : {s["n_cores"]}')
    print(f'  CPU mean / p95 / max : {c["mean_pct"]} / {c["p95_pct"]} / {c["max_pct"]} %')
    print(f'  CPU saturated (>={c["saturation_threshold_pct"]}%): '
          f'{c["saturation_fraction"]*100:.1f} % of run')
    print(f'  Workers mean / max   : {cc["mean_workers"]} / {cc["max_workers"]}')
    print(f'  Workers completed    : {cc["workers_completed"]}')
    print(f'  Throughput           : {tp["workers_per_hour"]} workers/h'
          + (f'   (mean lifetime {tp["mean_worker_lifetime_s"]} s)'
             if tp["mean_worker_lifetime_s"] else ''))
    print(f'  Tree RSS max         : {s["memory"]["tree_rss_max_mb"]:.0f} MB')
    if g['vendor']:
        print(f'  GPU util mean / max  : {g["util_mean_pct"]} / {g["util_max_pct"]} %'
              f'   VRAM max {g["mem_max_mb"]} MB')
    print('-' * 66)
    print(f'  time series : {csv_path}')
    print(f'  summary     : {json_path}')
    print('=' * 66 + '\n')


# ── Before/after comparison ──────────────────────────────────────────────────

def compare(before_path, after_path):
    with open(before_path) as f:
        b = json.load(f)
    with open(after_path) as f:
        a = json.load(f)

    def line(name, bv, av, better='lower', unit='', pct=False):
        if bv is None or av is None:
            print(f'  {name:<26}: {bv} -> {av}')
            return None
        delta = av - bv
        rel = (delta / bv * 100) if bv else float('inf')
        arrow = '↓' if delta < 0 else ('↑' if delta > 0 else '=')
        good = (delta <= 0) if better == 'lower' else (delta >= 0)
        tag = 'OK ' if good else 'REGRESSION'
        d = f'{delta:+.1f}{unit}'
        r = f' ({rel:+.1f}%)' if not pct else ''
        print(f'  {name:<26}: {bv}{unit} -> {av}{unit}  {arrow} {d}{r}   [{tag}]')
        return good

    print('\n' + '=' * 66)
    print(f'  COMPARISON   {b["label"]}  ->  {a["label"]}')
    print('=' * 66)

    print('\n  Throughput / speed (higher is better):')
    line('wall_time_s', b['wall_time_s'], a['wall_time_s'], 'lower', ' s')
    line('workers_per_hour', b['throughput']['workers_per_hour'],
         a['throughput']['workers_per_hour'], 'higher')
    line('mean_worker_lifetime_s', b['throughput']['mean_worker_lifetime_s'],
         a['throughput']['mean_worker_lifetime_s'], 'lower', ' s')

    print('\n  CPU pressure (must NOT get worse):')
    cpu_ok = []
    cpu_ok.append(line('cpu_mean_pct', b['cpu']['mean_pct'], a['cpu']['mean_pct'], 'lower', ' %'))
    cpu_ok.append(line('cpu_saturation_fraction',
                       b['cpu']['saturation_fraction'], a['cpu']['saturation_fraction'],
                       'lower', '', pct=True))

    print('\n  Concurrency / GPU:')
    line('mean_workers', b['concurrency']['mean_workers'],
         a['concurrency']['mean_workers'], 'higher')
    line('gpu_util_mean_pct', b['gpu']['util_mean_pct'],
         a['gpu']['util_mean_pct'], 'higher', ' %')

    print('=' * 66)
    # Guardrail: the optimization must not increase CPU pressure (Eje H §9.11.3).
    regressed = [ok for ok in cpu_ok if ok is False]
    if regressed:
        print('  ⚠  CPU pressure INCREASED — this change worsens the bottleneck.')
        print('=' * 66 + '\n')
        return 2
    print('  ✓  CPU pressure did not get worse.')
    print('=' * 66 + '\n')
    return 0


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Low-overhead profiler for parallel_train.py (Eje H §9.11).',
    )
    parser.add_argument('--label', default='run',
                        help='Name for this run (used in output filenames).')
    parser.add_argument('--outdir', default='.profile',
                        help='Directory for CSV/JSON outputs. Default: .profile')
    parser.add_argument('--interval', type=float, default=1.0,
                        help='Sampling interval in seconds (>=1 recommended). Default: 1.0')
    parser.add_argument('--saturation-threshold', type=float, default=90.0,
                        help='System CPU %% considered "saturated". Default: 90')
    parser.add_argument('--compare', nargs=2, metavar=('BEFORE.json', 'AFTER.json'),
                        help='Compare two summary JSONs instead of running a new profile.')
    parser.add_argument('passthrough', nargs=argparse.REMAINDER,
                        help='Args after `--` are forwarded to parallel_train.py.')
    args = parser.parse_args()

    if args.compare:
        sys.exit(compare(args.compare[0], args.compare[1]))

    # Strip a leading '--' separator from REMAINDER if present.
    passthrough = args.passthrough
    if passthrough and passthrough[0] == '--':
        passthrough = passthrough[1:]

    if args.interval < 0.5:
        print('[profiler] WARNING: interval < 0.5s adds measurable overhead; '
              'use >= 1.0s to avoid perturbing the host.', file=sys.stderr)

    sys.exit(run_and_profile(args, passthrough))


if __name__ == '__main__':
    main()
