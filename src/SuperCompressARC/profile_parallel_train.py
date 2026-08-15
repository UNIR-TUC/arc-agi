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
  * reads GPU stats preferentially via direct /sys/class/drm reads (no subprocess, no
    SMI-tool ioctl surface), falling back to a short-lived `rocm-smi`/`nvidia-smi` call
    only if sysfs is unavailable, and samples GPU stats on its own slow cadence
    (`--gpu-interval`, default 20 s) decoupled from the 1 s CPU/process loop.

GPU sampling safety note
-------------------------
Earlier versions of this harness polled `rocm-smi` every second for the whole run. On
this host (AMD RDNA4 + ROCm) that was observed to coincide with full kernel panics when
combined with many concurrent worker processes churning VRAM — repeated external SMI
subprocess spawns (fork/exec + ioctl-heavy queries), occasionally killed mid-query by the
`timeout=`, are suspected to destabilize the amdgpu driver under heavy concurrent load.
GPU sampling now prefers lightweight sysfs reads and runs far less often; use
`--gpu-mode off` to disable it completely if instability recurs.

What it measures
----------------
  * Wall-clock time of the whole run.
  * System CPU: overall %, per-core %, saturation fraction (time above threshold).
  * Process tree (parent parallel_train.py + spawned worker processes):
      - number of concurrently alive workers over time (effective parallelism),
      - aggregate CPU% and RSS of the tree,
      - per-worker lifetime, mean CPU%, peak RSS  -> "resources per parallel task".
  * Worker completions over time  -> task throughput proxy.
  * GPU utilization, memory-controller busy %, VRAM and board power (best effort).
  * From `run_metadata_{split}.json` written by parallel_train.py: planned training
    steps, pass@2 solved count and the exact Eje D acceleration configuration ->
    training throughput (steps/s), energy per 1k steps, and accuracy.

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

Eje D (§9.6) before/after workflow — `--accel-preset` forwards the matching flag
to parallel_train.py and labels the summary, so the two runs differ only in the
acceleration configuration:

    python profile_parallel_train.py --label d_baseline --accel-preset baseline \
        -- --split training --demo 20 --iterations 300
    python profile_parallel_train.py --label d_full --accel-preset full \
        -- --split training --demo 20 --iterations 300
    python profile_parallel_train.py --compare .profile/d_baseline_summary.json \
                                                .profile/d_full_summary.json

Compare two summaries (no training is launched):

    python profile_parallel_train.py --compare .profile/baseline_summary.json \
                                                .profile/optimized_summary.json

The comparison prints deltas and FAILS LOUDLY if the CPU bottleneck got worse
(exit code 2, so an "optimization" that actually increases CPU pressure is
caught) or if pass@2 accuracy dropped (exit code 3, so a speedup bought with
broken numerics is caught too).
"""

import argparse
import csv
import glob
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
    """Return (util_pct, mem_used_mb, mem_busy_pct, power_w) or Nones.

    Kept deliberately short-lived and wrapped in try/except: a flaky SMI call
    must never crash the harness nor block sampling for long. The SMI path does
    not report mem_busy/power here — those come from sysfs only.
    """
    try:
        if vendor == 'nvidia':
            out = subprocess.run(
                [path, '--query-gpu=utilization.gpu,memory.used,power.draw',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=4,
            ).stdout.strip()
            utils, mems, powers = [], [], []
            for line in out.splitlines():
                parts = line.split(',')
                utils.append(float(parts[0]))
                mems.append(float(parts[1]))
                if len(parts) > 2:
                    try:
                        powers.append(float(parts[2]))
                    except ValueError:
                        pass
            if not utils:
                return None, None, None, None
            return (max(utils), sum(mems), None,
                    sum(powers) if powers else None)
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
                    sum(mems) if mems else None,
                    None, None)
    except Exception:
        return None, None, None, None
    return None, None, None, None


# ── GPU sampling via sysfs (preferred: no subprocess, no SMI ioctl surface) ──

def _detect_gpu_sysfs():
    """Return a list of AMD GPU device dirs under /sys/class/drm, or [] if none found.

    Reading sysfs attributes directly avoids spawning an external SMI tool (no fork/exec,
    no ioctl-heavy queries) and is far less likely to contend with the amdgpu driver while
    many worker processes are concurrently allocating/freeing VRAM.
    """
    device_dirs = []
    try:
        for entry in sorted(glob.glob('/sys/class/drm/card[0-9]*')):
            device_dir = os.path.join(entry, 'device')
            vendor_path = os.path.join(device_dir, 'vendor')
            vram_path = os.path.join(device_dir, 'mem_info_vram_used')
            if not (os.path.exists(vendor_path) and os.path.exists(vram_path)):
                continue
            with open(vendor_path) as f:
                vendor_id = f.read().strip()
            if vendor_id.lower() == '0x1002':  # AMD
                device_dirs.append(device_dir)
    except Exception:
        return []
    return device_dirs


def _sample_gpu_sysfs(device_dirs):
    """Return (util_pct, mem_used_mb, mem_busy_pct, power_w) via sysfs, or Nones.

    Pure file reads, wrapped defensively: a missing/unreadable attribute on one GPU must
    never crash the harness nor block sampling.

    `mem_busy_percent` (memory-controller occupancy) and board power complement
    `gpu_busy_percent` as silicon-utilisation signals for Eje D: BF16 and kernel
    fusion should raise compute occupancy and lower energy per solved task.
    """
    utils, mems, mem_busy, powers = [], [], [], []
    for device_dir in device_dirs:
        try:
            with open(os.path.join(device_dir, 'mem_info_vram_used')) as f:
                mems.append(int(f.read().strip()) / 1024**2)
        except Exception:
            pass
        try:
            with open(os.path.join(device_dir, 'gpu_busy_percent')) as f:
                utils.append(float(f.read().strip()))
        except Exception:
            pass
        try:
            with open(os.path.join(device_dir, 'mem_busy_percent')) as f:
                mem_busy.append(float(f.read().strip()))
        except Exception:
            pass
        # Board power lives under device/hwmon/hwmon*/power1_average (microwatts).
        try:
            for power_path in glob.glob(
                    os.path.join(device_dir, 'hwmon', 'hwmon*', 'power1_average')):
                with open(power_path) as f:
                    powers.append(int(f.read().strip()) / 1e6)
                break
        except Exception:
            pass
    return (max(utils) if utils else None,
            sum(mems) if mems else None,
            max(mem_busy) if mem_busy else None,
            sum(powers) if powers else None)


def _detect_gpu_source(mode):
    """Resolve --gpu-mode to a concrete (source_type, payload) sampling source.

    source_type is one of 'sysfs', 'smi', or None (no GPU sampling). 'auto' prefers the
    low-risk sysfs path and only falls back to spawning an SMI tool if sysfs is unavailable.
    """
    if mode == 'off':
        return None, None
    if mode in ('auto', 'sysfs'):
        device_dirs = _detect_gpu_sysfs()
        if device_dirs:
            return 'sysfs', device_dirs
        if mode == 'sysfs':
            return None, None
    vendor, path = _detect_gpu_tool()
    if vendor:
        return 'smi', (vendor, path)
    return None, None


def _sample_gpu_unified(source_type, payload):
    """Dispatch to the resolved GPU sampling source. Never raises."""
    if source_type == 'sysfs':
        return _sample_gpu_sysfs(payload)
    if source_type == 'smi':
        vendor, path = payload
        return _sample_gpu(vendor, path)
    return None, None, None, None


# ── Run metadata ingestion (written by parallel_train.py) ─────────────────

def _collect_run_metadata(t0):
    """Read every run_metadata_{split}.json written by this run.

    parallel_train.py writes one file per split at the end of run_split(),
    describing what was actually executed (steps, tasks, solved count, exact
    acceleration config). Files older than the profiled run are ignored so a
    stale file from a previous experiment cannot pollute the summary.

    Returns:
        (list[dict], dict): per-split metadata, and derived aggregates
            (planned steps, solved counts, accel config).
    """
    metadata = []
    for path in sorted(glob.glob('run_metadata_*.json')):
        try:
            if os.path.getmtime(path) < t0 - 1:
                continue
            with open(path) as f:
                metadata.append(json.load(f))
        except Exception:
            continue

    total_steps = 0
    n_tasks = 0
    n_solved = 0
    phase1_s = 0.0
    phase2_s = 0.0
    min_steps = None
    have_solutions = False
    accel_cfg = None
    for m in metadata:
        steps = m.get('n_steps') or 0
        tasks = m.get('n_tasks') or 0
        total_steps += steps * tasks
        n_tasks += tasks
        if steps:
            min_steps = steps if min_steps is None else min(min_steps, steps)
        phase1_s += m.get('phase1_s') or 0.0
        phase2_s += m.get('phase2_s') or 0.0
        if m.get('n_solved') is not None:
            have_solutions = True
            n_solved += m['n_solved']
        accel_cfg = m.get('accel', accel_cfg)

    derived = {
        'planned_train_steps': total_steps,
        'n_tasks': n_tasks,
        'n_solved': n_solved if have_solutions else None,
        'min_n_steps': min_steps,
        'phase1_s': round(phase1_s, 1) if phase1_s else None,
        'phase2_s': round(phase2_s, 1) if phase2_s else None,
        'accel': accel_cfg,
    }
    return metadata, derived


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

# A direct child of parallel_train.py that never exceeds this much CPU is
# infrastructure (the spawn resource_tracker, the per-phase Manager server), not
# a task worker; real workers run at 100-200 %.
TASK_WORKER_CPU_PCT = 50.0


class TreeTracker:
    """Tracks the parent process and its (recursive) children, priming psutil's
    cpu_percent so subsequent reads are accurate, and records per-worker stats.

    Task workers are the *direct* children of parallel_train.py that actually
    burn CPU. Everything deeper is TorchInductor's compile pool, which spawns
    dozens of short-lived processes per task and used to inflate the concurrency
    and throughput metrics by an order of magnitude (149 "workers completed" for
    a 10-task run).
    """

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
                            'direct': proc.ppid() == self.root.pid,
                            'is_task': False,
                        }
            except psutil.Error:
                continue
        # mark workers that disappeared as completed
        for pid, rec in self.workers.items():
            if rec['end'] is None and pid not in live_pids:
                rec['end'] = t_rel
                self.completions.append((t_rel, pid))

    def sample(self):
        """Return (n_task_workers, n_helpers, tree_cpu_pct, tree_rss_mb)."""
        n_workers = 0
        n_helpers = 0
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
            rec = self.workers.get(pid)
            if rec is None or rec['end'] is not None:
                continue
            rec['cpu_samples'].append(cpu)
            rec['peak_rss'] = max(rec['peak_rss'], rss)
            # Sticky: once a direct child has done real work it stays a worker
            # even while it idles waiting on the GPU.
            if rec['direct'] and cpu >= TASK_WORKER_CPU_PCT:
                rec['is_task'] = True
            if rec['is_task']:
                n_workers += 1
            else:
                n_helpers += 1
        return n_workers, n_helpers, tree_cpu, tree_rss


# ── Main run+profile ─────────────────────────────────────────────────────────

def run_and_profile(args, passthrough):
    os.makedirs(args.outdir, exist_ok=True)
    prefix = os.path.join(args.outdir, args.label)
    csv_path = f'{prefix}_timeseries.csv'
    json_path = f'{prefix}_summary.json'

    n_cores = psutil.cpu_count(logical=True)
    gpu_source_type, gpu_payload = _detect_gpu_source(args.gpu_mode)
    gpu_label = {
        'sysfs': 'sysfs(amdgpu)',
        'smi': f'smi({gpu_payload[0] if gpu_payload else "?"})',
    }.get(gpu_source_type, 'off')

    cmd = [sys.executable, '-u', 'parallel_train.py'] + passthrough
    print(f'[profiler] launching: {" ".join(cmd)}')
    print(f'[profiler] cores={n_cores}  gpu={gpu_label}  '
          f'interval={args.interval}s  gpu_interval={args.gpu_interval}s  '
          f'accel={args.accel_preset}')

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
    helpers_series = []
    gpu_util_series = []
    gpu_mem_series = []
    gpu_mem_busy_series = []
    gpu_power_series = []
    gpu_energy_wh = 0.0
    last_gpu_sample_t = -float('inf')

    try:
        while child.poll() is None:
            loop_start = time.time()
            t_rel = loop_start - t0

            tracker.refresh(t_rel)
            percore = psutil.cpu_percent(percpu=True)
            sys_cpu = sum(percore) / len(percore) if percore else 0.0
            n_workers, n_helpers, tree_cpu, tree_rss = tracker.sample()
            vmem = psutil.virtual_memory()

            # GPU sampling runs on its own slow cadence (default 20s), decoupled from
            # the 1s CPU/process loop, to avoid contending with the GPU driver.
            if (gpu_source_type is not None
                    and (t_rel - last_gpu_sample_t) >= args.gpu_interval):
                dt = (t_rel - last_gpu_sample_t) if last_gpu_sample_t > -1e9 else 0.0
                last_gpu_sample_t = t_rel
                gpu_util, gpu_mem, gpu_mem_busy, gpu_power = _sample_gpu_unified(
                    gpu_source_type, gpu_payload)
                # Rectangular integration of board power over the sampling
                # interval -> energy, the basis of the energy-per-task metric.
                if gpu_power is not None and dt > 0:
                    gpu_energy_wh += gpu_power * dt / 3600.0
            else:
                gpu_util = gpu_mem = gpu_mem_busy = gpu_power = None

            rows.append({
                't_rel': round(t_rel, 2),
                'sys_cpu_pct': round(sys_cpu, 1),
                'percore_max_pct': round(max(percore), 1) if percore else 0.0,
                'percore_min_pct': round(min(percore), 1) if percore else 0.0,
                'n_workers': n_workers,
                'n_helper_procs': n_helpers,
                'tree_cpu_pct': round(tree_cpu, 1),
                'tree_rss_mb': round(tree_rss, 1),
                'host_mem_used_pct': vmem.percent,
                'gpu_util_pct': gpu_util,
                'gpu_mem_mb': round(gpu_mem, 1) if gpu_mem is not None else None,
                'gpu_mem_busy_pct': gpu_mem_busy,
                'gpu_power_w': round(gpu_power, 1) if gpu_power is not None else None,
            })
            sys_cpu_series.append(sys_cpu)
            percore_series.append(percore)
            workers_series.append(n_workers)
            helpers_series.append(n_helpers)
            if gpu_util is not None:
                gpu_util_series.append(gpu_util)
            if gpu_mem is not None:
                gpu_mem_series.append(gpu_mem)
            if gpu_mem_busy is not None:
                gpu_mem_busy_series.append(gpu_mem_busy)
            if gpu_power is not None:
                gpu_power_series.append(gpu_power)

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
            'task_worker': rec['is_task'],
            'lifetime_s': round(end - rec['start'], 1),
            'mean_cpu_pct': round(statistics.mean(cpu_samples), 1) if cpu_samples else None,
            'peak_cpu_pct': round(max(cpu_samples), 1) if cpu_samples else None,
            'peak_rss_mb': round(rec['peak_rss'], 1),
        })

    # Only real task workers count: Inductor's compile pool otherwise reports
    # ~150 "completed workers" for a 10-task run.
    completed = [r for r in worker_records
                 if r['task_worker'] and tracker.workers[r['pid']]['end'] is not None]
    worker_lifetimes = [r['lifetime_s'] for r in completed if r['lifetime_s'] > 0]

    # Metadata written by parallel_train.py: lets us report *training* throughput
    # (steps/s) and pass@2 accuracy, not just process counts.
    run_metadata, run_derived = _collect_run_metadata(t0)
    planned_steps = run_derived['planned_train_steps']
    n_solved = run_derived['n_solved']
    n_meta_tasks = run_derived['n_tasks']
    phase2_s = run_derived['phase2_s']

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
            # Inductor compile-pool processes: not tasks, but they consume the
            # host RAM and CPU that cap how many tasks can run at once.
            'mean_helper_procs': round(statistics.mean(helpers_series), 2) if helpers_series else 0,
            'max_helper_procs': max(helpers_series) if helpers_series else 0,
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
            'vendor': gpu_label,
            'util_mean_pct': round(statistics.mean(gpu_util_series), 1) if gpu_util_series else None,
            'util_max_pct': round(max(gpu_util_series), 1) if gpu_util_series else None,
            'mem_max_mb': round(max(gpu_mem_series), 1) if gpu_mem_series else None,
            'mem_busy_mean_pct': round(statistics.mean(gpu_mem_busy_series), 1) if gpu_mem_busy_series else None,
            'power_mean_w': round(statistics.mean(gpu_power_series), 1) if gpu_power_series else None,
            'power_max_w': round(max(gpu_power_series), 1) if gpu_power_series else None,
            'energy_wh': round(gpu_energy_wh, 3) if gpu_power_series else None,
        },
        'efficiency': {
            # Training throughput: planned Phase-2 steps over total wall time.
            # Comparable across A/B runs as long as both use the same split,
            # --demo, --iterations and Phase-1 cache state.
            'planned_train_steps': planned_steps or None,
            'steps_per_s_aggregate': (round(planned_steps / wall, 1)
                                      if planned_steps and wall > 0 else None),
            # Phase-2-only throughput: the metric that actually reflects training
            # speed. The aggregate one is dominated by Phase 1 whenever the
            # measurement phase is expensive, which is exactly when a comparison
            # matters most.
            'phase1_s': run_derived['phase1_s'],
            'phase2_s': phase2_s,
            'steps_per_s_phase2': (round(planned_steps / phase2_s, 2)
                                   if planned_steps and phase2_s else None),
            'energy_wh_per_worker': (round(gpu_energy_wh / len(completed), 4)
                                     if gpu_power_series and completed else None),
            'energy_wh_per_1k_steps': (round(gpu_energy_wh / (planned_steps / 1000.0), 4)
                                       if gpu_power_series and planned_steps else None),
        },
        'accuracy': {
            'n_solved': n_solved,
            'n_tasks': n_meta_tasks or None,
            'min_n_steps': run_derived['min_n_steps'],
            'solved_fraction': (round(n_solved / n_meta_tasks, 4)
                                if n_solved is not None and n_meta_tasks else None),
        },
        'accel_preset': args.accel_preset,
        'run_metadata': run_metadata,
        'workers': worker_records,
    }

    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    _print_summary(summary, csv_path, json_path)
    return 0 if return_code == 0 else 1


def _print_summary(s, csv_path, json_path):
    c, cc, tp, g = s['cpu'], s['concurrency'], s['throughput'], s['gpu']
    eff, acc = s['efficiency'], s['accuracy']
    print('\n' + '=' * 66)
    print(f'  PROFILE SUMMARY — {s["label"]}   (return code {s["return_code"]})')
    print('=' * 66)
    print(f'  Acceleration         : {s["accel_preset"]}')
    print(f'  Wall time            : {s["wall_time_s"]:.1f} s   ({s["wall_time_s"]/3600:.2f} h)')
    print(f'  Host cores           : {s["n_cores"]}')
    print(f'  CPU mean / p95 / max : {c["mean_pct"]} / {c["p95_pct"]} / {c["max_pct"]} %')
    print(f'  CPU saturated (>={c["saturation_threshold_pct"]}%): '
          f'{c["saturation_fraction"]*100:.1f} % of run')
    print(f'  Workers mean / max   : {cc["mean_workers"]} / {cc["max_workers"]}')
    print(f'  Workers completed    : {cc["workers_completed"]}')
    if cc.get('max_helper_procs'):
        print(f'  Compile helper procs : {cc["mean_helper_procs"]} mean / '
              f'{cc["max_helper_procs"]} max  (not counted as workers)')
    print(f'  Throughput           : {tp["workers_per_hour"]} workers/h'
          + (f'   (mean lifetime {tp["mean_worker_lifetime_s"]} s)'
             if tp["mean_worker_lifetime_s"] else ''))
    if eff['steps_per_s_aggregate'] is not None:
        print(f'  Train throughput     : {eff["steps_per_s_aggregate"]} steps/s'
              f'   ({eff["planned_train_steps"]} planned steps)')
    if eff.get('steps_per_s_phase2') is not None:
        print(f'  Phase 2 throughput   : {eff["steps_per_s_phase2"]} steps/s'
              f'   (phase1 {eff["phase1_s"]} s / phase2 {eff["phase2_s"]} s)')
    print(f'  Tree RSS max         : {s["memory"]["tree_rss_max_mb"]:.0f} MB')
    if g['vendor']:
        print(f'  GPU util mean / max  : {g["util_mean_pct"]} / {g["util_max_pct"]} %'
              f'   VRAM max {g["mem_max_mb"]} MB')
        if g['mem_busy_mean_pct'] is not None:
            print(f'  GPU mem busy mean    : {g["mem_busy_mean_pct"]} %')
        if g['power_mean_w'] is not None:
            print(f'  GPU power mean / max : {g["power_mean_w"]} / {g["power_max_w"]} W'
                  f'   energy {g["energy_wh"]} Wh')
    if acc['solved_fraction'] is not None:
        print(f'  Solved (pass@2)      : {acc["n_solved"]}/{acc["n_tasks"]}'
              f'   ({acc["solved_fraction"]*100:.1f} %)')
    print('-' * 66)
    print(f'  time series : {csv_path}')
    print(f'  summary     : {json_path}')
    print('=' * 66 + '\n')


# ── Before/after comparison ──────────────────────────────────────────────────

def compare(before_path, after_path, accuracy_tolerance=0.0, cpu_tolerance_pp=2.0):
    with open(before_path) as f:
        b = json.load(f)
    with open(after_path) as f:
        a = json.load(f)

    def get(d, *path, default=None):
        """Tolerant nested lookup: summaries written by older versions of this
        script simply report the metric as missing instead of crashing."""
        for key in path:
            if not isinstance(d, dict) or key not in d:
                return default
            d = d[key]
        return d

    def line(name, bv, av, better='lower', unit='', pct=False, tolerance=0.0):
        if bv is None or av is None:
            print(f'  {name:<26}: {bv} -> {av}')
            return None
        delta = av - bv
        rel = (delta / bv * 100) if bv else float('inf')
        if delta < 0:
            arrow = '↓'
        elif delta > 0:
            arrow = '↑'
        else:
            arrow = '='
        if better is None:
            # Informational only: the metric has no intrinsic "good" direction.
            good, tag = None, 'info'
        elif tolerance > 0 and abs(delta) <= tolerance:
            good, tag = True, 'OK~'
        else:
            good = (delta <= 0) if better == 'lower' else (delta >= 0)
            tag = 'OK ' if good else 'REGRESSION'
        d = f'{delta:+.1f}{unit}'
        r = f' ({rel:+.1f}%)' if not pct else ''
        print(f'  {name:<26}: {bv}{unit} -> {av}{unit}  {arrow} {d}{r}   [{tag}]')
        return good

    print('\n' + '=' * 66)
    print(f'  COMPARISON   {b["label"]}  ->  {a["label"]}')
    print(f'  accel preset {get(b, "accel_preset", default="?")}'
          f'  ->  {get(a, "accel_preset", default="?")}')
    print('=' * 66)

    print('\n  Throughput / speed (higher is better):')
    line('wall_time_s', b['wall_time_s'], a['wall_time_s'], 'lower', ' s')
    # Phase-2 throughput first: it isolates training speed from the Phase-1
    # measurement, which can dominate the wall clock and invert the verdict.
    line('steps_per_s_phase2',
         get(b, 'efficiency', 'steps_per_s_phase2'),
         get(a, 'efficiency', 'steps_per_s_phase2'), 'higher')
    line('phase1_s', get(b, 'efficiency', 'phase1_s'),
         get(a, 'efficiency', 'phase1_s'), 'lower', ' s')
    line('steps_per_s_aggregate',
         get(b, 'efficiency', 'steps_per_s_aggregate'),
         get(a, 'efficiency', 'steps_per_s_aggregate'), 'higher')
    line('workers_per_hour', b['throughput']['workers_per_hour'],
         a['throughput']['workers_per_hour'], 'higher')
    line('mean_worker_lifetime_s', b['throughput']['mean_worker_lifetime_s'],
         a['throughput']['mean_worker_lifetime_s'], 'lower', ' s')

    print(f'\n  CPU pressure (must NOT get worse; ±{cpu_tolerance_pp} pp is noise):')
    cpu_ok = []
    cpu_ok.append(line('cpu_mean_pct', b['cpu']['mean_pct'], a['cpu']['mean_pct'],
                       'lower', ' %', tolerance=cpu_tolerance_pp))
    cpu_ok.append(line('cpu_saturation_fraction',
                       b['cpu']['saturation_fraction'], a['cpu']['saturation_fraction'],
                       'lower', '', pct=True))

    print('\n  Silicon utilisation (higher is better):')
    line('mean_workers', b['concurrency']['mean_workers'],
         a['concurrency']['mean_workers'], 'higher')
    line('max_helper_procs', get(b, 'concurrency', 'max_helper_procs'),
         get(a, 'concurrency', 'max_helper_procs'), None)
    line('gpu_util_mean_pct', b['gpu']['util_mean_pct'],
         a['gpu']['util_mean_pct'], 'higher', ' %')
    line('gpu_mem_busy_mean_pct', get(b, 'gpu', 'mem_busy_mean_pct'),
         get(a, 'gpu', 'mem_busy_mean_pct'), 'higher', ' %')

    print('\n  Energy efficiency (lower is better; mean power is informational —\n'
          '  drawing more watts is fine if the work per watt-hour improves):')
    line('gpu_power_mean_w', get(b, 'gpu', 'power_mean_w'),
         get(a, 'gpu', 'power_mean_w'), None, ' W')
    line('energy_wh_per_1k_steps', get(b, 'efficiency', 'energy_wh_per_1k_steps'),
         get(a, 'efficiency', 'energy_wh_per_1k_steps'), 'lower', ' Wh')
    line('energy_wh_per_worker', get(b, 'efficiency', 'energy_wh_per_worker'),
         get(a, 'efficiency', 'energy_wh_per_worker'), 'lower', ' Wh')

    print('\n  Accuracy (must NOT get worse):')
    b_solved = get(b, 'accuracy', 'solved_fraction')
    a_solved = get(a, 'accuracy', 'solved_fraction')
    line('solved_fraction', b_solved, a_solved, 'higher', '', pct=True)
    accuracy_regressed = (
        b_solved is not None and a_solved is not None
        and (b_solved - a_solved) > accuracy_tolerance
    )
    if b_solved is None or a_solved is None:
        print('  (no ground-truth solutions in these runs — accuracy not checked)')
    else:
        n_after = get(a, 'accuracy', 'n_tasks')
        min_steps = get(a, 'accuracy', 'min_n_steps')
        if n_after and n_after < 50:
            print(f'  NOTE: only {n_after} tasks — pass@2 is noisy '
                  f'at this sample size; confirm on a larger split before concluding.')
        if min_steps and min_steps < 1000:
            print(f'  NOTE: only {min_steps} iterations/task — the reference setup uses '
                  f'1500-2000. Almost nothing is solved this early, so this run '
                  f'cannot support an accuracy claim either way.')

    print('=' * 66)
    # Guardrail 1: Eje D is semantics-preserving by design, so a drop in pass@2
    # outranks everything else — a `reduce-overhead` run once posted the best
    # throughput of a campaign while solving nothing at all.
    if accuracy_regressed:
        print(f'  ⚠  pass@2 DROPPED by {(b_solved - a_solved)*100:.1f} points '
              f'(tolerance {accuracy_tolerance*100:.1f}) — the speedup is not free.')
        print('=' * 66 + '\n')
        return 3
    # Guardrail 2: the optimization must not increase CPU pressure (Eje H §9.11.3).
    regressed = [ok for ok in cpu_ok if ok is False]
    if regressed:
        print('  ⚠  CPU pressure INCREASED — this change worsens the bottleneck.')
        print('=' * 66 + '\n')
        return 2
    print('  ✓  CPU pressure did not get worse and pass@2 held up.')
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
    parser.add_argument('--gpu-interval', type=float, default=20.0,
                        help='Seconds between GPU samples, decoupled from --interval. Kept '
                             'coarse on purpose: frequent external GPU polling concurrent with '
                             'heavy VRAM churn has been observed to destabilize the GPU driver '
                             'on this host. Default: 20.0')
    parser.add_argument('--gpu-mode', choices=['auto', 'sysfs', 'smi', 'off'], default='auto',
                        help='GPU sampling method. "sysfs" reads /sys/class/drm directly (no '
                             'subprocess, safest); "smi" spawns rocm-smi/nvidia-smi; "auto" '
                             'prefers sysfs and falls back to smi; "off" disables GPU sampling '
                             'entirely. Default: auto')
    parser.add_argument('--accel-preset',
                        choices=['baseline', 'bf16', 'compile', 'full'], default=None,
                        help='Eje D (§9.6) acceleration preset. Appends '
                             '"--accel-preset X" to the parallel_train.py command line and '
                             'records it in the summary, so before/after runs are labelled '
                             'consistently. Use "baseline" for the reference run and '
                             '"bf16"/"compile"/"full" for the accelerated ones. Individual '
                             'flags can still be passed through after `--`.')
    parser.add_argument('--accuracy-tolerance', type=float, default=0.0,
                        help='In --compare mode, how much pass@2 solved_fraction may drop '
                             'before the comparison is flagged as a regression (exit code 3). '
                             'Default: 0.0 (strict).')
    parser.add_argument('--cpu-tolerance-pp', type=float, default=2.0,
                        help='In --compare mode, how many percentage points cpu_mean_pct '
                             'may rise before it counts as a regression. Sub-point moves '
                             'are run-to-run noise. Default: 2.0.')
    parser.add_argument('--compare', nargs=2, metavar=('BEFORE.json', 'AFTER.json'),
                        help='Compare two summary JSONs instead of running a new profile.')
    parser.add_argument('passthrough', nargs=argparse.REMAINDER,
                        help='Args after `--` are forwarded to parallel_train.py.')
    args = parser.parse_args()

    if args.compare:
        sys.exit(compare(args.compare[0], args.compare[1],
                         args.accuracy_tolerance, args.cpu_tolerance_pp))

    # Strip a leading '--' separator from REMAINDER if present.
    passthrough = args.passthrough
    if passthrough and passthrough[0] == '--':
        passthrough = passthrough[1:]

    # --accel-preset is a convenience wrapper over the parallel_train.py flag; an
    # explicit --accel-preset in the passthrough always wins.
    if args.accel_preset and '--accel-preset' not in passthrough:
        passthrough = passthrough + ['--accel-preset', args.accel_preset]
    elif args.accel_preset is None:
        idx = passthrough.index('--accel-preset') if '--accel-preset' in passthrough else None
        args.accel_preset = (passthrough[idx + 1]
                             if idx is not None and idx + 1 < len(passthrough)
                             else 'unspecified')

    if args.interval < 0.5:
        print('[profiler] WARNING: interval < 0.5s adds measurable overhead; '
              'use >= 1.0s to avoid perturbing the host.', file=sys.stderr)

    sys.exit(run_and_profile(args, passthrough))


if __name__ == '__main__':
    main()
