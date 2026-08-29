"""
Parallel ARC-AGI task solver.

Solves as many puzzles as possible simultaneously by saturating GPU VRAM.

Phase 1 — runs 2 iterations on every puzzle to measure its VRAM footprint.
Phase 2 — greedy scheduler packs puzzles onto GPUs under the safe memory budget
           and trains each for 2000 iterations.

After Phase 2 the script saves:
  predictions_{split}.npz      — logger data for list_solved_puzzles.py
  submission_{split}.json      — Kaggle-format predictions
    .log/arc_training_{split}_*.log  — full log (DEBUG to file, INFO to console)
  last_results.txt             — live-updating list of solved tasks

Usage
-----
  python parallel_train.py                    # run training + evaluation + test
  python parallel_train.py --split training   # run one split only
"""

# We run 2 steps of every puzzle to determine how much memory each puzzle uses.
# We run 2000 steps per task at optimal puzzle parallelization under memory constraint.
# We have changed layers.direction_share() to make it run faster, ~5-10% speedup.

import os
import sys
import time
import json
import shutil
import argparse
import multiprocessing
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch

try:
    import psutil
except ImportError:  # the host-RAM guards degrade to no-ops
    psutil = None

import preprocessing
import train
import arc_compressor
import initializers
import multitensor_systems
import layers
import solution_selection
import visualization
import solve_task
import arc_logging
import accel

# ── Global PyTorch settings (must run at import time for the main process) ──
multiprocessing.set_start_method('spawn', force=True)
torch.set_default_dtype(torch.float32)
torch.set_default_device('cuda')
torch.backends.cudnn.benchmark = True
# NOTE (Eje D, §9.2): `torch.backends.cuda.matmul.allow_tf32` used to be set here.
# It is a **no-op on ROCm/RDNA4** (no TF32 hardware). The portable equivalent is
# `torch.set_float32_matmul_precision(...)`, which accel.configure_process()
# applies per worker process according to --matmul-precision (default 'highest',
# i.e. exactly the previous FP32 behaviour).


# ── Live terminal progress line ──────────────────────────────────────────────

def _print_progress_line(task_names, tasks_started, tasks_finished, progress_dict,
                          n_iterations, n_tasks, task_start_times=None):
    """Overwrite a single terminal line with the %-progress of every task
    currently running in parallel. Piggybacks on the scheduler's existing
    1s poll tick, so it adds no extra polling or logging overhead.

    A task shows `init` until it reports its first step: without that, a run
    wedged before training starts looks exactly like one that is merely slow.
    """
    running = [i for i in range(n_tasks) if tasks_started[i] and not tasks_finished[i]]
    if not running:
        return
    now = time.time()
    parts, rates, worst_remaining = [], [], 0
    for i in running:
        name = task_names[i]
        step = int(progress_dict.get(name, -1))
        if step < 0:
            parts.append(f'{name}:init')
            continue
        parts.append(f'{name}:{100.0 * step / max(n_iterations, 1):3.0f}%')
        elapsed = now - task_start_times[i] if task_start_times else 0.0
        if step > 0 and elapsed > 0:
            rates.append(step / elapsed)
            worst_remaining = max(worst_remaining, (n_iterations - step) / (step / elapsed))
    head = f'{len(running)} running'
    if rates:
        head += f', {sum(rates):.1f} it/s, eta {worst_remaining / 60:.0f}m'
    line = f'[{head}] ' + '  '.join(parts)
    width = shutil.get_terminal_size((120, 20)).columns
    sys.stdout.write('\r' + line[:width - 1].ljust(width - 1))
    sys.stdout.flush()


# ── Crash-resumable per-task results ─────────────────────────────────────────

def _safe_task_name(task_name):
    """Reject anything that is not a plain ARC task id before it reaches a path."""
    if not task_name or not all(ch.isalnum() or ch in '-_' for ch in task_name):
        raise ValueError(f'unsafe task name for a file path: {task_name!r}')
    return task_name


def _partial_dir(split):
    return os.path.join('.partial', split)


def save_task_partial(split, task_name, n_steps, solution, logger_data,
                      arc_logger=None):
    """Persist one finished task so an interrupted split can be resumed.

    A 400-task split takes tens of hours and its results otherwise live only in
    the Manager dict, so any failure loses the whole run.
    """
    if not solution:
        return
    try:
        arc_logger.info(f'Saving partial result for {task_name}')
        directory = _partial_dir(split)
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f'{_safe_task_name(task_name)}.json')
        tmp = path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump({'n_steps': n_steps,
                       'solution': solution,
                       'logger': logger_data}, f)
        os.replace(tmp, path)
    except Exception as exc:
        if arc_logger is not None:
            arc_logger.warning(f'Could not save partial result for {task_name}: {exc}')


def load_task_partials(split, task_names, n_steps):
    """Return (solutions, loggers) for tasks already completed at this n_steps."""
    arc_logger.info(f'Loading partial results for split {split}')
    solutions, loggers = {}, {}
    directory = _partial_dir(split)
    if not os.path.isdir(directory):
        return solutions, loggers
    for name in task_names:
        try:
            path = os.path.join(directory, f'{_safe_task_name(name)}.json')
            if not os.path.exists(path):
                continue
            with open(path, 'r') as f:
                payload = json.load(f)
        except Exception:
            continue
        # A --demo 300-iteration partial must never satisfy a 1500-iteration run.
        if payload.get('n_steps') != n_steps or not payload.get('solution'):
            continue
        solutions[name] = payload['solution']
        if payload.get('logger'):
            loggers[name] = payload['logger']
    return solutions, loggers


# A worker needs a couple of minutes of Inductor work to reach its peak RSS, so
# recently launched ones are charged in full instead of trusting `available`.
_HOST_MEM_RAMP_S = 120


def _host_mem_available_gb():
    """Free host RAM right now, or None if psutil is unavailable.

    Deliberately `available` and not `total`: Ubuntu plus whatever else runs on
    the box keeps several GB the workers will never get.
    """
    if psutil is None:
        return None
    try:
        return psutil.virtual_memory().available / 1024**3
    except Exception:
        return None


def _host_memory_cap(n_cpus, gb_per_worker, reserve_gb, arc_logger=None):
    """Lower the concurrency cap so N workers fit in the free host RAM.

    The scheduler packs on VRAM only, but compiled workers peak at 5-9 GB RSS
    against 1.8 GB eager, so host RAM is what actually binds once the VRAM
    over-estimate is removed.
    """
    if not gb_per_worker or gb_per_worker <= 0:
        return n_cpus
    available_gb = _host_mem_available_gb()
    if available_gb is None:
        return n_cpus
    budget_gb = available_gb - max(reserve_gb, 0.0)
    cap = max(1, int(budget_gb / gb_per_worker))
    if arc_logger is not None:
        arc_logger.info(
            f'Host RAM: {available_gb:.0f} GB free − {reserve_gb:.0f} GB reserved '
            f'= {budget_gb:.0f} GB for workers → cap {cap} '
            f'(at {gb_per_worker} GB/worker)'
        )
    return min(n_cpus, cap)


def _host_mem_admits(gb_per_worker, reserve_gb, n_unaccounted):
    """True if launching one more worker still leaves `reserve_gb` free.

    Runtime backstop for an unattended multi-day run: it catches a wrong
    `--host-mem-per-worker-gb`, an unusually large task, or anything else the
    operator starts on the box mid-run.
    """
    if reserve_gb <= 0:
        return True
    available_gb = _host_mem_available_gb()
    if available_gb is None:
        return True
    pending_gb = n_unaccounted * max(gb_per_worker or 0.0, 0.0)
    return available_gb - pending_gb >= reserve_gb


def _cache_free_gb(cache_dir):
    """Return free space on the cache filesystem, or None when disabled."""
    if not cache_dir:
        return None
    return shutil.disk_usage(cache_dir).free / 1024**3


# ── Core scheduler ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ResourceLimits:
    """Host and device limits that keep an unattended multi-day run alive."""

    host_mem_per_worker_gb: float = 0.0
    host_mem_reserve_gb: float = 8.0
    vram_margin_frac: float = 0.0
    stall_timeout_s: float = 1800.0
    cache_min_free_gb: float = 0.0


@dataclass(frozen=True)
class ParallelRunOptions:
    """Optional configuration for :func:`parallelize_runs`."""

    collect_logger_data: bool = False
    track_progress: bool = False
    arc_logger: object = None
    solutions_json: Optional[dict] = None
    task_original_idx: Optional[dict] = None
    n_original_tasks: Optional[int] = None
    quiet: bool = False
    verbose: bool = False
    postprocess_stride: int = 1
    accel_config: Optional[dict] = None
    partial_split: Optional[str] = None
    partial_n_steps: Optional[int] = None
    host_mem_per_worker_gb: float = 0.0
    host_mem_reserve_gb: float = 8.0
    stall_timeout_s: float = 1800.0
    cache_dir: Optional[str] = None
    cache_min_free_gb: float = 0.0

def parallelize_runs(
    gpu_quotas,
    task_usages,
    n_iterations,
    task_names,
    split,
    n_tasks,
    n_gpus,
    n_cpus,
    options=None,
):
    """
    Spawn worker processes to solve ARC-AGI tasks, greedily filling GPU memory.

    Args:
        gpu_quotas (list[float])     : available quota per GPU (bytes or task-slots).
        task_usages (list[float])    : quota consumed by each task.
        n_iterations (int)           : training steps per task.
        task_names (list[str])       : puzzle names in scheduling order.
        split (str)                  : 'training' | 'evaluation' | 'test'.
        n_tasks (int)                : len(task_names).
        n_gpus (int)                 : number of CUDA devices.
        n_cpus (int)                 : CPU core count (caps concurrent processes).
        options (ParallelRunOptions|None): optional logging, progress, persistence,
            acceleration, and host-memory settings.

    Returns:
        memory_dict    (dict[str, int])  : peak VRAM per task (bytes).
        solutions_dict (dict)            : Kaggle-format predictions per task.
        loggers_data   (dict)            : logger data per task (empty if not collected).
        time_taken     (float)           : wall-clock seconds.
    """
    options = options or ParallelRunOptions()
    collect_logger_data = options.collect_logger_data
    track_progress = options.track_progress
    arc_logger = options.arc_logger
    solutions_json = options.solutions_json
    task_original_idx = options.task_original_idx
    n_original_tasks = options.n_original_tasks
    quiet = options.quiet
    verbose = options.verbose
    postprocess_stride = options.postprocess_stride
    accel_config = options.accel_config
    partial_split = options.partial_split
    partial_n_steps = options.partial_n_steps
    host_mem_per_worker_gb = options.host_mem_per_worker_gb
    host_mem_reserve_gb = options.host_mem_reserve_gb
    stall_timeout_s = options.stall_timeout_s
    cache_dir = options.cache_dir
    cache_min_free_gb = options.cache_min_free_gb

    t = time.time()
    gpu_quotas = gpu_quotas[:]
    n_cpus = _host_memory_cap(n_cpus, host_mem_per_worker_gb,
                              host_mem_reserve_gb, arc_logger)
    n_disp = n_original_tasks if n_original_tasks is not None else n_tasks

    tasks_started    = [False] * n_tasks
    tasks_finished   = [False] * n_tasks
    processes        = [None]  * n_tasks
    process_gpu_ids  = [None]  * n_tasks
    task_start_times = [None]  * n_tasks
    task_last_pct    = {}   # task_name → last 10%-bucket logged
    task_last_step   = {}   # task_name → (step, when it last changed)
    recent_launches  = []   # start times of workers still ramping up their RSS
    mem_blocked      = False
    cache_blocked    = False

    with multiprocessing.Manager() as manager:

        # ── Shared inter-process structures ──────────────────────────────
        memory_dict    = manager.dict()
        solutions_dict = manager.dict()
        error_queue    = manager.Queue()
        _loggers_dict  = manager.dict() if collect_logger_data else None
        _progress_dict = manager.dict() if track_progress       else None

        # ── Main monitoring loop ──────────────────────────────────────────
        while not all(tasks_finished):

            # Check for errors propagated from workers
            if not error_queue.empty():
                if arc_logger is not None:
                    arc_logger.close_dashboard()
                raise ValueError(error_queue.get())

            # ── Detect finished tasks ─────────────────────────────────
            for i in range(n_tasks):
                if tasks_started[i] and not tasks_finished[i]:
                    processes[i].join(timeout=0)
                    if not processes[i].is_alive():
                        tasks_finished[i] = True
                        gpu_quotas[process_gpu_ids[i]] += task_usages[i]

                        elapsed  = time.time() - task_start_times[i]
                        peak_mb  = memory_dict.get(task_names[i], 0) / 1024**2
                        orig_idx = (task_original_idx.get(task_names[i], i)
                                    if task_original_idx else i)

                        # Check whether this task was solved
                        solved_info = _check_solved(
                            task_names[i], solutions_json, solutions_dict
                        )

                        if partial_split is not None:
                            save_task_partial(
                                partial_split, task_names[i], partial_n_steps,
                                solutions_dict.get(task_names[i]),
                                (_loggers_dict.get(task_names[i])
                                 if _loggers_dict is not None else None),
                                arc_logger,
                            )

                        if arc_logger is not None:
                            arc_logger.log_task_finished(
                                task_names[i], orig_idx, n_disp,
                                elapsed, peak_mb, solved_info, quiet=quiet,
                                has_ground_truth=solutions_json is not None,
                            )
                            if solved_info is not None:
                                arc_logger.write_solved_result(
                                    task_names[i], orig_idx, solved_info
                                )

                        if verbose and not (
                            arc_logger is not None and arc_logger.dashboard.enabled
                        ):
                            status = f'[SOLVED @{solved_info}]' if solved_info else ''
                            print(task_names[i], 'finished on gpu',
                                  process_gpu_ids[i],
                                  f'quota={gpu_quotas[process_gpu_ids[i]]:.0f}',
                                  status)

            # ── Report training progress for running tasks ────────────
            # File/debug log: throttled to once per 10% bucket per task
            # (cheap, useful for post-run analysis).
            progress_snapshot = {}
            if _progress_dict is not None:
                for i in range(n_tasks):
                    if tasks_started[i] and not tasks_finished[i]:
                        name   = task_names[i]
                        step   = int(_progress_dict.get(name, 0))
                        progress_snapshot[name] = step
                        if arc_logger is None:
                            continue
                        bucket = int(step / max(n_iterations, 1) * 10) * 10
                        if bucket > task_last_pct.get(name, -1):
                            task_last_pct[name] = bucket
                            arc_logger.log_task_progress(name, step, n_iterations)

            # Terminal: a single overwriting status line with every running
            # task's live %, redrawn on the existing 1s tick below — no extra
            # polling, so no measurable performance impact.
            if _progress_dict is not None:
                if arc_logger is not None and arc_logger.dashboard.enabled:
                    arc_logger.render_progress(progress_snapshot, n_iterations)
                elif verbose:
                    _print_progress_line(
                        task_names, tasks_started, tasks_finished,
                        _progress_dict, n_iterations, n_tasks, task_start_times,
                    )

            # ── Stall watchdog ────────────────────────────────────────
            # A wedged worker spins on the GPU forever and takes the whole
            # campaign with it (j_base_50, 2026-08-27). Kill it and move on;
            # with no partial written, --resume retries it next time.
            if _progress_dict is not None and stall_timeout_s > 0:
                now = time.time()
                for i in range(n_tasks):
                    if not tasks_started[i] or tasks_finished[i]:
                        continue
                    name = task_names[i]
                    step = int(_progress_dict.get(name, -1))
                    last = task_last_step.get(name)
                    if last is None or step != last[0]:
                        task_last_step[name] = (step, now)
                        continue
                    if now - last[1] <= stall_timeout_s:
                        continue
                    where = f'step {step}' if step >= 0 else 'before the training loop'
                    if arc_logger is not None:
                        arc_logger.warning(
                            f'{name} stalled {where} for '
                            f'{stall_timeout_s:.0f}s — terminating it'
                        )
                    processes[i].terminate()
                    processes[i].join(timeout=30)
                    if processes[i].is_alive():
                        processes[i].kill()
                    tasks_finished[i] = True
                    gpu_quotas[process_gpu_ids[i]] += task_usages[i]

            # ── Schedule new tasks ────────────────────────────────────
            # One launch per tick: ten workers racing into the HIP allocator in
            # the same instant wedged the driver on a near-full card.
            recent_launches = [ts for ts in recent_launches
                               if time.time() - ts < _HOST_MEM_RAMP_S]
            cache_free_gb = _cache_free_gb(cache_dir)
            if (cache_min_free_gb > 0 and cache_free_gb is not None
                    and cache_free_gb < cache_min_free_gb):
                if not cache_blocked and arc_logger is not None:
                    arc_logger.warning(
                        f'Holding back new tasks: Inductor cache filesystem has '
                        f'{cache_free_gb:.1f} GB free, below the '
                        f'{cache_min_free_gb:.1f} GB minimum'
                    )
                cache_blocked = True
                running = (sum(map(int, tasks_started))
                           - sum(map(int, tasks_finished)))
                if running == 0:
                    raise RuntimeError(
                        f'Inductor cache filesystem below free-space minimum: '
                        f'{cache_free_gb:.1f} GB free < '
                        f'{cache_min_free_gb:.1f} GB required at {cache_dir}'
                    )
                time.sleep(1)
                continue
            if cache_blocked and arc_logger is not None:
                arc_logger.info('Inductor cache space recovered — scheduling resumed')
            cache_blocked = False
            launched = False
            for gpu_id in range(n_gpus):
                if launched:
                    break
                for i in range(n_tasks):
                    if tasks_started[i]:
                        continue
                    running = (sum(map(int, tasks_started))
                               - sum(map(int, tasks_finished)))
                    if running >= n_cpus:
                        break
                    if gpu_quotas[gpu_id] < task_usages[i]:
                        continue
                    if not _host_mem_admits(host_mem_per_worker_gb,
                                            host_mem_reserve_gb,
                                            len(recent_launches) + 1):
                        if not mem_blocked and arc_logger is not None:
                            arc_logger.warning(
                                f'Holding back new tasks: host RAM down to '
                                f'{_host_mem_available_gb():.0f} GB free, '
                                f'reserve is {host_mem_reserve_gb:.0f} GB'
                            )
                        mem_blocked = True
                        break
                    if mem_blocked and arc_logger is not None:
                        arc_logger.info('Host RAM recovered — scheduling resumed')
                    mem_blocked = False
                    gpu_quotas[gpu_id] -= task_usages[i]
                    task_start_times[i] = time.time()

                    orig_idx = (task_original_idx.get(task_names[i], i)
                                if task_original_idx else i)

                    worker_args = (
                        task_names[i], split, 1e20, n_iterations,
                        gpu_id, memory_dict, solutions_dict, error_queue,
                        _loggers_dict, _progress_dict, postprocess_stride,
                        accel_config,
                    )
                    p = multiprocessing.Process(
                        target=solve_task.solve_task, args=worker_args
                    )
                    p.start()
                    processes[i]       = p
                    tasks_started[i]   = True
                    process_gpu_ids[i] = gpu_id
                    recent_launches.append(time.time())
                    launched = True

                    if arc_logger is not None:
                        arc_logger.log_task_started(
                            task_names[i], orig_idx, n_disp, gpu_id,
                            quiet=quiet,
                        )

                    if verbose and not (
                        arc_logger is not None and arc_logger.dashboard.enabled
                    ):
                        print(task_names[i], 'started on gpu', gpu_id,
                            f'quota={gpu_quotas[gpu_id]:.0f}')

            time.sleep(1)

        if (_progress_dict is not None and verbose
            and not (arc_logger is not None and arc_logger.dashboard.enabled)):
            sys.stdout.write('\n')
            sys.stdout.flush()

        # Final error scan
        if not error_queue.empty():
            if arc_logger is not None:
                arc_logger.close_dashboard()
            raise ValueError(error_queue.get())

        # ── Collect results before Manager shuts down ─────────────────
        memory_dict_out    = dict(memory_dict)
        solutions_dict_out = dict(solutions_dict)
        loggers_data = dict(_loggers_dict) if _loggers_dict is not None else {}

    time_taken = time.time() - t
    if arc_logger is not None:
        arc_logger.debug(f'parallelize_runs done in {time_taken:.1f}s')
    if verbose:
        print('All jobs finished in', time_taken, 'seconds.')

    return memory_dict_out, solutions_dict_out, loggers_data, time_taken


# ── Phase 1 memory-measurement cache ────────────────────────────────────────

def _cache_path(split):
    return f'memory_cache_{split}.json'


def _gpu_fingerprint(n_gpus, accel_config=None):
    """Identity of the measurement environment. Includes the accel config because
    BF16 autocast and torch.compile change each task's VRAM footprint (Eje D),
    so a baseline measurement must not be reused for an accelerated run."""
    return {
        'n_gpus':            n_gpus,
        'gpu_names':         [torch.cuda.get_device_name(i) for i in range(n_gpus)],
        'gpu_vram_total_gb': [
            round(torch.cuda.mem_get_info(i)[1] / 1024**3, 2)
            for i in range(n_gpus)
        ],
        'torch_version':     torch.__version__,
        'accel':             accel.AccelConfig.from_dict(accel_config).to_dict(),
    }


def load_memory_cache(split, n_gpus, required_task_names, accel_config=None):
    """Return {task_name: mem_bytes} from disk if the cache matches the
    current GPU fingerprint and covers every required task, else None."""
    path = _cache_path(split)
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            cache = json.load(f)
        if cache.get('fingerprint') != _gpu_fingerprint(n_gpus, accel_config):
            return None
        measurements = cache.get('measurements', {})
        if not all(name in measurements for name in required_task_names):
            return None
        return {name: int(measurements[name]) for name in required_task_names}
    except Exception:
        return None


def save_memory_cache(split, n_gpus, memory_dict, accel_config=None):
    path = _cache_path(split)
    cache = {
        'fingerprint':  _gpu_fingerprint(n_gpus, accel_config),
        'created_at':   time.strftime('%Y-%m-%d %H:%M:%S'),
        'measurements': {k: int(v) for k, v in memory_dict.items()},
    }
    with open(path, 'w') as f:
        json.dump(cache, f, indent=2)


# ── Per-split runner ─────────────────────────────────────────────────────────

def run_split(split, n_gpus, n_cpus, arc_logger, solutions_json, demo_n=None,
              postprocess_stride=4, accel_cfg=None, n_steps=1500,
              compile_memory_factor=1.2, resume=False, limits=None):
    """
    Execute the full two-phase pipeline for one split and save all outputs.

    Args:
        demo_n (int|None): if set, use only the first demo_n tasks (smoke-test mode).
        postprocess_stride (int): Phase 2 only — run full pass@2 postprocessing every
            N steps instead of every step (Eje H, H3). Default 4. Phase 1's 2-step
            memory measurement always uses the Logger default (1) since it's too
            short to matter.
        accel_cfg (accel.AccelConfig|None): Eje D acceleration settings, applied in
            every worker process. None => untouched baseline.
        n_steps (int): Phase 2 training iterations per task.
        compile_memory_factor (float): multiplier applied to the (eager) Phase-1
            measurements when Phase 2 runs compiled. See below.
        resume (bool): reuse per-task results left in .partial/{split}/ by an
            earlier interrupted run at the same n_steps.
        limits (ResourceLimits|None): host-RAM caps, VRAM packing margin and the
            stall watchdog timeout.

    Returns:
        n_solved (int)         : tasks solved (always 0 for 'test').
        n_tasks (int)          : total tasks in this split.
        elapsed_sec (float)    : wall-clock seconds.
        predictions_file (str) : path to the saved .npz file.
    """
    split_start = time.time()
    limits = limits or ResourceLimits()
    accel_cfg = accel.AccelConfig.from_dict(accel_cfg)
    accel_config = accel_cfg.to_dict()
    # Phase 1 never compiles (Eje D): compiling to run 2 iterations cost ~1200 s
    # per task and was 92 % of a profiling run's wall time.
    measure_cfg = accel.for_measurement(accel_cfg)
    measure_config = measure_cfg.to_dict()
    compiled_phase2 = accel_cfg.compile_mode != 'off'
    cache_dir = (
        os.path.abspath(os.environ['TORCHINDUCTOR_CACHE_DIR'])
        if compiled_phase2 and os.environ.get('TORCHINDUCTOR_CACHE_DIR')
        else None
    )
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_free_gb = _cache_free_gb(cache_dir)
        arc_logger.info(
            f'Inductor cache: {cache_dir} ({cache_free_gb:.1f} GB free, '
            f'minimum {limits.cache_min_free_gb:.1f} GB)'
        )
        if (limits.cache_min_free_gb > 0
                and cache_free_gb < limits.cache_min_free_gb):
            raise RuntimeError(
                f'Inductor cache filesystem below free-space minimum: '
                f'{cache_free_gb:.1f} GB free < '
                f'{limits.cache_min_free_gb:.1f} GB required at {cache_dir}'
            )

    # ── Load challenge names in original JSON order ──────────────────
    with open(f'dataset/arc-agi_{split}_challenges.json', 'r') as f:
        problems = json.load(f)
    original_task_names = list(problems.keys())
    del problems

    if demo_n is not None:
        n_total_in_split = len(original_task_names)
        original_task_names = original_task_names[:demo_n]
        arc_logger.warning(
            f'DEMO MODE — running {len(original_task_names)} of '
            f'{n_total_in_split} tasks in the {split} split'
        )

    n_tasks = len(original_task_names)

    arc_logger.log_run_start(n_tasks, n_gpus)
    arc_logger.info(f'Acceleration (Eje D): {accel_cfg.summary()}')

    # ── Phase 1: measure VRAM footprint (2 iterations per task), or load
    #    from a cached measurement keyed to the current GPU fingerprint ──
    gpu_memory_quotas = [torch.cuda.mem_get_info(i)[0] for i in range(n_gpus)]

    cached_memory_dict = load_memory_cache(
        split, n_gpus, original_task_names, measure_config
    )
    if cached_memory_dict is not None:
        t_p1 = 0.0
        arc_logger.log_phase(
            f'Phase 1 — SKIPPED — loaded {len(cached_memory_dict)} task '
            f'measurements from {_cache_path(split)}'
        )
        arc_logger.info(
            f'GPU fingerprint matches cached run. '
            f'Delete {_cache_path(split)} to force re-measurement.'
        )
        memory_dict = cached_memory_dict
    else:
        arc_logger.log_phase(
            f'Phase 1 — Memory measurement  (2 iterations × {n_tasks} tasks)'
        )
        gpu_task_quotas = [1] * n_gpus  # one task at a time → clean individual measurements

        memory_dict, _, _, t_p1 = parallelize_runs(
            gpu_task_quotas,
            [1] * n_tasks,
            2,
            original_task_names,
            split,
            n_tasks, n_gpus, n_cpus,
            ParallelRunOptions(
                arc_logger=arc_logger,
                n_original_tasks=n_tasks,
                quiet=True,  # phase-1 task events → DEBUG only
                verbose=True,
                accel_config=measure_config,
            ),
        )
        arc_logger.info(f'Phase 1 complete in {t_p1:.1f}s')
        save_memory_cache(split, n_gpus, memory_dict, measure_config)
        arc_logger.info(f'Saved Phase 1 measurements to {_cache_path(split)}')

    # Phase 1 measured the *eager* footprint. Compiled tasks were later measured
    # to occupy essentially the same VRAM (~0.95 GB against ~0.89 GB eager: 3
    # resident compiled tasks held 2.84-2.90 GB of whole-GPU memory), so this
    # only carries a small safety margin. The old 4.5x default came from a
    # *compiled* Phase 1 whose reading still included Inductor autotuning
    # buffers, and it throttled Phase 2 to 3 concurrent tasks instead of 10.
    if compiled_phase2 and abs((compile_memory_factor or 1.0) - 1.0) > 1e-9:
        memory_dict = {name: int(mem * compile_memory_factor)
                       for name, mem in memory_dict.items()}
        arc_logger.info(
            f'Compiled Phase 2: scaled eager Phase-1 measurements by '
            f'{compile_memory_factor}x  (mean '
            f'{sum(memory_dict.values()) / max(len(memory_dict), 1) / 1024**2:.0f} MB/task)'
        )

    # Sort tasks by decreasing VRAM so the greedy scheduler fills GPUs tightly
    sorted_tasks      = sorted(memory_dict.items(), key=lambda x: x[1], reverse=True)
    sorted_names      = [name for name, _ in sorted_tasks]
    sorted_mem_usages = [mem  for _, mem  in sorted_tasks]

    # Skip tasks a previous interrupted run already finished at this n_steps.
    resume_solutions, resume_loggers = (
        load_task_partials(split, original_task_names, n_steps) if resume
        else ({}, {})
    )
    if resume_solutions:
        arc_logger.info(
            f'Resume: {len(resume_solutions)}/{n_tasks} tasks already complete in '
            f'{_partial_dir(split)} — Phase 2 will run the remaining '
            f'{n_tasks - len(resume_solutions)}'
        )
        pending = [(name, mem) for name, mem in zip(sorted_names, sorted_mem_usages)
                   if name not in resume_solutions]
        sorted_names      = [name for name, _ in pending]
        sorted_mem_usages = [mem  for _, mem  in pending]

    # Map task_name → original JSON index (used in log messages)
    task_original_idx = {name: idx for idx, name in enumerate(original_task_names)}

    # ── Phase 2: full training (n_steps comes from --iterations) ────────

    # The flat 1 GiB floor is enough on the measured evidence: the same 10 tasks
    # at 300 and at 1500 iterations peaked at 8890 and 9021 MB, so a 2-iteration
    # Phase-1 reading under-reports the steady state by only ~1.5 %.
    # --vram-margin-frac raises the margin proportionally if a bigger split
    # turns out to need it.
    safe_gpu_memory_quotas = [q - max(1 * 1024**3, int(q * limits.vram_margin_frac))
                              for q in gpu_memory_quotas]
    arc_logger.info(
        'Phase 2 VRAM budget: '
        + ', '.join(f'GPU{i}={safe_gpu_memory_quotas[i]/1024**3:.2f} GB '
                    f'of {gpu_memory_quotas[i]/1024**3:.2f} GB free'
                    for i in range(n_gpus))
    )

    arc_logger.log_phase(
        f'Phase 2 — Full training  ({n_steps} iterations × {len(sorted_names)} tasks)'
    )

    if sorted_names:
        _, solutions_dict, loggers_data, t_p2 = parallelize_runs(
            safe_gpu_memory_quotas,
            sorted_mem_usages,
            n_steps,
            sorted_names,
            split,
            len(sorted_names), n_gpus, n_cpus,
            ParallelRunOptions(
                collect_logger_data=True,
                track_progress=True,
                arc_logger=arc_logger,
                solutions_json=solutions_json,
                task_original_idx=task_original_idx,
                n_original_tasks=n_tasks,
                quiet=False,
                verbose=True,
                postprocess_stride=postprocess_stride,
                accel_config=accel_config,
                partial_split=split,
                partial_n_steps=n_steps,
                host_mem_per_worker_gb=limits.host_mem_per_worker_gb,
                host_mem_reserve_gb=limits.host_mem_reserve_gb,
                stall_timeout_s=limits.stall_timeout_s,
                cache_dir=cache_dir,
                cache_min_free_gb=limits.cache_min_free_gb,
            ),
        )
    else:
        solutions_dict, loggers_data, t_p2 = {}, {}, 0.0
    solutions_dict = {**resume_solutions, **solutions_dict}
    loggers_data   = {**resume_loggers,   **loggers_data}
    arc_logger.info(f'Phase 2 complete in {t_p2:.1f}s')

    # ── Save predictions_{split}.npz in original JSON task order ─────
    predictions_file = f'predictions_{split}.npz'
    contrib_logs, picks_histories = [], []
    missing = 0
    for name in original_task_names:
        if name in loggers_data:
            contrib_logs.append(loggers_data[name]['solution_contributions_log'])
            picks_histories.append(loggers_data[name]['solution_picks_history'])
        else:
            arc_logger.warning(f'No logger data for task {name} — empty placeholder inserted')
            contrib_logs.append([])
            picks_histories.append([])
            missing += 1

    np.savez(
        predictions_file,
        solution_contribution_logs=np.array(contrib_logs,    dtype=object),
        solution_picks_histories  =np.array(picks_histories, dtype=object),
    )
    arc_logger.info(
        f'Saved {predictions_file}'
        + (f'  ({missing} tasks with missing data)' if missing else '')
    )

    # ── Save submission_{split}.json (Kaggle format) ─────────────────
    submission_file = f'submission_{split}.json'
    with open(submission_file, 'w') as f:
        json.dump(solutions_dict, f, indent=4)
    arc_logger.info(f'Saved {submission_file}')

    # ── Count solved tasks ────────────────────────────────────────────
    n_solved = 0
    if solutions_json is not None:
        for task_name, pred in solutions_dict.items():
            true_sol = solutions_json.get(task_name)
            if true_sol and pred and _check_all_examples(pred, true_sol) is not None:
                n_solved += 1

    elapsed = time.time() - split_start
    arc_logger.log_run_summary(n_solved, n_tasks, elapsed, predictions_file)
    arc_logger.finalize_results(n_solved, n_tasks, elapsed)

    # ── Save run_metadata_{split}.json ────────────────────────────────
    # Machine-readable description of *what* was run, consumed by
    # profile_parallel_train.py to report throughput-per-step, accuracy and the
    # exact acceleration configuration of each A/B run (Eje D).
    metadata_file = f'run_metadata_{split}.json'
    with open(metadata_file, 'w') as f:
        json.dump({
            'split':              split,
            'timestamp':          time.strftime('%Y-%m-%d %H:%M:%S'),
            'n_tasks':            n_tasks,
            'n_steps':            n_steps,
            'n_solved':           n_solved if solutions_json is not None else None,
            'elapsed_s':          round(elapsed, 1),
            'n_gpus':             n_gpus,
            'max_workers':        n_cpus,
            'postprocess_stride': postprocess_stride,
            # Phase split matters: aggregate wall-clock metrics are meaningless
            # when Phase 1 dominates (a compiled Phase 1 once took 92 % of a run).
            'phase1_s':           round(t_p1, 1),
            'phase2_s':           round(t_p2, 1),
            'resumed_tasks':      len(resume_solutions),
            'limits':             asdict(limits),
            'inductor_cache': {
                'path': cache_dir,
                'free_gb_at_end': (
                    round(_cache_free_gb(cache_dir), 1) if cache_dir else None
                ),
            },
            'compile_memory_factor': (compile_memory_factor if compiled_phase2 else None),
            'accel':              accel_cfg.describe(),
            'accel_measurement':  measure_cfg.describe(),
            'backend':            accel.backend_info(),
        }, f, indent=2)
    arc_logger.info(f'Saved {metadata_file}')

    return n_solved, n_tasks, elapsed, predictions_file


# ── Solved-status helpers ─────────────────────────────────────────────────────

def _check_solved(task_name, solutions_json, solutions_dict):
    """Return 1 if attempt_1 correct, 2 if attempt_2 correct, None otherwise."""
    if solutions_json is None:
        return None
    true_sol = solutions_json.get(task_name)
    if true_sol is None:
        return None
    pred = solutions_dict.get(task_name)
    if not pred:
        return None
    return _check_all_examples(pred, true_sol)


def _check_all_examples(pred, true_sol):
    """Check guess@1 then guess@2 across every test example."""
    try:
        if all(pred[j]['attempt_1'] == true_sol[j] for j in range(len(true_sol))):
            return 1
        if all(pred[j]['attempt_2'] == true_sol[j] for j in range(len(true_sol))):
            return 2
    except (IndexError, KeyError, TypeError):
        pass
    return None


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Parallel ARC-AGI task solver — generates predictions.npz.'
    )
    parser.add_argument(
        '--split',
        type=str,
        choices=['training', 'evaluation', 'test', 'all'],
        default='all',
        help=(
            'Dataset split to train.  '
            '"all" runs training → evaluation → test in sequence.  '
            'Default: all'
        ),
    )
    parser.add_argument(
        '--demo',
        type=int,
        default=None,
        metavar='N',
        help=(
            'Smoke-test mode: run only the first N tasks per split.  '
            'Example: --demo 20 finishes in ~40 min instead of ~20 h.'
        ),
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=None,
        metavar='N',
        help=(
            'Cap on concurrent task processes (Eje H, §9.11.3 step 1). Default: '
            'all logical CPU cores, same as before. Lowering this can increase '
            'aggregate throughput and reduce CPU dispatch contention when the '
            'host is CPU-bound rather than GPU-bound — sweep e.g. 4/8/12/16 with '
            'profile_parallel_train.py to find the sweet spot for this host.'
        ),
    )
    parser.add_argument(
        '--postprocess-stride',
        type=int,
        default=4,
        metavar='K',
        help=(
            'Run full pass@2 candidate postprocessing/scoring every K training '
            'steps instead of every step (Eje H, H3). Reduces GPU->CPU syncs and '
            'Python-side recoloring overhead. Default: 4.'
        ),
    )
    parser.add_argument(

        '--no-tui',
        action='store_true',
        help='Disable the interactive ANSI dashboard and use line-oriented output.',
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=2000,
        metavar='N',
        help=(
            'Phase 2 training steps per task. Lower values make before/after A/B '
            'profiling runs cheap. Default: 2000.'
        ),
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help=(
            'Reuse per-task results left in .partial/{split}/ by an earlier '
            'interrupted run at the same --iterations. Results are always '
            'written there; this flag only controls whether they are consumed. '
            'Delete the directory to force a clean run.'
        ),
    )
    parser.add_argument(
        '--host-mem-per-worker-gb',
        type=float,
        default=0.0,
        metavar='GB',
        help=(
            'Expected peak host RSS per worker; >0 caps Phase-2 concurrency so '
            'the workers fit in the RAM that is actually free. The scheduler '
            'otherwise packs on VRAM only, and compiled workers peak at 5-9 GB '
            'RSS against 1.8 GB eager. Suggested: 6 with a warm Inductor cache, '
            '9 with a cold one. Default: 0 (disabled).'
        ),
    )
    parser.add_argument(
        '--host-mem-reserve-gb',
        type=float,
        default=8.0,
        metavar='GB',
        help=(
            'Host RAM never handed to workers, for the OS and anything else '
            'running on the box. Budgets are computed from free memory minus '
            'this, and no task starts if launching it would eat into it. '
            'Default: 8.0. Use 0 to disable the guard entirely.'
        ),
    )
    parser.add_argument(
        '--vram-margin-frac',
        type=float,
        default=0.0,
        metavar='F',
        help=(
            'Withhold this fraction of free VRAM from the Phase-2 packer, on top '
            'of the 1 GiB floor. Default 0: the floor alone covers the ~1.5 %% by '
            'which a 2-iteration Phase-1 reading under-reports a 1500-step run. '
            'Raise it only if a full split shows VRAM pressure.'
        ),
    )
    parser.add_argument(
        '--task-stall-timeout',
        type=float,
        default=1800.0,
        metavar='S',
        help=(
            'Terminate a worker that reports no progress for this many seconds. '
            'Without it one wedged task hangs an 800-task campaign forever; the '
            'task is simply retried on the next --resume. Default: 1800. '
            'Use 0 to disable.'
        ),
    )
    parser.add_argument(
        '--cache-min-free-gb',
        type=float,
        default=0.0,
        metavar='GB',
        help=(
            'Stop launching new Phase-2 tasks when the filesystem containing '
            'the Inductor cache has less than this much free space. Running '
            'tasks finish and persist before the run exits. Default: 0 '
            '(disabled).'
        ),
    )

    # ── Eje D — computational efficiency / silicon utilisation (§9.6) ──────
    accel_group = parser.add_argument_group(
        'Eje D — acceleration (all OFF by default: baseline behaviour)'
    )
    accel_group.add_argument(
        '--accel-preset',
        choices=sorted(accel.PRESETS),
        default='baseline',
        help=(
            'Bundle of acceleration settings. "baseline" = untouched. "compile" '
            '= torch.compile (Inductor->Triton-ROCm) + host tuning, the only '
            'configuration measured to win at 1500 iterations (-51 %% wall, '
            '+105 %% steps/s, pass@2 unharmed). "full" = compile plus the '
            'expandable_segments allocator. "bf16" = BF16 autocast, kept for '
            'experiments only: it measured 34 %% SLOWER at 1500 iterations '
            'because this workload is launch-bound, not FLOP-bound. Individual '
            'flags below override the preset.'
        ),
    )
    accel_group.add_argument(
        '--amp',
        choices=accel.AMP_CHOICES,
        default=None,
        help=(
            'Mixed precision for the forward pass (§9.6.1). BF16 routes matmuls '
            'to the RDNA4 Matrix cores while the KL and the residual stream stay '
            'in FP32. fp16 is NOT recommended: no loss scaling is implemented.'
        ),
    )
    accel_group.add_argument(
        '--compile',
        dest='compile_mode',
        choices=accel.COMPILE_CHOICES,
        default=None,
        help=(
            'torch.compile mode for model.forward (§9.6.2). Uses dynamic=False '
            'because each worker solves a single task with static shapes. '
            '"reduce-overhead" enables HIP graphs: fastest measured mode, but '
            'its outputs live in a replayed static pool — accel clones them, and '
            'pass@2 must be re-verified before trusting it.'
        ),
    )
    accel_group.add_argument(
        '--matmul-precision',
        choices=accel.MATMUL_PRECISION_CHOICES,
        default=None,
        help=(
            'torch.set_float32_matmul_precision value, the ROCm-correct '
            'replacement for the no-op allow_tf32 (§9.2). Default: highest.'
        ),
    )
    accel_group.add_argument(
        '--threads-per-worker',
        type=int,
        default=None,
        metavar='N',
        help=(
            'torch.set_num_threads per worker process (0 = leave untouched). '
            'With many concurrent workers the default intra-op pools '
            'oversubscribe the host CPU; 1 is usually best.'
        ),
    )
    accel_group.add_argument(
        '--alloc-conf',
        type=str,
        default=None,
        metavar='CONF',
        help=(
            'PYTORCH_HIP_ALLOC_CONF / PYTORCH_CUDA_ALLOC_CONF value, e.g. '
            '"expandable_segments:True" to cut allocator fragmentation and fit '
            'more concurrent tasks in VRAM (§9.6.3).'
        ),
    )
    accel_group.add_argument(
        '--inductor-cache-dir',
        type=str,
        default=None,
        metavar='DIR',
        help='Persistent TorchInductor cache directory, reused across runs.',
    )
    accel_group.add_argument(
        '--compile-threads',
        type=int,
        default=None,
        metavar='N',
        help=(
            'TORCHINDUCTOR_COMPILE_THREADS per worker. Phase 1 no longer '
            'compiles, so Phase 2 can afford a small pool; single-threaded '
            'Inductor was measured at ~1200 s per task.'
        ),
    )
    accel_group.add_argument(
        '--memory-planning',
        dest='memory_planning',
        action='store_true',
        default=None,
        help=(
            'Enable torch._inductor.config.memory_planning (buffer reuse across '
            'the fused graph). Measured to change neither speed nor VRAM on this '
            'workload; toggling it also partitions the Inductor cache key, which '
            'forces a full recompile.'
        ),
    )
    accel_group.add_argument(
        '--no-memory-planning',
        dest='memory_planning',
        action='store_false',
        help='Disable Inductor memory planning (overrides the preset).',
    )
    accel_group.add_argument(
        '--compile-memory-factor',
        type=float,
        default=1.2,
        metavar='F',
        help=(
            'Multiplier applied to the eager Phase-1 VRAM measurements when '
            'Phase 2 runs compiled. Compiled and eager footprints were measured '
            'to be within ~7 %% of each other (~0.95 vs ~0.89 GB/task), so this '
            'is a safety margin, not a correction. Default: 1.2. The former 4.5 '
            'default came from a compiled Phase 1 and throttled Phase 2 to 3 '
            'concurrent tasks instead of 10.'
        ),
    )
    args = parser.parse_args()

    accel_cfg = accel.config_from_preset(
        args.accel_preset,
        amp=args.amp,
        compile_mode=args.compile_mode,
        matmul_precision=args.matmul_precision,
        threads_per_worker=args.threads_per_worker,
        alloc_conf=args.alloc_conf,
        inductor_cache_dir=args.inductor_cache_dir,
        compile_threads=args.compile_threads,
        memory_planning=args.memory_planning,
    )
    # Export the tuned environment before any worker is spawned, so children
    # inherit it (the allocator reads it at first allocation).
    accel.configure_parent(accel_cfg)

    limits = ResourceLimits(
        host_mem_per_worker_gb=args.host_mem_per_worker_gb,
        host_mem_reserve_gb=args.host_mem_reserve_gb,
        vram_margin_frac=args.vram_margin_frac,
        stall_timeout_s=args.task_stall_timeout,
        cache_min_free_gb=args.cache_min_free_gb,
    )

    splits_to_run = (
        ['training', 'evaluation', 'test'] if args.split == 'all'
        else [args.split]
    )

    overall_start = time.time()
    n_cpus = args.max_workers if args.max_workers else multiprocessing.cpu_count()
    n_gpus = torch.cuda.device_count()

    # Initialise the shared results file once for the whole run
    # Use a Windows-safe timestamp for filenames (':' is invalid on NTFS paths).
    current_date_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    print(f'ARC-AGI parallel training started at {current_date_time}')
    results_file = f'results_{current_date_time}.txt'
    arc_logging.init_results_file(results_file)

    print(f'\nStarting ARC-AGI parallel training — splits: {splits_to_run}')
    print(f'GPUs: {n_gpus}   CPU cores: {n_cpus}')
    print(f'Acceleration: {accel_cfg.summary()}\n')

    total_solved = 0
    total_tasks  = 0

    for split in splits_to_run:

        # Load ground-truth solutions (not available for 'test')
        solutions_json = None
        solutions_path = f'dataset/arc-agi_{split}_solutions.json'
        if os.path.exists(solutions_path):
            with open(solutions_path, 'r') as f:
                solutions_json = json.load(f)

        # Per-split logger (.log directory, shared last_results.txt)
        arc_logger = arc_logging.ArcLogger(
            split, log_dir='.', results_file=results_file,
            enable_tui=False if args.no_tui else None,
        )

        try:
            n_solved, n_tasks, elapsed, pred_file = run_split(
                split, n_gpus, n_cpus, arc_logger, solutions_json,
                demo_n=args.demo,
                postprocess_stride=args.postprocess_stride,
                accel_cfg=accel_cfg,
                n_steps=args.iterations,
                compile_memory_factor=args.compile_memory_factor,
                resume=args.resume,
                limits=limits,
            )
        except BaseException:
            arc_logger.close_dashboard()
            raise
        
        total_solved += n_solved
        total_tasks  += n_tasks

        print(
            f'\n[{split}] done — {n_solved}/{n_tasks} solved in '
            f'{elapsed:.1f}s.  Predictions: {pred_file}\n'
        )

    overall_elapsed = time.time() - overall_start
    print(
        f'All splits complete — {total_solved}/{total_tasks} tasks solved  '
        f'in {overall_elapsed:.1f}s ({overall_elapsed / 3600:.2f}h)'
    )

    with open('timing_result.txt', 'w') as f:
        f.write(f'Splits run   : {", ".join(splits_to_run)}\n')
        f.write(f'Total solved : {total_solved}/{total_tasks}\n')
        f.write(f'Total time   : {overall_elapsed:.1f}s\n')
