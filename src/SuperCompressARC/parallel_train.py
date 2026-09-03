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
import queue
import signal
import traceback
from contextlib import contextmanager
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
import task_persistence

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


# Keep these local names for the existing run_split call sites and log messages.
_partial_dir = task_persistence.partial_dir
load_task_partials = task_persistence.load_task_partials


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


class WorkerFailure(RuntimeError):
    """A task worker failed, so the current attempt must be resumed."""

    def __init__(self, message, *, task_name=None, gpu_id=None, exit_code=None,
                 last_step=None, stage=None, compile_mode=None,
                 exception_type=None, recoverable_task=False, seed=0,
                 job_id=None):
        super().__init__(message)
        self.task_name = task_name
        self.gpu_id = gpu_id
        self.exit_code = exit_code
        self.last_step = last_step
        self.stage = stage
        self.compile_mode = compile_mode
        self.exception_type = exception_type
        self.recoverable_task = recoverable_task
        self.seed = seed
        self.job_id = job_id or task_name

    def recovery_record(self):
        return {
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'stage': self.stage,
            'job_id': self.job_id,
            'seed': self.seed,
            'gpu_id': self.gpu_id,
            'exit_code': self.exit_code,
            'last_step': self.last_step,
            'compile_mode': self.compile_mode,
            'exception_type': self.exception_type,
            'message': str(self)[-4000:],
        }


def _drain_worker_errors(error_queue):
    records = []
    while True:
        try:
            records.append(error_queue.get_nowait())
        except queue.Empty:
            return records


def _step_location(step):
    return f'after step {step}' if step is not None and step >= 0 else 'before training'


def _format_reported_worker_error(record):
    if not isinstance(record, dict):
        return f'Worker reported an exception:\n{record}'
    task_name = record.get('task_name', '<unknown>')
    job_id = record.get('job_id', task_name)
    gpu_id = record.get('gpu_id', '?')
    pid = record.get('pid', '?')
    location = _step_location(record.get('last_step'))
    traceback_text = record.get('traceback') or '<no traceback reported>'
    return (
        f'Worker {job_id} on GPU {gpu_id} (pid {pid}) failed {location}:\n'
        f'{traceback_text.rstrip()}'
    )


def _worker_failure_from_report(record):
    message = _format_reported_worker_error(record)
    if not isinstance(record, dict):
        return WorkerFailure(message)
    stage = record.get('stage')
    return WorkerFailure(
        message,
        task_name=record.get('task_name'),
        gpu_id=record.get('gpu_id'),
        exit_code=record.get('exit_code', 1),
        last_step=record.get('last_step'),
        stage=stage,
        compile_mode=record.get('compile_mode'),
        exception_type=record.get('exception_type'),
        recoverable_task=stage in ('training', 'postprocess'),
        seed=record.get('seed', 0),
        job_id=record.get('job_id'),
    )


def _format_worker_exit(task_name, gpu_id, exitcode, last_step):
    location = _step_location(last_step)
    if exitcode is None:
        detail = 'ended without an exit code'
    elif exitcode < 0:
        signum = -exitcode
        try:
            signal_name = signal.Signals(signum).name
        except ValueError:
            signal_name = f'signal {signum}'
        detail = f'was terminated by signal {signum} ({signal_name})'
        if signum == signal.SIGKILL:
            detail += (
                '; no Python traceback is possible. This may indicate the OS '
                'OOM killer or another external kill'
            )
    else:
        detail = f'exited with code {exitcode} without reporting a traceback'
    return f'Worker {task_name} on GPU {gpu_id} {detail} {location}'


def _worker_failure_from_exit(task_name, gpu_id, exitcode, last_step,
                              compile_mode=None, seed=0, job_id=None):
    stage = 'training' if last_step is not None and last_step >= 0 else 'setup'
    return WorkerFailure(
        _format_worker_exit(job_id or task_name, gpu_id, exitcode, last_step),
        task_name=task_name,
        gpu_id=gpu_id,
        exit_code=exitcode,
        last_step=last_step,
        stage=stage,
        compile_mode=compile_mode,
        recoverable_task=stage == 'training',
        seed=seed,
        job_id=job_id,
    )


def _validate_worker_outputs(task_name, memory_dict, solutions_dict,
                             loggers_dict, require_logger):
    missing = []
    if task_name not in memory_dict:
        missing.append('memory measurement')
    if not solutions_dict.get(task_name):
        missing.append('solution')
    if require_logger and not task_persistence.is_complete_logger(
        loggers_dict.get(task_name) if loggers_dict is not None else None
    ):
        missing.append('logger data')
    if missing:
        raise WorkerFailure(
            f'Worker {task_name} exited successfully but did not publish: '
            + ', '.join(missing)
        )


def _stop_process(process, timeout=30):
    if process is None:
        return
    if process.is_alive():
        process.terminate()
        process.join(timeout=timeout)
    if process.is_alive():
        process.kill()
        process.join(timeout=timeout)
    else:
        process.join(timeout=0)


def _terminate_active_processes(processes):
    for process in processes:
        try:
            _stop_process(process)
        except Exception:
            pass


@contextmanager
def _worker_process_guard(processes, arc_logger):
    try:
        yield
    except BaseException:
        _terminate_active_processes(processes)
        if arc_logger is not None:
            arc_logger.close_dashboard()
        raise


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
    task_accel_configs: Optional[dict] = None
    partial_split: Optional[str] = None
    partial_n_steps: Optional[int] = None
    host_mem_per_worker_gb: float = 0.0
    host_mem_reserve_gb: float = 8.0
    stall_timeout_s: float = 1800.0
    cache_dir: Optional[str] = None
    cache_min_free_gb: float = 0.0
    job_specs: Optional[dict] = None
    partial_fingerprint: Optional[str] = None
    state_dir: Optional[str] = None


def _effective_task_accel_config(task_name, default_config, task_configs=None):
    task_configs = task_configs or {}
    selected = task_configs.get(task_name, default_config)
    return accel.AccelConfig.from_dict(selected).to_dict()


def _resolve_job(job_id, job_specs=None):
    if not job_specs:
        return job_id, 0
    spec = job_specs.get(job_id)
    if spec is None:
        raise ValueError(f'missing job specification for {job_id!r}')
    return spec['task_name'], int(spec['seed'])


def _seed_job_id(task_name, seed):
    return f'{task_name}__seed_{seed}'


def _expand_seed_jobs(task_names, task_usages, seeds, completed_job_ids=()):
    completed_job_ids = set(completed_job_ids)
    job_ids = []
    job_usages = []
    job_specs = {}
    for task_name, task_usage in zip(task_names, task_usages):
        for seed in seeds:
            job_id = _seed_job_id(task_name, seed)
            job_specs[job_id] = {'task_name': task_name, 'seed': seed}
            if job_id in completed_job_ids:
                continue
            job_ids.append(job_id)
            job_usages.append(task_usage)
    return job_ids, job_usages, job_specs


def _merge_seed_results(task_names, seeds, job_solutions, job_loggers):
    solutions = {}
    loggers = {}
    missing = []
    for task_name in task_names:
        seed_loggers = {}
        for seed in seeds:
            job_id = _seed_job_id(task_name, seed)
            if not job_solutions.get(job_id) or not task_persistence.is_complete_logger(
                job_loggers.get(job_id)
            ):
                missing.append(job_id)
                continue
            seed_loggers[seed] = job_loggers[job_id]
        if len(seed_loggers) != len(seeds):
            continue
        solution, logger_data = solution_selection.merge_seed_logger_data(
            seed_loggers
        )
        solutions[task_name] = solution
        loggers[task_name] = logger_data
    if missing:
        raise WorkerFailure(
            'Cannot merge incomplete Eje B seed jobs: ' + ', '.join(missing)
        )
    return solutions, loggers

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
    if n_gpus < 1:
        raise ValueError('parallel ARC training requires at least one GPU')
    if n_cpus < 1:
        raise ValueError('parallel ARC training requires at least one worker')

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
    task_accel_configs = options.task_accel_configs
    partial_split = options.partial_split
    partial_n_steps = options.partial_n_steps
    host_mem_per_worker_gb = options.host_mem_per_worker_gb
    host_mem_reserve_gb = options.host_mem_reserve_gb
    stall_timeout_s = options.stall_timeout_s
    cache_dir = options.cache_dir
    cache_min_free_gb = options.cache_min_free_gb
    job_specs = options.job_specs
    partial_fingerprint = options.partial_fingerprint
    state_dir = options.state_dir

    t = time.time()
    gpu_quotas = gpu_quotas[:]
    n_cpus = _host_memory_cap(n_cpus, host_mem_per_worker_gb,
                              host_mem_reserve_gb, arc_logger)
    n_disp = n_original_tasks if n_original_tasks is not None else n_tasks

    tasks_started    = [False] * n_tasks
    tasks_finished   = [False] * n_tasks
    processes        = [None]  * n_tasks
    process_gpu_ids  = [None]  * n_tasks
    process_compile_modes = [None] * n_tasks
    task_start_times = [None]  * n_tasks
    task_last_pct    = {}   # task_name → last 10%-bucket logged
    task_last_step   = {}   # task_name → (step, when it last changed)
    recent_launches  = []   # start times of workers still ramping up their RSS
    mem_blocked      = False
    cache_blocked    = False

    with multiprocessing.Manager() as manager, _worker_process_guard(
        processes, arc_logger,
    ):

        # ── Shared inter-process structures ──────────────────────────────
        memory_dict    = manager.dict()
        solutions_dict = manager.dict()
        error_queue    = manager.Queue()
        _loggers_dict  = manager.dict() if collect_logger_data else None
        _progress_dict = manager.dict() if track_progress       else None

        # ── Main monitoring loop ──────────────────────────────────────────
        while not all(tasks_finished):

            reported_errors = _drain_worker_errors(error_queue)
            if reported_errors:
                raise _worker_failure_from_report(reported_errors[0])

            # ── Detect finished tasks ─────────────────────────────────
            for i in range(n_tasks):
                if tasks_started[i] and not tasks_finished[i]:
                    process = processes[i]
                    process.join(timeout=0)
                    if not process.is_alive():
                        reported_errors = _drain_worker_errors(error_queue)
                        if reported_errors:
                            raise _worker_failure_from_report(reported_errors[0])
                        if process.exitcode != 0:
                            task_name, seed = _resolve_job(
                                task_names[i], job_specs
                            )
                            last_step = (
                                int(_progress_dict.get(task_names[i], -1))
                                if _progress_dict is not None else None
                            )
                            raise _worker_failure_from_exit(
                                task_name, process_gpu_ids[i],
                                process.exitcode, last_step,
                                process_compile_modes[i],
                                seed=seed,
                                job_id=task_names[i],
                            )
                        _validate_worker_outputs(
                            task_names[i], memory_dict, solutions_dict,
                            _loggers_dict, collect_logger_data,
                        )
                        tasks_finished[i] = True
                        gpu_quotas[process_gpu_ids[i]] += task_usages[i]

                        elapsed  = time.time() - task_start_times[i]
                        peak_mb  = memory_dict.get(task_names[i], 0) / 1024**2
                        task_name, seed = _resolve_job(task_names[i], job_specs)
                        orig_idx = (task_original_idx.get(task_name, i)
                                    if task_original_idx else i)

                        # Check whether this task was solved
                        solved_info = _check_solved(
                            task_name,
                            solutions_json,
                            {task_name: solutions_dict[task_names[i]]},
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
            # campaign with it. Abort this attempt so the wrapper can resume it.
            if _progress_dict is not None and stall_timeout_s > 0:
                now = time.time()
                for i in range(n_tasks):
                    if not tasks_started[i] or tasks_finished[i]:
                        continue
                    name = task_names[i]
                    task_name, seed = _resolve_job(name, job_specs)
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
                            f'{stall_timeout_s:.0f}s — aborting this attempt'
                        )
                    _stop_process(processes[i])
                    raise WorkerFailure(
                        f'Worker {name} on GPU {process_gpu_ids[i]} stalled '
                        f'{where} for {stall_timeout_s:.0f}s and was terminated; '
                        f'the task will be retried from its last completed partial',
                        task_name=task_name,
                        gpu_id=process_gpu_ids[i],
                        exit_code=None,
                        last_step=step,
                        stage='training',
                        compile_mode=process_compile_modes[i],
                        recoverable_task=True,
                        seed=seed,
                        job_id=name,
                    )

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

                    job_id = task_names[i]
                    task_name, seed = _resolve_job(job_id, job_specs)
                    orig_idx = (task_original_idx.get(task_name, i)
                                if task_original_idx else i)
                    task_accel_config = _effective_task_accel_config(
                        task_name, accel_config, task_accel_configs,
                    )

                    worker_args = (
                        task_name, split, 1e20, n_iterations,
                        gpu_id, memory_dict, solutions_dict, error_queue,
                        _loggers_dict, _progress_dict, postprocess_stride,
                        task_accel_config, partial_split, partial_n_steps,
                        seed, job_id, partial_fingerprint, state_dir,
                    )
                    p = multiprocessing.Process(
                        target=solve_task.solve_task, args=worker_args
                    )
                    p.start()
                    processes[i]       = p
                    tasks_started[i]   = True
                    process_gpu_ids[i] = gpu_id
                    process_compile_modes[i] = task_accel_config['compile_mode']
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

        reported_errors = _drain_worker_errors(error_queue)
        if reported_errors:
            raise _worker_failure_from_report(reported_errors[0])

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
    fingerprint = _gpu_fingerprint(n_gpus, accel_config)
    measurements = {}
    try:
        with open(path, 'r') as f:
            existing = json.load(f)
        if existing.get('fingerprint') == fingerprint:
            measurements.update(existing.get('measurements', {}))
    except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    measurements.update({k: int(v) for k, v in memory_dict.items()})
    cache = {
        'fingerprint':  fingerprint,
        'created_at':   time.strftime('%Y-%m-%d %H:%M:%S'),
        'measurements': measurements,
    }
    with open(path, 'w') as f:
        json.dump(cache, f, indent=2)


# ── Per-split runner ─────────────────────────────────────────────────────────


def _validate_recovery_task_ids(task_names, eager_tasks, retry_quarantined):
    known = set(task_names)
    unknown = (set(eager_tasks) | set(retry_quarantined)) - known
    if unknown:
        raise ValueError(
            'unknown recovery task id(s): ' + ', '.join(sorted(unknown))
        )


def _parse_task_id_list(value):
    if not value:
        return set()
    task_ids = set()
    for raw_task_id in value.split(','):
        task_id = raw_task_id.strip()
        if not task_id:
            continue
        try:
            task_ids.add(task_persistence.safe_task_name(task_id))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc
    return task_ids


def _parse_seed_list(value):
    seeds = []
    for item in value.split(','):
        item = item.strip()
        if not item:
            continue
        try:
            seed = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f'invalid seed {item!r}; expected comma-separated integers'
            ) from exc
        if seed < 0:
            raise argparse.ArgumentTypeError('seeds must be non-negative')
        seeds.append(seed)
    if not seeds:
        raise argparse.ArgumentTypeError('at least one seed is required')
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError('seeds must not contain duplicates')
    return tuple(seeds)


def _recovery_task_sets(task_names, recovery_entries, eager_tasks,
                        retry_quarantined):
    """Resolve durable and operator-requested task recovery policy."""
    _validate_recovery_task_ids(task_names, eager_tasks, retry_quarantined)
    quarantined = {
        name for name, entry in recovery_entries.items()
        if name in task_names and entry.get('state') == 'quarantined'
    }
    invalid_retries = set(retry_quarantined) - quarantined
    if invalid_retries:
        raise ValueError(
            'task id(s) are not quarantined: '
            + ', '.join(sorted(invalid_retries))
        )
    automatic_eager = {
        name for name, entry in recovery_entries.items()
        if name in task_names
        and entry.get('state') in ('retry_eager', 'recovered_eager')
    }
    effective_eager = (
        set(eager_tasks) | automatic_eager | set(retry_quarantined)
    )
    skipped = quarantined - set(retry_quarantined)
    effective_eager -= skipped
    return effective_eager, skipped


def _task_eager_config(accel_config):
    return {**accel.AccelConfig.from_dict(accel_config).to_dict(),
            'compile_mode': 'off'}


def _activate_quarantined_retries(split, n_steps, task_names,
                                  recovery_entries, arc_logger=None):
    """Make an operator-requested eager retry durable across attempts."""
    for task_name in task_names:
        entry = task_persistence.update_task_recovery(
            split, task_name, n_steps, 'retry_eager',
        )
        recovery_entries[task_name] = entry
    if task_names and arc_logger is not None:
        arc_logger.info(
            'Reactivated quarantined tasks for eager retry: '
            + ', '.join(sorted(task_names))
        )


def _record_task_recovery_failure(split, n_steps, failure, enabled, arc_logger):
    """Persist the next recovery state, returning whether it was handled."""
    if (
        not enabled
        or not isinstance(failure, WorkerFailure)
        or not failure.recoverable_task
        or not failure.task_name
        or failure.compile_mode not in accel.COMPILE_CHOICES
    ):
        return False
    state = (
        'quarantined' if failure.compile_mode == 'off'
        else 'retry_eager'
    )
    task_persistence.update_task_recovery(
        split, failure.task_name, n_steps, state,
        failure.recovery_record(),
    )
    if arc_logger is not None:
        if state == 'retry_eager':
            action = 'will retry eagerly in a fresh attempt'
        else:
            action = 'quarantined after an eager failure'
        arc_logger.warning(f'{failure.task_name}: {action}')
    return True


def _quarantined_fallback(n_test):
    """Return the solver's deterministic initial guess for a skipped task."""
    return [
        {
            'attempt_1': [[0, 0], [0, 0]],
            'attempt_2': [[0, 0], [0, 0]],
        }
        for _ in range(n_test)
    ]


def _apply_quarantined_fallbacks(solutions, loggers, task_names,
                                 task_test_counts):
    for task_name in task_names:
        solutions[task_name] = _quarantined_fallback(
            task_test_counts[task_name]
        )
        loggers[task_name] = {
            'solution_contributions_log': [],
            'solution_picks_history': [],
        }


def run_split(split, n_gpus, n_cpus, arc_logger, solutions_json, demo_n=None,
              postprocess_stride=4, accel_cfg=None, n_steps=1500,
              compile_memory_factor=1.2, resume=False, limits=None,
              recover_task_failures=False, eager_tasks=None,
              retry_quarantined=None, task_ids=None, output_dir='.',
              state_dir=None):
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
        recover_task_failures (bool): persist task-level compiled-to-eager and
            eager-to-quarantine transitions before aborting a failed attempt.
        eager_tasks (set[str]|None): task ids forced to use eager forward mode.
        retry_quarantined (set[str]|None): quarantined task ids to retry eagerly.

    Returns:
        n_solved (int)         : tasks solved (always 0 for 'test').
        n_tasks (int)          : total tasks in this split.
        elapsed_sec (float)    : wall-clock seconds.
        predictions_file (str) : path to the saved .npz file.
    """
    split_start = time.time()
    os.makedirs(output_dir, exist_ok=True)
    limits = limits or ResourceLimits()
    eager_tasks = set(eager_tasks or ())
    retry_quarantined = set(retry_quarantined or ())
    accel_cfg = accel.AccelConfig.from_dict(accel_cfg)
    accel_config = accel_cfg.to_dict()
    partial_fingerprint = accel_cfg.algorithm_fingerprint(
        n_steps, postprocess_stride
    )
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
    task_test_counts = {
        name: len(problem.get('test', ()))
        for name, problem in problems.items()
    }
    del problems

    if task_ids is not None:
        task_ids = set(task_ids)
        unknown = task_ids - set(original_task_names)
        if unknown:
            raise ValueError(
                'unknown task id(s) for split '
                f'{split}: {", ".join(sorted(unknown))}'
            )
        original_task_names = [
            name for name in original_task_names if name in task_ids
        ]

    if demo_n is not None:
        n_total_in_split = len(original_task_names)
        original_task_names = original_task_names[:demo_n]
        arc_logger.warning(
            f'DEMO MODE — running {len(original_task_names)} of '
            f'{n_total_in_split} tasks in the {split} split'
        )

    n_tasks = len(original_task_names)
    recovery_entries = task_persistence.load_task_recovery(split, n_steps)
    effective_eager_tasks, quarantined_tasks = _recovery_task_sets(
        original_task_names, recovery_entries, eager_tasks, retry_quarantined,
    )
    _activate_quarantined_retries(
        split, n_steps, retry_quarantined, recovery_entries, arc_logger,
    )
    resume_job_solutions = {}
    resume_job_loggers = {}
    if resume and accel_cfg.eje_b:
        arc_logger.info(f'Loading Eje B seed partials for split {split}')
        seed_solutions, seed_loggers = task_persistence.load_seed_partials(
            split,
            original_task_names,
            n_steps,
            accel_cfg.seeds,
            partial_fingerprint,
            state_dir=state_dir,
        )
        resume_job_solutions = {
            _seed_job_id(task_name, seed): solution
            for (task_name, seed), solution in seed_solutions.items()
        }
        resume_job_loggers = {
            _seed_job_id(task_name, seed): logger_data
            for (task_name, seed), logger_data in seed_loggers.items()
        }
        resume_solutions, resume_loggers = {}, {}
    elif resume:
        arc_logger.info(f'Loading partial results for split {split}')
        resume_solutions, resume_loggers = load_task_partials(
            split, original_task_names, n_steps,
        )
    else:
        resume_solutions, resume_loggers = {}, {}
    resolved_quarantines = quarantined_tasks & set(resume_solutions)
    for task_name in resolved_quarantines:
        task_persistence.update_task_recovery(
            split, task_name, n_steps, 'recovered_eager',
        )
    quarantined_tasks -= resolved_quarantines
    runnable_task_names = [
        name for name in original_task_names if name not in quarantined_tasks
    ]
    if effective_eager_tasks:
        if accel_cfg.eje_b:
            raise ValueError(
                'Eje B requires compile for every seed job; remove eager '
                'recovery state before running this task set'
            )
        arc_logger.info(
            'Task-specific eager mode: '
            + ', '.join(sorted(effective_eager_tasks))
        )
    if quarantined_tasks:
        arc_logger.warning(
            'Skipping quarantined tasks: '
            + ', '.join(sorted(quarantined_tasks))
        )

    arc_logger.log_run_start(n_tasks, n_gpus)
    arc_logger.info(f'Acceleration (Eje D): {accel_cfg.summary()}')

    # ── Phase 1: measure VRAM footprint (2 iterations per task), or load
    #    from a cached measurement keyed to the current GPU fingerprint ──
    gpu_memory_quotas = [torch.cuda.mem_get_info(i)[0] for i in range(n_gpus)]

    cached_memory_dict = load_memory_cache(
        split, n_gpus, runnable_task_names, measure_config
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
            f'Phase 1 — Memory measurement  '
            f'(2 iterations × {len(runnable_task_names)} tasks)'
        )
        gpu_task_quotas = [1] * n_gpus  # one task at a time → clean individual measurements

        memory_dict, _, _, t_p1 = parallelize_runs(
            gpu_task_quotas,
            [1] * len(runnable_task_names),
            2,
            runnable_task_names,
            split,
            len(runnable_task_names), n_gpus, n_cpus,
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

    # Skip tasks a previous interrupted baseline run already finished at this
    # n_steps. Eje B resumes individual seed jobs after expansion below.
    if resume_solutions and not accel_cfg.eje_b:
        remaining = n_tasks - len(resume_solutions) - len(quarantined_tasks)
        arc_logger.info(
            f'Resume: {len(resume_solutions)}/{n_tasks} tasks already complete in '
            f'{_partial_dir(split)} — Phase 2 will run the remaining '
            f'{remaining}'
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

    phase2_names = sorted_names
    phase2_usages = sorted_mem_usages
    job_specs = None
    if accel_cfg.eje_b:
        phase2_names, phase2_usages, job_specs = _expand_seed_jobs(
            sorted_names,
            sorted_mem_usages,
            accel_cfg.seeds,
            resume_job_solutions,
        )
        arc_logger.info(
            f'Eje B seed jobs: {len(resume_job_solutions)} resumed, '
            f'{len(phase2_names)} pending'
        )

    arc_logger.log_phase(
        f'Phase 2 — Full training  '
        f'({n_steps} iterations × {len(phase2_names)} jobs)'
    )

    if phase2_names:
        eager_config = (
            _task_eager_config(accel_config)
            if effective_eager_tasks else None
        )
        task_accel_configs = {
            name: eager_config for name in effective_eager_tasks
            if name in sorted_names and eager_config is not None
        }
        try:
            _, phase2_solutions, phase2_loggers, t_p2 = parallelize_runs(
                safe_gpu_memory_quotas,
                phase2_usages,
                n_steps,
                phase2_names,
                split,
                len(phase2_names), n_gpus, n_cpus,
                ParallelRunOptions(
                    collect_logger_data=True,
                    track_progress=True,
                    arc_logger=arc_logger,
                    solutions_json=(None if accel_cfg.eje_b else solutions_json),
                    task_original_idx=task_original_idx,
                    n_original_tasks=n_tasks,
                    quiet=False,
                    verbose=True,
                    postprocess_stride=postprocess_stride,
                    accel_config=accel_config,
                    task_accel_configs=task_accel_configs,
                    partial_split=split,
                    partial_n_steps=n_steps,
                    host_mem_per_worker_gb=limits.host_mem_per_worker_gb,
                    host_mem_reserve_gb=limits.host_mem_reserve_gb,
                    stall_timeout_s=limits.stall_timeout_s,
                    cache_dir=cache_dir,
                    cache_min_free_gb=limits.cache_min_free_gb,
                    job_specs=job_specs,
                    partial_fingerprint=(
                        partial_fingerprint if accel_cfg.eje_b else None
                    ),
                    state_dir=state_dir,
                ),
            )
        except WorkerFailure as failure:
            if accel_cfg.eje_b:
                arc_logger.warning(
                    f'{failure.job_id}: Eje B is compile-only; refusing '
                    'compiled-to-eager recovery'
                )
            else:
                _record_task_recovery_failure(
                    split, n_steps, failure, recover_task_failures, arc_logger,
                )
            raise
    else:
        phase2_solutions, phase2_loggers, t_p2 = {}, {}, 0.0

    trained_seed_jobs = len(phase2_solutions)
    seed_solved_task_ids = {}
    if accel_cfg.eje_b:
        all_job_solutions = {
            **resume_job_solutions,
            **phase2_solutions,
        }
        all_job_loggers = {
            **resume_job_loggers,
            **phase2_loggers,
        }
        solutions_dict, loggers_data = _merge_seed_results(
            runnable_task_names,
            accel_cfg.seeds,
            all_job_solutions,
            all_job_loggers,
        )
        if solutions_json is not None:
            for seed in accel_cfg.seeds:
                solved_ids = []
                for task_name in runnable_task_names:
                    job_id = _seed_job_id(task_name, seed)
                    if _check_all_examples(
                        all_job_solutions[job_id], solutions_json[task_name]
                    ) is not None:
                        solved_ids.append(task_name)
                seed_solved_task_ids[str(seed)] = solved_ids
        trained_this_attempt = len({
            job_specs[job_id]['task_name']
            for job_id in phase2_solutions
        }) if job_specs else 0
    else:
        trained_this_attempt = len(phase2_solutions)
        solutions_dict = {**resume_solutions, **phase2_solutions}
        loggers_data = {**resume_loggers, **phase2_loggers}
    arc_logger.info(f'Phase 2 complete in {t_p2:.1f}s')

    recovered_eager_tasks = {
        name for name in effective_eager_tasks
        if solutions_dict.get(name)
    }
    for task_name in recovered_eager_tasks:
        if recovery_entries.get(task_name, {}).get('state') != 'recovered_eager':
            task_persistence.update_task_recovery(
                split, task_name, n_steps, 'recovered_eager',
            )

    fallback_quarantined = {
        name for name in quarantined_tasks if not solutions_dict.get(name)
    }
    _apply_quarantined_fallbacks(
        solutions_dict, loggers_data, fallback_quarantined, task_test_counts,
    )
    if fallback_quarantined:
        arc_logger.warning(
            f'{n_tasks - len(fallback_quarantined)} tasks have real results; '
            f'{len(fallback_quarantined)} quarantined task(s) use the '
            f'deterministic 2x2-zero fallback: '
            + ', '.join(sorted(fallback_quarantined))
        )

    missing_solutions = [
        name for name in original_task_names if not solutions_dict.get(name)
    ]
    missing_loggers = [
        name for name in original_task_names
        if not task_persistence.is_complete_logger(loggers_data.get(name))
    ]
    if missing_solutions or missing_loggers:
        details = []
        if missing_solutions:
            details.append(f'{len(missing_solutions)} missing solutions')
        if missing_loggers:
            details.append(f'{len(missing_loggers)} missing logger records')
        raise WorkerFailure(
            'Refusing to write incomplete split outputs: ' + ', '.join(details)
        )
    solutions_dict = {
        name: solutions_dict[name] for name in original_task_names
    }

    # ── Save predictions_{split}.npz in original JSON task order ─────
    predictions_file = os.path.join(output_dir, f'predictions_{split}.npz')
    contrib_logs, picks_histories, eje_b_diagnostics = [], [], []
    missing = 0
    for name in original_task_names:
        if name in loggers_data:
            contrib_logs.append(loggers_data[name]['solution_contributions_log'])
            picks_histories.append(loggers_data[name]['solution_picks_history'])
            eje_b_diagnostics.append(
                loggers_data[name].get('seed_loggers')
                or loggers_data[name].get('eje_b_diagnostics')
            )
        else:
            arc_logger.warning(f'No logger data for task {name} — empty placeholder inserted')
            contrib_logs.append([])
            picks_histories.append([])
            eje_b_diagnostics.append(None)
            missing += 1

    np.savez(
        predictions_file,
        solution_contribution_logs=np.array(contrib_logs,    dtype=object),
        solution_picks_histories  =np.array(picks_histories, dtype=object),
        eje_b_diagnostics=np.array(eje_b_diagnostics, dtype=object),
    )
    arc_logger.info(
        f'Saved {predictions_file}'
        + (f'  ({missing} tasks with missing data)' if missing else '')
    )

    # ── Save submission_{split}.json (Kaggle format) ─────────────────
    submission_file = os.path.join(output_dir, f'submission_{split}.json')
    with open(submission_file, 'w') as f:
        json.dump(solutions_dict, f, indent=4)
    arc_logger.info(f'Saved {submission_file}')

    # ── Count solved tasks ────────────────────────────────────────────
    n_solved = _count_solved_tasks(
        solutions_dict, solutions_json, fallback_quarantined,
    )

    elapsed = time.time() - split_start
    arc_logger.log_run_summary(n_solved, n_tasks, elapsed, predictions_file)
    arc_logger.finalize_results(n_solved, n_tasks, elapsed)

    # ── Save run_metadata_{split}.json ────────────────────────────────
    # Machine-readable description of *what* was run, consumed by
    # profile_parallel_train.py to report throughput-per-step, accuracy and the
    # exact acceleration configuration of each A/B run (Eje D).
    metadata_file = os.path.join(output_dir, f'run_metadata_{split}.json')
    with open(metadata_file, 'w') as f:
        json.dump({
            'split':              split,
            'timestamp':          time.strftime('%Y-%m-%d %H:%M:%S'),
            'n_tasks':            n_tasks,
            'task_ids':           original_task_names,
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
            'trained_this_attempt': trained_this_attempt,
            'trained_seed_jobs':   trained_seed_jobs,
            'resumed_seed_jobs':   len(resume_job_solutions),
            'total_seed_jobs':     n_tasks * len(accel_cfg.seeds),
            'total_optimizer_steps': n_tasks * len(accel_cfg.seeds) * n_steps,
            'optimizer_steps_this_attempt': trained_seed_jobs * n_steps,
            'seeds':               list(accel_cfg.seeds),
            'eje_b_fingerprint':   partial_fingerprint,
            'seed_solved_task_ids': seed_solved_task_ids,
            'output_dir':          os.path.abspath(output_dir),
            'state_dir':           (
                os.path.abspath(state_dir) if state_dir else None
            ),
            'real_result_tasks':  n_tasks - len(fallback_quarantined),
            'degraded':           bool(fallback_quarantined),
            'recovery': {
                'enabled': recover_task_failures,
                'manifest': task_persistence.recovery_path(split),
                'explicit_eager_tasks': sorted(eager_tasks),
                'effective_eager_tasks': sorted(effective_eager_tasks),
                'recovered_eager_tasks': sorted(recovered_eager_tasks),
                'quarantined_tasks': sorted(fallback_quarantined),
                'fallback_strategy': (
                    '2x2_zero_initial_guess' if fallback_quarantined else None
                ),
            },
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


def _count_solved_tasks(solutions_dict, solutions_json, excluded_tasks=()):
    if solutions_json is None:
        return 0
    excluded_tasks = set(excluded_tasks)
    return sum(
        1 for task_name, pred in solutions_dict.items()
        if task_name not in excluded_tasks
        and solutions_json.get(task_name)
        and pred
        and _check_all_examples(pred, solutions_json[task_name]) is not None
    )


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
        '--task-ids',
        type=_parse_task_id_list,
        default=None,
        metavar='ID1,ID2,...',
        help=(
            'Run only these comma-separated task IDs, preserving dataset order. '
            'Mutually exclusive with --demo; intended for reproducible '
            'regression panels.'
        ),
    )
    parser.add_argument(
        '--output-dir',
        default='.',
        metavar='DIR',
        help=(
            'Directory for submission, predictions and run metadata. '
            'Default: repository root.'
        ),
    )
    parser.add_argument(
        '--state-dir',
        default=None,
        metavar='DIR',
        help=(
            'Root directory for fingerprinted Eje B seed partials. Default: '
            'the existing .partial directory.'
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
        '--recover-task-failures',
        action='store_true',
        help=(
            'Persist recoverable task failures across attempts. A compiled '
            'task is retried eagerly in a fresh worker; an eager task that '
            'also fails is quarantined so the remaining split can finish.'
        ),
    )
    parser.add_argument(
        '--eager-tasks',
        type=_parse_task_id_list,
        default=set(),
        metavar='TASK_IDS',
        help=(
            'Comma-separated task ids that must run without torch.compile. '
            'Other tasks retain the selected acceleration preset.'
        ),
    )
    parser.add_argument(
        '--retry-quarantined',
        type=_parse_task_id_list,
        default=set(),
        metavar='TASK_IDS',
        help=(
            'Comma-separated quarantined task ids to retry eagerly. Use for '
            'one deliberate recovery attempt after investigating the failure.'
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
    accel_group.add_argument(
        '--eje-b',
        action='store_true',
        help=(
            'Enable Eje B robustness: scheduled per-leaf free bits, '
            'hard-example curriculum, and multi-seed evidence fusion. This is '
            'only valid with --accel-preset compile.'
        ),
    )
    accel_group.add_argument(
        '--seeds',
        type=_parse_seed_list,
        default=None,
        metavar='S0,S1,...',
        help='Independent Eje B seeds. Default with --eje-b: 0,1,2,3.',
    )
    accel_group.add_argument(
        '--kl-free-bits-initial',
        type=float,
        default=None,
        metavar='NATS',
        help=(
            'Initial per-leaf free-bits threshold, decayed linearly to zero. '
            'Default with --eje-b: 2.0.'
        ),
    )
    accel_group.add_argument(
        '--curriculum',
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            'Enable hard-example demonstration weighting. Enabled by default '
            'with --eje-b; use --no-curriculum for ablation.'
        ),
    )
    accel_group.add_argument(
        '--curriculum-beta-max',
        type=float,
        default=None,
        metavar='BETA',
        help='Final curriculum softmax strength. Default: 1.0.',
    )
    accel_group.add_argument(
        '--curriculum-ema-decay',
        type=float,
        default=None,
        metavar='DECAY',
        help='EMA decay for historical per-example difficulty. Default: 0.9.',
    )
    args = parser.parse_args()

    if args.demo is not None and args.task_ids is not None:
        parser.error('--demo and --task-ids are mutually exclusive')
    if args.eje_b and args.accel_preset != 'compile':
        parser.error('Eje B must run with --accel-preset compile')

    seeds = args.seeds if args.seeds is not None else (
        (0, 1, 2, 3) if args.eje_b else (0,)
    )
    kl_free_bits_initial = (
        args.kl_free_bits_initial
        if args.kl_free_bits_initial is not None
        else (2.0 if args.eje_b else 0.0)
    )
    curriculum = (
        args.curriculum
        if args.curriculum is not None
        else args.eje_b
    )

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
        eje_b=args.eje_b,
        seeds=seeds,
        kl_free_bits_initial=kl_free_bits_initial,
        curriculum=curriculum,
        curriculum_beta_max=args.curriculum_beta_max,
        curriculum_ema_decay=args.curriculum_ema_decay,
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
    split_task_ids = {}
    for selected_split in splits_to_run:
        with open(
            f'dataset/arc-agi_{selected_split}_challenges.json', 'r'
        ) as handle:
            split_task_ids[selected_split] = set(json.load(handle))
    _validate_recovery_task_ids(
        set().union(*split_task_ids.values()),
        args.eager_tasks,
        args.retry_quarantined,
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
                recover_task_failures=args.recover_task_failures,
                eager_tasks=args.eager_tasks & split_task_ids[split],
                retry_quarantined=(
                    args.retry_quarantined & split_task_ids[split]
                ),
                task_ids=(
                    args.task_ids & split_task_ids[split]
                    if args.task_ids is not None else None
                ),
                output_dir=args.output_dir,
                state_dir=args.state_dir,
            )
        except BaseException:
            arc_logger.log_run_failed(traceback.format_exc())
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

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'timing_result.txt'), 'w') as f:
        f.write(f'Splits run   : {", ".join(splits_to_run)}\n')
        f.write(f'Total solved : {total_solved}/{total_tasks}\n')
        f.write(f'Total time   : {overall_elapsed:.1f}s\n')
