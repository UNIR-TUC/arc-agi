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

import numpy as np
import torch

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

# ── Global PyTorch settings (must run at import time for the main process) ──
multiprocessing.set_start_method('spawn', force=True)
torch.set_default_dtype(torch.float32)
torch.set_default_device('cuda')
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


# ── Live terminal progress line ──────────────────────────────────────────────

def _print_progress_line(task_names, tasks_started, tasks_finished, progress_dict,
                          n_iterations, n_tasks):
    """Overwrite a single terminal line with the %-progress of every task
    currently running in parallel. Piggybacks on the scheduler's existing
    1s poll tick, so it adds no extra polling or logging overhead."""
    running = [i for i in range(n_tasks) if tasks_started[i] and not tasks_finished[i]]
    if not running:
        return
    parts = []
    for i in running:
        name = task_names[i]
        step = int(progress_dict.get(name, 0))
        pct  = 100.0 * step / max(n_iterations, 1)
        parts.append(f'{name}:{pct:3.0f}%')
    line = f'[{len(running)} running] ' + '  '.join(parts)
    width = shutil.get_terminal_size((120, 20)).columns
    sys.stdout.write('\r' + line[:width - 1].ljust(width - 1))
    sys.stdout.flush()


# ── Core scheduler ───────────────────────────────────────────────────────────

def parallelize_runs(
    gpu_quotas,
    task_usages,
    n_iterations,
    task_names,
    split,
    n_tasks,
    n_gpus,
    n_cpus,
    collect_logger_data=False,
    track_progress=False,
    arc_logger=None,
    solutions_json=None,
    task_original_idx=None,
    n_original_tasks=None,
    quiet=False,
    verbose=False,
    postprocess_stride=1,
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
        collect_logger_data (bool)   : if True, store solution logs for predictions.npz.
        track_progress (bool)        : if True, workers report step % to main process.
        arc_logger (ArcLogger|None)  : logger; None silences per-task events.
        solutions_json (dict|None)   : ground-truth solutions for solved-status check.
        task_original_idx (dict|None): task_name → index in the original JSON order.
        n_original_tasks (int|None)  : total tasks in original order (for display).
        quiet (bool)                 : task-start/finish events go to DEBUG not INFO.
        verbose (bool)               : print raw status to stdout.
        postprocess_stride (int)     : run full pass@2 postprocessing every N steps
            (Eje H, H3) instead of every step. Forwarded to solve_task.solve_task.

    Returns:
        memory_dict    (dict[str, int])  : peak VRAM per task (bytes).
        solutions_dict (dict)            : Kaggle-format predictions per task.
        loggers_data   (dict)            : logger data per task (empty if not collected).
        time_taken     (float)           : wall-clock seconds.
    """
    t = time.time()
    gpu_quotas = gpu_quotas[:]
    n_disp = n_original_tasks if n_original_tasks is not None else n_tasks

    tasks_started    = [False] * n_tasks
    tasks_finished   = [False] * n_tasks
    processes        = [None]  * n_tasks
    process_gpu_ids  = [None]  * n_tasks
    task_start_times = [None]  * n_tasks
    task_last_pct    = {}   # task_name → last 10%-bucket logged

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

                        if arc_logger is not None:
                            arc_logger.log_task_finished(
                                task_names[i], orig_idx, n_disp,
                                elapsed, peak_mb, solved_info, quiet=quiet,
                            )
                            if solved_info is not None:
                                arc_logger.write_solved_result(
                                    task_names[i], orig_idx, solved_info
                                )

                        if verbose:
                            status = f'[SOLVED @{solved_info}]' if solved_info else ''
                            print(task_names[i], 'finished on gpu',
                                  process_gpu_ids[i],
                                  f'quota={gpu_quotas[process_gpu_ids[i]]:.0f}',
                                  status)

            # ── Report training progress for running tasks ────────────
            # File/debug log: throttled to once per 10% bucket per task
            # (cheap, useful for post-run analysis).
            if _progress_dict is not None and arc_logger is not None:
                for i in range(n_tasks):
                    if tasks_started[i] and not tasks_finished[i]:
                        name   = task_names[i]
                        step   = int(_progress_dict.get(name, 0))
                        bucket = int(step / max(n_iterations, 1) * 10) * 10
                        if bucket > task_last_pct.get(name, -1):
                            task_last_pct[name] = bucket
                            arc_logger.log_task_progress(name, step, n_iterations)

            # Terminal: a single overwriting status line with every running
            # task's live %, redrawn on the existing 1s tick below — no extra
            # polling, so no measurable performance impact.
            if _progress_dict is not None and verbose:
                _print_progress_line(
                    task_names, tasks_started, tasks_finished,
                    _progress_dict, n_iterations, n_tasks,
                )

            # ── Schedule new tasks ────────────────────────────────────
            for gpu_id in range(n_gpus):
                for i in range(n_tasks):
                    if tasks_started[i]:
                        continue
                    enough_quota = gpu_quotas[gpu_id] >= task_usages[i]
                    running = (sum(map(int, tasks_started))
                               - sum(map(int, tasks_finished)))
                    enough_cpus = running < n_cpus
                    if enough_quota and enough_cpus:
                        gpu_quotas[gpu_id] -= task_usages[i]
                        task_start_times[i] = time.time()

                        orig_idx = (task_original_idx.get(task_names[i], i)
                                    if task_original_idx else i)

                        worker_args = (
                            task_names[i], split, 1e20, n_iterations,
                            gpu_id, memory_dict, solutions_dict, error_queue,
                            _loggers_dict, _progress_dict, postprocess_stride,
                        )
                        p = multiprocessing.Process(
                            target=solve_task.solve_task, args=worker_args
                        )
                        p.start()
                        processes[i]       = p
                        tasks_started[i]   = True
                        process_gpu_ids[i] = gpu_id

                        if arc_logger is not None:
                            arc_logger.log_task_started(
                                task_names[i], orig_idx, n_disp, gpu_id,
                                quiet=quiet,
                            )
                        if verbose:
                            print(task_names[i], 'started on gpu', gpu_id,
                                  f'quota={gpu_quotas[gpu_id]:.0f}')

            time.sleep(1)

        if _progress_dict is not None and verbose:
            sys.stdout.write('\n')
            sys.stdout.flush()

        # Final error scan
        if not error_queue.empty():
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


def _gpu_fingerprint(n_gpus):
    return {
        'n_gpus':            n_gpus,
        'gpu_names':         [torch.cuda.get_device_name(i) for i in range(n_gpus)],
        'gpu_vram_total_gb': [
            round(torch.cuda.mem_get_info(i)[1] / 1024**3, 2)
            for i in range(n_gpus)
        ],
        'torch_version':     torch.__version__,
    }


def load_memory_cache(split, n_gpus, required_task_names):
    """Return {task_name: mem_bytes} from disk if the cache matches the
    current GPU fingerprint and covers every required task, else None."""
    path = _cache_path(split)
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            cache = json.load(f)
        if cache.get('fingerprint') != _gpu_fingerprint(n_gpus):
            return None
        measurements = cache.get('measurements', {})
        if not all(name in measurements for name in required_task_names):
            return None
        return {name: int(measurements[name]) for name in required_task_names}
    except Exception:
        return None


def save_memory_cache(split, n_gpus, memory_dict):
    path = _cache_path(split)
    cache = {
        'fingerprint':  _gpu_fingerprint(n_gpus),
        'created_at':   time.strftime('%Y-%m-%d %H:%M:%S'),
        'measurements': {k: int(v) for k, v in memory_dict.items()},
    }
    with open(path, 'w') as f:
        json.dump(cache, f, indent=2)


# ── Per-split runner ─────────────────────────────────────────────────────────

def run_split(split, n_gpus, n_cpus, arc_logger, solutions_json, demo_n=None,
              postprocess_stride=4):
    """
    Execute the full two-phase pipeline for one split and save all outputs.

    Args:
        demo_n (int|None): if set, use only the first demo_n tasks (smoke-test mode).
        postprocess_stride (int): Phase 2 only — run full pass@2 postprocessing every
            N steps instead of every step (Eje H, H3). Default 4. Phase 1's 2-step
            memory measurement always uses the Logger default (1) since it's too
            short to matter.

    Returns:
        n_solved (int)         : tasks solved (always 0 for 'test').
        n_tasks (int)          : total tasks in this split.
        elapsed_sec (float)    : wall-clock seconds.
        predictions_file (str) : path to the saved .npz file.
    """
    split_start = time.time()

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

    # ── Phase 1: measure VRAM footprint (2 iterations per task), or load
    #    from a cached measurement keyed to the current GPU fingerprint ──
    gpu_memory_quotas = [torch.cuda.mem_get_info(i)[0] for i in range(n_gpus)]

    cached_memory_dict = load_memory_cache(split, n_gpus, original_task_names)
    if cached_memory_dict is not None:
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
            arc_logger=arc_logger,
            n_original_tasks=n_tasks,
            quiet=True,       # phase-1 task events → DEBUG only (not cluttering console)
            verbose=True,
        )
        arc_logger.info(f'Phase 1 complete in {t_p1:.1f}s')
        save_memory_cache(split, n_gpus, memory_dict)
        arc_logger.info(f'Saved Phase 1 measurements to {_cache_path(split)}')

    # Sort tasks by decreasing VRAM so the greedy scheduler fills GPUs tightly
    sorted_tasks      = sorted(memory_dict.items(), key=lambda x: x[1], reverse=True)
    sorted_names      = [name for name, _ in sorted_tasks]
    sorted_mem_usages = [mem  for _, mem  in sorted_tasks]

    # Map task_name → original JSON index (used in log messages)
    task_original_idx = {name: idx for idx, name in enumerate(original_task_names)}

    # ── Phase 2: full 2000-step training ─────────────────────────────
    # n_steps = 2000
    n_steps = 1500

    # Phase 1 measurements now capture `total_vram - free_now` (solve_task.py),
    # which already includes the per-process HIP/CUDA context (~470 MB each).
    # The original 4 GB margin was compensating for that unmeasured overhead;
    # 1 GB now suffices: system/display GPU use (~90 MB idle) + allocator
    # fragmentation + measurement variance.
    safe_gpu_memory_quotas = [q - 1 * 1024**3 for q in gpu_memory_quotas]
    arc_logger.debug(
        f'Phase 2 free VRAM: '
        + ', '.join(f'GPU{i}={safe_gpu_memory_quotas[i]/1024**3:.2f} GB'
                   for i in range(n_gpus))
    )

    arc_logger.log_phase(
        f'Phase 2 — Full training  ({n_steps} iterations × {n_tasks} tasks)'
    )

    _, solutions_dict, loggers_data, t_p2 = parallelize_runs(
        safe_gpu_memory_quotas,
        sorted_mem_usages,
        n_steps,
        sorted_names,
        split,
        n_tasks, n_gpus, n_cpus,
        collect_logger_data=True,
        track_progress=True,
        arc_logger=arc_logger,
        solutions_json=solutions_json,
        task_original_idx=task_original_idx,
        n_original_tasks=n_tasks,
        quiet=False,
        verbose=True,
        postprocess_stride=postprocess_stride,
    )
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
    args = parser.parse_args()

    splits_to_run = (
        ['training', 'evaluation', 'test'] if args.split == 'all'
        else [args.split]
    )

    overall_start = time.time()
    n_cpus = args.max_workers if args.max_workers else multiprocessing.cpu_count()
    n_gpus = torch.cuda.device_count()

    # Initialise the shared results file once for the whole run
    current_date_time = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())
    print(f'ARC-AGI parallel training started at {current_date_time}')
    results_file = f'results_{current_date_time}.txt'
    arc_logging.init_results_file(results_file)

    print(f'\nStarting ARC-AGI parallel training — splits: {splits_to_run}')
    print(f'GPUs: {n_gpus}   CPU cores: {n_cpus}\n')

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
            split, log_dir='.', results_file=results_file
        )

        n_solved, n_tasks, elapsed, pred_file = run_split(
            split, n_gpus, n_cpus, arc_logger, solutions_json,
            demo_n=args.demo,
            postprocess_stride=args.postprocess_stride,
        )
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
