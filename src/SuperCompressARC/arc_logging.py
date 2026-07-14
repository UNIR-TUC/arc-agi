"""
Logging infrastructure for parallel ARC-AGI training runs.

Writes every event to a timestamped .log file inside a project-root `.log`
directory (DEBUG level, i.e. all detail) and mirrors key events to the
terminal (INFO level).

Solved tasks are appended in real time to `last_results.txt` so you can
monitor progress without waiting for the full run to finish.
"""

import logging
import os
import time

import torch


class ArcLogger:
    """
    Per-split logger. Creates one .log file per split.
    All loggers in a run share the same `last_results.txt`.
    """

    def __init__(self, split, log_dir='.', results_file='last_results.txt'):
        self.split = split
        project_root = os.path.dirname(os.path.abspath(__file__))
        default_log_dir = os.path.join(project_root, '.log')
        self.log_dir = default_log_dir if log_dir in (None, '', '.') else log_dir
        self.results_file = results_file
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(
            self.log_dir,
            f'arc_training_{split}_{timestamp}.log',
        )
        self._setup_logger(split, timestamp)

    # ------------------------------------------------------------------
    # Internal setup
    # ------------------------------------------------------------------

    def _setup_logger(self, split, timestamp):
        name = f'arc_{split}_{timestamp}'
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False  # don't bubble up to root logger

        fmt = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )

        # File handler — captures everything (DEBUG and above)
        fh = logging.FileHandler(self.log_file, mode='a', encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        self.logger.addHandler(fh)

        # Console handler — shows INFO and above
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        self.logger.addHandler(ch)

    # ------------------------------------------------------------------
    # Raw log methods (pass-through to stdlib logger)
    # ------------------------------------------------------------------

    def info(self, msg):    self.logger.info(msg)
    def debug(self, msg):   self.logger.debug(msg)
    def warning(self, msg): self.logger.warning(msg)
    def error(self, msg):   self.logger.error(msg)

    # ------------------------------------------------------------------
    # Structured event methods
    # ------------------------------------------------------------------

    def log_run_start(self, n_tasks, n_gpus):
        self.logger.info('=' * 72)
        self.logger.info(f'ARC-AGI Parallel Training  |  split={self.split}')
        self.logger.info(f'  Tasks       : {n_tasks}')
        self.logger.info(f'  GPUs        : {n_gpus}')
        self.logger.info(f'  Log file    : {self.log_file}')
        self.logger.info(f'  Results file: {self.results_file}')
        self.logger.info('=' * 72)
        self._log_all_gpus(n_gpus)

    def log_phase(self, description):
        self.logger.info('-' * 72)
        self.logger.info(f'PHASE: {description}')
        self.logger.info('-' * 72)

    def log_task_started(self, task_name, orig_idx, n_total, gpu_id, quiet=False):
        msg = f'START [{orig_idx + 1:>3}/{n_total}] {task_name}  →  GPU {gpu_id}'
        if quiet:
            self.logger.debug(msg)
        else:
            self.logger.info(msg)
        self._log_single_gpu(gpu_id)

    def log_task_progress(self, task_name, step, total_steps):
        pct = 100.0 * step / max(total_steps, 1)
        self.logger.debug(
            f'PROG  {task_name}  step {step:>4}/{total_steps}  ({pct:5.1f}%)'
        )

    def log_task_finished(self, task_name, orig_idx, n_total,
                          elapsed_sec, peak_mb, solved_info, quiet=False):
        if solved_info is not None:
            status = f'SOLVED @guess{solved_info}'
        else:
            status = 'not solved'
        msg = (
            f'DONE  [{orig_idx + 1:>3}/{n_total}] {task_name}  '
            f'{elapsed_sec:6.1f}s  peak={peak_mb:6.0f} MB  [{status}]'
        )
        if quiet:
            self.logger.debug(msg)
        else:
            self.logger.info(msg)

    def log_run_summary(self, n_solved, n_total, elapsed_sec, predictions_file=None):
        pct = 100.0 * n_solved / n_total if n_total > 0 else 0.0
        self.logger.info('=' * 72)
        self.logger.info(
            f'RUN COMPLETE — {n_solved}/{n_total} tasks solved ({pct:.1f}%)  '
            f'in {elapsed_sec:.1f}s ({elapsed_sec / 3600:.2f}h)'
        )
        if predictions_file:
            self.logger.info(f'  Predictions saved : {predictions_file}')
        self.logger.info('=' * 72)

    def log_gpu_stats(self, gpu_id):
        self._log_single_gpu(gpu_id)

    # ------------------------------------------------------------------
    # last_results.txt helpers
    # ------------------------------------------------------------------

    def write_solved_result(self, task_name, orig_idx, guess_number):
        """Append a solved-task line to last_results.txt."""
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        line = (
            f'{ts} | {self.split:<12} | '
            f'[{orig_idx + 1:>4}] {task_name:<45} | guess@{guess_number}\n'
        )
        with open(self.results_file, 'a', encoding='utf-8') as f:
            f.write(line)

    def finalize_results(self, n_solved, n_total, elapsed_sec):
        """Write a footer for this split to last_results.txt."""
        pct = 100.0 * n_solved / n_total if n_total > 0 else 0.0
        with open(self.results_file, 'a', encoding='utf-8') as f:
            f.write(
                f'\n--- {self.split} complete: {n_solved}/{n_total} solved '
                f'({pct:.1f}%)  {elapsed_sec:.1f}s ---\n\n'
            )

    # ------------------------------------------------------------------
    # Private GPU helpers
    # ------------------------------------------------------------------

    def _log_all_gpus(self, n_gpus):
        for gpu_id in range(n_gpus):
            self._log_single_gpu(gpu_id)

    def _log_single_gpu(self, gpu_id):
        try:
            free_b, total_b = torch.cuda.mem_get_info(gpu_id)
            used_b = total_b - free_b
            alloc = torch.cuda.memory_allocated(gpu_id)
            rsvd  = torch.cuda.memory_reserved(gpu_id)
            self.logger.debug(
                f'  GPU {gpu_id} | '
                f'Used: {used_b / 1024**3:.2f}/{total_b / 1024**3:.2f} GB | '
                f'Allocated: {alloc / 1024**2:.1f} MB | '
                f'Reserved: {rsvd / 1024**2:.1f} MB'
            )
        except Exception as exc:
            self.logger.debug(f'  GPU {gpu_id} | stats unavailable: {exc}')


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def init_results_file(results_file='last_results.txt'):
    """
    Call once at the very start of a run to clear the file and write a header.
    Each split's ArcLogger then appends to it.
    """
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write('ARC-AGI Parallel Training — Solved Tasks\n')
        f.write(f'Run started: {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write('=' * 90 + '\n')
        f.write(
            f'{"Timestamp":<21}  {"Split":<12}  {"#":<6}  '
            f'{"Task Name":<45}  Guess\n'
        )
        f.write('-' * 90 + '\n\n')


def get_gpu_stats(n_gpus):
    """
    Return a dict keyed by gpu_id with VRAM statistics from torch.cuda.
    Safe to call even if some GPUs are unavailable.
    """
    stats = {}
    for gpu_id in range(n_gpus):
        try:
            free_b, total_b = torch.cuda.mem_get_info(gpu_id)
            stats[gpu_id] = {
                'used_gb':      (total_b - free_b) / 1024**3,
                'total_gb':      total_b            / 1024**3,
                'allocated_mb': torch.cuda.memory_allocated(gpu_id) / 1024**2,
                'reserved_mb':  torch.cuda.memory_reserved(gpu_id)  / 1024**2,
            }
        except Exception:
            stats[gpu_id] = {}
    return stats
