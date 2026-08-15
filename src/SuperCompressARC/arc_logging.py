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
import re
import shutil
import sys
import time
from collections import deque

import torch


_ANSI_RE = re.compile(r'\x1b\[[0-9;?]*[ -/]*[@-~]')


class TerminalDashboard:
    """ANSI-only dashboard updated by the scheduler's existing poll tick."""

    RESET = '\x1b[0m'
    HOME = '\x1b[H'
    CLEAR = '\x1b[2J'
    HIDE_CURSOR = '\x1b[?25l'
    SHOW_CURSOR = '\x1b[?25h'
    COLORS = {
        'border': '\x1b[38;5;141m', 'cyan': '\x1b[38;5;81m',
        'blue': '\x1b[38;5;117m', 'green': '\x1b[38;5;84m',
        'magenta': '\x1b[38;5;213m', 'pink': '\x1b[38;5;205m',
        'lavender': '\x1b[38;5;183m', 'white': '\x1b[38;5;255m',
        'muted': '\x1b[38;5;245m', 'failed': '\x1b[38;5;203m',
    }
    STATE_COLORS = {
        'WAITING': 'muted', 'SEARCHING': 'cyan', 'SOLVING': 'blue',
        'VERIFYING': 'lavender', 'SUCCESS': 'green', 'FAILED': 'failed',
    }

    def __init__(self, split, stream=None, enabled=None, refresh_interval=1.0):
        self.split = split
        self.stream = stream or sys.stdout
        self.enabled = self._detect_support() if enabled is None else bool(enabled)
        self.refresh_interval = max(float(refresh_interval), 0.25)
        self.phase = 'WAITING'
        self.phase_description = 'initializing'
        self.n_tasks = 0
        self.n_gpus = 0
        self.total_steps = 1
        self.started_at = time.monotonic()
        self.tasks = {}
        self.completed = deque(maxlen=200)
        self.speed_history = deque([0.0], maxlen=120)
        self._last_sample_time = None
        self._last_total_steps = 0
        self._last_render_time = 0.0
        self._last_frame = None
        self._active = False

    def _detect_support(self):
        encoding = getattr(self.stream, 'encoding', '') or ''
        return (
            hasattr(self.stream, 'isatty') and self.stream.isatty()
            and os.environ.get('TERM', '').lower() != 'dumb'
            and not os.environ.get('NO_COLOR') and 'UTF' in encoding.upper()
        )

    @property
    def active(self):
        return self.enabled and self._active

    def start(self, n_tasks, n_gpus, total_steps=1):
        self.n_tasks = n_tasks
        self.n_gpus = n_gpus
        self.total_steps = max(total_steps, 1)
        self.started_at = time.monotonic()
        if self.enabled and not self._active:
            self.stream.write(self.CLEAR + self.HOME + self.HIDE_CURSOR)
            self.stream.flush()
            self._active = True

    def close(self):
        if self.active:
            self.stream.write(self.RESET + self.SHOW_CURSOR + '\n')
            self.stream.flush()
        self._active = False

    def set_phase(self, state, description, total_steps=None):
        self.phase = state
        self.phase_description = description
        if total_steps is not None:
            self.total_steps = max(total_steps, 1)

    def task_started(self, name, original_index, gpu_id, state='SOLVING'):
        self.tasks[name] = {
            'name': name, 'index': original_index, 'gpu': gpu_id, 'step': 0,
            'state': state, 'started_at': time.monotonic(), 'elapsed': 0.0,
            'peak_mb': 0.0, 'result': None, 'finished': False,
        }

    def task_finished(self, name, elapsed, peak_mb, solved_info,
                      has_ground_truth=True):
        task = self.tasks.setdefault(
            name, {'name': name, 'index': 0, 'gpu': 0, 'step': 0}
        )
        if task.get('state') == 'SEARCHING':
            del self.tasks[name]
            return
        task.update({
            'elapsed': elapsed, 'peak_mb': peak_mb, 'step': self.total_steps,
            'state': ('SUCCESS' if solved_info is not None else
                      'FAILED' if has_ground_truth else 'VERIFYING'),
            'result': solved_info, 'finished': True,
        })
        self.completed.appendleft(dict(task))

    def update_progress(self, progress, now=None):
        now = time.monotonic() if now is None else now
        for name, step in progress.items():
            if name in self.tasks:
                self.tasks[name]['step'] = min(int(step), self.total_steps)
                if step >= self.total_steps:
                    self.tasks[name]['state'] = 'VERIFYING'
        total = sum(task.get('step', 0) for task in self.tasks.values())
        if self._last_sample_time is not None and now > self._last_sample_time:
            speed = max(0.0, total - self._last_total_steps) / (now - self._last_sample_time)
            self.speed_history.append(speed)
        self._last_sample_time = now
        self._last_total_steps = total

    def render(self, width=None, height=None, now=None, force=False):
        if not self.enabled:
            return ''
        now = time.monotonic() if now is None else now
        if not force and now - self._last_render_time < self.refresh_interval:
            return ''
        size = shutil.get_terminal_size((120, 32))
        width, height = max(40, width or size.columns), max(12, height or size.lines)
        frame = self._build_frame(width, height, now)
        self._last_render_time = now
        if not force and frame == self._last_frame:
            return ''
        self._last_frame = frame
        if self.active:
            self.stream.write(self.HOME + frame)
            self.stream.flush()
        return frame

    def _build_frame(self, width, height, now):
        if width >= 120 and height >= 30:
            top_height = 8
            lower_height = height - top_height
            left_width = max(58, int(width * 0.62))
            right_width = width - left_width - 1
            active_height = max(7, lower_height // 2)
            completed_height = lower_height - active_height
            lines = self._panel('arc-agi solver', self._global_rows(width - 2, now), width, top_height)
            lines += self._join(
                self._panel('active tasks', self._active_rows(left_width - 2, active_height - 2), left_width, active_height),
                self._panel('current task / grid info', self._current_rows(right_width - 2, active_height - 2, now), right_width, active_height),
            )
            lines += self._join(
                self._panel('completed tasks + scores', self._completed_rows(left_width - 2, completed_height - 2), left_width, completed_height),
                self._panel('iteration / seconds', self._speed_rows(right_width - 2, completed_height - 2), right_width, completed_height),
            )
        elif width >= 82 and height >= 22:
            top_height, footer_height = 7, 5
            active_height = height - top_height - footer_height
            lines = self._panel('arc-agi solver', self._global_rows(width - 2, now), width, top_height)
            lines += self._panel('active tasks', self._active_rows(width - 2, active_height - 2), width, active_height)
            lines += self._panel('completed / speed', self._summary_rows(width - 2), width, footer_height)
        else:
            rows = self._global_rows(width - 2, now)[:3]
            rows += self._active_rows(width - 2, max(1, height - 5))
            lines = self._panel('arc solver', rows, width, height)
        return '\n'.join(self._fit(line, width) for line in lines[:height])

    def _global_rows(self, inner_width, now):
        done = len(self.completed)
        active = sum(not task.get('finished', False) for task in self.tasks.values())
        steps = sum(task.get('step', 0) for task in self.tasks.values())
        pct = min(100.0, 100.0 * steps / max(self.n_tasks * self.total_steps, 1))
        bar_width = max(8, min(42, inner_width - 28))
        return [
            f'{self._state(self.phase)}  split {self._color("white", self.split)}  tasks {done:>3}/{self.n_tasks:<3}  active {active:<2}  gpu {self.n_gpus:<2}  up {self._duration(now - self.started_at)}',
            f'global {self._bar(pct, bar_width)} {pct:5.1f}%  {self.speed_history[-1]:7.1f} iter/s',
            f'iter/s   {self._sparkline(max(8, inner_width - 11))}',
            self._color('muted', self.phase_description),
        ]

    def _active_rows(self, inner_width, limit):
        active = [task for task in self.tasks.values() if not task.get('finished', False)]
        if not active:
            return [self._color('muted', '· waiting for task dispatch')]
        bar_width = max(5, min(20, inner_width - 43))
        name_width = max(8, inner_width - bar_width - 35)
        rows = []
        for task in active[:limit]:
            pct = 100.0 * task.get('step', 0) / self.total_steps
            rows.append(
                f'{self._truncate(task["name"], name_width):<{name_width}} '
                f'{self._bar(pct, bar_width)} {pct:5.1f}% g{task.get("gpu", 0)} '
                f'{self._state(task.get("state", "WAITING"))}'
            )
        return rows

    def _current_rows(self, inner_width, limit, now):
        active = [task for task in self.tasks.values() if not task.get('finished', False)]
        if not active:
            return [self._color('muted', 'No active task')]
        task = max(active, key=lambda item: item.get('step', 0))
        speed = self.speed_history[-1]
        remaining = max(0, self.total_steps - task.get('step', 0))
        rows = [
            f'name  {self._color("white", self._truncate(task["name"], inner_width - 6))}',
            f'state {self._state(task.get("state", "WAITING"))}',
            f'gpu   {task.get("gpu", 0)}',
            f'iter  {task.get("step", 0):>5}/{self.total_steps:<5}',
            f'time  {self._duration(now - task.get("started_at", now))}',
            f'eta   {self._duration(remaining / speed) if speed > 0 else "--:--:--"}',
        ]
        return rows[:limit]

    def _completed_rows(self, inner_width, limit):
        if not self.completed:
            return [self._color('muted', '· no completed tasks')]
        name_width = max(8, inner_width - 34)
        rows = []
        for task in list(self.completed)[:limit]:
            result = f'guess@{task["result"]}' if task.get('result') else '--'
            rows.append(
                f'{self._truncate(task["name"], name_width):<{name_width}} '
                f'{self._state(task.get("state", "FAILED"))} {result:>7} '
                f'{task.get("elapsed", 0):6.1f}s {task.get("peak_mb", 0):5.0f}M'
            )
        return rows

    def _speed_rows(self, inner_width, limit):
        values = list(self.speed_history)
        rows = [
            f'current {self._color("cyan", f"{values[-1]:8.1f}")} iter/s',
            f'average {self._color("blue", f"{sum(values) / len(values):8.1f}")} iter/s',
            f'peak    {self._color("magenta", f"{max(values):8.1f}")} iter/s',
            self._sparkline(max(8, inner_width)),
        ]
        return rows[:limit]

    def _summary_rows(self, inner_width):
        solved = sum(task.get('state') == 'SUCCESS' for task in self.completed)
        failed = sum(task.get('state') == 'FAILED' for task in self.completed)
        return [
            f'{self._state("SUCCESS")} {solved:<3}  {self._state("FAILED")} {failed:<3}  '
            f'current {self.speed_history[-1]:.1f} iter/s  {self._sparkline(max(8, inner_width // 3))}'
        ]

    def _panel(self, title, rows, width, height):
        width, height = max(width, 4), max(height, 2)
        label = f' {title} '
        border = self.COLORS['border']
        lines = [border + '┌' + label + '─' * max(0, width - len(label) - 2) + '┐' + self.RESET]
        for row in rows[:height - 2]:
            lines.append(border + '│' + self.RESET + self._fit(row, width - 2) + border + '│' + self.RESET)
        while len(lines) < height - 1:
            lines.append(border + '│' + self.RESET + ' ' * (width - 2) + border + '│' + self.RESET)
        lines.append(border + '└' + '─' * (width - 2) + '┘' + self.RESET)
        return lines

    @staticmethod
    def _join(left, right):
        return [a + ' ' + b for a, b in zip(left, right)]

    def _bar(self, pct, width):
        filled = min(width, max(0, int(round(width * pct / 100.0))))
        return self._color('green', '█' * filled) + self._color('muted', '░' * (width - filled))

    def _sparkline(self, width):
        values = list(self.speed_history)[-width:]
        values = [0.0] * (width - len(values)) + values
        maximum = max(values, default=0.0) or 1.0
        blocks = '·▁▂▃▄▅▆▇█'
        return self._color('cyan', ''.join(blocks[min(8, int(value / maximum * 8))] for value in values))

    def _state(self, state):
        return self._color(self.STATE_COLORS.get(state, 'white'), state)

    def _color(self, name, text):
        return self.COLORS[name] + str(text) + self.RESET

    @staticmethod
    def _duration(seconds):
        seconds = max(0, int(seconds or 0))
        return f'{seconds // 3600:02}:{seconds % 3600 // 60:02}:{seconds % 60:02}'

    @staticmethod
    def _truncate(text, width):
        text = str(text)
        return text if len(text) <= width else text[:max(0, width - 1)] + '…'

    @classmethod
    def _fit(cls, text, width):
        visible = len(_ANSI_RE.sub('', text))
        if visible <= width:
            return text + ' ' * (width - visible)
        return cls._truncate(_ANSI_RE.sub('', text), width)


class ArcLogger:
    """
    Per-split logger. Creates one .log file per split.
    All loggers in a run share the same `last_results.txt`.
    """

    def __init__(self, split, log_dir='.', results_file='last_results.txt',
                 enable_tui=None, stream=None):
        self.split = split
        self.dashboard = TerminalDashboard(split, stream=stream, enabled=enable_tui)
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
        if self.dashboard.enabled:
            ch.addFilter(lambda record: not self.dashboard.active)
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
        self.dashboard.start(n_tasks, n_gpus)

    def log_phase(self, description):
        state = 'SEARCHING' if 'Phase 1' in description else 'SOLVING'
        self.dashboard.set_phase(state, description)
        self.logger.info('-' * 72)
        self.logger.info(f'PHASE: {description}')
        self.logger.info('-' * 72)
        self.dashboard.render(force=True)

    def log_task_started(self, task_name, orig_idx, n_total, gpu_id, quiet=False):
        state = 'SEARCHING' if self.dashboard.phase == 'SEARCHING' else 'SOLVING'
        self.dashboard.task_started(task_name, orig_idx, gpu_id, state=state)
        msg = f'START [{orig_idx + 1:>3}/{n_total}] {task_name}  →  GPU {gpu_id}'
        if quiet:
            self.logger.debug(msg)
        else:
            self.logger.info(msg)
        self._log_single_gpu(gpu_id)
        self.dashboard.render()

    def log_task_progress(self, task_name, step, total_steps):
        pct = 100.0 * step / max(total_steps, 1)
        self.logger.debug(
            f'PROG  {task_name}  step {step:>4}/{total_steps}  ({pct:5.1f}%)'
        )

    def log_task_finished(self, task_name, orig_idx, n_total,
                          elapsed_sec, peak_mb, solved_info, quiet=False,
                          has_ground_truth=True):
        self.dashboard.task_finished(
            task_name, elapsed_sec, peak_mb, solved_info,
            has_ground_truth=has_ground_truth,
        )
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
        self.dashboard.render()

    def log_run_summary(self, n_solved, n_total, elapsed_sec, predictions_file=None):
        self.dashboard.close()
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

    def render_progress(self, progress, total_steps, force=False):
        """Render one frame from progress data already read by the scheduler."""
        self.dashboard.total_steps = max(total_steps, 1)
        self.dashboard.update_progress(progress)
        return self.dashboard.render(force=force)

    def close_dashboard(self):
        self.dashboard.close()

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
