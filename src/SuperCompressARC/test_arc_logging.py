import io
import os
import tempfile
import unittest
from unittest import mock

from arc_logging import TerminalDashboard, _ANSI_RE


class FakeTerminal(io.StringIO):
    encoding = 'UTF-8'

    def isatty(self):
        return True


class TerminalDashboardTests(unittest.TestCase):
    def make_dashboard(self):
        dashboard = TerminalDashboard(
            'training', stream=FakeTerminal(), enabled=True,
            refresh_interval=1.0,
        )
        dashboard.start(4, 1, total_steps=1500)
        dashboard.set_phase('SOLVING', 'Phase 2 — Full training', 1500)
        dashboard.task_started('ARC_001', 0, 0)
        dashboard.task_started('ARC_002_with_a_long_name', 1, 0)
        return dashboard

    def assert_frame_size(self, frame, width, height):
        lines = frame.splitlines()
        self.assertEqual(len(lines), height)
        for line in lines:
            self.assertLessEqual(len(_ANSI_RE.sub('', line)), width)

    def test_renders_full_compact_and_minimal_layouts(self):
        dashboard = self.make_dashboard()
        dashboard.update_progress({'ARC_001': 700, 'ARC_002_with_a_long_name': 300}, now=10.0)

        for width, height in ((160, 36), (100, 25), (70, 18)):
            with self.subTest(width=width, height=height):
                frame = dashboard.render(width, height, now=11.0, force=True)
                self.assert_frame_size(frame, width, height)
                plain = _ANSI_RE.sub('', frame)
                self.assertIn('arc', plain.lower())
                self.assertIn('SOLVING', plain)

    def test_speed_uses_existing_progress_samples(self):
        dashboard = self.make_dashboard()
        dashboard.update_progress({'ARC_001': 100}, now=10.0)
        dashboard.update_progress({'ARC_001': 300}, now=12.0)
        self.assertEqual(dashboard.speed_history[-1], 100.0)

    def test_finished_task_records_success_and_failure(self):
        dashboard = self.make_dashboard()
        dashboard.task_finished('ARC_001', 12.5, 700.0, 1)
        dashboard.task_finished('ARC_002_with_a_long_name', 14.0, 710.0, None)
        states = [task['state'] for task in dashboard.completed]
        self.assertEqual(states, ['FAILED', 'SUCCESS'])

    def test_search_measurement_does_not_pollute_completed_results(self):
        dashboard = self.make_dashboard()
        dashboard.task_started('measure_only', 3, 0, state='SEARCHING')
        dashboard.task_finished('measure_only', 1.0, 500.0, None)
        self.assertNotIn('measure_only', dashboard.tasks)
        self.assertFalse(dashboard.completed)

    def test_test_split_is_not_marked_failed_without_ground_truth(self):
        dashboard = self.make_dashboard()
        dashboard.task_finished(
            'ARC_001', 10.0, 600.0, None, has_ground_truth=False,
        )
        self.assertEqual(dashboard.completed[0]['state'], 'VERIFYING')
        active_text = _ANSI_RE.sub('', '\n'.join(dashboard._active_rows(70, 5)))
        self.assertNotIn('ARC_001', active_text)
        self.assertIn('ARC_002', active_text)

    def test_close_restores_cursor(self):
        dashboard = self.make_dashboard()
        dashboard.close()
        self.assertIn(TerminalDashboard.SHOW_CURSOR, dashboard.stream.getvalue())
        self.assertFalse(dashboard.active)

    def test_disabled_dashboard_produces_no_frame(self):
        dashboard = TerminalDashboard('training', stream=io.StringIO(), enabled=False)
        self.assertEqual(dashboard.render(force=True), '')

    def test_auto_detection_honors_no_color(self):
        stream = FakeTerminal()
        with mock.patch.dict(os.environ, {'TERM': 'xterm-256color', 'NO_COLOR': '1'}):
            dashboard = TerminalDashboard('training', stream=stream)
        self.assertFalse(dashboard.enabled)

    def test_progress_bar_has_fixed_visible_width(self):
        dashboard = self.make_dashboard()
        for percentage in (0, 50, 100):
            bar = dashboard._bar(percentage, 20)
            self.assertEqual(len(_ANSI_RE.sub('', bar)), 20)

    def test_run_failure_is_written_to_structured_log(self):
        from arc_logging import ArcLogger

        with tempfile.TemporaryDirectory() as temp_dir:
            logger = ArcLogger(
                'failure_test', log_dir=temp_dir, enable_tui=False,
            )
            logger.log_run_failed('Traceback\nMemoryError: injected')
            for handler in logger.logger.handlers:
                handler.flush()

            with open(logger.log_file, encoding='utf-8') as handle:
                contents = handle.read()

        self.assertIn('RUN FAILED', contents)
        self.assertIn('MemoryError: injected', contents)


if __name__ == '__main__':
    unittest.main()