import json
from pathlib import Path
import tempfile
import unittest

import task_selection


class TaskSelectionTests(unittest.TestCase):
    def test_axis_d_history_selects_260_known_tasks_in_dataset_order(self):
        root = Path(__file__).resolve().parent
        selected = task_selection.select_unresolved_task_ids(
            root / 'dataset' / 'arc-agi_training_challenges.json',
            root / 'all_results_axis_D.txt',
            expected_excluded=140,
            expected_remaining=260,
        )

        known = list(json.loads(
            (root / 'dataset' / 'arc-agi_training_challenges.json').read_text(
                encoding='utf-8'
            )
        ))
        completed = set(task_selection._completed_task_ids(
            root / 'all_results_axis_D.txt', 'training'
        ))
        self.assertEqual(selected, [task_id for task_id in known if task_id not in completed])

    def test_rejects_duplicate_and_unknown_historical_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            challenges = root / 'challenges.json'
            results = root / 'results.txt'
            challenges.write_text(
                json.dumps({'known': {}, 'other': {}}), encoding='utf-8'
            )
            results.write_text(
                '2026-09-05 00:00:00 | training | [ 1] known | guess@1\n'
                '2026-09-05 00:00:01 | training | [ 2] known | guess@2\n',
                encoding='utf-8',
            )
            with self.assertRaisesRegex(ValueError, 'duplicate task id'):
                task_selection.select_unresolved_task_ids(challenges, results)

            results.write_text(
                '2026-09-05 00:00:00 | training | [ 1] missing | guess@1\n',
                encoding='utf-8',
            )
            with self.assertRaisesRegex(ValueError, 'unknown task id'):
                task_selection.select_unresolved_task_ids(challenges, results)


if __name__ == '__main__':
    unittest.main()