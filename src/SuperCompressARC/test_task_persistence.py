import json
import os
import tempfile
import unittest
from unittest import mock

import task_persistence


class TaskPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.previous_cwd = os.getcwd()
        os.chdir(self.temp_dir.name)
        self.addCleanup(os.chdir, self.previous_cwd)
        self.solution = [{'attempt_1': [[1]], 'attempt_2': [[2]]}]
        self.logger_data = {
            'solution_contributions_log': [],
            'solution_picks_history': [],
        }

    def test_atomic_round_trip_preserves_existing_schema(self):
        path = task_persistence.save_task_partial(
            'training', '007bbfb7', 2000, self.solution, self.logger_data,
        )

        with open(path, encoding='utf-8') as handle:
            payload = json.load(handle)
        self.assertEqual(
            set(payload), {'n_steps', 'solution', 'logger'},
        )
        self.assertFalse(os.path.exists(path + '.tmp'))
        self.assertEqual(
            task_persistence.load_task_partials(
                'training', ['007bbfb7'], 2000,
            ),
            ({'007bbfb7': self.solution}, {'007bbfb7': self.logger_data}),
        )

    def test_load_rejects_wrong_iterations_corruption_and_incomplete_logger(self):
        directory = task_persistence.partial_dir('training')
        os.makedirs(directory)
        records = {
            'wrong_steps': {
                'n_steps': 300,
                'solution': self.solution,
                'logger': self.logger_data,
            },
            'missing_logger': {
                'n_steps': 2000,
                'solution': self.solution,
                'logger': None,
            },
        }
        for name, payload in records.items():
            with open(os.path.join(directory, f'{name}.json'), 'w') as handle:
                json.dump(payload, handle)
        with open(os.path.join(directory, 'wrong_shape.json'), 'w') as handle:
            json.dump([], handle)
        with open(os.path.join(directory, 'corrupt.json'), 'w') as handle:
            handle.write('{')

        solutions, loggers = task_persistence.load_task_partials(
            'training', list(records) + ['wrong_shape', 'corrupt'], 2000,
        )
        self.assertEqual(solutions, {})
        self.assertEqual(loggers, {})

    def test_save_rejects_unsafe_or_incomplete_results(self):
        with self.assertRaises(ValueError):
            task_persistence.save_task_partial(
                'training', '../escape', 2000,
                self.solution, self.logger_data,
            )
        with self.assertRaises(ValueError):
            task_persistence.save_task_partial(
                'training', 'safe_name', 2000, [], self.logger_data,
            )
        with self.assertRaises(ValueError):
            task_persistence.save_task_partial(
                'training', 'safe_name', 2000, self.solution, {},
            )

    def test_write_failure_is_not_swallowed_and_cleans_temporary_file(self):
        with mock.patch('task_persistence.os.replace', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'):
                task_persistence.save_task_partial(
                    'training', '007bbfb7', 2000,
                    self.solution, self.logger_data,
                )

        path = os.path.join(
            task_persistence.partial_dir('training'), '007bbfb7.json',
        )
        self.assertFalse(os.path.exists(path))
        self.assertFalse(os.path.exists(path + '.tmp'))

    def test_recovery_manifest_round_trip_is_iteration_scoped(self):
        failure = {
            'stage': 'training',
            'exit_code': -6,
            'last_step': 10,
        }

        task_persistence.update_task_recovery(
            'training', 'fcb5c309', 2000, 'retry_eager', failure,
        )
        task_persistence.update_task_recovery(
            'training', 'fcb5c309', 1500, 'quarantined', failure,
        )
        task_persistence.update_task_recovery(
            'training', 'fcb5c309', 2000, 'recovered_eager',
        )

        current = task_persistence.load_task_recovery('training', 2000)
        older = task_persistence.load_task_recovery('training', 1500)
        self.assertEqual(current['fcb5c309']['state'], 'recovered_eager')
        self.assertEqual(current['fcb5c309']['failures'], [failure])
        self.assertEqual(older['fcb5c309']['state'], 'quarantined')
        self.assertFalse(any(
            '.tmp.' in name
            for name in os.listdir(task_persistence.partial_dir('training'))
        ))

    def test_recovery_manifest_rejects_corruption_and_unknown_state(self):
        directory = task_persistence.partial_dir('training')
        os.makedirs(directory)
        with open(task_persistence.recovery_path('training'), 'w') as handle:
            handle.write('{')

        with self.assertRaisesRegex(ValueError, 'invalid task recovery manifest'):
            task_persistence.load_task_recovery('training', 2000)
        with self.assertRaisesRegex(ValueError, 'unknown task recovery state'):
            task_persistence.update_task_recovery(
                'evaluation', '007bbfb7', 2000, 'unknown',
            )

    def test_recovery_manifest_rejects_unknown_schema_version(self):
        directory = task_persistence.partial_dir('training')
        os.makedirs(directory)
        with open(
            task_persistence.recovery_path('training'), 'w', encoding='utf-8',
        ) as handle:
            json.dump({'schema_version': 2, 'iterations': {}}, handle)

        with self.assertRaisesRegex(ValueError, 'invalid task recovery manifest'):
            task_persistence.load_task_recovery('training', 2000)

    def test_recovery_manifest_write_failure_keeps_previous_state(self):
        task_persistence.update_task_recovery(
            'training', 'fcb5c309', 2000, 'retry_eager',
        )
        with mock.patch('task_persistence.os.replace', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'):
                task_persistence.update_task_recovery(
                    'training', 'fcb5c309', 2000, 'quarantined',
                )

        state = task_persistence.load_task_recovery('training', 2000)
        self.assertEqual(state['fcb5c309']['state'], 'retry_eager')
        self.assertFalse(any(
            '.tmp.' in name
            for name in os.listdir(task_persistence.partial_dir('training'))
        ))


if __name__ == '__main__':
    unittest.main()