import json
import os
import tempfile
import unittest

import profile_parallel_train


def summary(eje_b, compile_mode='default', task_ids=None, n_steps=2000,
            postprocess_stride=4):
    return {'run_metadata': [{
        'split': 'training',
        'task_ids': task_ids or ['007bbfb7'],
        'n_steps': n_steps,
        'postprocess_stride': postprocess_stride,
        'accel': {
            'eje_b': eje_b,
            'compile_mode': compile_mode,
        },
    }]}


class EjeBComparisonContractTests(unittest.TestCase):
    def test_matching_compile_runs_are_comparable(self):
        self.assertEqual(
            profile_parallel_train._eje_b_comparison_errors(
                summary(False), summary(True)
            ),
            [],
        )

    def test_non_compile_or_different_workload_is_rejected(self):
        errors = profile_parallel_train._eje_b_comparison_errors(
            summary(False, compile_mode='off'),
            summary(True, task_ids=['00d62c1b']),
        )

        self.assertIn('before run is not compile=default', errors)
        self.assertIn('task_ids differs between runs', errors)

    def test_non_eje_b_comparison_keeps_legacy_behavior(self):
        self.assertEqual(
            profile_parallel_train._eje_b_comparison_errors(
                {'run_metadata': []}, {'run_metadata': []}
            ),
            [],
        )

    def test_metadata_prefers_steps_executed_in_resumed_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'run_metadata_training.json')
            with open(path, 'w', encoding='utf-8') as handle:
                json.dump({
                    'n_steps': 2000,
                    'n_tasks': 11,
                    'total_optimizer_steps': 88000,
                    'optimizer_steps_this_attempt': 24000,
                }, handle)

            _, derived = profile_parallel_train._collect_run_metadata(
                0, directory
            )

        self.assertEqual(derived['planned_train_steps'], 24000)


if __name__ == '__main__':
    unittest.main()