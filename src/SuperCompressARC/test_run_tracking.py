import json
import os
import tempfile
import unittest

import run_tracking


class RunTrackingTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.events = os.path.join(self.temp_dir.name, 'attempts.jsonl')

    def test_split_summary_accounts_for_retries_and_preserves_metadata_fields(self):
        run_tracking.record_attempt_start(
            self.events, 'run-1', 'campaign-1', 'training', 1, 10, 'run.log',
        )
        run_tracking.record_attempt_end(
            self.events, 'run-1', 'campaign-1', 'training', 1, 20, 1,
        )
        run_tracking.record_attempt_start(
            self.events, 'run-1', 'campaign-1', 'training', 2, 50, 'run.log',
        )
        run_tracking.record_attempt_end(
            self.events, 'run-1', 'campaign-1', 'training', 2, 70, 0,
        )
        metadata_path = os.path.join(self.temp_dir.name, 'metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as handle:
            json.dump({'elapsed_s': 20.0, 'phase2_s': 19.0}, handle)
        summary_path = os.path.join(self.temp_dir.name, 'summary.json')
        latest_path = os.path.join(self.temp_dir.name, 'latest.json')

        summary = run_tracking.write_split_summary(
            self.events, summary_path, latest_path, metadata_path,
            'run-1', 'campaign-1', 'training', 0, 80, 'success', 'run.log',
        )

        self.assertEqual(summary['attempt_count'], 2)
        self.assertEqual(summary['retry_count'], 1)
        self.assertEqual(summary['cumulative_active_s'], 30.0)
        self.assertEqual(summary['total_wall_s'], 80.0)
        self.assertEqual(summary['retry_wait_and_overhead_s'], 50.0)
        with open(metadata_path, encoding='utf-8') as handle:
            metadata = json.load(handle)
        self.assertEqual(metadata['elapsed_s'], 20.0)
        self.assertEqual(metadata['phase2_s'], 19.0)
        self.assertEqual(metadata['runner']['total_wall_s'], 80.0)
        self.assertTrue(os.path.exists(latest_path))
        self.assertFalse(any('.tmp.' in name for name in os.listdir(self.temp_dir.name)))

    def test_degraded_metadata_is_propagated_to_split_summary(self):
        run_tracking.record_attempt_start(
            self.events, 'run-1', 'campaign-1', 'training', 1, 10, 'run.log',
        )
        run_tracking.record_attempt_end(
            self.events, 'run-1', 'campaign-1', 'training', 1, 20, 0,
        )
        metadata_path = os.path.join(self.temp_dir.name, 'metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as handle:
            json.dump({
                'degraded': True,
                'real_result_tasks': 399,
                'recovery': {'quarantined_tasks': ['fcb5c309']},
            }, handle)

        summary = run_tracking.write_split_summary(
            self.events,
            os.path.join(self.temp_dir.name, 'summary.json'),
            None,
            metadata_path,
            'run-1', 'campaign-1', 'training', 0, 30,
            'success', 'run.log',
        )

        self.assertTrue(summary['degraded'])
        self.assertEqual(summary['quarantined_tasks'], ['fcb5c309'])
        self.assertEqual(summary['real_result_tasks'], 399)

    def test_unfinished_attempt_is_counted_until_wrapper_end(self):
        run_tracking.record_attempt_start(
            self.events, 'run-1', 'campaign-1', 'evaluation', 1, 10, 'run.log',
        )

        summary = run_tracking.build_split_summary(
            run_tracking.load_events(self.events),
            'run-1', 'campaign-1', 'evaluation', 0, 25,
            'interrupted', 'run.log',
        )

        self.assertEqual(summary['cumulative_active_s'], 15.0)
        self.assertEqual(summary['attempts'][0]['outcome'], 'interrupted')
        self.assertIsNone(summary['attempts'][0]['exit_code'])

    def test_success_without_fresh_metadata_is_rejected(self):
        with self.assertRaisesRegex(FileNotFoundError, 'did not write'):
            run_tracking.write_split_summary(
                self.events,
                os.path.join(self.temp_dir.name, 'summary.json'),
                None,
                os.path.join(self.temp_dir.name, 'missing-metadata.json'),
                'run-1', 'campaign-1', 'training', 0, 1,
                'success', 'run.log',
            )

    def test_subsecond_stale_metadata_is_rejected(self):
        metadata_path = os.path.join(self.temp_dir.name, 'metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as handle:
            json.dump({'elapsed_s': 1.0}, handle)
        os.utime(metadata_path, (1000.0, 1000.0))

        with self.assertRaisesRegex(ValueError, 'metadata is stale'):
            run_tracking.write_split_summary(
                self.events,
                os.path.join(self.temp_dir.name, 'summary.json'),
                None,
                metadata_path,
                'run-1', 'campaign-1', 'training', 1000.5, 1002,
                'success', 'run.log',
            )

    def test_campaign_aggregates_split_active_and_wall_time(self):
        paths = []
        for split, active, wall in (
            ('training', 30.0, 80.0),
            ('evaluation', 40.0, 50.0),
        ):
            path = os.path.join(self.temp_dir.name, f'{split}.json')
            run_tracking.atomic_write_json(path, {
                'campaign_id': 'campaign-1',
                'split': split,
                'status': 'success',
                'attempt_count': 2,
                'retry_count': 1,
                'cumulative_active_s': active,
                'total_wall_s': wall,
                'degraded': split == 'evaluation',
                'quarantined_tasks': (
                    ['fcb5c309'] if split == 'evaluation' else []
                ),
                'outputs': {},
            })
            paths.append(path)

        summary = run_tracking.build_campaign_summary(
            paths, 'campaign-1', 0, 150, 'success',
        )

        self.assertEqual(summary['split_count'], 2)
        self.assertEqual(summary['attempt_count'], 4)
        self.assertEqual(summary['cumulative_active_s'], 70.0)
        self.assertEqual(summary['total_wall_s'], 150.0)
        self.assertTrue(summary['degraded'])
        self.assertEqual(
            summary['splits'][1]['quarantined_tasks'], ['fcb5c309'],
        )

    def test_campaign_rejects_summary_from_another_run(self):
        path = os.path.join(self.temp_dir.name, 'training.json')
        run_tracking.atomic_write_json(path, {
            'campaign_id': 'stale-campaign',
            'split': 'training',
            'status': 'success',
            'attempt_count': 1,
            'retry_count': 0,
            'cumulative_active_s': 1.0,
            'total_wall_s': 1.0,
            'outputs': {},
        })

        with self.assertRaisesRegex(ValueError, 'stale-campaign'):
            run_tracking.build_campaign_summary(
                [path], 'campaign-1', 0, 2, 'failed',
            )


if __name__ == '__main__':
    unittest.main()