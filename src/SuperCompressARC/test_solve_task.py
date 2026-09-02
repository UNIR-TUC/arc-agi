import unittest
from unittest import mock

import gpu_memory


class GpuMemoryMetricsTests(unittest.TestCase):
    def test_separates_allocator_peaks_from_device_usage(self):
        cuda = mock.Mock()
        cuda.mem_get_info.return_value = (4_000, 10_000)
        cuda.max_memory_allocated.return_value = 1_500
        cuda.max_memory_reserved.return_value = 2_500

        metrics = gpu_memory.capture(cuda)

        cuda.synchronize.assert_called_once_with()
        self.assertEqual(metrics, {
            'peak_allocated_bytes': 1_500,
            'peak_reserved_bytes': 2_500,
            'device_used_bytes': 6_000,
        })

    def test_summarizes_phase2_worker_metrics(self):
        metadata = [{
            'gpu_memory': {
                'phase2_workers': {
                    'task-a': {
                        'peak_allocated_bytes': 100,
                        'peak_reserved_bytes': 200,
                        'device_used_bytes': 600,
                    },
                    'task-b': {
                        'peak_allocated_bytes': 150,
                        'peak_reserved_bytes': 250,
                        'device_used_bytes': 500,
                    },
                },
            },
        }]

        summary = gpu_memory.summarize_run_metadata(metadata)

        self.assertEqual(summary, {
            'worker_reports': 2,
            'process_peak_allocated_max_bytes': 150,
            'process_peak_reserved_max_bytes': 250,
            'device_used_at_worker_exit_max_bytes': 600,
        })

    def test_summarizes_legacy_metadata_as_empty(self):
        summary = gpu_memory.summarize_run_metadata([{'n_tasks': 10}])

        self.assertEqual(summary, {
            'worker_reports': 0,
            'process_peak_allocated_max_bytes': None,
            'process_peak_reserved_max_bytes': None,
            'device_used_at_worker_exit_max_bytes': None,
        })


if __name__ == '__main__':
    unittest.main()