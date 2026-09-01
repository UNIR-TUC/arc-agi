import unittest
import warnings
from unittest import mock

import accel


class EagerFallbackTests(unittest.TestCase):
    def test_benign_compile_failure_falls_back_permanently(self):
        compiled = mock.Mock(side_effect=RuntimeError('unsupported graph'))
        eager = mock.Mock(return_value='eager-result')
        wrapper = accel._EagerFallback(compiled, eager)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self.assertEqual(wrapper(), 'eager-result')
            self.assertEqual(wrapper(), 'eager-result')

        compiled.assert_called_once_with()
        self.assertEqual(eager.call_count, 2)
        self.assertIn('falling back to eager', str(caught[0].message))

    def test_illegal_memory_access_never_calls_eager(self):
        error = RuntimeError(
            'CUDA driver error: 700; an illegal memory access was encountered'
        )
        compiled = mock.Mock(side_effect=error)
        eager = mock.Mock()
        wrapper = accel._EagerFallback(compiled, eager)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with self.assertRaisesRegex(RuntimeError, 'driver error: 700'):
                wrapper()
            with self.assertRaisesRegex(RuntimeError, 'driver error: 700'):
                wrapper()

        eager.assert_not_called()
        compiled.assert_called_once_with()
        self.assertIn('refusing eager fallback', str(caught[0].message))

    def test_fatal_error_markers_are_case_insensitive(self):
        self.assertTrue(accel._is_fatal_accelerator_error(
            RuntimeError('HIPERRORILLEGALADDRESS')
        ))
        self.assertFalse(accel._is_fatal_accelerator_error(
            RuntimeError('graph break is unsupported')
        ))


if __name__ == '__main__':
    unittest.main()