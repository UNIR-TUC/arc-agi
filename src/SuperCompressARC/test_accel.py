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


class AccelConfigTests(unittest.TestCase):
    def test_eje_b_eager_override_is_internal_and_keeps_fingerprint(self):
        compiled = accel.AccelConfig(
            compile_mode='default',
            eje_b=True,
            seeds=(0, 1, 2, 3),
            kl_free_bits_initial=2.0,
            curriculum=True,
        )
        eager = accel.AccelConfig.from_dict({
            **compiled.to_dict(),
            'compile_mode': 'off',
            '_eje_b_eager_override': True,
        })

        self.assertEqual(eager.compile_mode, 'off')
        self.assertNotIn('_eje_b_eager_override', compiled.to_dict())
        self.assertTrue(eager.to_dict()['_eje_b_eager_override'])
        self.assertEqual(
            eager.algorithm_fingerprint(2000, 4),
            compiled.algorithm_fingerprint(2000, 4),
        )

    def test_eje_b_compile_off_requires_internal_override(self):
        with self.assertRaisesRegex(ValueError, 'requires the Eje D'):
            accel.AccelConfig(
                compile_mode='off',
                eje_b=True,
                seeds=(0, 1, 2, 3),
            )

        with self.assertRaisesRegex(ValueError, 'requires eje_b'):
            accel.AccelConfig(
                compile_mode='off',
                _eje_b_eager_override=True,
            )


if __name__ == '__main__':
    unittest.main()