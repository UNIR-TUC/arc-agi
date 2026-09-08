import unittest

import torch

import accel
import train


def eje_b_config(**overrides):
    values = {
        'eje_b': True,
        'seeds': (0, 1, 2, 3),
        'kl_free_bits_initial': 2.0,
        'curriculum': True,
    }
    values.update(overrides)
    return accel.config_from_preset('compile', **values)


class EjeBConfigTests(unittest.TestCase):
    def test_eje_b_requires_compile_default(self):
        with self.assertRaisesRegex(ValueError, "requires the Eje D 'compile'"):
            accel.config_from_preset('baseline', eje_b=True)
        with self.assertRaisesRegex(ValueError, "requires the Eje D 'compile'"):
            accel.config_from_preset(
                'compile', eje_b=True, compile_mode='reduce-overhead'
            )

    def test_b_options_require_eje_b_switch(self):
        with self.assertRaisesRegex(ValueError, 'require eje_b=True'):
            accel.config_from_preset('compile', seeds=(0, 1))
        with self.assertRaisesRegex(ValueError, 'require eje_b=True'):
            accel.config_from_preset(
                'compile', curriculum_ema_decay=0.8
            )

    def test_serialization_and_schedule_endpoints(self):
        config = accel.AccelConfig.from_dict(eje_b_config().to_dict())
        self.assertEqual(config.seeds, (0, 1, 2, 3))
        self.assertEqual(config.kl_free_bits_at_step(0, 2000), 2.0)
        self.assertEqual(config.kl_free_bits_at_step(1999, 2000), 0.0)
        self.assertEqual(config.curriculum_beta_at_step(0, 2000), 0.0)
        self.assertEqual(config.curriculum_beta_at_step(1999, 2000), 1.0)

    def test_measurement_disables_compile_and_eje_b(self):
        measurement = accel.for_measurement(eje_b_config())
        self.assertEqual(measurement.compile_mode, 'off')
        self.assertFalse(measurement.eje_b)
        self.assertEqual(measurement.seeds, (0,))
        self.assertEqual(measurement.kl_free_bits_initial, 0.0)
        self.assertFalse(measurement.curriculum)

    def test_fingerprint_tracks_algorithm_not_execution_tuning(self):
        config = eje_b_config()
        tuned = eje_b_config(
            threads_per_worker=2,
            alloc_conf='expandable_segments:True',
        )

        fingerprint = config.algorithm_fingerprint(2000, 4)

        self.assertEqual(
            fingerprint, tuned.algorithm_fingerprint(2000, 4)
        )
        self.assertNotEqual(
            fingerprint,
            eje_b_config(kl_free_bits_initial=1.0).algorithm_fingerprint(
                2000, 4
            ),
        )
        self.assertNotEqual(
            fingerprint, config.algorithm_fingerprint(2000, 1)
        )


class FreeBitsTests(unittest.TestCase):
    def test_floor_is_per_leaf_and_stops_only_compression_gradient(self):
        below = torch.tensor(0.5, device='cpu', requires_grad=True)
        above = torch.tensor([0.75, 1.25], device='cpu', requires_grad=True)

        raw, effective, n_below = train._sum_kl_with_free_bits(
            [below, above], 1.0
        )
        effective.backward()

        self.assertEqual(raw.item(), 2.5)
        self.assertEqual(effective.item(), 3.0)
        self.assertEqual(n_below.item(), 1)
        self.assertEqual(below.grad.item(), 0.0)
        self.assertTrue(torch.equal(above.grad, torch.ones_like(above)))


class CurriculumTests(unittest.TestCase):
    def test_weights_start_uniform_then_prefer_harder_example(self):
        config = eje_b_config()
        initial = train._curriculum_weights(
            None, 2, config, 0, 2000, torch.device('cpu'), torch.float32
        )
        weighted = train._curriculum_weights(
            torch.tensor([1.0, 3.0], device='cpu'),
            2,
            config,
            1999,
            2000,
            torch.device('cpu'),
            torch.float32,
        )

        self.assertTrue(torch.equal(
            initial, torch.ones(2, device='cpu')
        ))
        self.assertTrue(torch.allclose(
            weighted.sum(), torch.tensor(2.0, device='cpu')
        ))
        self.assertGreater(weighted[1].item(), weighted[0].item())

    def test_state_tracks_detached_per_pixel_loss(self):
        state = train.EjeBTrainingState()
        pair_losses = [
            torch.tensor(4.0, device='cpu', requires_grad=True),
            torch.tensor(9.0, device='cpu', requires_grad=True),
        ]

        train._update_curriculum_state(state, pair_losses, [2, 3], 0.9)

        self.assertTrue(torch.equal(
            state.demo_loss_ema, torch.tensor([2.0, 3.0], device='cpu')
        ))
        self.assertFalse(state.demo_loss_ema.requires_grad)


if __name__ == '__main__':
    unittest.main()