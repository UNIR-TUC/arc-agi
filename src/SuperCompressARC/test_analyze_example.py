import argparse
import json
import os
import tempfile
import unittest

import matplotlib
matplotlib.use('Agg')

import analyze_example
import visualization


class ParseArgsTests(unittest.TestCase):
    def test_defaults_without_eje_b_keep_legacy_layout(self):
        args = analyze_example.parse_args(['--task-name', 'abc'])
        self.assertEqual(args.accel_preset, 'baseline')
        self.assertEqual(args.seeds, (0,))
        self.assertEqual(args.run_label, 'baseline')
        self.assertEqual(args.kl_free_bits_initial, 0.0)
        self.assertFalse(args.curriculum)
        self.assertEqual(
            analyze_example.seed_output_dir(args, 0),
            os.path.join('results', 'abc'),
        )

    def test_eje_b_defaults_select_compile_and_subfolder(self):
        args = analyze_example.parse_args(
            ['--task-name', 'abc', '--eje-b', '--seeds', '0,2']
        )
        self.assertEqual(args.accel_preset, 'compile')
        self.assertEqual(args.seeds, (0, 2))
        self.assertEqual(args.kl_free_bits_initial, 2.0)
        self.assertTrue(args.curriculum)
        self.assertEqual(args.postprocess_stride, 4)
        self.assertEqual(
            analyze_example.seed_output_dir(args, 2),
            os.path.join('results', 'abc', 'eje_b', 'seed_2'),
        )
        self.assertEqual(
            analyze_example.fused_output_dir(args),
            os.path.join('results', 'abc', 'eje_b', 'fused'),
        )

    def test_invalid_combinations_are_rejected(self):
        invalid = [
            ['--seeds', '0,1'],
            ['--curriculum'],
            ['--kl-free-bits-initial', '1'],
            ['--eje-b', '--accel-preset', 'baseline'],
            ['--eje-b', '--seed', '1', '--seeds', '0,1'],
            ['--postprocess-stride', '0'],
            ['--iterations', '20'],
            ['--eje-b', '--seeds', '1,1'],
            ['--eje-b', '--run-label', '../x'],
        ]
        for argv in invalid:
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                analyze_example.parse_args(argv)

    def test_single_seed_flag_feeds_seed_tuple(self):
        args = analyze_example.parse_args(['--eje-b', '--seed', '3'])
        self.assertEqual(args.seeds, (3,))

    def test_pass2_curve_marks_steps_with_the_true_hash(self):
        self.assertEqual(
            analyze_example.pass2_curve([[1, 2], [3, 4], [5, 3]], 3), [0, 1, 1]
        )


class PlotHelperTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def path(self, name):
        return os.path.join(self.tmp.name, name)

    def test_eje_b_plots_are_written(self):
        steps = 30
        kl_curves = {
            '[0, 1, 0, 0, 0]': [5.0 - 0.1 * i for i in range(steps)],
            '[1, 0, 0, 1, 0]': [0.5] * steps,
        }
        floor = [2.0 * (1 - i / (steps - 1)) for i in range(steps)]
        special = {'dims': [(0, 1, 0, 0, 0)], 'colors': [(1, 0, 0)]}
        visualization.plot_kl_components(
            kl_curves, self.path('kl.png'), special_colors=special,
            free_bits_curve=floor,
        )
        visualization.plot_kl_components_seeds(
            {0: kl_curves, 1: {k: [v * 2 for v in c] for k, c in kl_curves.items()}},
            self.path('klseeds.png'), special_colors=special, free_bits_curve=floor,
        )
        visualization.plot_kl_vs_reconstruction(
            kl_curves, [3.0] * steps, self.path('rec.png'),
            effective_total_KL_curve=[6.0] * steps,
        )
        visualization.plot_eje_b_free_bits([1] * steps, floor, self.path('fb.png'))
        visualization.plot_eje_b_curriculum(
            [[0.5, 1.5]] * steps, self.path('cur.png')
        )
        visualization.plot_pass2_curves(
            {'seed 0': [0, 1, 1], 'fused': [0, 0, 1]}, self.path('p2.png'),
            highlight='fused',
        )
        visualization.plot_fused_solution(
            [{'attempt_1': [[1, 2], [0, 1]], 'attempt_2': [[0]]}],
            self.path('fused.png'), title='t',
        )
        for name in ('kl', 'klseeds', 'rec', 'fb', 'cur', 'p2', 'fused'):
            self.assertGreater(os.path.getsize(self.path(name + '.png')), 0)


class FuseSeedsTests(unittest.TestCase):
    def test_fusion_writes_plots_and_attempts(self):
        solution = (((1,),),)
        hashed = hash(solution)

        def logger_data():
            return {
                'solution_contributions_log': [[(hashed, -1.0)]] * 3,
                'solution_picks_history': [[hashed, hashed]] * 3,
                'candidate_evidence': [{
                    'hash': hashed,
                    'solution': [[[1]]],
                    'score': -1.0,
                    'first_step': 0,
                    'source_index': 0,
                }],
            }

        args = argparse.Namespace(
            task_name='abc', run_label='eje_b', seeds=(0, 1),
        )
        results = [
            {'seed': seed, 'pass2': [1, 1, 1], 'logger_data': logger_data(),
             'KL_curves': {'[0, 1, 0, 0, 0]': [1.0, 2.0, 3.0]},
             'free_bits_curve': [2.0, 1.0, 0.0]}
            for seed in (0, 1)
        ]
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                analyze_example.fuse_seeds(args, hashed, results)
                folder = analyze_example.fused_output_dir(args)
                with open(os.path.join(folder, 'abc_fused_attempts.json')) as f:
                    saved = json.load(f)
                self.assertTrue(saved['final_pass_at_2'])
                self.assertEqual(saved['attempts'][0]['attempt_1'], [[1]])
                self.assertTrue(os.path.exists(
                    os.path.join(folder, 'abc_pass2_curves.png')))
                self.assertTrue(os.path.exists(
                    os.path.join(folder, 'abc_fused_solution.png')))
                self.assertTrue(os.path.exists(
                    os.path.join(folder, 'abc_KL_components_seeds.png')))
            finally:
                os.chdir(cwd)


if __name__ == '__main__':
    unittest.main()
