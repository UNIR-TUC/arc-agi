import json
import os
import queue
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import parallel_train
import solve_task


class RecordingQueue:
    def __init__(self):
        self.records = []

    def put(self, record):
        self.records.append(record)


class FakeProcess:
    def __init__(self, stubborn=False):
        self.alive = True
        self.stubborn = stubborn
        self.terminate_calls = 0
        self.kill_calls = 0
        self.join_calls = []

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.terminate_calls += 1
        if not self.stubborn:
            self.alive = False

    def kill(self):
        self.kill_calls += 1
        self.alive = False

    def join(self, timeout=None):
        self.join_calls.append(timeout)


class SolveTaskFailureTests(unittest.TestCase):
    def test_setup_failure_is_reported_and_reraised(self):
        error_queue = RecordingQueue()
        with mock.patch.object(
            solve_task.accel, 'configure_process',
            side_effect=RuntimeError('injected setup failure'),
        ):
            with self.assertRaisesRegex(RuntimeError, 'injected setup failure'):
                solve_task.solve_task(
                    '007bbfb7', 'training', 1e20, 2000, 0,
                    {}, {}, error_queue,
                )

        self.assertEqual(len(error_queue.records), 1)
        failure = error_queue.records[0]
        self.assertEqual(failure['task_name'], '007bbfb7')
        self.assertEqual(failure['gpu_id'], 0)
        self.assertEqual(failure['last_step'], -1)
        self.assertEqual(failure['stage'], 'setup')
        self.assertEqual(failure['exception_type'], 'RuntimeError')
        self.assertIn('RuntimeError: injected setup failure', failure['traceback'])

    def test_success_persists_before_publishing_manager_outputs(self):
        memory_dict = {}
        solutions_dict = {}
        loggers_dict = {}
        error_queue = RecordingQueue()
        task = SimpleNamespace(n_test=1)
        model = SimpleNamespace(weights_list=[])
        logger = SimpleNamespace(
            solution_most_frequent=(((1,),),),
            solution_second_most_frequent=(((2,),),),
            solution_contributions_log=[],
            solution_picks_history=[],
            total_KL_curve=[],
            effective_total_KL_curve=[],
            kl_free_bits_curve=[],
            n_kl_below_floor_curve=[],
            curriculum_weights_curve=[],
            candidate_evidence=mock.Mock(return_value=[]),
            materialize_curves=mock.Mock(),
        )

        def assert_not_published(*args, **kwargs):
            self.assertEqual(memory_dict, {})
            self.assertEqual(solutions_dict, {})
            self.assertEqual(loggers_dict, {})

        with (
            mock.patch('builtins.open', mock.mock_open(read_data='{"007bbfb7": {}}')),
            mock.patch.object(solve_task.accel, 'configure_process'),
            mock.patch.object(solve_task.accel, 'apply'),
            mock.patch.object(solve_task.accel, 'compile_report', return_value=None),
            mock.patch.object(solve_task.preprocessing, 'Task', return_value=task),
            mock.patch.object(
                solve_task.arc_compressor, 'ARCCompressor', return_value=model,
            ),
            mock.patch.object(solve_task.solution_selection, 'Logger', return_value=logger),
            mock.patch.object(solve_task.train, 'take_step'),
            mock.patch.object(solve_task.torch, 'set_default_device'),
            mock.patch.object(solve_task.torch.cuda, 'set_device'),
            mock.patch.object(solve_task.torch.cuda, 'reset_peak_memory_stats'),
            mock.patch.object(solve_task.torch.cuda, 'synchronize'),
            mock.patch.object(
                solve_task.torch.cuda, 'mem_get_info', return_value=(900, 1000),
            ),
            mock.patch.object(solve_task.torch.cuda, 'empty_cache'),
            mock.patch.object(solve_task.torch.optim, 'Adam', return_value=object()),
            mock.patch.object(solve_task.gc, 'collect'),
            mock.patch.object(
                solve_task.task_persistence, 'save_task_partial',
                side_effect=assert_not_published,
            ) as save_partial,
        ):
            solve_task.solve_task(
                '007bbfb7', 'training', 1e20, 1, 0,
                memory_dict, solutions_dict, error_queue,
                loggers_dict=loggers_dict,
                partial_split='training', partial_n_steps=1,
            )

        save_partial.assert_called_once()
        self.assertEqual(memory_dict['007bbfb7'], 100)
        self.assertTrue(solutions_dict['007bbfb7'])
        self.assertIn('007bbfb7', loggers_dict)
        self.assertEqual(error_queue.records, [])


class SchedulerFailureTests(unittest.TestCase):
    def test_saving_filtered_measurements_preserves_existing_cache_entries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            previous_cwd = os.getcwd()
            os.chdir(temp_dir)
            try:
                fingerprint = {'gpu': 'test'}
                with open('memory_cache_training.json', 'w') as handle:
                    json.dump({
                        'fingerprint': fingerprint,
                        'measurements': {'quarantined': 100},
                    }, handle)
                with mock.patch.object(
                    parallel_train, '_gpu_fingerprint',
                    return_value=fingerprint,
                ):
                    parallel_train.save_memory_cache(
                        'training', 1, {'runnable': 200}, {},
                    )

                with open('memory_cache_training.json') as handle:
                    measurements = json.load(handle)['measurements']
                self.assertEqual(
                    measurements, {'quarantined': 100, 'runnable': 200},
                )
            finally:
                os.chdir(previous_cwd)

    def test_quarantined_retry_activation_survives_an_unrelated_restart(self):
        recovery_entries = {'fcb5c309': {'state': 'quarantined'}}
        persisted = {
            'state': 'retry_eager',
            'updated_at': 'now',
            'failures': [],
        }
        with mock.patch.object(
            parallel_train.task_persistence, 'update_task_recovery',
            return_value=persisted,
        ) as update:
            parallel_train._activate_quarantined_retries(
                'training', 2000, {'fcb5c309'}, recovery_entries,
            )

        update.assert_called_once_with(
            'training', 'fcb5c309', 2000, 'retry_eager',
            state_dir=None,
        )
        self.assertEqual(recovery_entries['fcb5c309'], persisted)

    def test_quarantined_run_writes_ordered_degraded_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            previous_cwd = os.getcwd()
            os.chdir(temp_dir)
            try:
                os.makedirs('dataset')
                with open(
                    'dataset/arc-agi_training_challenges.json',
                    'w', encoding='utf-8',
                ) as handle:
                    json.dump({
                        'fcb5c309': {'test': [{'input': [[1]]}]},
                        'real': {'test': [{'input': [[2]]}]},
                    }, handle)
                logger = SimpleNamespace(
                    log_run_start=mock.Mock(),
                    info=mock.Mock(),
                    warning=mock.Mock(),
                    log_phase=mock.Mock(),
                    log_run_summary=mock.Mock(),
                    finalize_results=mock.Mock(),
                )
                real_solution = [{
                    'attempt_1': [[2]],
                    'attempt_2': [[2]],
                }]
                real_logger = {
                    'solution_contributions_log': [],
                    'solution_picks_history': [],
                }
                with (
                    mock.patch.object(
                        parallel_train.task_persistence,
                        'load_task_recovery',
                        return_value={
                            'fcb5c309': {'state': 'quarantined'},
                        },
                    ),
                    mock.patch.object(
                        parallel_train, 'load_task_partials',
                        return_value=(
                            {'real': real_solution},
                            {'real': real_logger},
                        ),
                    ),
                    mock.patch.object(
                        parallel_train, 'load_memory_cache',
                        return_value={'real': 100},
                    ),
                    mock.patch.object(
                        parallel_train.torch.cuda, 'mem_get_info',
                        return_value=(2 * 1024**3, 2 * 1024**3),
                    ),
                    mock.patch.object(
                        parallel_train.accel, 'backend_info',
                        return_value={},
                    ),
                ):
                    n_solved, n_tasks, _, _ = parallel_train.run_split(
                        'training', 1, 1, logger,
                        {
                            'fcb5c309': [[[0, 0], [0, 0]]],
                            'real': [[[2]]],
                        },
                        accel_cfg=parallel_train.accel.AccelConfig(),
                        n_steps=2000,
                        resume=True,
                        recover_task_failures=True,
                    )

                with open('submission_training.json', encoding='utf-8') as handle:
                    submission = json.load(handle)
                with open('run_metadata_training.json', encoding='utf-8') as handle:
                    metadata = json.load(handle)
                self.assertEqual(list(submission), ['fcb5c309', 'real'])
                self.assertEqual(n_solved, 1)
                self.assertEqual(n_tasks, 2)
                self.assertTrue(metadata['degraded'])
                self.assertEqual(metadata['real_result_tasks'], 1)
                self.assertEqual(
                    metadata['recovery']['quarantined_tasks'], ['fcb5c309'],
                )
            finally:
                os.chdir(previous_cwd)

    def test_quarantine_fallback_cannot_count_as_solved(self):
        fallback = parallel_train._quarantined_fallback(1)
        truth = [[[0, 0], [0, 0]]]

        self.assertEqual(
            parallel_train._count_solved_tasks(
                {'failed': fallback}, {'failed': truth}, {'failed'},
            ),
            0,
        )
        self.assertEqual(
            parallel_train._count_solved_tasks(
                {'real': fallback}, {'real': truth}, set(),
            ),
            1,
        )

    def test_quarantine_fallback_matches_initial_guess_and_logger_schema(self):
        solutions = {'real': [{'attempt_1': [[1]], 'attempt_2': [[2]]}]}
        loggers = {'real': {
            'solution_contributions_log': [1],
            'solution_picks_history': [2],
        }}

        parallel_train._apply_quarantined_fallbacks(
            solutions, loggers, {'failed'}, {'failed': 2},
        )

        self.assertEqual(len(solutions['failed']), 2)
        self.assertEqual(
            solutions['failed'][0],
            {
                'attempt_1': [[0, 0], [0, 0]],
                'attempt_2': [[0, 0], [0, 0]],
            },
        )
        self.assertTrue(parallel_train.task_persistence.is_complete_logger(
            loggers['failed']
        ))
        self.assertEqual(solutions['real'][0]['attempt_1'], [[1]])

    def test_recovery_policy_combines_durable_and_operator_state(self):
        entries = {
            'auto_eager': {'state': 'retry_eager'},
            'known_eager': {'state': 'recovered_eager'},
            'skip_me': {'state': 'quarantined'},
            'retry_me': {'state': 'quarantined'},
        }

        eager, skipped = parallel_train._recovery_task_sets(
            list(entries) + ['explicit'], entries, {'explicit'}, {'retry_me'},
        )

        self.assertEqual(
            eager, {'auto_eager', 'known_eager', 'retry_me', 'explicit'},
        )
        self.assertEqual(skipped, {'skip_me'})

    def test_task_id_list_parser_trims_and_deduplicates(self):
        self.assertEqual(
            parallel_train._parse_task_id_list(' fcb5c309,007bbfb7,fcb5c309 '),
            {'fcb5c309', '007bbfb7'},
        )
        with self.assertRaisesRegex(Exception, 'unsafe task name'):
            parallel_train._parse_task_id_list('../escape')

    def test_recovery_policy_rejects_unknown_and_non_quarantined_retry(self):
        with self.assertRaisesRegex(ValueError, 'unknown recovery task'):
            parallel_train._recovery_task_sets(
                ['known'], {}, {'unknown'}, set(),
            )
        with self.assertRaisesRegex(ValueError, 'not quarantined'):
            parallel_train._recovery_task_sets(
                ['known'], {}, set(), {'known'},
            )

    def test_compiled_failure_demotes_then_eager_failure_quarantines(self):
        with mock.patch.object(
            parallel_train.task_persistence, 'update_task_recovery',
        ) as update:
            compiled_failure = parallel_train.WorkerFailure(
                'compiled failed', task_name='fcb5c309', gpu_id=0,
                exit_code=-6, last_step=10, stage='training',
                compile_mode='default', recoverable_task=True,
            )
            eager_failure = parallel_train.WorkerFailure(
                'eager failed', task_name='fcb5c309', gpu_id=0,
                exit_code=-6, last_step=10, stage='training',
                compile_mode='off', recoverable_task=True,
            )

            self.assertTrue(parallel_train._record_task_recovery_failure(
                'training', 2000, compiled_failure, True, None,
            ))
            self.assertTrue(parallel_train._record_task_recovery_failure(
                'training', 2000, eager_failure, True, None,
            ))

        self.assertEqual(update.call_args_list[0].args[3], 'retry_eager')
        self.assertEqual(update.call_args_list[1].args[3], 'quarantined')

    def test_eje_b_failure_is_recorded_for_eager_recovery(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            previous_cwd = os.getcwd()
            os.chdir(temp_dir)
            try:
                os.makedirs('dataset')
                with open(
                    'dataset/arc-agi_training_challenges.json', 'w',
                    encoding='utf-8',
                ) as handle:
                    json.dump({'fcb5c309': {'test': [{'input': [[1]]}]}}, handle)

                logger = SimpleNamespace(
                    info=mock.Mock(),
                    warning=mock.Mock(),
                    log_run_start=mock.Mock(),
                    log_phase=mock.Mock(),
                )
                failure = parallel_train.WorkerFailure(
                    'compiled failed', task_name='fcb5c309', gpu_id=0,
                    exit_code=-6, last_step=10, stage='training',
                    compile_mode='default', recoverable_task=True,
                    seed=0, job_id='fcb5c309__seed_0',
                )
                config = parallel_train.accel.AccelConfig(
                    compile_mode='default', eje_b=True,
                    seeds=(0, 1, 2, 3), kl_free_bits_initial=2.0,
                    curriculum=True,
                )
                with (
                    mock.patch.object(
                        parallel_train.task_persistence,
                        'load_task_recovery', return_value={},
                    ),
                    mock.patch.object(
                        parallel_train, 'load_memory_cache',
                        return_value={'fcb5c309': 100},
                    ),
                    mock.patch.object(
                        parallel_train.torch.cuda, 'mem_get_info',
                        return_value=(2 * 1024**3, 2 * 1024**3),
                    ),
                    mock.patch.object(
                        parallel_train, 'parallelize_runs',
                        side_effect=failure,
                    ),
                    mock.patch.object(
                        parallel_train.task_persistence,
                        'update_task_recovery',
                    ) as update,
                ):
                    with self.assertRaisesRegex(
                        parallel_train.WorkerFailure, 'compiled failed',
                    ):
                        parallel_train.run_split(
                            'training', 1, 1, logger, None,
                            accel_cfg=config, n_steps=2000,
                            recover_task_failures=True,
                            state_dir='state',
                        )
            finally:
                os.chdir(previous_cwd)

        self.assertEqual(update.call_args.args[:4], (
            'training', 'fcb5c309', 2000, 'retry_eager',
        ))

    def test_non_task_failure_is_not_recorded_for_recovery(self):
        failure = parallel_train.WorkerFailure(
            'persist failed', task_name='fcb5c309', stage='persistence',
            compile_mode='off', recoverable_task=False,
        )
        with mock.patch.object(
            parallel_train.task_persistence, 'update_task_recovery',
        ) as update:
            self.assertFalse(parallel_train._record_task_recovery_failure(
                'training', 2000, failure, True, None,
            ))
        update.assert_not_called()

    def test_task_acceleration_override_only_changes_named_task(self):
        default = {'compile_mode': 'default', 'matmul_precision': 'high'}
        eager = {'compile_mode': 'off', 'matmul_precision': 'high'}

        selected = parallel_train._effective_task_accel_config(
            'fcb5c309', default, {'fcb5c309': eager},
        )
        untouched = parallel_train._effective_task_accel_config(
            '007bbfb7', default, {'fcb5c309': eager},
        )

        self.assertEqual(selected['compile_mode'], 'off')
        self.assertEqual(untouched['compile_mode'], 'default')
        self.assertEqual(selected['matmul_precision'], 'high')

    def test_eje_b_task_eager_override_preserves_algorithm_settings(self):
        default = parallel_train.accel.AccelConfig(
            compile_mode='default',
            eje_b=True,
            seeds=(0, 1, 2, 3),
            kl_free_bits_initial=2.0,
            curriculum=True,
        )
        eager = parallel_train._task_eager_config(default.to_dict())
        selected = parallel_train._effective_task_accel_config(
            'fcb5c309', default.to_dict(), {'fcb5c309': eager},
        )
        untouched = parallel_train._effective_task_accel_config(
            '007bbfb7', default.to_dict(), {'fcb5c309': eager},
        )

        self.assertEqual(selected['compile_mode'], 'off')
        self.assertTrue(selected['_eje_b_eager_override'])
        self.assertEqual(selected['seeds'], (0, 1, 2, 3))
        self.assertEqual(selected['kl_free_bits_initial'], 2.0)
        self.assertTrue(selected['curriculum'])
        self.assertEqual(untouched['compile_mode'], 'default')

    def test_seed_jobs_preserve_usage_and_skip_only_completed_jobs(self):
        job_ids, usages, specs = parallel_train._expand_seed_jobs(
            ['task_a', 'task_b'], [100, 200], (0, 1), {'task_a__seed_0'},
        )

        self.assertEqual(
            job_ids, ['task_a__seed_1', 'task_b__seed_0', 'task_b__seed_1']
        )
        self.assertEqual(usages, [100, 200, 200])
        self.assertEqual(
            parallel_train._resolve_job('task_b__seed_1', specs),
            ('task_b', 1),
        )

    def test_worker_failure_keeps_seed_job_identity(self):
        failure = parallel_train._worker_failure_from_report({
            'task_name': '007bbfb7',
            'job_id': '007bbfb7__seed_3',
            'seed': 3,
            'gpu_id': 0,
            'last_step': 12,
            'stage': 'training',
            'compile_mode': 'default',
            'exception_type': 'RuntimeError',
            'traceback': 'RuntimeError: failed',
        })

        self.assertEqual(failure.task_name, '007bbfb7')
        self.assertEqual(failure.job_id, '007bbfb7__seed_3')
        self.assertEqual(failure.seed, 3)
        self.assertEqual(failure.recovery_record()['seed'], 3)

    def test_scheduler_rejects_zero_gpus_before_starting_manager(self):
        with mock.patch.object(
            parallel_train.multiprocessing, 'Manager',
            side_effect=AssertionError('manager must not start'),
        ):
            with self.assertRaisesRegex(ValueError, 'at least one GPU'):
                parallel_train.parallelize_runs(
                    [], [1], 2, ['007bbfb7'], 'training', 1, 0, 1,
                )

    def test_signal_exit_diagnoses_possible_oom_without_claiming_it(self):
        failure = parallel_train._worker_failure_from_exit(
            '007bbfb7', 0, -9, 420,
        )
        message = str(failure)

        self.assertIn('SIGKILL', message)
        self.assertIn('no Python traceback is possible', message)
        self.assertIn('may indicate the OS OOM killer', message)
        self.assertIn('after step 420', message)
        self.assertTrue(failure.recoverable_task)
        self.assertEqual(failure.task_name, '007bbfb7')
        self.assertEqual(failure.stage, 'training')

    def test_zero_exit_requires_all_expected_outputs(self):
        with self.assertRaisesRegex(
            parallel_train.WorkerFailure,
            'solution, logger data',
        ):
            parallel_train._validate_worker_outputs(
                '007bbfb7', {'007bbfb7': 123}, {}, {}, True,
            )

        parallel_train._validate_worker_outputs(
            '007bbfb7',
            {'007bbfb7': 123},
            {'007bbfb7': [{'attempt_1': [[1]], 'attempt_2': [[2]]}]},
            {'007bbfb7': {
                'solution_contributions_log': [],
                'solution_picks_history': [],
            }},
            True,
        )

    def test_structured_queue_error_keeps_task_and_traceback(self):
        error_queue = queue.Queue()
        error_queue.put({
            'task_name': '007bbfb7',
            'gpu_id': 0,
            'pid': 42,
            'last_step': 99,
            'stage': 'training',
            'compile_mode': 'default',
            'exception_type': 'RuntimeError',
            'traceback': 'Traceback\nRuntimeError: failed',
        })

        records = parallel_train._drain_worker_errors(error_queue)
        failure = parallel_train._worker_failure_from_report(records[0])
        message = str(failure)
        self.assertEqual(len(records), 1)
        self.assertIn('007bbfb7', message)
        self.assertIn('after step 99', message)
        self.assertIn('RuntimeError: failed', message)
        self.assertTrue(failure.recoverable_task)
        self.assertEqual(failure.compile_mode, 'default')
        self.assertEqual(failure.recovery_record()['last_step'], 99)

    def test_attempt_failure_terminates_and_kills_stubborn_peers(self):
        process = FakeProcess(stubborn=True)

        with self.assertRaisesRegex(parallel_train.WorkerFailure, 'injected'):
            with parallel_train._worker_process_guard([process], None):
                raise parallel_train.WorkerFailure('injected')

        self.assertEqual(process.terminate_calls, 1)
        self.assertEqual(process.kill_calls, 1)
        self.assertFalse(process.is_alive())


if __name__ == '__main__':
    unittest.main()