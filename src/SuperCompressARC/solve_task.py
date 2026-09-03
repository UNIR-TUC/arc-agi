import os
import sys
import time
import json
import importlib
import gc
import multiprocessing
import random
import tqdm
import traceback

import numpy as np
import torch

import preprocessing
import train
import arc_compressor
import initializers
import multitensor_systems
import layers
import solution_selection
import visualization
import accel
import task_persistence

"""
A script that solves one puzzle, to be imported and used with parallel_train.py and multiprocessing.
"""

def solve_task(task_name, split, time_limit, n_train_iterations, gpu_id,
               memory_dict, solutions_dict, error_queue, loggers_dict=None,
               progress_dict=None, postprocess_stride=1, accel_config=None,
               partial_split=None, partial_n_steps=None, seed=0, job_id=None,
               partial_fingerprint=None, state_dir=None):
    """
    Solves a puzzle.
    Args:
        task_name (str): The name of the puzzle to solve.
        split (str): 'training', 'evaluation', or 'test'
        time_limit (float): An end time that will cause training to exit early if reached.
        n_train_iterations (int): The number of iterations to train for.
        gpu_id (int): The GPU number to run the solver on.
        memory_dict (multiprocessing.Dict[str, int]): An inter-process shared dict that we
            can store the amount of memory taken by this job in.
        solutions_dict (multiprocessing.Dict[str, list[Dict[str, list[list[int]]]]]): An
            inter-process shared dict that we can store the solution in.
        error_queue (multiprocessing.Queue[Exception]): An inter-process shared queue to
            put errors in when an exception occurs.
        loggers_dict (multiprocessing.Dict, optional): Shared dict to store
            solution_contributions_log and solution_picks_history for predictions.npz.
        progress_dict (multiprocessing.Dict, optional): Shared dict updated every 100
            training steps so the parent process can log percentage progress.
        postprocess_stride (int): Run full pass@2 candidate postprocessing every N
            training steps instead of every step (Eje H, H3). Default 1 (no change).
        accel_config (dict, optional): Serialized accel.AccelConfig (Eje D, §9.6):
            BF16 autocast, torch.compile and host/silicon tuning. None or an
            all-defaults config reproduces the untouched baseline.
        partial_split (str, optional): Persist a final Phase-2 result for this split
            before publishing worker success. None disables persistence.
        partial_n_steps (int, optional): Iteration count stored with the partial.
    """

    result_key = job_id or task_name
    last_step = -1
    failure_stage = 'setup'
    effective_compile_mode = 'unknown'
    try:

        # Eje D: must run before any GPU tensor exists, so allocator/Inductor
        # environment variables and the matmul precision policy take effect.
        accel_cfg = accel.configure_process(accel_config)
        effective_compile_mode = accel_cfg.compile_mode

        torch.set_default_device('cuda')
        torch.cuda.set_device(gpu_id)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.reset_peak_memory_stats()  # Measure the memory used.

        # Get the task
        with open(f'dataset/arc-agi_{split}_challenges.json', 'r') as f:
            problems = json.load(f)
        task = preprocessing.Task(task_name, problems[task_name], None)
        del problems

        # Set up the training
        model = arc_compressor.ARCCompressor(task)
        # Eje D: rebinds model.forward with BF16 autocast / torch.compile.
        # No-op when accel is disabled; never changes the model's parameters.
        accel.apply(model, accel_cfg)
        optimizer = torch.optim.Adam(model.weights_list, lr=0.01, betas=(0.5, 0.9))
        train_history_logger = solution_selection.Logger(task, postprocess_stride=postprocess_stride)
        eje_b_state = train.EjeBTrainingState() if accel_cfg.curriculum else None
        train_history_logger.solution_most_frequent = tuple(((0, 0), (0, 0)) for example_num in range(task.n_test))
        train_history_logger.solution_second_most_frequent = tuple(((0, 0), (0, 0)) for example_num in range(task.n_test))

        # Training loop
        failure_stage = 'training'
        if progress_dict is not None:
            progress_dict[result_key] = 0   # reached the loop: no longer "init"
        for train_step in range(n_train_iterations):
            train.take_step(
                task,
                model,
                optimizer,
                train_step,
                train_history_logger,
                accel_config=accel_cfg,
                n_train_iterations=n_train_iterations,
                eje_b_state=eje_b_state,
            )
            last_step = train_step
            # Every 10 steps, not 100: the parent's stall watchdog needs finer
            # resolution than a slow task's 100-step interval.
            if progress_dict is not None and train_step % 10 == 0:
                progress_dict[result_key] = train_step
            if time.time() > time_limit:
                break

        if progress_dict is not None:
            progress_dict[result_key] = last_step + 1

        # Batch-convert accumulated GPU scalar tensors to floats in a single sync
        # (Eje H, H2) instead of one sync per training step.
        failure_stage = 'postprocess'
        train_history_logger.materialize_curves()

        # Eje D: where did torch.compile spend its time? Compilation dominates
        # this workload, so the breakdown drives which mitigation to pursue.
        report = accel.compile_report(accel_cfg)
        if report:
            print(f'[accel][{task_name}] torch.compile report: {report}', flush=True)

        # Get the solution
        example_list = []
        for example_num in range(task.n_test):
            attempt_1 = [list(row) for row in train_history_logger.solution_most_frequent[example_num]]
            attempt_2 = [list(row) for row in train_history_logger.solution_second_most_frequent[example_num]]
            example_list.append({'attempt_1': attempt_1, 'attempt_2': attempt_2})

        logger_data = None
        if loggers_dict is not None:
            logger_data = {
                'solution_contributions_log': train_history_logger.solution_contributions_log,
                'solution_picks_history':     train_history_logger.solution_picks_history,
                'candidate_evidence':         train_history_logger.candidate_evidence(),
                'eje_b_diagnostics': {
                    'total_KL_raw': train_history_logger.total_KL_curve,
                    'total_KL_effective': (
                        train_history_logger.effective_total_KL_curve
                    ),
                    'kl_free_bits': train_history_logger.kl_free_bits_curve,
                    'n_kl_below_floor': (
                        train_history_logger.n_kl_below_floor_curve
                    ),
                    'curriculum_weights': (
                        train_history_logger.curriculum_weights_curve
                    ),
                    'seed': seed,
                },
            }

        # Measure actual GPU memory BEFORE cleanup: includes HIP/CUDA context +
        # PyTorch reserved allocator pool + active tensors.  Much more accurate
        # than max_memory_allocated(), which misses the per-process context overhead
        # (~200-400 MB on ROCm/RDNA4) that caused Phase 2 to schedule too many tasks.
        torch.cuda.synchronize()
        free_now, total_vram = torch.cuda.mem_get_info()
        task_peak_memory = total_vram - free_now

        if partial_split is not None:
            failure_stage = 'persistence'
            if partial_n_steps is None:
                raise ValueError('partial_n_steps is required with partial_split')
            if partial_fingerprint is None:
                task_persistence.save_task_partial(
                    partial_split, task_name, partial_n_steps,
                    example_list, logger_data,
                )
            else:
                task_persistence.save_seed_partial(
                    partial_split, task_name, partial_n_steps, seed,
                    partial_fingerprint, example_list, logger_data,
                    state_dir=state_dir,
                )

        failure_stage = 'publish'
        del task
        del model
        del optimizer
        del train_history_logger
        torch.cuda.empty_cache()
        gc.collect()

        # Store the result
        memory_dict[result_key] = task_peak_memory
        solutions_dict[result_key] = example_list
        if loggers_dict is not None:
            loggers_dict[result_key] = logger_data

    except BaseException as exc:
        failure = {
            'task_name': task_name,
            'job_id': result_key,
            'seed': seed,
            'gpu_id': gpu_id,
            'pid': os.getpid(),
            'last_step': last_step,
            'stage': failure_stage,
            'compile_mode': effective_compile_mode,
            'exception_type': type(exc).__name__,
            'traceback': traceback.format_exc(),
        }
        try:
            error_queue.put(failure)
        except BaseException:
            pass
        raise
