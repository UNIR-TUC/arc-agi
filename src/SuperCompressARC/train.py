import time

import numpy as np
import torch

import preprocessing
import arc_compressor
import initializers
import multitensor_systems
import layers
import solution_selection
import visualization
import accel


"""
This file trains a model for every ARC-AGI task in a split.
"""

np.random.seed(0)
torch.manual_seed(0)


class EjeBTrainingState:
    """Mutable per-task state used by the hard-example curriculum."""

    def __init__(self):
        self.demo_loss_ema = None


def _sum_kl_with_free_bits(KL_amounts, free_bits):
    """Return raw/effective KL totals using one free-bits floor per leaf."""
    total_raw = 0
    total_effective = 0
    below_floor = []
    for KL_amount in KL_amounts:
        leaf_KL = torch.sum(KL_amount)
        total_raw = total_raw + leaf_KL
        total_effective = total_effective + torch.clamp(
            leaf_KL, min=free_bits
        )
        below_floor.append(leaf_KL.detach() < free_bits)
    n_below_floor = torch.stack(below_floor).sum()
    return total_raw, total_effective, n_below_floor


def _curriculum_weights(previous_losses, n_train, config, train_step,
                        n_train_iterations, device, dtype):
    """Compute mean-one weights from detached historical per-pixel losses."""
    if previous_losses is None or n_train <= 1:
        return torch.ones(n_train, device=device, dtype=dtype)
    beta = config.curriculum_beta_at_step(train_step, n_train_iterations)
    difficulty = previous_losses.detach()
    difficulty = difficulty - torch.mean(difficulty)
    difficulty = difficulty / (
        torch.std(difficulty, correction=0) + 1e-6
    )
    return n_train * torch.softmax(beta * difficulty, dim=0)


def _update_curriculum_state(state, pair_losses, pixel_counts, decay):
    current_losses = torch.stack([
        pair_loss.detach() / max(pixel_count, 1)
        for pair_loss, pixel_count in zip(pair_losses, pixel_counts)
    ])
    if state.demo_loss_ema is None:
        state.demo_loss_ema = current_losses
    else:
        state.demo_loss_ema = (
            decay * state.demo_loss_ema + (1 - decay) * current_losses
        )


def mask_select_logprobs(mask, length):
    """
    Figure out the unnormalized log probability of taking each slice given the output mask.

    Vectorized (Eje H, H1): logprob(offset) = -sum(mask[:offset]) +
    sum(mask[offset:offset+length]) - sum(mask[offset+length:]). With
    prefix = cumsum(mask) prepended with a 0 and total = prefix[-1]:
        sum(mask[offset:offset+length]) = prefix[offset+length] - prefix[offset]
        sum(mask[offset+length:])        = total - prefix[offset+length]
    so logprob(offset) = 2*prefix[offset+length] - 2*prefix[offset] - total, for every
    offset at once via one cumsum + vectorized slicing instead of a Python loop.
    """
    n_offsets = mask.shape[0] - length + 1
    prefix = torch.cat([torch.zeros_like(mask[:1]), torch.cumsum(mask, dim=0)], dim=0)
    total = prefix[-1]
    logprobs = 2*prefix[length:length+n_offsets] - 2*prefix[:n_offsets] - total
    log_partition = torch.logsumexp(logprobs, dim=0)
    return log_partition, logprobs

def _grid_reconstruction_error(task, logits, x_mask, y_mask, example_num,
                               in_out_mode, train_step):
    """Return the reconstruction error for one visible input/output grid."""
    grid_size_uncertain = not (
        task.in_out_same_size
        or task.all_out_same_size and in_out_mode == 1
        or task.all_in_same_size and in_out_mode == 0
    )
    if grid_size_uncertain:
        coefficient = 0.01**max(0, 1-train_step/100)
    else:
        coefficient = 1
    logits_slice = logits[example_num, :, :, :, in_out_mode]
    problem_slice = task.problem[example_num, :, :, in_out_mode]
    output_shape = task.shapes[example_num][in_out_mode]
    x_log_partition, x_logprobs = mask_select_logprobs(
        coefficient*x_mask[example_num, :, in_out_mode], output_shape[0]
    )
    y_log_partition, y_logprobs = mask_select_logprobs(
        coefficient*y_mask[example_num, :, in_out_mode], output_shape[1]
    )
    if grid_size_uncertain:
        x_log_partitions = []
        y_log_partitions = []
        for length in range(1, x_mask.shape[1]+1):
            x_log_partitions.append(mask_select_logprobs(
                coefficient*x_mask[example_num, :, in_out_mode], length
            )[0])
        for length in range(1, y_mask.shape[1]+1):
            y_log_partitions.append(mask_select_logprobs(
                coefficient*y_mask[example_num, :, in_out_mode], length
            )[0])
        x_log_partition = torch.logsumexp(
            torch.stack(x_log_partitions, dim=0), dim=0
        )
        y_log_partition = torch.logsumexp(
            torch.stack(y_log_partitions, dim=0), dim=0
        )

    n_x_offsets = x_logprobs.shape[0]
    n_y_offsets = y_logprobs.shape[0]
    target_crop = problem_slice[:output_shape[0], :output_shape[1]]
    logits_crops = logits_slice.unfold(1, output_shape[0], 1).unfold(
        2, output_shape[1], 1
    )
    logits_crops = logits_crops.permute(1, 2, 0, 3, 4).reshape(
        n_x_offsets*n_y_offsets,
        logits_slice.shape[0],
        output_shape[0],
        output_shape[1],
    )
    target_batch = target_crop.unsqueeze(0).expand(
        n_x_offsets*n_y_offsets, -1, -1
    )
    ce = torch.nn.functional.cross_entropy(
        logits_crops, target_batch, reduction='none'
    )
    ce_sum = ce.sum(dim=(1, 2)).reshape(n_x_offsets, n_y_offsets)
    logprobs = (
        x_logprobs[:, None] - x_log_partition
        + y_logprobs[None, :] - y_log_partition
        - ce_sum
    )
    if grid_size_uncertain:
        coefficient = 0.1**max(0, 1-train_step/100)
    else:
        coefficient = 1
    logprob = torch.logsumexp(
        coefficient*logprobs, dim=(0, 1)
    ) / coefficient
    return -logprob


def take_step(task, model, optimizer, train_step, train_history_logger,
              accel_config=None, n_train_iterations=None,
              eje_b_state=None):
    """
    Runs a forward pass of the model on the ARC-AGI task.
    Args:
        task (Task): The ARC-AGI task containing the problem.
        model (ArcCompressor): The VAE decoder model to run the forward pass with.
        optimizer (torch.optim.Optimizer): The optimizer used to take the step on the model weights.
        train_step (int): The training iteration number.
        train_history_logger (Logger): A logger object used for logging the forward pass outputs
                of the model, as well as accuracy and other things.
    """

    config = accel.AccelConfig.from_dict(accel_config)
    if config.eje_b and not n_train_iterations:
        raise ValueError('n_train_iterations is required when Eje B is enabled')
    if config.curriculum and eje_b_state is None:
        raise ValueError('eje_b_state is required when curriculum is enabled')

    optimizer.zero_grad()
    logits, x_mask, y_mask, KL_amounts, KL_names, = model.forward()
    logits = torch.cat([torch.zeros_like(logits[:,:1,:,:]), logits], dim=1)  # add black color to logits

    # Compute the total KL loss
    kl_free_bits = 0.0
    n_kl_below_floor = None
    if config.eje_b:
        kl_free_bits = config.kl_free_bits_at_step(
            train_step, n_train_iterations
        )
        total_KL, effective_total_KL, n_kl_below_floor = (
            _sum_kl_with_free_bits(KL_amounts, kl_free_bits)
        )
    else:
        total_KL = 0
        for KL_amount in KL_amounts:
            total_KL = total_KL + torch.sum(KL_amount)
        effective_total_KL = total_KL

    # Compute the reconstruction error
    curriculum_weights = None
    if config.curriculum:
        pair_losses = [0 for _ in range(task.n_train)]
        pixel_counts = [0 for _ in range(task.n_train)]
        test_input_error = 0
        for example_num in range(task.n_examples):
            modes = range(2) if example_num < task.n_train else range(1)
            for in_out_mode in modes:
                grid_error = _grid_reconstruction_error(
                    task, logits, x_mask, y_mask,
                    example_num, in_out_mode, train_step,
                )
                if example_num < task.n_train:
                    pair_losses[example_num] = (
                        pair_losses[example_num] + grid_error
                    )
                    shape = task.shapes[example_num][in_out_mode]
                    pixel_counts[example_num] += shape[0] * shape[1]
                else:
                    test_input_error = test_input_error + grid_error

        curriculum_weights = _curriculum_weights(
            eje_b_state.demo_loss_ema,
            task.n_train,
            config,
            train_step,
            n_train_iterations,
            logits.device,
            logits.dtype,
        )
        reconstruction_error = test_input_error
        for example_num, pair_loss in enumerate(pair_losses):
            reconstruction_error = (
                reconstruction_error
                + curriculum_weights[example_num] * pair_loss
            )
        _update_curriculum_state(
            eje_b_state,
            pair_losses,
            pixel_counts,
            config.curriculum_ema_decay,
        )
    else:
        reconstruction_error = 0
        for example_num in range(task.n_examples):
            for in_out_mode in range(2):
                if example_num >= task.n_train and in_out_mode == 1:
                    continue
                reconstruction_error = (
                    reconstruction_error
                    + _grid_reconstruction_error(
                        task, logits, x_mask, y_mask,
                        example_num, in_out_mode, train_step,
                    )
                )

    loss = effective_total_KL + 10*reconstruction_error
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Performance recording
    train_history_logger.log(train_step,
                             logits,
                             x_mask,
                             y_mask,
                             KL_amounts,
                             KL_names,
                             total_KL,
                             reconstruction_error,
                             loss,
                             effective_total_KL=effective_total_KL,
                             kl_free_bits=kl_free_bits,
                             n_kl_below_floor=n_kl_below_floor,
                             curriculum_weights=curriculum_weights)


if __name__ == "__main__":
    start_time = time.time()

    task_nums = list(range(400))
    split = "training"  # "training", "evaluation, or "test"

    # Preprocess all tasks, make models, optimizers, and loggers. Make plots.
    tasks = preprocessing.preprocess_tasks(split, task_nums)
    models = []
    optimizers = []
    train_history_loggers = []
    for task in tasks:
        model = arc_compressor.ARCCompressor(task)
        models.append(model)
        optimizer = torch.optim.Adam(model.weights_list, lr=0.01, betas=(0.5, 0.9))
        optimizers.append(optimizer)
        train_history_logger = solution_selection.Logger(task)
        visualization.plot_problem(train_history_logger)
        train_history_loggers.append(train_history_logger)

    # Get the solution hashes so that we can check for correctness
    true_solution_hashes = [task.solution_hash for task in tasks]

    # Train the models one by one
    for i, (task, model, optimizer, train_history_logger) in enumerate(zip(tasks, models, optimizers, train_history_loggers)):
        n_iterations = 2000
        # n_iterations = 1500
        for train_step in range(n_iterations):
            take_step(task, model, optimizer, train_step, train_history_logger)
        train_history_logger.materialize_curves()
        visualization.plot_solution(train_history_logger)
        solution_selection.save_predictions(train_history_loggers[:i+1])
        solution_selection.plot_accuracy(true_solution_hashes)

    # Write down how long it all took
    with open('timing_result.txt', 'w') as f:
        f.write("Time elapsed in seconds: " + str(time.time() - start_time))
