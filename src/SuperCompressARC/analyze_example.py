import argparse
import csv
import gc
import json
import os
import random
import time
from tqdm import tqdm

import numpy as np
import torch
import matplotlib.pyplot as plt

import train
import preprocessing
import arc_compressor
import initializers
import multitensor_systems
import layers
import solution_selection
import visualization
import accel


"""
This file allows you to train one model on one task, and see plots of what
the process and end result looks like. You can input the training split and
the task code, and it will:
- Train a model for 1500 steps,
- Plot sampled solutions from the model at every 50 steps,
- Plot the KL and reconstruction error over time,
- Plot the contribution of each tensor shape to the KL over time,
- Show top principal components of each tensor that still contributes to
  the KL at the end of training.

With --eje-b (Eje B, compile preset only) it additionally trains every seed in
--seeds, annotates the step plots with the free-bits floor and curriculum
weights, adds free-bits, curriculum and pass@2 plots, and fuses the seeds with
the same policy as parallel_train.py. Eje B results go to
results/<task>/<run-label>/seed_<s>/ and .../fused/ so earlier plots in
results/<task>/ are not overwritten.
"""

# For some reason trying to set the seed doesn't actually fix results.
# Just run things over and over again until you see desired interesting behaviors.
np.random.seed(0)
torch.manual_seed(0)
torch.set_default_dtype(torch.float32)

WARMUP_STEPS = 20
PLOT_INTERVAL = 50
DEFAULT_POSTPROCESS_STRIDE = 4  # same default as parallel_train.py
DEFAULT_INDUCTOR_CACHE_DIR = '/mnt/supercompressarc-cache/.inductor_cache'

# For specific tasks that we found interesting, color some KL components differently.
SPECIAL_CURVE_COLORS = {
    '272f95fa': {
        'dims': [(1,0,0,1,0), (1,0,0,0,1), (0,1,1,0,0), (0,1,0,0,0)],
        'colors': [(1, 0, 0), (0, 1, 0), (0, 0.5, 1), (0.5, 0, 1)]
    },
    '6cdd2623': {
        'dims': [(1,0,0,1,0), (1,0,0,0,1), (1,1,0,0,0), (1,0,0,1,1), (0,0,1,0,0)],
        'colors': [(1, 0.6, 0), (0, 1, 0), (0, 0.5, 1), (0.5, 0, 1), (1, 0, 0.5)]
    },
    '41e4d17e': {
        'dims': [(1,0,0,1,1), (0,1,0,0,0)],
        'colors': [(1, 0, 0), (0, 0, 1)]
    },
    '6d75e8bb': {
        'dims': [(1,0,0,1,0), (1,0,0,0,1), (1,0,0,1,1), (0,1,0,0,0)],
        'colors': [(1, 0, 0), (0, 1, 0), (0, 0.5, 1), (0.5, 0, 1)]
    }
}


def _parse_seed_list(text):
    try:
        seeds = tuple(int(item) for item in text.split(','))
    except ValueError:
        raise argparse.ArgumentTypeError(f'invalid seed list: {text!r}')
    if any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError(
            'seeds must be unique non-negative integers'
        )
    return seeds


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Train and analyse one ARC-AGI task.',
    )
    parser.add_argument(
        '--task-name',
        default='007bbfb7',
        help='ARC task identifier. Default: 007bbfb7.',
    )
    parser.add_argument(
        '--split',
        choices=('training', 'evaluation', 'test'),
        default='training',
        help='Dataset split containing the task. Default: training.',
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=2000,
        help='Number of training steps. Must be greater than 20. Default: 2000.',
    )
    parser.add_argument(
        '--accel-preset',
        choices=sorted(accel.PRESETS),
        default=None,
        help=(
            'Acceleration preset from accel.py. "compile" enables the '
            'recommended torch.compile configuration. Default: compile with '
            '--eje-b, baseline otherwise.'
        ),
    )
    parser.add_argument(
        '--inductor-cache-dir',
        default=DEFAULT_INDUCTOR_CACHE_DIR,
        metavar='DIR',
        help=(
            'Persistent TorchInductor cache directory used by compiled presets. '
            f'Default: {DEFAULT_INDUCTOR_CACHE_DIR}. An existing '
            'TORCHINDUCTOR_CACHE_DIR environment variable takes precedence.'
        ),
    )
    parser.add_argument(
        '--run-label',
        default=None,
        help=(
            'Label used in the timing CSV name and, with --eje-b, as the '
            'results sub-folder. Defaults to "eje_b" with --eje-b and to the '
            'accel preset otherwise.'
        ),
    )
    parser.add_argument(
        '--postprocess-stride',
        type=int,
        default=DEFAULT_POSTPROCESS_STRIDE,
        metavar='N',
        help=(
            'Run the pass@2 candidate postprocessing (guess 1/2) every N '
            f'steps (Eje H). Default: {DEFAULT_POSTPROCESS_STRIDE}; use 1 to '
            'match runs made before this option existed.'
        ),
    )
    parser.add_argument(
        '--eje-b',
        action='store_true',
        help=(
            'Enable Eje B: free bits, hard-example curriculum, multi-seed '
            'fusion and the extra plots. Requires --accel-preset compile.'
        ),
    )
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for a single run. Default: 0.',
    )
    seed_group.add_argument(
        '--seeds',
        type=_parse_seed_list,
        default=None,
        metavar='S0,S1,...',
        help='Independent seeds trained in sequence and fused. Requires --eje-b.',
    )
    parser.add_argument(
        '--kl-free-bits-initial',
        type=float,
        default=None,
        metavar='NATS',
        help='Initial per-leaf free-bits floor, decayed to 0. Default with --eje-b: 2.0.',
    )
    parser.add_argument(
        '--curriculum',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Hard-example demonstration weighting. Default with --eje-b: on.',
    )
    args = parser.parse_args(argv)

    if args.iterations <= WARMUP_STEPS:
        parser.error(f'--iterations must be greater than {WARMUP_STEPS}')
    if args.postprocess_stride < 1:
        parser.error('--postprocess-stride must be at least 1')
    if args.seed is not None and args.seed < 0:
        parser.error('--seed must be non-negative')
    if not args.eje_b:
        for flag, value in (('--seeds', args.seeds),
                            ('--kl-free-bits-initial', args.kl_free_bits_initial),
                            ('--curriculum', args.curriculum)):
            if value is not None:
                parser.error(f'{flag} requires --eje-b')

    if args.accel_preset is None:
        args.accel_preset = 'compile' if args.eje_b else 'baseline'
    if args.eje_b and args.accel_preset != 'compile':
        parser.error('Eje B must run with --accel-preset compile')

    if args.seeds is None:
        args.seeds = (args.seed if args.seed is not None else 0,)
    if args.eje_b:
        if args.kl_free_bits_initial is None:
            args.kl_free_bits_initial = 2.0
        if args.curriculum is None:
            args.curriculum = True
    else:
        args.kl_free_bits_initial = 0.0
        args.curriculum = False

    if args.run_label is None:
        args.run_label = 'eje_b' if args.eje_b else args.accel_preset
    if os.path.basename(args.run_label) != args.run_label or args.run_label in ('', '.', '..'):
        parser.error('--run-label must be a plain name without path separators')
    return args


def seed_output_dir(args, seed):
    folder = os.path.join('results', args.task_name)
    if args.eje_b:
        folder = os.path.join(folder, args.run_label, f'seed_{seed}')
    return folder


def fused_output_dir(args):
    return os.path.join('results', args.task_name, args.run_label, 'fused')


def configure_acceleration(args):
    preset_cfg = accel.config_from_preset(
        args.accel_preset,
        eje_b=args.eje_b,
        seeds=args.seeds if args.eje_b else (0,),
        kl_free_bits_initial=args.kl_free_bits_initial,
        curriculum=args.curriculum,
    )
    if preset_cfg.compile_mode != 'off':
        preset_cfg = accel.AccelConfig.from_dict({
            **preset_cfg.to_dict(),
            'inductor_cache_dir': args.inductor_cache_dir,
        })
    return accel.configure_process(preset_cfg)


def pass2_curve(picks_history, true_hash):
    """1 at every step where one of the two picked solutions is the ground truth."""
    return [int(true_hash in pair) for pair in picks_history]


def step_annotation(logger, seed, train_step):
    """Eje B state at the last logged step; reads GPU scalars, so call it sparingly."""
    text = (f'seed {seed} | step {train_step + 1} | '
            f'free-bits floor {float(logger.kl_free_bits_curve[-1]):.2f} | '
            f'leaves below floor {int(logger.n_kl_below_floor_curve[-1])}')
    if logger.curriculum_weights_curve:
        weights = logger.curriculum_weights_curve[-1].detach().cpu().tolist()
        text += ' | demo weights [' + ', '.join(f'{w:.2f}' for w in weights) + ']'
    return text


def save_representation_plots(task, folder, task_name, target_capacities,
                              decode_weights, multiposteriors):
    # Get the learned representation tensors
    samples = []
    for i in range(100):
        sample, KL_amounts, KL_names = layers.decode_latents(target_capacities,
                                           decode_weights, multiposteriors)
        samples.append(sample)

    def average_samples(dims, *items):
        mean = torch.mean(torch.stack(items, dim=0), dim=0).detach().cpu().numpy()
        all_but_last_dim = tuple(range(len(mean.shape) - 1))
        mean = mean - np.mean(mean, axis=all_but_last_dim)
        return mean
    means = multitensor_systems.multify(average_samples)(*samples)

    # Figure out which tensors contain significant information
    dims_to_plot = []
    for KL_amount, KL_name in zip(KL_amounts, KL_names):
        dims = tuple(eval(KL_name))
        if torch.sum(KL_amount).detach().cpu().numpy() > 1:
            dims_to_plot.append(dims)

    # Show the top principal components of the significant tensors.
    color_names = ['black', 'blue', 'red', 'green', 'yellow', 'gray', 'magenta', 'orange', 'light blue', 'brown']
    restricted_color_names = [color_names[i] for i in task.colors]
    restricted_color_codes = [tuple((visualization.color_list[i]/255).tolist())
                              for i in task.colors]
    for dims in dims_to_plot:
        tensor = means[dims]

        orig_shape = tensor.shape
        if len(orig_shape) == 2:
            tensor = tensor[None,:,:]
        orig_shape = tensor.shape
        if len(orig_shape) == 3:
            tensor = np.reshape(tensor, (-1, orig_shape[-1]))
            U, S, Vh = np.linalg.svd(tensor)  # Get top 3 principal components
            for component_num in range(min(3, U.shape[1])):
                component = np.reshape(U[:,component_num], orig_shape[:-1])
                component = component / np.max(np.abs(component))
                strength = S[component_num] / tensor.shape[0]  # Calculate component strength

                # Show the component
                fig, ax = plt.subplots()
                ax.imshow(component, cmap='gray', vmin=-1, vmax=1)

                # Pick the axis labels
                axis_names = ['example', 'color', 'direction', 'height', 'width']
                tensor_name = '_'.join([axis_name
                    for axis_name, axis_exists in zip(axis_names, dims) if axis_exists])
                if sum(dims) == 2:
                    x_dim = [axis_names[i] for i, dim in enumerate(dims) if dim][0]
                    y_dim = [axis_names[i] for i, dim in enumerate(dims) if dim][1]
                else:
                    x_dim = None
                    y_dim = [axis_names[i] for i, dim in enumerate(dims) if dim][0]
                plt.ylabel(x_dim)
                plt.xlabel(y_dim)

                if x_dim is None:
                    ax.set_yticks([])
                    ax.set_xticks([], minor=True)
                if y_dim is None:
                    ax.set_xticks([])
                    ax.set_xticks([], minor=True)

                # Set the tick labels
                # Tick labels for example axis
                if x_dim == 'example':
                    ax.set_yticks(np.arange(task.n_examples))
                if y_dim == 'example':
                    ax.set_xticks(np.arange(task.n_examples))

                # Tick labels for color axis
                if x_dim == 'color':
                    ax.set_yticks(np.arange(len(restricted_color_names[1:])))
                    ax.set_yticklabels(restricted_color_names[1:])
                    for ticklabel, tickcolor in zip(ax.get_yticklabels(), restricted_color_codes[1:]):
                        ticklabel.set_color(tickcolor)
                        ticklabel.set_fontweight("bold")
                if y_dim == 'color':
                    ax.set_xticks(np.arange(len(restricted_color_names[1:])))
                    ax.set_xticklabels(restricted_color_names[1:])
                    for ticklabel, tickcolor in zip(ax.get_xticklabels(), restricted_color_codes[1:]):
                        ticklabel.set_color(tickcolor)
                        ticklabel.set_fontweight("bold")

                # Tick labels for direction axis
                direction_names = ["↓", "↘", "→", "↗", "↑", "↖", "←", "↙"]
                if x_dim == 'direction':
                    ax.set_yticks(np.arange(8))
                    ax.set_yticklabels(direction_names)
                    ax.tick_params(axis='y', which='major', labelsize=22)
                if y_dim == 'direction':
                    ax.set_xticks(np.arange(8))
                    ax.set_xticklabels(direction_names)
                    ax.tick_params(axis='x', which='major', labelsize=22)

                # Standard tick labels for height and width axes

                ax.set_title('component' + str(component_num) + ', strength = ' + str(float(strength)))
                plt.savefig(os.path.join(folder, task_name + '_' + tensor_name + '_component_' + str(component_num) + '.png'), bbox_inches='tight')
                plt.close()

        # Plot an ({example, color, direction}, x, y) tensor with subplots
        elif len(orig_shape) == 4 and dims[3] == 1 and dims[4] == 1:
            tensor = np.reshape(tensor, (-1, orig_shape[-1]))
            U, S, Vh = np.linalg.svd(tensor)  # Get the top 3 principal components
            for component_num in range(min(3, U.shape[1])):
                component = np.reshape(U[:,component_num], orig_shape[:-1])
                component = component / np.max(np.abs(component))
                strength = S[component_num] / tensor.shape[0]
                n_plots = orig_shape[0]

                # Make the subplots
                fig, axs = plt.subplots(1, n_plots)
                for plot_idx in range(n_plots):
                    ax = axs[plot_idx]
                    ax.imshow(component[plot_idx,:,:], cmap='gray', vmin=-1, vmax=1)

                    # Get the axis labels
                    axis_names = ['example', 'color', 'direction', 'height', 'width']
                    tensor_name = '_'.join([axis_name
                        for axis_name, axis_exists in zip(axis_names, dims) if axis_exists])
                    ax_dim = [axis_names[i] for i, dim in enumerate(dims) if dim][0]
                    x_dim = [axis_names[i] for i, dim in enumerate(dims) if dim][1]
                    y_dim = [axis_names[i] for i, dim in enumerate(dims) if dim][2]
                    ax.set_ylabel(x_dim)
                    ax.set_xlabel(y_dim)

                    # Standard tick labels for height and width axes

                    # Label the subplots
                    if ax_dim == 'example':
                        ax.set_title('example ' + str(plot_idx))
                    elif ax_dim == 'color':
                        ax.set_title(restricted_color_names[plot_idx],
                                     color=restricted_color_codes[plot_idx],
                                     fontweight="bold")
                    elif ax_dim == 'direction':
                        direction_names = ["↓", "↘", "→", "↗", "↑", "↖", "←", "↙"]
                        ax.set_title(direction_names[plot_idx], fontsize=22)

                plt.subplots_adjust(wspace=1)
                fig.suptitle('component ' + str(component_num) + ', strength = ' + str(float(strength)))
                plt.subplots_adjust(top=1.4)
                plt.savefig(os.path.join(folder, task_name + '_' + tensor_name + '_component_' + str(component_num) + '.png'), bbox_inches='tight')
                plt.close()


def run_seed(args, accel_cfg, seed):
    """Train one seed, write all its plots and return plain data for the fusion step."""
    task_name = args.task_name
    folder = seed_output_dir(args, seed)
    os.makedirs(folder, exist_ok=True)
    print('Performing a training run on task', task_name, 'with seed', seed,
          'and placing the results in', folder)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Preprocess the task, set up the training
    task = preprocessing.preprocess_tasks(args.split, [task_name])[0]
    model = arc_compressor.ARCCompressor(task)
    accel.apply(model, accel_cfg)
    optimizer = torch.optim.Adam(model.weights_list, lr=0.01, betas=(0.5, 0.9))
    train_history_logger = solution_selection.Logger(
        task, postprocess_stride=args.postprocess_stride
    )

    # ── Training with per-step CPU / wall-clock profiling ────────────────────
    n_iterations = args.iterations
    eje_b_state = train.EjeBTrainingState() if accel_cfg.curriculum else None
    step_profile  = []  # (step, wall_s, cpu_s)

    t_wall = time.perf_counter()
    t_cpu  = time.process_time()

    for train_step in tqdm(range(n_iterations), desc=f'seed {seed}'):
        train.take_step(
            task,
            model,
            optimizer,
            train_step,
            train_history_logger,
            accel_config=accel_cfg,
            n_train_iterations=n_iterations,
            eje_b_state=eje_b_state,
        )

        # Flush GPU so each slice covers one complete step (CPU work + GPU kernels).
        torch.cuda.synchronize()
        t_wall_now = time.perf_counter()
        t_cpu_now  = time.process_time()

        if train_step >= WARMUP_STEPS:
            step_profile.append((train_step, t_wall_now - t_wall, t_cpu_now - t_cpu))

        t_wall, t_cpu = t_wall_now, t_cpu_now

        # Plot solutions every 50 steps
        if (train_step+1) % PLOT_INTERVAL == 0:
            annotation = (step_annotation(train_history_logger, seed, train_step)
                          if accel_cfg.eje_b else None)
            for extension in ('png', 'pdf'):
                visualization.plot_solution(train_history_logger,
                    fname=os.path.join(folder, f'{task_name}_at_{train_step+1} steps.{extension}'),
                    task_name=task_name,
                    annotation=annotation)

    train_history_logger.materialize_curves()
    compile_report = accel.compile_report(accel_cfg)
    if compile_report:
        print(f'[accel][{task_name}] torch.compile report: {compile_report}', flush=True)

    # ── Timing summary ────────────────────────────────────────────────────────
    wall_arr  = np.array([r[1] for r in step_profile])
    cpu_arr   = np.array([r[2] for r in step_profile])
    cpu_ratio = cpu_arr / np.maximum(wall_arr, 1e-9)

    print(f'\n{"="*62}')
    print(f'  Performance Summary — label: {args.run_label}   seed: {seed}')
    print(f'  Task: {task_name}   Steps profiled: {len(step_profile)}')
    print(f'{"="*62}')
    print(f'  Wall time / step : {wall_arr.mean()*1e3:7.2f} ms  ±  {wall_arr.std()*1e3:.2f} ms')
    print(f'  CPU  time / step : {cpu_arr.mean()*1e3:7.2f} ms  ±  {cpu_arr.std()*1e3:.2f} ms')
    print(f'  CPU / wall ratio : {cpu_ratio.mean()*100:6.1f} %')
    print(f'  Throughput       : {1/wall_arr.mean():7.1f} steps/sec')
    print(f'  Total wall time  : {wall_arr.sum():7.1f} s  (profiled steps)')
    print(f'{"="*62}\n')

    # ── Export CSV ────────────────────────────────────────────────────────────
    effective_cache_dir = os.environ.get('TORCHINDUCTOR_CACHE_DIR')
    csv_path = os.path.join(folder, f'{task_name}_step_times_{args.run_label}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'step', 'wall_s', 'cpu_s', 'cpu_pct',
            'accel_preset', 'accel_summary', 'inductor_cache_dir', 'seed',
        ])
        for step, wall_s, cpu_s in step_profile:
            writer.writerow([step, f'{wall_s:.6f}', f'{cpu_s:.6f}',
                             f'{cpu_s / max(wall_s, 1e-9) * 100:.1f}',
                             args.accel_preset, accel_cfg.summary(),
                             effective_cache_dir or '', seed])
    print(f'  Step timing saved → {csv_path}')

    # Save the metrics, model weights, and learned representations.
    npz_path = os.path.join(folder, task_name + '_KL_curves.npz')
    np.savez(npz_path,
             KL_curves={key:np.array(val) for key, val in train_history_logger.KL_curves.items()},
             total_KL_curve=np.array(train_history_logger.total_KL_curve),
             effective_total_KL_curve=np.array(
                 train_history_logger.effective_total_KL_curve),
             kl_free_bits_curve=np.array(
                 train_history_logger.kl_free_bits_curve),
             n_kl_below_floor_curve=np.array(
                 train_history_logger.n_kl_below_floor_curve),
             curriculum_weights_curve=np.array(
                 train_history_logger.curriculum_weights_curve),
             reconstruction_error_curve=np.array(train_history_logger.reconstruction_error_curve),
             seed=seed,
             multiposteriors=model.multiposteriors,
             target_capacities=model.target_capacities,
             decode_weights=model.decode_weights)

    # Load the metrics, model weights, and learned representations.
    stored_data = np.load(npz_path, allow_pickle=True)
    KL_curves = stored_data['KL_curves'][()]
    reconstruction_error_curve = stored_data['reconstruction_error_curve']
    multiposteriors = stored_data['multiposteriors'][()]
    target_capacities = stored_data['target_capacities'][()]
    decode_weights = stored_data['decode_weights'][()]

    # Plot the KL curves over time.
    ylim = (0.3, 4e4) if task_name == '6cdd2623' else None
    visualization.plot_kl_components(
        KL_curves,
        os.path.join(folder, task_name + '_KL_components.png'),
        special_colors=SPECIAL_CURVE_COLORS.get(task_name),
        free_bits_curve=(train_history_logger.kl_free_bits_curve
                         if accel_cfg.eje_b else None),
        ylim=ylim,
    )

    # Plot the KL vs reconstruction error
    visualization.plot_kl_vs_reconstruction(
        KL_curves,
        reconstruction_error_curve,
        os.path.join(folder, task_name + '_KL_vs_reconstruction.png'),
        effective_total_KL_curve=(train_history_logger.effective_total_KL_curve
                                  if accel_cfg.eje_b else None),
        reconstruction_label=('reconstruction error (curriculum-weighted)'
                              if accel_cfg.curriculum else 'reconstruction error'),
    )

    pass2 = None
    if accel_cfg.eje_b:
        visualization.plot_eje_b_free_bits(
            train_history_logger.n_kl_below_floor_curve,
            train_history_logger.kl_free_bits_curve,
            os.path.join(folder, task_name + '_eje_b_free_bits.png'),
        )
        if accel_cfg.curriculum:
            visualization.plot_eje_b_curriculum(
                train_history_logger.curriculum_weights_curve,
                os.path.join(folder, task_name + '_eje_b_curriculum.png'),
            )
        if task.solution_hash is not None:
            pass2 = pass2_curve(train_history_logger.solution_picks_history,
                                task.solution_hash)
            visualization.plot_pass2_curves(
                {f'seed {seed}': pass2},
                os.path.join(folder, task_name + '_pass2_curve.png'),
            )

    save_representation_plots(task, folder, task_name, target_capacities,
                              decode_weights, multiposteriors)

    return {
        'seed': seed,
        'pass2': pass2,
        'KL_curves': KL_curves,
        'free_bits_curve': list(train_history_logger.kl_free_bits_curve),
        'logger_data': {
            'solution_contributions_log': train_history_logger.solution_contributions_log,
            'solution_picks_history': train_history_logger.solution_picks_history,
            'candidate_evidence': train_history_logger.candidate_evidence(),
        },
    }


def fuse_seeds(args, true_hash, seed_results):
    """Merge the seeds like parallel_train.py and plot the fused pass@2 evidence."""
    folder = fused_output_dir(args)
    os.makedirs(folder, exist_ok=True)
    attempts, merged = solution_selection.merge_seed_logger_data(
        {result['seed']: result['logger_data'] for result in seed_results}
    )

    fused_correct = None
    if true_hash is not None:
        fused_correct = pass2_curve(merged['solution_picks_history'], true_hash)
        curves = {f"seed {result['seed']}": result['pass2'] for result in seed_results}
        curves['fused'] = fused_correct
        visualization.plot_pass2_curves(
            curves, os.path.join(folder, args.task_name + '_pass2_curves.png'),
            highlight='fused',
        )
    visualization.plot_kl_components_seeds(
        {result['seed']: result['KL_curves'] for result in seed_results},
        os.path.join(folder, args.task_name + '_KL_components_seeds.png'),
        special_colors=SPECIAL_CURVE_COLORS.get(args.task_name),
        free_bits_curve=seed_results[0]['free_bits_curve'],
        ylim=(0.3, 4e4) if args.task_name == '6cdd2623' else None,
    )
    visualization.plot_fused_solution(
        attempts,
        os.path.join(folder, args.task_name + '_fused_solution.png'),
        title=f'{args.task_name} - fused seeds {",".join(str(s) for s in args.seeds)}',
    )
    with open(os.path.join(folder, args.task_name + '_fused_attempts.json'), 'w') as f:
        json.dump({
            'merge_policy': merged['merge_policy'],
            'seeds': list(args.seeds),
            'final_pass_at_2': None if fused_correct is None else bool(fused_correct[-1]),
            'attempts': attempts,
        }, f)

    if fused_correct is not None:
        for result in seed_results:
            print(f"  seed {result['seed']}: pass@2 = {bool(result['pass2'][-1])}")
        print(f'  fused: pass@2 = {bool(fused_correct[-1])}')
    print(f'  Fused results saved → {folder}')


def main(argv=None):
    args = parse_args(argv)
    accel_cfg = configure_acceleration(args)
    torch.set_default_device('cuda')

    print('Acceleration:', accel_cfg.summary())
    if accel_cfg.compile_mode != 'off':
        print('Inductor cache:', os.environ.get('TORCHINDUCTOR_CACHE_DIR'))

    # Some interesting tasks: 272f95fa, 6d75e8bb, 6cdd2623, 41e4d17e, 2bee17df
    # 228f6490, 508bd3b6, 2281f1f4, ecdecbb3
    problem_task = preprocessing.preprocess_tasks(args.split, [args.task_name])[0]
    visualization.plot_problem(solution_selection.Logger(problem_task))

    seed_results = []
    for seed in args.seeds:
        seed_results.append(run_seed(args, accel_cfg, seed))
        # Drop the previous model's compiled state before the next seed starts.
        gc.collect()
        torch.cuda.empty_cache()
        if accel_cfg.compile_mode != 'off':
            torch._dynamo.reset()

    if accel_cfg.eje_b and len(seed_results) > 1:
        fuse_seeds(args, problem_task.solution_hash, seed_results)

    print('done')


if __name__ == "__main__":
    main()
