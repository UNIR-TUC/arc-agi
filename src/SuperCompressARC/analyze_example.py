import csv
import os
import pickle
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
"""

# For some reason trying to set the seed doesn't actually fix results.
# Just run things over and over again until you see desired interesting behaviors.
np.random.seed(0)
torch.manual_seed(0)
torch.set_default_dtype(torch.float32)
torch.set_default_device('cuda')

if __name__ == "__main__":

    # Change RUN_LABEL between runs to distinguish baseline vs optimised results.
    # Workflow:
    #   1. git stash  →  RUN_LABEL = 'baseline'  →  run
    #   2. git stash pop  →  RUN_LABEL = 'optimised'  →  run
    #   3. Load both CSVs into a DataFrame to plot comparisons.
    RUN_LABEL = 'baseline'

    # Some interesting tasks: 272f95fa, 6d75e8bb, 6cdd2623, 41e4d17e, 2bee17df
    # 228f6490, 508bd3b6, 2281f1f4, ecdecbb3
    # split = input('Enter which split you want to find the task in (training, evaluation, test): ')
    split = 'training'
    # task_name = input('Enter which task you want to analyze (eg. 272f95fa): ')
    task_name = '6d75e8bb'
    folder = 'results/' + task_name + '/'
    print('Performing a training run on task', task_name,
          'and placing the results in', folder)
    os.makedirs(folder, exist_ok=True)

    # Preprocess the task, set up the training
    task = preprocessing.preprocess_tasks(split, [task_name])[0]
    model = arc_compressor.ARCCompressor(task)
    optimizer = torch.optim.Adam(model.weights_list, lr=0.01, betas=(0.5, 0.9))
    try:
        train_history_logger = solution_selection.Logger(
            task, tracking_interval=1, enable_curves=True
        )
    except TypeError:
        # Compatibility with the pre-optimisation Logger (no extra params).
        train_history_logger = solution_selection.Logger(task)
    visualization.plot_problem(train_history_logger)

    # ── Training with per-step CPU / wall-clock profiling ────────────────────
    n_iterations = 1500
    WARMUP_STEPS = 20   # skip first N steps (CUDA / JIT warm-up)
    step_profile  = []  # (step, wall_s, cpu_s)

    t_wall = time.perf_counter()
    t_cpu  = time.process_time()

    for train_step in tqdm(range(n_iterations)):
        train.take_step(task, model, optimizer, train_step, train_history_logger)

        # Flush GPU so each slice covers one complete step (CPU work + GPU kernels).
        torch.cuda.synchronize()
        t_wall_now = time.perf_counter()
        t_cpu_now  = time.process_time()

        if train_step >= WARMUP_STEPS:
            step_profile.append((train_step, t_wall_now - t_wall, t_cpu_now - t_cpu))

        t_wall, t_cpu = t_wall_now, t_cpu_now

        # Plot solutions every 50 steps
        if (train_step+1) % 50 == 0:
            visualization.plot_solution(train_history_logger,
                fname=folder + task_name + '_at_' + str(train_step+1) + ' steps.png',
                task_name=task_name)
            visualization.plot_solution(train_history_logger,
                fname=folder + task_name + '_at_' + str(train_step+1) + ' steps.pdf',
                task_name=task_name)

    # ── Timing summary ────────────────────────────────────────────────────────
    wall_arr  = np.array([r[1] for r in step_profile])
    cpu_arr   = np.array([r[2] for r in step_profile])
    cpu_ratio = cpu_arr / np.maximum(wall_arr, 1e-9)

    print(f'\n{"="*62}')
    print(f'  Performance Summary — label: {RUN_LABEL}')
    print(f'  Task: {task_name}   Steps profiled: {len(step_profile)}')
    print(f'{"="*62}')
    print(f'  Wall time / step : {wall_arr.mean()*1e3:7.2f} ms  ±  {wall_arr.std()*1e3:.2f} ms')
    print(f'  CPU  time / step : {cpu_arr.mean()*1e3:7.2f} ms  ±  {cpu_arr.std()*1e3:.2f} ms')
    print(f'  CPU / wall ratio : {cpu_ratio.mean()*100:6.1f} %')
    print(f'  Throughput       : {1/wall_arr.mean():7.1f} steps/sec')
    print(f'  Total wall time  : {wall_arr.sum():7.1f} s  (profiled steps)')
    print(f'{"="*62}\n')

    # ── Export CSV ────────────────────────────────────────────────────────────
    csv_path = os.path.join(folder, f'{task_name}_step_times_{RUN_LABEL}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'wall_s', 'cpu_s', 'cpu_pct'])
        for step, wall_s, cpu_s in step_profile:
            writer.writerow([step, f'{wall_s:.6f}', f'{cpu_s:.6f}',
                             f'{cpu_s / max(wall_s, 1e-9) * 100:.1f}'])
    print(f'  Step timing saved → {csv_path}')

    # Save the metrics, model weights, and learned representations.
    np.savez(folder + task_name + '_KL_curves.npz',
             KL_curves={key:np.array(val) for key, val in train_history_logger.KL_curves.items()},
             reconstruction_error_curve=np.array(train_history_logger.reconstruction_error_curve),
             multiposteriors=model.multiposteriors,
             target_capacities=model.target_capacities,
             decode_weights=model.decode_weights)

    # Load the metrics, model weights, and learned representations.
    stored_data = np.load(folder + task_name + '_KL_curves.npz', allow_pickle=True)
    KL_curves = stored_data['KL_curves'][()]
    reconstruction_error_curve = stored_data['reconstruction_error_curve']
    multiposteriors = stored_data['multiposteriors'][()]
    target_capacities = stored_data['target_capacities'][()]
    decode_weights = stored_data['decode_weights'][()]
    
    # Plot the KL curves over time.
    # For specific tasks that we found interesting, we wrote some
    # code to color interesting KL components differently.
    special_curve_colors = {
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
    fig, ax = plt.subplots()
    for component_name, curve in KL_curves.items():
        line_color = (0.5, 0.5, 0.5)
        label = None
        if task_name in special_curve_colors:
            dims_list = special_curve_colors[task_name]['dims']
            colors_list = special_curve_colors[task_name]['colors']
            for dims, color in zip(dims_list, colors_list):
                if tuple(eval(component_name)) == dims:
                    line_color = color
                    axis_names = ['example', 'color', 'direction', 'height', 'width']
                    axis_names = [axis_name
                        for axis_name, axis_exists in zip(axis_names, dims) if axis_exists]
                    label = '(' + ', '.join(axis_names) + ', channel)'
        ax.plot(np.arange(curve.shape[0]), curve, color=line_color, label=label)
    if task_name == '6cdd2623':
        ax.set_ylim((0.3, 4e4))

    ax.legend()
    plt.yscale('log')
    plt.xlabel('step')
    plt.ylabel('KL contribution')
    ax.grid(which='both', linestyle='-', linewidth='0.5', color='gray')
    plt.savefig(folder + task_name + '_KL_components.png', bbox_inches='tight')
    plt.close()

    # Plot the KL vs reconstruction error
    fig, ax = plt.subplots()
    total_KL = 0
    for component_name, curve in KL_curves.items():
        total_KL = total_KL + curve
    fig, ax = plt.subplots()
    ax.plot(np.arange(total_KL.shape[0]), total_KL, label='KL from z', color='k')
    ax.plot(np.arange(reconstruction_error_curve.shape[0]),
            reconstruction_error_curve, label='reconstruction error', color='r')
    ax.legend()
    plt.yscale('log')
    plt.xlabel('step')
    plt.ylabel('total KL or reconstruction error')
    ax.grid(which='both', linestyle='-', linewidth='0.5', color='gray')
    plt.savefig(folder + task_name + '_KL_vs_reconstruction.png', bbox_inches='tight')
    plt.close()

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
                plt.savefig(folder + task_name + '_' + tensor_name + '_component_' + str(component_num) + '.png', bbox_inches='tight')
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
                plt.savefig(folder + task_name + '_' + tensor_name + '_component_' + str(component_num) + '.png', bbox_inches='tight')
                plt.close()


print('done')
