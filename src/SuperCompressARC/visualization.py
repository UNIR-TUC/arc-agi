import ast
import os

import matplotlib.pyplot as plt
import numpy as np
import torch


"""
This file trains a model for every ARC-AGI task in a split.
"""

np.random.seed(0)
torch.manual_seed(0)


color_list = np.array([
    [0, 0, 0],  # black
    [30, 147, 255],  # blue
    [249, 60, 49],  # red
    [79, 204, 48],  # green
    [255, 220, 0],  # yellow
    [153, 153, 153],  # gray
    [229, 58, 163],  # magenta
    [255, 133, 27],  # orange
    [135, 216, 241],  # light blue
    [146, 18, 49],  # brown
])

def convert_color(grid):  # grid dims must end in c
    return np.clip(np.matmul(grid, color_list), 0, 255).astype(np.uint8)

def plot_problem(logger, task_name=None):
    """
    Draw a plot of an ARC-AGI problem, and save it in plots/
    Args:
        logger (Logger): A logger object used to log model outputs for the ARC-AGI task.
    """

    # Put all the grids beside one another on one grid
    n_train = logger.task.n_train
    n_test = logger.task.n_test
    n_examples = logger.task.n_examples
    n_x = logger.task.n_x
    n_y = logger.task.n_y
    pixels = 255+np.zeros([n_train+n_test, 2*n_x+2, 2, 2*n_y+8, 3], dtype=np.uint8)
    for example_num in range(n_examples):
        if example_num < n_train:
            subsplit = 'train'
            subsplit_example_num = example_num
        else:
            subsplit = 'test'
            subsplit_example_num = example_num - n_train
        for mode_num, mode in enumerate(('input', 'output')):
            if subsplit == 'test' and mode == 'output':
                continue
            grid = np.array(logger.task.unprocessed_problem[subsplit][subsplit_example_num][mode])  # x, y
            grid = (np.arange(10)==grid[:,:,None]).astype(np.float32)  # x, y, c
            grid = convert_color(grid)  # x, y, c
            repeat_grid = np.repeat(grid, 2, axis=0)
            repeat_grid = np.repeat(repeat_grid, 2, axis=1)
            pixels[example_num,n_x+1-grid.shape[0]:n_x+1+grid.shape[0],mode_num,n_y+4-grid.shape[1]:n_y+4+grid.shape[1],:] = repeat_grid
    pixels = pixels.reshape([(n_train+n_test)*(2*n_x+2), 2*(2*n_y+8), 3])
    
    os.makedirs("plots/", exist_ok=True)

    # Plot the combined grid and make gray dividers between the grid cells, arrows, and a question mark for unsolved examples.
    fig, ax = plt.subplots()
    ax.imshow(pixels, aspect='equal', interpolation='none')
    for example_num in range(n_examples):
        for mode_num, mode in enumerate(('input', 'output')):
            if example_num < n_train:
                subsplit = 'train'
                subsplit_example_num = example_num
            else:
                subsplit = 'test'
                subsplit_example_num = example_num - n_train
            ax.arrow((2*n_y+8)-3-0.5, (2*n_x+2)*example_num+1+n_x-0.5, 6, 0, width=0.5, fc='k', ec='k', length_includes_head=True)
            if subsplit == 'test' and mode == 'output':
                ax.text((2*n_y+8)+4+n_y-0.5, (2*n_x+2)*example_num+1+n_x-0.5, '?', size='xx-large', ha='center', va='center')
                continue
            grid = np.array(logger.task.unprocessed_problem[subsplit][subsplit_example_num][mode])  # x, y
            for xline in range(grid.shape[0]+1):
                ax.plot(((2*n_y+8)*mode_num+4+n_y-grid.shape[1]-0.5, (2*n_y+8)*mode_num+4+n_y+grid.shape[1]-0.5),
                        ((2*n_x+2)*example_num+1+n_x-grid.shape[0]+2*xline-0.5,)*2,
                        color=(59/255, 59/255, 59/255),
                        linewidth=0.3)
            for yline in range(grid.shape[1]+1):
                ax.plot(((2*n_y+8)*mode_num+4+n_y-grid.shape[1]+2*yline-0.5,)*2,
                        ((2*n_x+2)*example_num+1+n_x-grid.shape[0]-0.5, (2*n_x+2)*example_num+1+n_x+grid.shape[0]-0.5),
                        color=(59/255, 59/255, 59/255),
                        linewidth=0.3)
    plt.axis('off')
    if task_name is None:
        task_name = logger.task.task_name
    plt.savefig('plots/' + task_name + '_problem.png', bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_solution(logger, fname=None, task_name=None, annotation=None):
    """
    Draw a plot of a model's solution to an ARC-AGI problem, and save it in plots/
    Draws four plots: A model output sample, the mean of samples, and the top two most common samples.
    Args:
        logger (Logger): A logger object used to log model outputs for the ARC-AGI task.
    """
    n_train = logger.task.n_train
    n_test = logger.task.n_test
    n_examples = logger.task.n_examples
    n_x = logger.task.n_x
    n_y = logger.task.n_y

    # Four plotted solutions
    solutions_list = [
            torch.softmax(logger.current_logits, dim=1).cpu().numpy(),
            torch.softmax(logger.ema_logits, dim=1).cpu().numpy(),
            logger.solution_most_frequent,
            logger.solution_second_most_frequent,
            ]
    masks_list = [
            (logger.current_x_mask, logger.current_y_mask),
            (logger.ema_x_mask, logger.ema_y_mask),
            None,
            None,
            ]
    solutions_labels = [
            'sample',
            'sample average',
            'guess 1',
            'guess 2',
            ]
    n_plotted_solutions = len(solutions_list)

    # Put all the grids beside one another on one grid
    pixels = 255+np.zeros([n_test, 2*n_x+2, n_plotted_solutions, 2*n_y+8, 3], dtype=np.uint8)
    shapes = []
    for subsplit_example_num in range(n_test):
        subsplit = 'test'
        example_num = subsplit_example_num + n_train
        shapes.append([])

        for solution_num, (solution, masks, label) in enumerate(zip(solutions_list, masks_list, solutions_labels)):
            grid = np.array(solution[subsplit_example_num])  # c, x, y if 'sample' in label else x, y, c
            if 'sample' in label:
                grid = np.einsum('dxy,dc->xyc', grid, color_list[logger.task.colors])  # x, y, c
                if logger.task.in_out_same_size or logger.task.all_out_same_size:
                    x_length = logger.task.shapes[example_num][1][0]
                    y_length = logger.task.shapes[example_num][1][1]
                else:
                    x_length = None
                    y_length = None
                x_start, x_end = logger._best_slice_point(masks[0][subsplit_example_num,:], x_length)
                y_start, y_end = logger._best_slice_point(masks[1][subsplit_example_num,:], y_length)
                grid = grid[x_start:x_end,y_start:y_end,:]  # x, y, c
                grid = np.clip(grid, 0, 255).astype(np.uint8)
            else:
                grid = (np.arange(10)==grid[:,:,None]).astype(np.float32)  # x, y, c
                grid = convert_color(grid)  # x, y, c

            shapes[subsplit_example_num].append((grid.shape[0], grid.shape[1]))
            repeat_grid = np.repeat(grid, 2, axis=0)
            repeat_grid = np.repeat(repeat_grid, 2, axis=1)
            pixels[subsplit_example_num,n_x+1-grid.shape[0]:n_x+1+grid.shape[0],solution_num,n_y+4-grid.shape[1]:n_y+4+grid.shape[1],:] = repeat_grid

    pixels = pixels.reshape([n_test*(2*n_x+2), n_plotted_solutions*(2*n_y+8), 3])
    
    # Plot the combined grid and make gray dividers between the grid cells, and labels.
    fig, ax = plt.subplots()
    ax.imshow(pixels, aspect='equal', interpolation='none')
    for subsplit_example_num in range(n_test):
        for solution_num in range(n_plotted_solutions):
            subsplit = 'test'
            grid = np.array(solutions_list[solution_num][subsplit_example_num])  # x, y
            shape = shapes[subsplit_example_num][solution_num]
            for xline in range(shape[0]+1):
                ax.plot(((2*n_y+8)*solution_num+4+n_y-shape[1]-0.5, (2*n_y+8)*solution_num+4+n_y+shape[1]-0.5),
                        ((2*n_x+2)*subsplit_example_num+1+n_x-shape[0]+2*xline-0.5,)*2,
                        color=(59/255, 59/255, 59/255),
                        linewidth=0.3)
            for yline in range(shape[1]+1):
                ax.plot(((2*n_y+8)*solution_num+4+n_y-shape[1]+2*yline-0.5,)*2,
                        ((2*n_x+2)*subsplit_example_num+1+n_x-shape[0]-0.5, (2*n_x+2)*subsplit_example_num+1+n_x+shape[0]-0.5),
                        color=(59/255, 59/255, 59/255),
                        linewidth=0.3)
    for solution_num, solution_label in enumerate(solutions_labels):
        ax.text((2*n_y+8)*solution_num+4+n_y-0.5, -3, solution_label, size='xx-small', ha='center', va='center')
    if annotation:
        ax.set_title(annotation, size='xx-small', pad=14)
    plt.axis('off')
    if fname is None:
        if task_name is None:
            task_name = logger.task.task_name
        fname = 'plots/' + task_name + '_solutions.pdf'
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close()


def _log_axis_setup(ax, xlabel, ylabel):
    ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(which='both', linestyle='-', linewidth='0.5', color='gray')


def _kl_line_style(component_name, special_colors):
    line_color = (0.5, 0.5, 0.5)
    label = None
    if special_colors is not None:
        dims = tuple(ast.literal_eval(component_name))
        for special_dims, color in zip(special_colors['dims'], special_colors['colors']):
            if dims == special_dims:
                line_color = color
                axis_names = ['example', 'color', 'direction', 'height', 'width']
                axis_names = [axis_name
                    for axis_name, axis_exists in zip(axis_names, dims) if axis_exists]
                label = '(' + ', '.join(axis_names) + ', channel)'
    return line_color, label


def plot_kl_components(KL_curves, fname, special_colors=None,
                       free_bits_curve=None, ylim=None):
    """Plot the raw KL of every multitensor leaf; with Eje B, overlay the free-bits floor."""
    fig, ax = plt.subplots()
    for component_name, curve in KL_curves.items():
        line_color, label = _kl_line_style(component_name, special_colors)
        ax.plot(np.arange(len(curve)), curve, color=line_color, label=label)
    if free_bits_curve is not None:
        floor = np.asarray(free_bits_curve, dtype=float)
        floor = np.where(floor > 0, floor, np.nan)  # log axis cannot show tau == 0
        ax.plot(np.arange(len(floor)), floor, 'r--', linewidth=1.5,
                label='free-bits floor (tau)')
    if ylim is not None:
        ax.set_ylim(ylim)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    _log_axis_setup(ax, 'step', 'KL contribution')
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)


def plot_kl_components_seeds(KL_curves_by_seed, fname, special_colors=None,
                             free_bits_curve=None, ylim=None):
    """Per-leaf KL across seeds: mean line with a min-max band."""
    fig, ax = plt.subplots()
    all_curves = list(KL_curves_by_seed.values())
    for component_name in all_curves[0]:
        stack = np.stack([np.asarray(curves[component_name], dtype=float)
                          for curves in all_curves])
        steps = np.arange(stack.shape[1])
        line_color, label = _kl_line_style(component_name, special_colors)
        ax.fill_between(steps, stack.min(axis=0), stack.max(axis=0),
                        color=line_color, alpha=0.2, linewidth=0)
        ax.plot(steps, stack.mean(axis=0), color=line_color, label=label)
    if free_bits_curve is not None:
        floor = np.asarray(free_bits_curve, dtype=float)
        floor = np.where(floor > 0, floor, np.nan)  # log axis cannot show tau == 0
        ax.plot(np.arange(len(floor)), floor, 'r--', linewidth=1.5,
                label='free-bits floor (tau)')
    if ylim is not None:
        ax.set_ylim(ylim)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    ax.set_title('KL per leaf, mean and min-max over seeds ' +
                 ','.join(str(seed) for seed in KL_curves_by_seed), size='small')
    _log_axis_setup(ax, 'step', 'KL contribution')
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)


def plot_kl_vs_reconstruction(KL_curves, reconstruction_error_curve, fname,
                              effective_total_KL_curve=None,
                              reconstruction_label='reconstruction error'):
    total_KL = 0
    for curve in KL_curves.values():
        total_KL = total_KL + np.asarray(curve, dtype=float)
    fig, ax = plt.subplots()
    ax.plot(np.arange(total_KL.shape[0]), total_KL, label='KL from z', color='k')
    if effective_total_KL_curve is not None:
        effective = np.asarray(effective_total_KL_curve, dtype=float)
        ax.plot(np.arange(effective.shape[0]), effective, '--', color='tab:blue',
                label='effective KL (free bits)')
    reconstruction = np.asarray(reconstruction_error_curve, dtype=float)
    ax.plot(np.arange(reconstruction.shape[0]), reconstruction,
            label=reconstruction_label, color='r')
    ax.legend()
    _log_axis_setup(ax, 'step', 'total KL or reconstruction error')
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)


def plot_eje_b_free_bits(n_below_floor_curve, free_bits_curve, fname):
    """Leaves whose KL is under the free-bits floor, with the floor schedule."""
    fig, ax = plt.subplots()
    count = np.asarray(n_below_floor_curve, dtype=float)
    ax.plot(np.arange(count.shape[0]), count, color='k', label='leaves below floor')
    ax.set_xlabel('step')
    ax.set_ylabel('leaves below floor')
    ax.grid(linestyle='-', linewidth='0.5', color='gray')
    twin = ax.twinx()
    floor = np.asarray(free_bits_curve, dtype=float)
    twin.plot(np.arange(floor.shape[0]), floor, 'r--', label='tau (nats)')
    twin.set_ylabel('free-bits floor tau (nats)')
    handles = ax.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + twin.get_legend_handles_labels()[1]
    ax.legend(handles, labels)
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)


def plot_eje_b_curriculum(weights_curve, fname):
    """Per-demonstration curriculum weights over training (mean is 1 by construction)."""
    weights = np.asarray(weights_curve, dtype=float)
    fig, ax = plt.subplots()
    for demo_num in range(weights.shape[1]):
        ax.plot(np.arange(weights.shape[0]), weights[:, demo_num],
                label='demo ' + str(demo_num))
    ax.axhline(1.0, color='gray', linestyle=':')
    ax.legend()
    ax.set_xlabel('step')
    ax.set_ylabel('curriculum weight')
    ax.grid(linestyle='-', linewidth='0.5', color='gray')
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)


def plot_pass2_curves(curves, fname, highlight=None):
    """Per-step pass@2 correctness (0/1) for each labelled curve, vertically offset to stay legible."""
    fig, ax = plt.subplots()
    for curve_num, (label, curve) in enumerate(curves.items()):
        curve = np.asarray(curve, dtype=float)
        is_highlight = label == highlight
        ax.step(np.arange(curve.shape[0]), curve + 0.04*curve_num, where='post',
                color='k' if is_highlight else None,
                linewidth=2.0 if is_highlight else 1.0, label=label)
    ax.set_ylim(-0.05, 1.05 + 0.04*len(curves))
    ax.legend(loc='center right')
    ax.set_xlabel('step')
    ax.set_ylabel('pass@2 correct (curves offset)')
    ax.grid(linestyle='-', linewidth='0.5', color='gray')
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)


def plot_fused_solution(attempts, fname, title=None):
    """Draw the two fused guesses (attempt_1, attempt_2) for every test input."""
    n_test = len(attempts)
    fig, axes = plt.subplots(n_test, 2, squeeze=False)
    for example_num, attempt_pair in enumerate(attempts):
        for column, key in enumerate(('attempt_1', 'attempt_2')):
            ax = axes[example_num][column]
            grid = np.array(attempt_pair[key])
            image = convert_color((np.arange(10) == grid[:, :, None]).astype(np.float32))
            ax.imshow(image, interpolation='none')
            ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
            ax.grid(which='minor', color=(59/255, 59/255, 59/255), linewidth=0.3)
            ax.tick_params(which='both', bottom=False, left=False,
                           labelbottom=False, labelleft=False)
            ax.set_title('guess ' + str(column + 1), size='small')
    if title:
        fig.suptitle(title, size='small')
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)


