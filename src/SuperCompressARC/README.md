<a href="https://iliao2345.github.io/blog_posts/arc_agi_without_pretraining/arc_agi_without_pretraining.html"></a>
This is the code base for the [ARC-AGI Without Pretraining](https://iliao2345.github.io/blog_posts/arc_agi_without_pretraining/arc_agi_without_pretraining.html) project. The Kaggle competition template version can be found [here](https://www.kaggle.com/code/iliao2345/arc-agi-without-pretraining/notebook?scriptVersionId=232760209).

# Installation

```
> git clone https://github.com/iliao2345/CompressARC.git
> cd CompressARC
> python -m venv arc_agi_without_pretraining
> source arc_agi_without_pretraining/bin/activate
> pip install -r requirements.txt
```

# How to solve an ARC-AGI task

Run `analyze_example.py` to initialize a new model and train from scratch:

```
> python analyze_example.py
Enter which split you want to find the task in (training, evaluation, test): <split>
Enter which task you want to analyze (eg. 272f95fa): <task>
Performing a training run on task <task> and placing the results in <task>/
|100%|███████████████████████████████████████████████| 1500/1500 [12:22<00:00, 2.01it/s]
done
```

The code will create a folder `<task>/` and put plots there after 1500 steps of training:

- solutions at every 50 steps
- interpretable tensors of task representations
- graph of each tensor's contribution to the KL over time
- graph of the KL vs reconstruction error over time

Most tasks may take up to 20 minutes to run, on one NVIDIA GeForce RTX 4070 GPU.

# How to see which puzzles were solved, using the run information in this repo

Run `python list_solved_puzzles.py results_for_the_blog_post/predictions_training.npz training 2000`

This prints out a table indicating

- names of solved puzzles
- how many guesses would have been required (up to 4000) to solve each puzzle
- total number of solved puzzles with 2 guesses
- total number of solved puzzles

You can change the 2000 to any number of inference-time steps, if you want to observe partway through the solution process. You can also change `predictions_training.npz` and `training` to `predictions_evaluation.npz` and `evaluation` to see which evaluation puzzles were solved as well. If you want the results in a text file instead, you can pipe the output into a file:

```
python list_solved_puzzles.py results_for_the_blog_post/predictions_training.npz training 2000 >> result.txt
```

# Running All Tasks and Debugging Guide

## Running all tasks in one command

Two scripts already support running the full training split (400 tasks) in a single command. They differ only in execution strategy — the algorithm, number of iterations (2000), and optimizer settings are identical for every task.

### Sequential — `train.py`

```
> python train.py
```

Trains all 400 tasks **one at a time**. Safe and simple, but slow.

| Property        | Value                                                                       |
| --------------- | --------------------------------------------------------------------------- |
| Wall-clock time | ~130 h (400 × ~20 min)                                                     |
| GPU VRAM needed | ~0.5–1 GB (one task at a time)                                             |
| Output          | `timing_result.txt`, per-task plots, rolling `predictions_training.npz` |
| Accuracy saved? | Yes — updates after every task finishes                                    |

### Parallel — `parallel_train.py` (recommended)

```
> python parallel_train.py
```

Runs as many tasks simultaneously as your GPU VRAM allows, then refills slots as tasks finish. 4–10× faster than sequential.

| Property        | Value                                                                                                                                           |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| Wall-clock time | ~15–30 h                                                                                                                                       |
| GPU VRAM needed | up to full GPU memory (scheduler auto-limits)                                                                                                   |
| Output          | `submission.json` (Kaggle-format)                                                                                                             |
| How it works    | Phase 1: runs 2 iterations per task to measure VRAM footprint. Phase 2: greedy scheduler packs tasks onto the GPU under the safe memory budget. |

### Changing the split

Both scripts hardcode `split = "training"` near the top of their `__main__` block. To run evaluation or test tasks, edit that line:

```python
# in train.py or parallel_train.py, find this line and change it:
split = "training"  # change to "evaluation" or "test"
```

### Are results the same between the two scripts?

Statistically equivalent, but **not bit-for-bit identical**. Both run the same algorithm for 2000 iterations per task.The differences are:

- **Random initialization**: `train.py` advances a single global seed (set once to 0) sequentially across all tasks. `parallel_train.py` spawns each task in a fresh subprocess; each subprocess re-imports `arc_compressor.py`, which resets the seed to 0 — so parallel tasks start from approximately the same seed and diverge only from different task architectures.
- **GPU non-determinism**: when multiple tasks share the GPU concurrently, floating-point operation ordering is not guaranteed.

The expected **pass@2 rate across the full split is the same** (~34.75 % on training). Individual task solutions may differ between runs.

---

## Step-by-step debugging guide

> **GPU required.** All scripts call `torch.set_default_device('cuda')` at import time. Running on a machine without a CUDA GPU will fail immediately. Verify with: `python -c "import torch; print(torch.cuda.is_available())"`.

Use `analyze_example.py` as the debug entry point — it runs a single task interactively, produces plots, and is the easiest place to insert breakpoints.

```
> python analyze_example.py
Enter which split you want to find the task in (training, evaluation, test): training
Enter which task you want to analyze (eg. 272f95fa): 272f95fa
```

Some known-interesting tasks to start with: `272f95fa`, `6d75e8bb`, `6cdd2623`, `41e4d17e`, `2bee17df`.

### Execution flow and what to inspect at each step

**Step 1 — Task preprocessing** (`preprocessing.py`)

```python
task = preprocessing.preprocess_tasks('training', ['272f95fa'])[0]
```

Key attributes to inspect:

| Attribute                         | What it tells you                                                                                                 |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `task.n_examples`               | Total examples (train + test pairs)                                                                               |
| `task.n_train`, `task.n_test` | Split counts                                                                                                      |
| `task.shapes`                   | `[[in_shape, out_shape], ...]` per example. Check that output shapes are predicted correctly for test examples. |
| `task.n_colors`                 | Number of non-black colors found in the task                                                                      |
| `task.in_out_same_size`         | If`True`, output grid = input grid size (simplifies loss)                                                       |
| `task.problem`                  | Tensor`[n_examples, n_x, n_y, 2]` — channel 0 = input, channel 1 = output                                      |

**Step 2 — Model creation** (`arc_compressor.py`)

```python
model = arc_compressor.ARCCompressor(task)
print(sum(w.numel() for w in model.weights_list))  # should be ~76 000
```

- `model.weights_list` — shared model weights (~76 K params). These are passed to the optimizer.
- `model.multiposteriors` — task-specific latent means/variances. **Not** in `weights_list`; their shape scales with `n_examples × n_colors × n_x × n_y`.

**Step 3 — Optimizer**

```python
optimizer = torch.optim.Adam(model.weights_list, lr=0.01, betas=(0.5, 0.9))
```

`betas=(0.5, 0.9)` is intentionally non-standard: lower β₁ makes the optimizer react faster to gradient changes on small, task-specific datasets.

**Step 4 — Logger** (`solution_selection.py`)

```python
logger = solution_selection.Logger(task)
```

Tracks two pass@2 candidates per step: the current VAE sample and an EMA (decay=0.97) of logits. After training, `logger.solution_most_frequent` and `logger.solution_second_most_frequent` are the final predictions.

**Step 5 — Training step** (`train.py → take_step`)

```python
train.take_step(task, model, optimizer, train_step=0, train_history_logger=logger)
```

The forward pass returns:

| Variable                 | Shape / type                        | What to watch                                            |
| ------------------------ | ----------------------------------- | -------------------------------------------------------- |
| `logits`               | `[n_ex, n_colors+1, n_x, n_y, 2]` | Color predictions for input/output grids                 |
| `x_mask`, `y_mask`   | `[n_ex, n_x/n_y, 2]`              | Grid boundary predictions                                |
| `KL_amounts`           | list of 27 tensors                  | One KL value per multitensor component                   |
| `KL_names`             | list of 27 strings                  | Which dims each KL corresponds to, e.g.`"[1,1,0,1,1]"` |
| `total_KL`             | scalar                              | Should decrease over training                            |
| `reconstruction_error` | scalar                              | Should decrease over training                            |

**Step 6 — Monitor posterior collapse**

The most common failure mode is **posterior collapse**: 14 of the 27 multitensor components drop to KL ≈ 0 and never recover, meaning the model stops using those information channels.

```python
# After a training step, print per-tensor KL values:
for name, kl in zip(KL_names, KL_amounts):
    print(f"{name}: {kl.sum().item():.4f}")
```

If nearly all values are near zero after step 50–100, the run is unlikely to find a solution. Re-run with a different seed or see the architecture docs for the free-bits mitigation proposal.

### Common failure causes

| Symptom                                                                | Likely cause                                                                                                                                                |
| ---------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `RuntimeError: CUDA error` or `AssertionError` on import           | No CUDA GPU available. All modules set`torch.set_default_device('cuda')` at import time.                                                                  |
| Results differ across runs with same seed                              | Known issue — noted in`analyze_example.py` comments. Seeds don't fully fix results.                                                                      |
| Loss stays high / all KL near zero after ~100 steps                    | Posterior collapse. Normal for many tensors to collapse; problematic if all do.                                                                             |
| `grid_size_uncertain = True` in loss loop                            | Output size prediction uncertain (neither same-size nor all-same-output-size). The loss uses a curriculum coefficient for the first 100 steps to stabilize. |
| `submission.json` missing some task keys after `parallel_train.py` | A subprocess crashed silently. Check`error_queue` propagation or run `python train.py` for a sequential run with full tracebacks.                       |

# Tips for Reading the Code

A basic description of the code files in this repo:

**For running via command line:**

- `analyze_example.py`: Demonstrates how to solve one ARC-AGI problem using our method, with visualizations of learned task representations and plots of metrics.
- `plot_problems.py`: Plots all of the ARC-AGI problems in a split.
- `plot_accuracy.py`: Plots pass@n accuracies during/after a bulk training run with `train.py`.
- `train.py`: Trains a model for every task in a split, plotting the accuracy. Contains code that computes the loss function. Defaults to the training split.
- `parallel_train.py`: A multiprocessing program that schedules as many puzzles as possible in a split to be solved at the same time through `solve_task.py`, while maximizing the GPU memory usage. Defaults to the training split.
- `scoring.py`: A script for scoring the results of `parallel_train.py`, which are better-formatted for Kaggle submissions.

**Functionality, not for running via command line:**

- `arc_compressor.py`: The network architecture and forward pass.
- `initializers.py`: Model initialization, and handling of equivariances via weight tying.
- `layers.py`: Implementation of individual layers in the forward pass.
- `multitensor_systems.py`: Handling multitensors.
- `preprocessing.py`: Converting the dataset into a form usable by the repo.
- `solution_selection.py`: Logging metrics and converting model outputs into solution predictions.
- `visualization.py`: Drawing problems and solutions.
- `solve_task.py`: A job script to solve one ARC-AGI problem on a specified GPU.

**Some classes that the repo defines and uses:**

- `MultiTensorSystem` (in `multitensor_systems.py`): A class that can spawn MultiTensors using stored dimensional information.
- `MultiTensor` (in `multitensor_systems.py`): Container class for groups of tensors.
- `Logger` (in `solution_selection.py`): For postprocessing of solutions outputted by the model, and their collection over time during training.
- `Task` (in `preprocessing.py`): Contains information about an ARC-AGI task, such as grid dimensions and masks, pixel colors, etc.
- `ARCCompressor` (in `arc_compressor.py`): Model class, with forward pass.
- `Initializer` (in `initializers.py`): For initializing model weights.

**Some repo-specific language that we use for variable naming, etc.**

- `dims` refers to a length 5 list of zeros and ones, and refers to the presence/absence of each of the five multitensor dimensions $(example, color, direction, height, width)$. Channel dimension is implicitly included.
- `axis` always refers to the index of some dim in a tensor. For example, in a $(example, color, height)$ tensor, the $height$ dim is the 2nd axis, whereas for the $(height, width)$ tensor, it is the 0th axis.
- This repo uses `x` and `y` to refer to the $height$ and $width$ dimensions, respectively.
- The `@multitensor_systems.multify` decorator takes a function and modifies it to apply it once for every tensor in a multitensor. If the input is a tensor/object, then the new input is now a multitensor/multiobject. The function must be written with additional parameter `dims`.
- The `@layers.add_residual` decorator takes a function and creates a residual connection around it, with projections to/from the input/output of the function and the residual stream. Optional parameters are added for using biases for the projections, using pre-norm, and post-norm.
- The `@layers.only_do_for_certain_shapes(*shapes)` decorator takes a function with `dims` as its first input, and applies the function only if `dims` is in `shapes`. Else, it applies the identity function. Useful for chaining with the `@multitensor_systems.multify` decorator.

Code for different files may be written in slightly different styles due to polishing of individual code files by ChatGPT.

# Citation

If you'd like to cite this blog post, use the following entry:

```
@online{liao2025arcagiwithoutpretraining,
	author = {Isaac Liao and Albert Gu},
	title = {ARC-AGI Without Pretraining},
	year = {2025},
	url = {https://iliao2345.github.io/blog_posts/arc_agi_without_pretraining/arc_agi_without_pretraining.html},
}
```
