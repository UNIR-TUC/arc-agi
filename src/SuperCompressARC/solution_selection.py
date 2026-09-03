import matplotlib.pyplot as plt
import numpy as np
import torch

np.random.seed(0)
torch.manual_seed(0)

class Logger:
    """
    This class contains functionalities relating to the recording of model outputs, postprocessing,
    selection of most frequently sampled/highest scoring solutions, accuracy computations, and more.
    """
    ema_decay = 0.97

    def __init__(self, task, postprocess_stride=1):
        self.task = task
        self.postprocess_stride = postprocess_stride
        self.KL_curves = {}
        self.total_KL_curve = []
        self.effective_total_KL_curve = []
        self.kl_free_bits_curve = []
        self.n_kl_below_floor_curve = []
        self.curriculum_weights_curve = []
        self.reconstruction_error_curve = []
        self.loss_curve = []

        n_test, n_colors, n_x, n_y = task.n_test, task.n_colors, task.n_x, task.n_y
        shape = (n_test, n_colors + 1, n_x, n_y)

        self.current_logits = torch.zeros(shape)
        self.current_x_mask = torch.zeros((n_test, n_x))
        self.current_y_mask = torch.zeros((n_test, n_y))

        self.ema_logits = torch.zeros(shape)
        self.ema_x_mask = torch.zeros((n_test, n_x))
        self.ema_y_mask = torch.zeros((n_test, n_y))

        self.solution_hashes_count = {}
        self.solutions_by_hash = {}
        self.solution_first_seen = {}
        self.solution_most_frequent = None
        self.solution_second_most_frequent = None

        self.solution_contributions_log = []
        self.solution_picks_history = []

    def log(self, train_step, logits, x_mask, y_mask, KL_amounts, KL_names,
            total_KL, reconstruction_error, loss, effective_total_KL=None,
            kl_free_bits=0.0, n_kl_below_floor=None,
            curriculum_weights=None):
        """Logs training progress and tracks solutions from one forward pass."""
        if train_step == 0:
            self.KL_curves = {KL_name: [] for KL_name in KL_names}

        # Eje H, H2: accumulate as detached GPU tensors instead of syncing to CPU
        # every step (`.cpu().numpy()` forces a cudaSynchronize). Call
        # materialize_curves() once (e.g. at the end of training) to convert to
        # plain floats in a single batched sync.
        for KL_amount, KL_name in zip(KL_amounts, KL_names):
            self.KL_curves[KL_name].append(KL_amount.detach().sum())

        self.total_KL_curve.append(total_KL.detach())
        effective_total_KL = (
            total_KL if effective_total_KL is None else effective_total_KL
        )
        self.effective_total_KL_curve.append(effective_total_KL.detach())
        self.kl_free_bits_curve.append(torch.as_tensor(
            kl_free_bits, device=total_KL.device
        ).detach())
        if n_kl_below_floor is not None:
            self.n_kl_below_floor_curve.append(n_kl_below_floor.detach())
        if curriculum_weights is not None:
            self.curriculum_weights_curve.append(curriculum_weights.detach())
        self.reconstruction_error_curve.append(reconstruction_error.detach())
        self.loss_curve.append(loss.detach())

        self._track_solution(train_step, logits.detach(), x_mask.detach(), y_mask.detach())

    def materialize_curves(self):
        """Convert accumulated GPU scalar tensors into plain Python floats in a
        single batched host sync (Eje H, H2), instead of syncing on every
        training step. Idempotent: a no-op if a curve is empty or already
        materialized."""
        for name, values in self.KL_curves.items():
            if values and isinstance(values[0], torch.Tensor):
                self.KL_curves[name] = torch.stack(values).cpu().tolist()
        for attr in (
            'total_KL_curve',
            'effective_total_KL_curve',
            'kl_free_bits_curve',
            'n_kl_below_floor_curve',
            'curriculum_weights_curve',
            'reconstruction_error_curve',
            'loss_curve',
        ):
            values = getattr(self, attr)
            if values and isinstance(values[0], torch.Tensor):
                setattr(self, attr, torch.stack(values).cpu().tolist())

    def _track_solution(self, train_step, logits, x_mask, y_mask):
        """Postprocess and score solutions and keep track of the top two solutions with highest scores."""
        self.current_logits = logits[self.task.n_train:, :, :, :, 1]  # example, color, x, y
        self.current_x_mask = x_mask[self.task.n_train:, :, 1]  # example, x
        self.current_y_mask = y_mask[self.task.n_train:, :, 1]  # example, y

        self.ema_logits = self.ema_decay * self.ema_logits + (1 - self.ema_decay) * self.current_logits
        self.ema_x_mask = self.ema_decay * self.ema_x_mask + (1 - self.ema_decay) * self.current_x_mask
        self.ema_y_mask = self.ema_decay * self.ema_y_mask + (1 - self.ema_decay) * self.current_y_mask

        # Eje H, H3: the expensive part below (argmax/uncertainty, then a
        # .cpu().numpy() sync + Python pixel-recoloring loops, run twice per
        # call) only needs to happen every `postprocess_stride` steps for the
        # pass@2 evidence to converge; always run on step 0 so a valid solution
        # exists from the start.
        if self.postprocess_stride <= 1 or train_step % self.postprocess_stride == 0:
            solution_contributions = []
            for source_index, (logits, x_mask_set, y_mask_set) in enumerate([  # Add two potential solutions: sample and mean.
                (self.current_logits, self.current_x_mask, self.current_y_mask),
                (self.ema_logits, self.ema_x_mask, self.ema_y_mask)
            ]):

                # Get the solution and the score.
                solution, uncertainty = self._postprocess_solution(logits, x_mask_set, y_mask_set)
                hashed_solution = hash(solution)
                previous_solution = self.solutions_by_hash.get(hashed_solution)
                if previous_solution is not None and previous_solution != solution:
                    raise RuntimeError('solution hash collision while tracking candidates')
                self.solutions_by_hash[hashed_solution] = solution
                self.solution_first_seen.setdefault(
                    hashed_solution, (train_step, source_index)
                )
                score = -10*uncertainty
                if train_step < 150:
                    score = score - 10
                if logits is self.ema_logits:
                    score = score - 4

                # Accumulate scores for solutions.
                solution_contributions.append((hashed_solution, score))
                self.solution_hashes_count[hashed_solution] = float(np.logaddexp(
                    self.solution_hashes_count.get(hashed_solution, -np.inf), score))

                self._update_most_frequent_solutions(hashed_solution, solution)
        else:
            # Neutral no-op contribution: np.logaddexp(x, -inf) == x exactly, so
            # downstream re-accumulators (plot_accuracy.py, list_solved_puzzles.py)
            # that logaddexp-sum every logged iteration are unaffected, and the
            # log stays one entry per iteration.
            solution_contributions = [(0, -np.inf), (0, -np.inf)]

        self.solution_contributions_log.append(solution_contributions)
        self.solution_picks_history.append([hash(sol) for sol in [
            self.solution_most_frequent, self.solution_second_most_frequent]])

    def candidate_evidence(self):
        """Return every canonical candidate and its accumulated score."""
        evidence = []
        for hashed_solution, solution in self.solutions_by_hash.items():
            first_step, source_index = self.solution_first_seen[hashed_solution]
            evidence.append({
                'hash': hashed_solution,
                'solution': _solution_to_lists(solution),
                'score': self.solution_hashes_count[hashed_solution],
                'first_step': first_step,
                'source_index': source_index,
            })
        return evidence

    def _update_most_frequent_solutions(self, hashed, solution):
        """Keeps track of the top two solutions with highest scores."""
        if self.solution_most_frequent is None:
            self.solution_most_frequent = solution
        if self.solution_second_most_frequent is None:
            self.solution_second_most_frequent = solution

        if hashed != hash(self.solution_most_frequent):
            if self.solution_hashes_count[hashed] >= self.solution_hashes_count.get(
                    hash(self.solution_second_most_frequent), -np.inf):
                self.solution_second_most_frequent = solution
                if self.solution_hashes_count[hashed] >= self.solution_hashes_count.get(
                        hash(self.solution_most_frequent), -np.inf):
                    self.solution_second_most_frequent = self.solution_most_frequent
                    self.solution_most_frequent = solution

    def best_crop(self, prediction, x_mask, x_length, y_mask, y_length):
        x_start, x_end = self._best_slice_point(x_mask, x_length)
        y_start, y_end = self._best_slice_point(y_mask, y_length)
        return prediction[..., x_start:x_end, y_start:y_end]

    def _best_slice_point(self, mask, length):
        if self.task.in_out_same_size or self.task.all_out_same_size:
            search_lengths = [length]
        else:
            search_lengths = list(range(1, mask.shape[0]+1))
        max_logprob, best_slice_start, best_slice_end = None, None, None

        for length in search_lengths:
            logprobs = torch.stack([
                -torch.sum(mask[:offset]) + torch.sum(mask[offset:offset + length]) - torch.sum(mask[offset + length:])
                for offset in range(mask.shape[0] - length + 1)
            ])
            if max_logprob is None or torch.max(logprobs) > max_logprob:
                max_logprob = torch.max(logprobs)
                best_slice_start = torch.argmax(logprobs).item()
                best_slice_end = best_slice_start + length

        return best_slice_start, best_slice_end

    def _postprocess_solution(self, prediction, x_mask, y_mask):  # prediction must be example, color, x, y
        """Postprocess a solution and compute some variables that are used to calculate the score."""
        colors = torch.argmax(prediction, dim=1)  # example, x, y
        uncertainties = torch.logsumexp(prediction, dim=1) - torch.amax(prediction, dim=1)  # example, x, y
        solution_slices, uncertainty_values = [], []  # example, x, y; example

        for example_num in range(self.task.n_test):
            x_length = None
            y_length = None
            if self.task.in_out_same_size or self.task.all_out_same_size:
                x_length = self.task.shapes[self.task.n_train+example_num][1][0]
                y_length = self.task.shapes[self.task.n_train+example_num][1][1]
            solution_slice = self.best_crop(colors[example_num],
                                            x_mask[example_num],
                                            x_length,
                                            y_mask[example_num],
                                            y_length)  # x, y
            uncertainty_slice = self.best_crop(uncertainties[example_num],
                                               x_mask[example_num],
                                               x_length,
                                               y_mask[example_num],
                                               y_length)  # x, y

            solution_slices.append(solution_slice.cpu().numpy().tolist())
            uncertainty_values.append(float(np.mean(uncertainty_slice.cpu().numpy())))

        for example in solution_slices:
            for row in example:
                for i, val in enumerate(row):
                    row[i] = self.task.colors[val]

        solution_slices = tuple(tuple(tuple(row) for row in example) for example in solution_slices)
        return solution_slices, np.mean(uncertainty_values)


def _canonical_solution(solution):
    return tuple(
        tuple(tuple(int(value) for value in row) for row in example)
        for example in solution
    )


def _solution_to_lists(solution):
    return [[list(row) for row in example] for example in solution]


def merge_seed_logger_data(seed_loggers):
    """Merge independent seed evidence into one deterministic logical logger."""
    if not seed_loggers:
        raise ValueError('at least one seed logger is required')

    ordered = sorted(seed_loggers.items())
    lengths = {
        len(logger_data['solution_contributions_log'])
        for _, logger_data in ordered
    }
    if len(lengths) != 1:
        raise ValueError('seed loggers must contain the same number of steps')

    merged_contributions = []
    merged_picks = []
    running_scores = {}
    for train_step in range(lengths.pop()):
        step_contributions = []
        for _, logger_data in ordered:
            step_contributions.extend(
                logger_data['solution_contributions_log'][train_step]
            )
        merged_contributions.append(step_contributions)
        for hashed_solution, score in step_contributions:
            running_scores[hashed_solution] = float(np.logaddexp(
                running_scores.get(hashed_solution, -np.inf), score
            ))
        ranked_hashes = [
            hashed_solution
            for hashed_solution, _ in sorted(
                running_scores.items(),
                key=lambda item: (-item[1], item[0]),
            )
            if running_scores[hashed_solution] > -np.inf
        ]
        if not ranked_hashes:
            merged_picks.append([hash(None), hash(None)])
        elif len(ranked_hashes) == 1:
            merged_picks.append([ranked_hashes[0], ranked_hashes[0]])
        else:
            merged_picks.append(ranked_hashes[:2])

    merged_by_solution = {}
    solution_by_hash = {}
    for seed, logger_data in ordered:
        for candidate in logger_data.get('candidate_evidence', []):
            solution = _canonical_solution(candidate['solution'])
            hashed_solution = int(candidate['hash'])
            previous = solution_by_hash.get(hashed_solution)
            if previous is not None and previous != solution:
                raise RuntimeError('solution hash collision across seeds')
            solution_by_hash[hashed_solution] = solution
            first_seen = (
                int(candidate['first_step']),
                seed,
                int(candidate['source_index']),
            )
            record = merged_by_solution.get(solution)
            if record is None:
                merged_by_solution[solution] = {
                    'hash': hashed_solution,
                    'solution': solution,
                    'score': float(candidate['score']),
                    'first_seen': first_seen,
                }
            else:
                record['score'] = float(np.logaddexp(
                    record['score'], candidate['score']
                ))
                record['first_seen'] = min(record['first_seen'], first_seen)

    ranked_candidates = sorted(
        merged_by_solution.values(),
        key=lambda candidate: (
            -candidate['score'],
            candidate['first_seen'],
            candidate['hash'],
        ),
    )
    if not ranked_candidates:
        raise ValueError('seed loggers contain no candidate evidence')
    if len(ranked_candidates) == 1:
        ranked_candidates.append(ranked_candidates[0])

    attempts = []
    first_solution = ranked_candidates[0]['solution']
    second_solution = ranked_candidates[1]['solution']
    for example_num in range(len(first_solution)):
        attempts.append({
            'attempt_1': [list(row) for row in first_solution[example_num]],
            'attempt_2': [list(row) for row in second_solution[example_num]],
        })

    merged_evidence = [{
        'hash': candidate['hash'],
        'solution': _solution_to_lists(candidate['solution']),
        'score': candidate['score'],
        'first_seen': list(candidate['first_seen']),
    } for candidate in ranked_candidates]
    return attempts, {
        'solution_contributions_log': merged_contributions,
        'solution_picks_history': merged_picks,
        'candidate_evidence': merged_evidence,
        'seed_loggers': {str(seed): data for seed, data in ordered},
    }


def save_predictions(loggers, fname='predictions.npz'):
    """Saves solution score contributions and history of chosen solutions."""
    np.savez(fname,
             solution_contribution_logs=[logger.solution_contributions_log for logger in loggers],
             solution_picks_histories=[logger.solution_picks_history for logger in loggers])


def plot_accuracy(true_solution_hashes, fname='predictions.npz'):
    """Plots accuracy curve over training iterations."""
    stored_data = np.load(fname, allow_pickle=True)
    solution_picks_histories = stored_data['solution_picks_histories']

    n_tasks = len(solution_picks_histories)
    n_iterations = len(solution_picks_histories[0])

    correct = np.array([[
        int(any(hash_ == true_solution_hashes[task_num] for hash_ in solution_pair))
        for solution_pair in task_history
    ] for task_num, task_history in enumerate(solution_picks_histories)])

    accuracy_curve = correct.mean(axis=0)

    plt.figure()
    plt.plot(np.arange(n_iterations), accuracy_curve, 'k-')
    plt.savefig('accuracy_curve.pdf', bbox_inches='tight')
    plt.close()
