import unittest

import numpy as np

import solution_selection


def candidate(solution, score, first_step=0, source_index=0):
    canonical = solution_selection._canonical_solution(solution)
    return {
        'hash': hash(canonical),
        'solution': solution_selection._solution_to_lists(canonical),
        'score': score,
        'first_step': first_step,
        'source_index': source_index,
    }


def logger_data(contributions, candidates):
    return {
        'solution_contributions_log': contributions,
        'solution_picks_history': [[0, 0] for _ in contributions],
        'candidate_evidence': candidates,
    }


class SeedMergeTests(unittest.TestCase):
    def test_repeated_candidate_accumulates_evidence_and_wins(self):
        solution_a = (((1,),),)
        solution_b = (((2,),),)
        hash_a = hash(solution_a)
        hash_b = hash(solution_b)
        seeds = {
            0: logger_data(
                [[(hash_a, -1.0), (hash_b, -0.8)]],
                [candidate(solution_a, -1.0), candidate(solution_b, -0.8)],
            ),
            1: logger_data(
                [[(hash_a, -1.0), (hash_b, -np.inf)]],
                [candidate(solution_a, -1.0)],
            ),
        }

        attempts, merged = solution_selection.merge_seed_logger_data(seeds)

        self.assertEqual(attempts[0]['attempt_1'], [[1]])
        self.assertEqual(attempts[0]['attempt_2'], [[2]])
        self.assertEqual(len(merged['solution_contributions_log'][0]), 4)
        self.assertEqual(merged['solution_picks_history'][0][0], hash_a)
        expected = float(np.logaddexp(-1.0, -1.0))
        self.assertAlmostEqual(merged['candidate_evidence'][0]['score'], expected)

    def test_tie_breaking_is_stable_by_seed_and_first_observation(self):
        solution_a = (((1,),),)
        solution_b = (((2,),),)
        seeds = {
            2: logger_data(
                [[(hash(solution_b), -1.0), (0, -np.inf)]],
                [candidate(solution_b, -1.0)],
            ),
            1: logger_data(
                [[(hash(solution_a), -1.0), (0, -np.inf)]],
                [candidate(solution_a, -1.0)],
            ),
        }

        attempts, _ = solution_selection.merge_seed_logger_data(seeds)

        self.assertEqual(attempts[0]['attempt_1'], [[1]])
        self.assertEqual(attempts[0]['attempt_2'], [[2]])

    def test_earlier_step_beats_lower_seed_when_scores_tie(self):
        solution_a = (((1,),),)
        solution_b = (((2,),),)
        seeds = {
            0: logger_data(
                [[(0, -np.inf), (0, -np.inf)]] * 2,
                [candidate(solution_a, -1.0, first_step=1)],
            ),
            1: logger_data(
                [[(0, -np.inf), (0, -np.inf)]] * 2,
                [candidate(solution_b, -1.0, first_step=0)],
            ),
        }

        attempts, _ = solution_selection.merge_seed_logger_data(seeds)

        self.assertEqual(attempts[0]['attempt_1'], [[2]])

    def test_mismatched_training_lengths_are_rejected(self):
        solution = (((1,),),)
        one_step = logger_data(
            [[(hash(solution), -1.0), (0, -np.inf)]],
            [candidate(solution, -1.0)],
        )
        two_steps = logger_data(
            [[(hash(solution), -1.0), (0, -np.inf)]] * 2,
            [candidate(solution, -1.0)],
        )

        with self.assertRaisesRegex(ValueError, 'same number of steps'):
            solution_selection.merge_seed_logger_data({0: one_step, 1: two_steps})

    def test_second_attempt_preserves_confident_single_seed_diversity(self):
        consensus = (((1,),),)
        runner_up = (((2,),),)
        diverse = (((3,),),)
        seeds = {
            0: logger_data(
                [[(hash(consensus), 5.0), (hash(runner_up), 4.0)]],
                [candidate(consensus, 5.0), candidate(runner_up, 4.0)],
            ),
            1: logger_data(
                [[(hash(consensus), 5.0), (hash(runner_up), 4.0)]],
                [candidate(consensus, 5.0), candidate(runner_up, 4.0)],
            ),
            2: logger_data(
                [[(hash(diverse), 3.0), (0, -np.inf)]],
                [candidate(diverse, 3.0)],
            ),
        }

        attempts, merged = solution_selection.merge_seed_logger_data(seeds)

        self.assertEqual(attempts[0]['attempt_1'], [[1]])
        self.assertEqual(attempts[0]['attempt_2'], [[3]])
        self.assertEqual(
            merged['merge_policy'], solution_selection.SEED_MERGE_POLICY
        )
        self.assertEqual(
            merged['solution_picks_history'][-1],
            [hash(consensus), hash(diverse)],
        )


if __name__ == '__main__':
    unittest.main()