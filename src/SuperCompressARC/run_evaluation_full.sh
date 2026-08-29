#!/usr/bin/env bash
#
# Solve the complete 400-task `evaluation` split.
#
#   ./run_evaluation_full.sh
#
# ~30-35 h on the RX 9070 XT. Safe to Ctrl-C and relaunch: finished tasks are
# persisted per-task and skipped on the next run.
#
# Run this after run_training_full.sh: the two splits share .inductor_cache, so
# whatever kernels the training puzzles compiled are reused here for free.
#
# Outputs: submission_evaluation.json, predictions_evaluation.npz
#
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_full_split.sh" evaluation
