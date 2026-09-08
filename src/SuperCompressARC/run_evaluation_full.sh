#!/usr/bin/env bash
#
# Solve the complete 400-task `evaluation` split.
#
#   ./run_evaluation_full.sh
#
# ~30-35 h on the RX 9070 XT. Completed tasks are persisted before the worker
# exits and skipped after an automatic or manual restart. Tasks active during
# the failure restart from iteration zero.
#
# Run this after run_training_full.sh: the two splits share .inductor_cache, so
# whatever kernels the training puzzles compiled are reused here for free.
#
# Errors: .log/YYYY-MM-DD/ and run_logs/evaluation_*.log
# Timing: run_summary_evaluation.json
# Outputs: submission_evaluation.json, predictions_evaluation.npz
#
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_full_split.sh" evaluation
