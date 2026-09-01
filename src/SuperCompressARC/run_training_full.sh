#!/usr/bin/env bash
#
# Solve the complete 400-task `training` split.
#
#   ./run_training_full.sh
#
# ~30 h on the RX 9070 XT. Completed tasks are persisted before the worker
# exits and skipped after an automatic or manual restart. Recoverable compiled
# failures retry eagerly in a fresh attempt; an eager failure quarantines that
# task so the remaining split can finish.
#
# Force a known task eager: EAGER_TASKS=fcb5c309 ./run_training_full.sh
# Retry quarantine later:  RETRY_QUARANTINED_TASKS=fcb5c309 ./run_training_full.sh
#
# Errors: .log/YYYY-MM-DD/ and run_logs/training_*.log
# Timing: run_summary_training.json
# Outputs: submission_training.json, predictions_training.npz
#
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_full_split.sh" training
