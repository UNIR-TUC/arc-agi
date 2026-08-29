#!/usr/bin/env bash
#
# Solve the complete 400-task `training` split.
#
#   ./run_training_full.sh
#
# ~30 h on the RX 9070 XT. Safe to Ctrl-C and relaunch: finished tasks are
# persisted per-task and skipped on the next run.
#
# Outputs: submission_training.json, predictions_training.npz
#
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_full_split.sh" training
