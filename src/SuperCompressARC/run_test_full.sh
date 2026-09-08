#!/usr/bin/env bash
#
# Solve the complete 400-task `test` split with crash-safe retries.
# Test has no ground truth, so completion is reported without a solved count.
#
#   ./run_test_full.sh
#
# Errors: .log/YYYY-MM-DD/ and run_logs/test_*.log
# Timing: run_summary_test.json
# Outputs: submission_test.json, predictions_test.npz
#
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_full_split.sh" test