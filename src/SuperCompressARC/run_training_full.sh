#!/usr/bin/env bash
#
# Solve the `training` split with Eje B (four independent seeds).
#
#   ./run_training_full.sh
#   ./run_training_full.sh --axis-d-unsolved
#
# The optional axis-D mode excludes the 140 tasks solved by the previous D
# campaign and runs only the remaining 260 tasks (~2,080,000 optimizer steps).
# Completed seed jobs are persisted before the worker exits and skipped after an
# automatic or manual restart.
#
# Force a known task eager: EAGER_TASKS=fcb5c309 ./run_training_full.sh
# Retry quarantine later:  RETRY_QUARANTINED_TASKS=fcb5c309 ./run_training_full.sh
#
# The B campaign is isolated from the old D artifacts:
# Outputs: split_results/training/eje_b_*/
# State:   .partial_eje_b/training/
# Errors and timing are stored alongside the selected output directory.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="full"
while (( $# )); do
	case "$1" in
		--axis-d-unsolved)
			MODE="axis-d-unsolved"
			shift
			;;
		--help|-h)
			echo "usage: $0 [--axis-d-unsolved]"
			exit 0
			;;
		*)
			echo "usage: $0 [--axis-d-unsolved]" >&2
			exit 2
			;;
	esac
done

PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/arcagi/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
	echo "[run] ERROR: Python interpreter is not executable: ${PYTHON_BIN}" >&2
	exit 1
fi
if [[ ! -f "${SCRIPT_DIR}/task_selection.py" ]]; then
	echo "[run] ERROR: task_selection.py not found" >&2
	exit 1
fi

ACCEL_PRESET="${ACCEL_PRESET:-compile}"
if [[ "$ACCEL_PRESET" != "compile" ]]; then
	echo "[run] ERROR: Eje B requires ACCEL_PRESET=compile" >&2
	exit 2
fi
ITERATIONS="${ITERATIONS:-2000}"
SEEDS="${SEEDS:-0,1,2,3}"
POSTPROCESS_STRIDE="${POSTPROCESS_STRIDE:-4}"
STATE_DIR="${STATE_DIR:-.partial_eje_b}"
if [[ "$MODE" == "axis-d-unsolved" ]]; then
	OUTPUT_DIR="${OUTPUT_DIR:-split_results/training/eje_b_axis_d_unsolved}"
else
	OUTPUT_DIR="${OUTPUT_DIR:-split_results/training/eje_b_full}"
fi

export ACCEL_PRESET ITERATIONS SEEDS POSTPROCESS_STRIDE STATE_DIR OUTPUT_DIR
export EJE_B=1
export RUN_LOG_PATH="${RUN_LOG_PATH:-${OUTPUT_DIR}/run.log}"
export ATTEMPTS_FILE="${ATTEMPTS_FILE:-${OUTPUT_DIR}/attempts.jsonl}"
export SPLIT_SUMMARY_PATH="${SPLIT_SUMMARY_PATH:-${OUTPUT_DIR}/run_summary.json}"
export LATEST_SUMMARY_PATH="${LATEST_SUMMARY_PATH:-${OUTPUT_DIR}/run_summary_latest.json}"

if [[ "$MODE" == "axis-d-unsolved" ]]; then
	RESULTS_FILE="${AXIS_D_RESULTS_FILE:-${SCRIPT_DIR}/all_results_axis_D.txt}"
	if [[ ! -f "$RESULTS_FILE" ]]; then
		echo "[run] ERROR: historical results file not found: ${RESULTS_FILE}" >&2
		exit 1
	fi
	TASK_IDS="$(${PYTHON_BIN} "${SCRIPT_DIR}/task_selection.py" \
			--challenges "${SCRIPT_DIR}/dataset/arc-agi_training_challenges.json" \
			--results "$RESULTS_FILE" \
			--split training \
			--expected-excluded 140 \
			--expected-remaining 260)"
	export TASK_IDS
	echo "[run] axis-D unsolved mode: 260 validated training tasks"
else
	unset TASK_IDS
	echo "[run] full training mode: 400 tasks"
fi

exec "${SCRIPT_DIR}/run_full_split.sh" training
