#!/usr/bin/env bash
#
# Shared runner for a complete ARC-AGI split. Invoked by run_training_full.sh
# and run_evaluation_full.sh — keeping one copy means the two splits can never
# drift into non-comparable settings.
#
# No profiling, no telemetry: parallel_train.py's own .log/ and
# run_metadata_{split}.json are all that gets written.
#
# Settings come from the Eje D campaign (Docs/Architecture/EJE_D_IMPLEMENTACION.md).
# Override any of them from the environment, e.g.  MAX_WORKERS=8 ./run_training_full.sh
#
set -uo pipefail

SPLIT="${1:?usage: run_full_split.sh <training|evaluation>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ITERATIONS="${ITERATIONS:-1500}"

# 6, not 10: the concurrency sweep showed the GPU saturates here. Going to 10
# bought +3.9 % throughput — inside the +4-5 % that the Inductor cache gains on
# its own per repeat — while costing +77 % CPU and +30 % host RAM. At 10 workers
# a 50-task run already peaked at 89.5 % of the 96 GB host.
MAX_WORKERS="${MAX_WORKERS:-6}"

# 50 tasks at 10 workers used ~8.4 GB of real RAM each; the full split has larger
# puzzles and schedules the biggest first, so budget 9 and keep 12 GB for Ubuntu.
MEM_PER_WORKER_GB="${MEM_PER_WORKER_GB:-9}"
MEM_RESERVE_GB="${MEM_RESERVE_GB:-12}"

# A wedged worker would otherwise hold the whole campaign forever. Workers report
# every 10 steps, so this is ~2 orders of magnitude above a healthy interval.
STALL_TIMEOUT_S="${STALL_TIMEOUT_S:-2700}"

MAX_ATTEMPTS="${MAX_ATTEMPTS:-5}"

if [[ ! -f "dataset/arc-agi_${SPLIT}_challenges.json" ]]; then
  echo "[run] ERROR: dataset/arc-agi_${SPLIT}_challenges.json not found" >&2
  exit 1
fi

mkdir -p run_logs
LOG="run_logs/${SPLIT}_$(date +%Y%m%d_%H%M%S).log"

echo "[run] split=${SPLIT} iterations=${ITERATIONS} max_workers=${MAX_WORKERS}"
echo "[run] host RAM: ${MEM_PER_WORKER_GB} GB/worker, ${MEM_RESERVE_GB} GB reserved"
echo "[run] log: ${LOG}"
echo "[run] interrupt at any time — relaunching resumes from .partial/${SPLIT}/"

attempt=1
while (( attempt <= MAX_ATTEMPTS )); do
  echo "[run] ===== ${SPLIT}: attempt ${attempt}/${MAX_ATTEMPTS} at $(date '+%F %T') ====="
  python -u parallel_train.py \
      --split "${SPLIT}" \
      --iterations "${ITERATIONS}" \
      --accel-preset compile \
      --max-workers "${MAX_WORKERS}" \
      --host-mem-per-worker-gb "${MEM_PER_WORKER_GB}" \
      --host-mem-reserve-gb "${MEM_RESERVE_GB}" \
      --task-stall-timeout "${STALL_TIMEOUT_S}" \
      --resume 2>&1 | tee -a "${LOG}"
  rc=${PIPESTATUS[0]}

  if (( rc == 0 )); then
    echo "[run] ${SPLIT} complete at $(date '+%F %T')"
    echo "[run] results: submission_${SPLIT}.json  predictions_${SPLIT}.npz"
    exit 0
  fi

  echo "[run] exit=${rc}; retrying — --resume skips everything already solved" | tee -a "${LOG}"
  attempt=$(( attempt + 1 ))
  sleep 30
done

echo "[run] ${SPLIT} still failing after ${MAX_ATTEMPTS} attempts — see ${LOG}" >&2
echo "[run] partial results are preserved in .partial/${SPLIT}/" >&2
exit 1
