#!/usr/bin/env bash
#
# Shared runner for one complete ARC-AGI split. All split wrappers invoke this
# file so recovery, resource limits and timing never drift apart.
#
# Every Python failure is retried with --resume. Finished tasks are durable;
# tasks active at failure restart from iteration zero.
#
# Settings come from the Eje D campaign (Docs/Architecture/EJE_D_IMPLEMENTACION.md).
# Override any of them from the environment, e.g.  MAX_WORKERS=8 ./run_training_full.sh
# Task selection and recovery controls: TASK_IDS, OUTPUT_DIR, STATE_DIR,
# ACCEL_PRESET, EJE_B, SEEDS, RECOVER_TASK_FAILURES, EAGER_TASKS and
# RETRY_QUARANTINED_TASKS.
#
set -uo pipefail

SPLIT="${1:?usage: run_full_split.sh <training|evaluation|test>}"
case "$SPLIT" in
  training|evaluation|test) ;;
  *)
    echo "usage: run_full_split.sh <training|evaluation|test>" >&2
    exit 2
    ;;
esac
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/arcagi/bin/python}"
ITERATIONS="${ITERATIONS:-2000}"
POSTPROCESS_STRIDE="${POSTPROCESS_STRIDE:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-.}"
STATE_DIR="${STATE_DIR:-}"
TASK_IDS="${TASK_IDS:-}"
EJE_B="${EJE_B:-0}"
SEEDS="${SEEDS:-}"

# 6, not 10: the concurrency sweep showed the GPU saturates here. Going to 10
# bought +3.9 % throughput — inside the +4-5 % that the Inductor cache gains on
# its own per repeat — while costing +77 % CPU and +30 % host RAM. At 10 workers
# a 50-task run already peaked at 89.5 % of the 96 GB host.
MAX_WORKERS="${MAX_WORKERS:-8}"

# 50 tasks at 10 workers used ~8.4 GB of real RAM each; the full split has larger
# puzzles and schedules the biggest first, so budget 9 and keep 12 GB for Ubuntu.
MEM_PER_WORKER_GB="${MEM_PER_WORKER_GB:-11}"
MEM_RESERVE_GB="${MEM_RESERVE_GB:-14}"

# A wedged worker would otherwise hold the whole campaign forever. Workers report
# every 10 steps, so this is ~2 orders of magnitude above a healthy interval.
STALL_TIMEOUT_S="${STALL_TIMEOUT_S:-2700}"

ACCEL_PRESET="${ACCEL_PRESET:-compile}"
RECOVER_TASK_FAILURES="${RECOVER_TASK_FAILURES:-1}"
EAGER_TASKS="${EAGER_TASKS:-}"
RETRY_QUARANTINED_TASKS="${RETRY_QUARANTINED_TASKS:-}"

if [[ "$EJE_B" != "0" && "$EJE_B" != "1" ]]; then
  echo "[run] ERROR: EJE_B must be 0 or 1" >&2
  exit 2
fi

MAX_ATTEMPTS="${MAX_ATTEMPTS:-5}"
RETRY_DELAY_S="${RETRY_DELAY_S:-30}"

# TorchInductor can create hundreds of gigabytes of compiled artifacts across a
# full campaign. Keep them on the dedicated LPX and refuse to fall through to
# the root filesystem when that mount is absent.
CACHE_MOUNT="${CACHE_MOUNT:-/mnt/supercompressarc-cache}"
CACHE_EXPECTED_UUID="${CACHE_EXPECTED_UUID:-7171729e-8a92-40d6-a172-634a85f1ce7f}"
CACHE_EXPECTED_SERIAL="${CACHE_EXPECTED_SERIAL:-A5JQB532006XN1}"
INDUCTOR_CACHE_DIR="${INDUCTOR_CACHE_DIR:-${CACHE_MOUNT}/.inductor_cache}"
CACHE_WARN_FREE_GB="${CACHE_WARN_FREE_GB:-400}"
CACHE_MIN_FREE_GB="${CACHE_MIN_FREE_GB:-50}"
CACHE_MIN_FREE_INODES="${CACHE_MIN_FREE_INODES:-1000000}"
CACHE_PREFLIGHT_ONLY="${CACHE_PREFLIGHT_ONLY:-0}"

validate_cache_storage() {
  local mount_target source fstype options uuid parent_device serial free_gb free_inodes

  if [[ ! -d "$CACHE_MOUNT" ]]; then
    echo "[run] ERROR: cache mount point does not exist: ${CACHE_MOUNT}" >&2
    return 1
  fi

  mount_target="$(findmnt -n -T "$CACHE_MOUNT" -o TARGET 2>/dev/null)"
  source="$(findmnt -n -T "$CACHE_MOUNT" -o SOURCE 2>/dev/null)"
  fstype="$(findmnt -n -T "$CACHE_MOUNT" -o FSTYPE 2>/dev/null)"
  options="$(findmnt -n -T "$CACHE_MOUNT" -o OPTIONS 2>/dev/null)"
  uuid="$(findmnt -n -T "$CACHE_MOUNT" -o UUID 2>/dev/null)"

  if [[ "$mount_target" != "$CACHE_MOUNT" ]]; then
    echo "[run] ERROR: ${CACHE_MOUNT} is not a mount point; refusing to use ${mount_target:-unknown}" >&2
    return 1
  fi
  if [[ "$uuid" != "$CACHE_EXPECTED_UUID" ]]; then
    echo "[run] ERROR: cache UUID is ${uuid:-unknown}, expected ${CACHE_EXPECTED_UUID}" >&2
    return 1
  fi
  parent_device="$(lsblk -ndo PKNAME "$source" 2>/dev/null)"
  serial="$(lsblk -ndo SERIAL "/dev/${parent_device}" 2>/dev/null | xargs)"
  if [[ "$serial" != "$CACHE_EXPECTED_SERIAL" ]]; then
    echo "[run] ERROR: cache device serial is ${serial:-unknown}, expected ${CACHE_EXPECTED_SERIAL}" >&2
    return 1
  fi
  if [[ "$fstype" != "ext4" ]]; then
    echo "[run] ERROR: cache filesystem is ${fstype:-unknown}, expected ext4" >&2
    return 1
  fi
  if [[ ",${options}," == *,noexec,* ]]; then
    echo "[run] ERROR: cache filesystem is mounted noexec; Inductor must load compiled objects" >&2
    return 1
  fi

  mkdir -p "$INDUCTOR_CACHE_DIR" || return 1
  if [[ ! -w "$INDUCTOR_CACHE_DIR" || ! -x "$INDUCTOR_CACHE_DIR" ]]; then
    echo "[run] ERROR: cache directory is not writable/searchable: ${INDUCTOR_CACHE_DIR}" >&2
    return 1
  fi

  free_gb="$(df -BG --output=avail "$INDUCTOR_CACHE_DIR" | tail -n 1 | tr -dc '0-9')"
  free_inodes="$(df --output=iavail "$INDUCTOR_CACHE_DIR" | tail -n 1 | tr -dc '0-9')"
  if [[ -z "$free_gb" || -z "$free_inodes" ]]; then
    echo "[run] ERROR: could not determine cache filesystem capacity" >&2
    return 1
  fi
  if (( free_gb < CACHE_MIN_FREE_GB )); then
    echo "[run] ERROR: cache filesystem has ${free_gb} GB free; minimum is ${CACHE_MIN_FREE_GB} GB" >&2
    return 1
  fi
  if (( free_inodes < CACHE_MIN_FREE_INODES )); then
    echo "[run] ERROR: cache filesystem has ${free_inodes} free inodes; minimum is ${CACHE_MIN_FREE_INODES}" >&2
    return 1
  fi
  if (( free_gb < CACHE_WARN_FREE_GB )); then
    echo "[run] WARNING: cache filesystem has only ${free_gb} GB free"
  fi

  echo "[run] cache: ${INDUCTOR_CACHE_DIR} on ${source} (${serial}, ${uuid}, ${free_gb} GB free)"
}

if [[ ! -f "dataset/arc-agi_${SPLIT}_challenges.json" ]]; then
  echo "[run] ERROR: dataset/arc-agi_${SPLIT}_challenges.json not found" >&2
  exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[run] ERROR: Python interpreter is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "run_tracking.py" ]]; then
  echo "[run] ERROR: run_tracking.py not found" >&2
  exit 1
fi
if [[ ! "$MAX_ATTEMPTS" =~ ^[1-9][0-9]*$ ]]; then
  echo "[run] ERROR: MAX_ATTEMPTS must be a positive integer" >&2
  exit 1
fi
if [[ "$RECOVER_TASK_FAILURES" != "0" && "$RECOVER_TASK_FAILURES" != "1" ]]; then
  echo "[run] ERROR: RECOVER_TASK_FAILURES must be 0 or 1" >&2
  exit 1
fi

mkdir -p run_logs
if ! mkdir -p "$OUTPUT_DIR"; then
  echo "[run] ERROR: could not create output directory: ${OUTPUT_DIR}" >&2
  exit 1
fi
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
CAMPAIGN_ID="${CAMPAIGN_ID:-$RUN_ID}"
RUN_STARTED_EPOCH="$(date +%s.%N)"
LOG="${RUN_LOG_PATH:-run_logs/${SPLIT}_${RUN_ID}.log}"
ATTEMPTS_FILE="${ATTEMPTS_FILE:-run_logs/${SPLIT}_${RUN_ID}_attempts.jsonl}"
SUMMARY_FILE="${SPLIT_SUMMARY_PATH:-run_logs/${SPLIT}_${RUN_ID}_summary.json}"
LATEST_SUMMARY="${LATEST_SUMMARY_PATH:-run_summary_${SPLIT}.json}"
METADATA_FILE="${METADATA_PATH:-${OUTPUT_DIR}/run_metadata_${SPLIT}.json}"
FINAL_STATUS="failed"

finalize_tracking() {
  local original_rc="$1" ended_at
  trap - EXIT
  ended_at="$(date +%s.%N)"
  if ! "$PYTHON_BIN" run_tracking.py split-summary \
      --events "$ATTEMPTS_FILE" \
      --summary "$SUMMARY_FILE" \
      --latest-summary "$LATEST_SUMMARY" \
      --metadata "$METADATA_FILE" \
      --run-id "$RUN_ID" \
      --campaign-id "$CAMPAIGN_ID" \
      --split "$SPLIT" \
      --started-at "$RUN_STARTED_EPOCH" \
      --ended-at "$ended_at" \
      --status "$FINAL_STATUS" \
      --log-path "$LOG"; then
    echo "[run] ERROR: could not write run timing summary" >&2
    if (( original_rc == 0 )); then
      original_rc=1
    fi
  fi
  exit "$original_rc"
}

handle_signal() {
  FINAL_STATUS="interrupted"
  exit "$1"
}

trap 'finalize_tracking "$?"' EXIT
trap 'handle_signal 130' INT
trap 'handle_signal 143' TERM

echo "[run] split=${SPLIT} iterations=${ITERATIONS} max_workers=${MAX_WORKERS}"
echo "[run] run_id=${RUN_ID} campaign_id=${CAMPAIGN_ID}"
echo "[run] python: ${PYTHON_BIN}"
echo "[run] acceleration: ${ACCEL_PRESET}; task recovery=${RECOVER_TASK_FAILURES}"
echo "[run] eje_b=${EJE_B}; seeds=${SEEDS:-default}; postprocess_stride=${POSTPROCESS_STRIDE}"
echo "[run] output_dir=${OUTPUT_DIR}; state_dir=${STATE_DIR:-.partial}"
if [[ -n "$TASK_IDS" ]]; then
  echo "[run] task selection: explicit task IDs"
fi
if [[ -n "$EAGER_TASKS" ]]; then
  echo "[run] eager tasks: ${EAGER_TASKS}"
fi
echo "[run] host RAM: ${MEM_PER_WORKER_GB} GB/worker, ${MEM_RESERVE_GB} GB reserved"
echo "[run] cache limits: warn=${CACHE_WARN_FREE_GB} GB min=${CACHE_MIN_FREE_GB} GB inodes=${CACHE_MIN_FREE_INODES}"
echo "[run] log: ${LOG}"
echo "[run] interrupt at any time — relaunching resumes from ${STATE_DIR:-.partial}/${SPLIT}/"

export TORCHINDUCTOR_CACHE_DIR="$INDUCTOR_CACHE_DIR"

if (( CACHE_PREFLIGHT_ONLY )); then
  if ! validate_cache_storage; then
    exit 1
  fi
  FINAL_STATUS="preflight"
  echo "[run] cache preflight passed; no training started"
  exit 0
fi

attempt=1
while (( attempt <= MAX_ATTEMPTS )); do
  if ! validate_cache_storage; then
    echo "[run] cache storage validation failed; not retrying" >&2
    exit 1
  fi
  echo "[run] ===== ${SPLIT}: attempt ${attempt}/${MAX_ATTEMPTS} at $(date '+%F %T') ====="
  attempt_started_epoch="$(date +%s.%N)"
  if ! "$PYTHON_BIN" run_tracking.py attempt-start \
      --events "$ATTEMPTS_FILE" \
      --run-id "$RUN_ID" \
      --campaign-id "$CAMPAIGN_ID" \
      --split "$SPLIT" \
      --attempt "$attempt" \
      --at "$attempt_started_epoch" \
      --log-path "$LOG"; then
    echo "[run] ERROR: could not persist attempt start" >&2
    exit 1
  fi
  # Keep stdout on the terminal: piping it makes the dashboard's isatty() check fail.
  # Mirror stderr for tracebacks while ArcLogger keeps the full structured .log file.
  python_args=(
    --split "${SPLIT}"
    --iterations "${ITERATIONS}"
    --postprocess-stride "${POSTPROCESS_STRIDE}"
    --accel-preset "${ACCEL_PRESET}"
    --output-dir "${OUTPUT_DIR}"
    --max-workers "${MAX_WORKERS}"
    --host-mem-per-worker-gb "${MEM_PER_WORKER_GB}"
    --host-mem-reserve-gb "${MEM_RESERVE_GB}"
    --task-stall-timeout "${STALL_TIMEOUT_S}"
    --inductor-cache-dir "${INDUCTOR_CACHE_DIR}"
    --cache-min-free-gb "${CACHE_MIN_FREE_GB}"
    --resume
  )
  if [[ -n "$STATE_DIR" ]]; then
    python_args+=(--state-dir "$STATE_DIR")
  fi
  if [[ -n "$TASK_IDS" ]]; then
    python_args+=(--task-ids "$TASK_IDS")
  fi
  if (( EJE_B )); then
    python_args+=(--eje-b)
  fi
  if [[ -n "$SEEDS" ]]; then
    python_args+=(--seeds "$SEEDS")
  fi
  if (( RECOVER_TASK_FAILURES )); then
    python_args+=(--recover-task-failures)
  fi
  if [[ -n "$EAGER_TASKS" ]]; then
    python_args+=(--eager-tasks "$EAGER_TASKS")
  fi
  if [[ -n "$RETRY_QUARANTINED_TASKS" ]] && (( attempt == 1 )); then
    python_args+=(--retry-quarantined "$RETRY_QUARANTINED_TASKS")
  fi
  "$PYTHON_BIN" -u parallel_train.py "${python_args[@]}" \
      2> >(tee -a "${LOG}" >&2)
  rc=$?
  attempt_ended_epoch="$(date +%s.%N)"
  if ! "$PYTHON_BIN" run_tracking.py attempt-end \
      --events "$ATTEMPTS_FILE" \
      --run-id "$RUN_ID" \
      --campaign-id "$CAMPAIGN_ID" \
      --split "$SPLIT" \
      --attempt "$attempt" \
      --at "$attempt_ended_epoch" \
      --exit-code "$rc"; then
    echo "[run] ERROR: could not persist attempt result" >&2
    exit 1
  fi

  if (( rc == 0 )); then
    FINAL_STATUS="success"
    echo "[run] ${SPLIT} complete at $(date '+%F %T')"
    echo "[run] results: ${OUTPUT_DIR}/submission_${SPLIT}.json  ${OUTPUT_DIR}/predictions_${SPLIT}.npz"
    exit 0
  fi

  if (( rc == 130 || rc == 143 )); then
    FINAL_STATUS="interrupted"
    echo "[run] interrupted (exit=${rc}); completed task partials are preserved" | tee -a "${LOG}"
    exit "$rc"
  fi

  if (( rc == 137 )); then
    echo "[run] exit=137 (SIGKILL); no Python traceback is possible. This may be the OS OOM killer or another external kill." | tee -a "${LOG}"
  else
    echo "[run] exit=${rc}; Python and worker diagnostics are above" | tee -a "${LOG}"
  fi

  if (( attempt >= MAX_ATTEMPTS )); then
    break
  fi
  echo "[run] retrying in ${RETRY_DELAY_S}s — --resume skips every completed task" | tee -a "${LOG}"
  attempt=$(( attempt + 1 ))
  sleep "$RETRY_DELAY_S"
done

echo "[run] ${SPLIT} still failing after ${MAX_ATTEMPTS} attempts — see ${LOG}" >&2
echo "[run] partial results are preserved in ${STATE_DIR:-.partial}/${SPLIT}/" >&2
exit 1
