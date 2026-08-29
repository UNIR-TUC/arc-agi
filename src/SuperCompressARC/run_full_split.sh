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

PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/arcagi/bin/python}"
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

# TorchInductor can create hundreds of gigabytes of compiled artifacts across a
# full campaign. Keep them on the dedicated LPX and refuse to fall through to
# the root filesystem when that mount is absent.
CACHE_MOUNT="${CACHE_MOUNT:-/mnt/supercompressarc-cache}"
CACHE_EXPECTED_UUID="${CACHE_EXPECTED_UUID:-7171729e-8a92-40d6-a172-634a85f1ce7f}"
CACHE_EXPECTED_SERIAL="${CACHE_EXPECTED_SERIAL:-A5JQB532006XN1}"
INDUCTOR_CACHE_DIR="${INDUCTOR_CACHE_DIR:-${CACHE_MOUNT}/.inductor_cache}"
CACHE_WARN_FREE_GB="${CACHE_WARN_FREE_GB:-400}"
CACHE_MIN_FREE_GB="${CACHE_MIN_FREE_GB:-250}"
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

mkdir -p run_logs
LOG="run_logs/${SPLIT}_$(date +%Y%m%d_%H%M%S).log"

echo "[run] split=${SPLIT} iterations=${ITERATIONS} max_workers=${MAX_WORKERS}"
echo "[run] python: ${PYTHON_BIN}"
echo "[run] host RAM: ${MEM_PER_WORKER_GB} GB/worker, ${MEM_RESERVE_GB} GB reserved"
echo "[run] cache limits: warn=${CACHE_WARN_FREE_GB} GB min=${CACHE_MIN_FREE_GB} GB inodes=${CACHE_MIN_FREE_INODES}"
echo "[run] log: ${LOG}"
echo "[run] interrupt at any time — relaunching resumes from .partial/${SPLIT}/"

export TORCHINDUCTOR_CACHE_DIR="$INDUCTOR_CACHE_DIR"

if (( CACHE_PREFLIGHT_ONLY )); then
  validate_cache_storage
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
  "$PYTHON_BIN" -u parallel_train.py \
      --split "${SPLIT}" \
      --iterations "${ITERATIONS}" \
      --accel-preset compile \
      --max-workers "${MAX_WORKERS}" \
      --host-mem-per-worker-gb "${MEM_PER_WORKER_GB}" \
      --host-mem-reserve-gb "${MEM_RESERVE_GB}" \
      --task-stall-timeout "${STALL_TIMEOUT_S}" \
      --inductor-cache-dir "${INDUCTOR_CACHE_DIR}" \
      --cache-min-free-gb "${CACHE_MIN_FREE_GB}" \
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
