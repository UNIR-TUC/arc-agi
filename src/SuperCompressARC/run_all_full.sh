#!/usr/bin/env bash
#
# Run training, evaluation and test as one timed, crash-resumable campaign.
# Each split retries independently; the campaign stops if one exhausts its
# attempts or the operator interrupts it.
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/arcagi/bin/python}"
CAMPAIGN_ID="${CAMPAIGN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
CAMPAIGN_STARTED_EPOCH="$(date +%s.%N)"
CAMPAIGN_ARCHIVE="run_logs/campaign_${CAMPAIGN_ID}_summary.json"
CAMPAIGN_STATUS="failed"
SPLIT_SUMMARIES=()

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[run] ERROR: Python interpreter is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
mkdir -p run_logs

finalize_campaign() {
  local original_rc="$1" ended_at path
  local summary_args=()
  trap - EXIT
  ended_at="$(date +%s.%N)"
  for path in "${SPLIT_SUMMARIES[@]}"; do
    if [[ -f "$path" ]]; then
      summary_args+=(--split-summary "$path")
    fi
  done
  if ! "$PYTHON_BIN" run_tracking.py campaign-summary \
      "${summary_args[@]}" \
      --output campaign_summary.json \
      --archive "$CAMPAIGN_ARCHIVE" \
      --timing-result timing_result.txt \
      --campaign-id "$CAMPAIGN_ID" \
      --started-at "$CAMPAIGN_STARTED_EPOCH" \
      --ended-at "$ended_at" \
      --status "$CAMPAIGN_STATUS"; then
    echo "[run] ERROR: could not write campaign timing summary" >&2
    if (( original_rc == 0 )); then
      original_rc=1
    fi
  fi
  exit "$original_rc"
}

handle_signal() {
  CAMPAIGN_STATUS="interrupted"
  exit "$1"
}

trap 'finalize_campaign "$?"' EXIT
trap 'handle_signal 130' INT
trap 'handle_signal 143' TERM

echo "[run] campaign=${CAMPAIGN_ID} splits=training,evaluation,test"

for split in training evaluation test; do
  split_summary="run_logs/${split}_${CAMPAIGN_ID}_summary.json"
  SPLIT_SUMMARIES+=("$split_summary")
  echo "[run] ===== campaign ${CAMPAIGN_ID}: ${split} ====="
  RUN_ID="${CAMPAIGN_ID}_${split}" \
  CAMPAIGN_ID="$CAMPAIGN_ID" \
  SPLIT_SUMMARY_PATH="$split_summary" \
  PYTHON_BIN="$PYTHON_BIN" \
    "$SCRIPT_DIR/run_full_split.sh" "$split"
  rc=$?
  if (( rc != 0 )); then
    if (( rc == 130 || rc == 143 )); then
      CAMPAIGN_STATUS="interrupted"
    fi
    echo "[run] campaign stopped in ${split} (exit=${rc})" >&2
    exit "$rc"
  fi
done

CAMPAIGN_STATUS="success"
echo "[run] all splits complete; campaign summary: campaign_summary.json"
exit 0