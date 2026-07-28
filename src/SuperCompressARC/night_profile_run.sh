#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Obtain sudo credentials up front so the final shutdown can run unattended.
echo "[night-run] Requesting sudo credentials for final shutdown..."
if ! sudo -v; then
  echo "[night-run] ERROR: sudo authentication failed."
  exit 1
fi

# Keep the sudo timestamp fresh during the long training run.
(
  while true; do
    sudo -n true || exit 0
    sleep 60
  done
) &
SUDO_KEEPALIVE_PID=$!

cleanup() {
  kill "$SUDO_KEEPALIVE_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

remove_cache_if_exists() {
  if [[ -f memory_cache_training.json ]]; then
    echo "[night-run] Removing memory_cache_training.json"
    rm -f memory_cache_training.json
  else
    echo "[night-run] memory_cache_training.json not found; nothing to remove"
  fi
}

run_profile() {
  local label="$1"
  local preset="$2"

  echo "[night-run] Starting run: ${label} (${preset})"
  python profile_parallel_train.py \
    --label "$label" \
    --accel-preset "$preset" \
    -- --split training --demo 10 --iterations 300
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[night-run] Run ${label} failed with exit code ${rc}"
  else
    echo "[night-run] Run ${label} completed successfully"
  fi
  return $rc
}

status_baseline=0
status_bf16=0
status_full=0

run_profile "d_baseline" "baseline" || status_baseline=$?
remove_cache_if_exists

run_profile "d_bf16" "bf16" || status_bf16=$?
remove_cache_if_exists

run_profile "d_full" "full" || status_full=$?

echo "[night-run] Run summary: baseline=${status_baseline}, bf16=${status_bf16}, full=${status_full}"
echo "[night-run] All scheduled runs finished. Shutting down now..."

# Non-interactive shutdown should succeed because sudo timestamp was pre-validated
# and refreshed in the keepalive loop.
sudo -n shutdown -h now
