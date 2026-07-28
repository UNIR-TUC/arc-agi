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

purge_caches() {
  echo "[night-run] Purging Phase-1 measurement cache and Inductor cache"
  rm -f memory_cache_training.json
  rm -rf .inductor_cache
}

run_profile() {
  local label="$1"; shift
  echo "[night-run] ===== ${label} :: $* ====="
  python profile_parallel_train.py --label "$label" "$@"
  local rc=$?
  echo "[night-run] ${label} exit=${rc}"
  return $rc
}

# Start from a fully cold state so the compile cost is measured, not hidden.
# From here on the Phase-1 cache is NOT purged between runs: Phase 1 is always
# executed eagerly (accel.for_measurement), so the same measurement is valid for
# the eager and the compiled presets and is reused instead of re-paid.
purge_caches

failures=()
attempt() { run_profile "$@" || failures+=("$1"); }

# ── A) Isolate compilation from the concurrency confound ────────────────────
# The previous comparison changed both at once (compile lowered concurrency
# 10 -> 3 through VRAM), which made a per-task speedup look like a regression.
attempt e_base_mw4   --accel-preset baseline -- --split training --demo 10 --iterations 300 --max-workers 4
attempt e_base_mw10  --accel-preset baseline -- --split training --demo 10 --iterations 300 --max-workers 10

# ── B) Compile, cold vs warm Inductor cache (same command twice) ────────────
# Cold pays ~1200 s/task of Inductor work; warm ~250 s. This quantifies how much
# of the cost is recoverable by keeping .inductor_cache around.
attempt e_comp_cold  --accel-preset compile  -- --split training --demo 10 --iterations 300
attempt e_comp_warm  --accel-preset compile  -- --split training --demo 10 --iterations 300

# ── C) Can the compiled VRAM footprint be brought down? ─────────────────────
# Compilation raised per-task VRAM 0.91 -> 3.87 GB, which is what capped
# concurrency at 3. If memory planning / no expandable_segments recovers it,
# concurrency goes back up and the per-iteration gain becomes an aggregate gain.
attempt e_comp_noexp --accel-preset compile  -- --split training --demo 10 --iterations 300 --alloc-conf ""
attempt e_comp_nomp  --accel-preset compile  -- --split training --demo 10 --iterations 300 --no-memory-planning

# ── D) Amortisation: the regime where compilation should actually pay off ───
# Break-even was estimated at ~8800 iterations/task with a cold cache; at 1500
# (the real setting) with a warm cache it should win clearly.
attempt e_base_1500  --accel-preset baseline -- --split training --demo 10 --iterations 1500
attempt e_comp_1500  --accel-preset compile  -- --split training --demo 10 --iterations 1500

# ── E) HIP graphs: the right tool for a launch-bound workload ───────────────
attempt e_graphs     --accel-preset compile  -- --split training --demo 10 --iterations 1500 \
                        --compile reduce-overhead

# ── F) Control: confirm BF16 is not worth keeping ───────────────────────────
attempt e_bf16_1500  --accel-preset bf16     -- --split training --demo 10 --iterations 1500

if [[ ${#failures[@]} -eq 0 ]]; then
  echo "[night-run] All runs completed successfully"
else
  echo "[night-run] Runs with non-zero exit: ${failures[*]}"
fi

cat <<'EOF'
[night-run] Suggested comparisons once the machine is back up:
  python profile_parallel_train.py --compare .profile/e_base_mw4_summary.json  .profile/e_comp_cold_summary.json
  python profile_parallel_train.py --compare .profile/e_comp_cold_summary.json .profile/e_comp_warm_summary.json
  python profile_parallel_train.py --compare .profile/e_comp_warm_summary.json .profile/e_comp_noexp_summary.json
  python profile_parallel_train.py --compare .profile/e_base_1500_summary.json .profile/e_comp_1500_summary.json
  python profile_parallel_train.py --compare .profile/e_comp_1500_summary.json .profile/e_graphs_summary.json
  python profile_parallel_train.py --compare .profile/e_base_1500_summary.json .profile/e_bf16_1500_summary.json
Read `steps_per_s_phase2` as the primary metric; `steps_per_s_aggregate` still
includes the Phase-1 measurement.
EOF

echo "[night-run] All scheduled runs finished. Shutting down now..."

# Non-interactive shutdown should succeed because sudo timestamp was pre-validated
# and refreshed in the keepalive loop.
sudo -n shutdown -h now
