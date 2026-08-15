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

mkdir -p .profile/logs

run_profile() {
  local label="$1"; shift
  echo "[night-run] ===== ${label} :: $* ====="
  # stdout is teed because `[accel][<task>] compile times:` is printed by the
  # worker and never reaches the .log/ files or the summary JSON.
  python profile_parallel_train.py --label "$label" "$@" \
    2>&1 | tee ".profile/logs/${label}.log"
  local rc=${PIPESTATUS[0]}
  echo "[night-run] ${label} exit=${rc}"
  return $rc
}

failures=()
attempt() { run_profile "$@" || failures+=("$1"); }

# Discard the previous run's partial results: --resume exists to recover an
# interrupted 400-task campaign, not to shortcut an A/B measurement.
reset_partials() { rm -rf .partial; }

# Start from a fully cold state so the compile cost is measured, not hidden.
# From here on the Phase-1 cache is NOT purged between runs: Phase 1 is always
# executed eagerly (accel.for_measurement), so the same measurement is valid for
# the eager and the compiled presets and is reused instead of re-paid.
echo "[night-run] Purging Phase-1 measurement cache and Inductor cache"
rm -f memory_cache_training.json
rm -rf .inductor_cache
reset_partials

# ── G) Compiled concurrency sweep — the question that sizes the full run ────
# The previous campaign ran every compiled preset at 3 concurrent tasks because
# --compile-memory-factor defaulted to 4.5 while compiled VRAM turned out to be
# ~0.95 GB/task, essentially the eager figure. With the factor corrected to 1.2
# the scheduler admits ~10, and this sweep says how far throughput actually
# scales before the GPU or the host saturates.
#
# The host has 96 GB shared with Ubuntu and everything else running on it, and a
# compiled worker peaks at 5-9 GB RSS: 14 of them with a cold cache would swap.
# HOSTMEM caps the sweep at whatever the free RAM really allows, so the high
# points degrade gracefully instead of taking the box down overnight.
HOSTMEM="--host-mem-per-worker-gb 6 --host-mem-reserve-gb 10"

# The first run also pays the cold Inductor cache; the rest reuse it.
attempt g_comp_mw03 --accel-preset compile -- --split training --demo 10 --iterations 1500 --max-workers 3 $HOSTMEM
reset_partials
attempt g_comp_mw06 --accel-preset compile -- --split training --demo 10 --iterations 1500 --max-workers 6 $HOSTMEM
reset_partials
attempt g_comp_mw10 --accel-preset compile -- --split training --demo 10 --iterations 1500 --max-workers 10 $HOSTMEM
reset_partials
attempt g_comp_mw14 --accel-preset compile -- --split training --demo 10 --iterations 1500 --max-workers 14 $HOSTMEM
reset_partials

# ── H) Do HIP graphs still destroy pass@2 once the outputs are cloned? ──────
# reduce-overhead was the fastest mode of the whole campaign (7.34 steps/s) and
# solved 0/10 against 5/10 compiled. accel now clones the forward outputs out of
# the cudagraph static pool. If this still solves ~0, the culprit is the RNG
# captured inside the graph and reduce-overhead has to be abandoned.
attempt h_graphs_clone --accel-preset compile -- --split training --demo 10 --iterations 1500 \
                          --max-workers 10 --compile reduce-overhead $HOSTMEM
reset_partials

# ── I) Inductor cache warming curve, unconfounded ───────────────────────────
# Block C of the previous campaign attributed a 1989 s -> 1050 s drop to
# --alloc-conf "", but that flag is a no-op on the `compile` preset: what it
# really measured was a third consecutive warm-cache pass. The same command
# twice more, with nothing else changed, isolates the curve.
attempt i_warm_2 --accel-preset compile -- --split training --demo 10 --iterations 1500 --max-workers 10 $HOSTMEM
reset_partials
attempt i_warm_3 --accel-preset compile -- --split training --demo 10 --iterations 1500 --max-workers 10 $HOSTMEM
reset_partials

# ── J) Are the first 10 tasks representative of a 400-task split? ───────────
# Every projection so far extrapolates from tasks 1-10. Larger grids cost more
# VRAM and more time per iteration, so this calibrates the multiplier the
# full-run estimate needs.
attempt j_base_50 --accel-preset baseline -- --split training --demo 50 --iterations 1500 --max-workers 10 $HOSTMEM
reset_partials
attempt j_comp_50 --accel-preset compile  -- --split training --demo 50 --iterations 1500 --max-workers 10 $HOSTMEM
reset_partials

if [[ ${#failures[@]} -eq 0 ]]; then
  echo "[night-run] All runs completed successfully"
else
  echo "[night-run] Runs with non-zero exit: ${failures[*]}"
fi

cat <<'EOF'
[night-run] Suggested comparisons once the machine is back up:
  # G — how far does compiled throughput scale with concurrency?
  python profile_parallel_train.py --compare .profile/g_comp_mw03_summary.json .profile/g_comp_mw06_summary.json
  python profile_parallel_train.py --compare .profile/g_comp_mw06_summary.json .profile/g_comp_mw10_summary.json
  python profile_parallel_train.py --compare .profile/g_comp_mw10_summary.json .profile/g_comp_mw14_summary.json
  # H — did cloning the cudagraph outputs restore pass@2?
  python profile_parallel_train.py --compare .profile/g_comp_mw10_summary.json .profile/h_graphs_clone_summary.json
  # I — cache warming, with nothing else changing
  python profile_parallel_train.py --compare .profile/g_comp_mw10_summary.json .profile/i_warm_2_summary.json
  python profile_parallel_train.py --compare .profile/i_warm_2_summary.json  .profile/i_warm_3_summary.json
  # J — 50 tasks instead of 10
  python profile_parallel_train.py --compare .profile/j_base_50_summary.json .profile/j_comp_50_summary.json

Read `steps_per_s_phase2` as the primary metric; `steps_per_s_aggregate` still
includes the Phase-1 measurement. `mean_workers` now counts task workers only —
Inductor's compile pool is reported separately as max_helper_procs.
Per-task compile breakdowns are in .profile/logs/<label>.log.
EOF

echo "[night-run] All scheduled runs finished. Shutting down now..."

# Non-interactive shutdown should succeed because sudo timestamp was pre-validated
# and refreshed in the keepalive loop.
sudo -n shutdown -h now
