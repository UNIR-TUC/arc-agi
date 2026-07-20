#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_LOG_DIR=".profile/overnight_logs"
mkdir -p "$RUN_LOG_DIR"
RUN_LOG="$RUN_LOG_DIR/eje_d_overnight_$(date +%Y%m%d_%H%M%S).log"

SHUTDOWN_DELAY_MINUTES=2
KEEP_SUDO_ALIVE_PID=""

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$RUN_LOG"
}

cleanup() {
    if [[ -n "$KEEP_SUDO_ALIVE_PID" ]]; then
        kill "$KEEP_SUDO_ALIVE_PID" 2>/dev/null || true
    fi
}

cancelled() {
    log "Interrupted; shutdown will NOT be scheduled."
    cleanup
    exit 130
}

trap cancelled INT TERM
trap cleanup EXIT

schedule_shutdown() {
    log "Scheduling safe shutdown in ${SHUTDOWN_DELAY_MINUTES} minutes."
    if [[ "${EUID}" -eq 0 ]]; then
        shutdown -h "+${SHUTDOWN_DELAY_MINUTES}" "Eje D overnight profiling complete"
    else
        sudo shutdown -h "+${SHUTDOWN_DELAY_MINUTES}" "Eje D overnight profiling complete"
    fi
}

require_shutdown_privileges() {
    if [[ "${EUID}" -eq 0 ]]; then
        return 0
    fi

    log "Requesting sudo credentials now so shutdown can run unattended later."
    sudo -v || {
        log "Could not obtain sudo credentials; aborting before starting overnight run."
        exit 1
    }

    while true; do
        sudo -n true 2>/dev/null || exit
        sleep 60
    done &
    KEEP_SUDO_ALIVE_PID="$!"
}

run_profile() {
    local label="$1"
    shift

    log "Starting profile: ${label}"
    "$@" 2>&1 | tee -a "$RUN_LOG"
    local status=${PIPESTATUS[0]}

    if [[ "$status" -eq 0 ]]; then
        log "Profile ${label} completed successfully."
    else
        log "Profile ${label} failed with exit code ${status}. Continuing to shutdown scheduling."
    fi

    return "$status"
}

main() {
    log "Eje D overnight profiling started."
    log "Repository: ${SCRIPT_DIR}"
    log "Run log: ${RUN_LOG}"

    require_shutdown_privileges

    if [[ -f "arcagi/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source "arcagi/bin/activate"
        log "Activated virtualenv: arcagi"
    else
        log "Virtualenv arcagi/bin/activate not found; using current Python environment."
    fi

    local baseline_status=0
    local bf16_status=0

    run_profile "d_baseline" \
        python profile_parallel_train.py --label d_baseline --gpu-mode sysfs -- \
            --split training --demo 10 \
            --mixed-precision off --compile-forward off \
            --float32-matmul-precision highest
    baseline_status=$?

    run_profile "d_bf16" \
        python profile_parallel_train.py --label d_bf16 --gpu-mode sysfs -- \
            --split training --demo 10 \
            --mixed-precision bf16 --compile-forward off \
            --float32-matmul-precision high
    bf16_status=$?

    log "Profiles finished. baseline_exit=${baseline_status}, bf16_exit=${bf16_status}"
    log "Summary files expected: .profile/d_baseline_summary.json and .profile/d_bf16_summary.json"
    log "To cancel the scheduled shutdown after this script finishes, run: sudo shutdown -c"

    schedule_shutdown
}

main "$@"