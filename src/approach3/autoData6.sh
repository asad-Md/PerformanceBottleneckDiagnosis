#!/usr/bin/env bash

set -e

# ===== CONFIG =====
CURRENT_LABEL=""
ML_PID=""
STRESS_PID=""
PINNER_PID=""
INTERRUPTED=false

DURATION=1500   # 25 minutes per session

PINNER_CMD="sudo ./pinner --obj perf_monitor.bpf.o"
READER_CMD="sudo ./reader"

# label|ml command|stress-ng command
SESSIONS=(
"stream_memory_contention|python3 ml_stress.py --mode cnn --batch-size 128 --workers 2|stress-ng --cpu 0 --stream 0 --vm 1 --vm-bytes 400M"

"cache_contention|python3 ml_stress.py --mode cnn --batch-size 128 --workers 2|stress-ng --cpu 0 --cache 0 --stream 0 --vm 1 --vm-bytes 400M"

"context_switch_contention|python3 ml_stress.py --mode cnn --batch-size 128 --workers 2|stress-ng --cpu 0 --stream 0 --switch 4"
)

# ===== SUDO KEEP-ALIVE =====
echo "[+] Requesting sudo access..."
sudo -v

(
while true; do
    sudo -v
    sleep 60
done
) &
SUDO_KEEPALIVE_PID=$!

# ===== CTRL+C HANDLER =====
cleanup() {
    echo
    echo "[!] Ctrl+C detected — finishing current session safely..."

    INTERRUPTED=true

    # Stop ML
    if [ -n "$ML_PID" ]; then
        echo "[!] Stopping ML workload..."
        kill -INT "$ML_PID" 2>/dev/null || true
        pkill -INT -P "$ML_PID" 2>/dev/null || true
        wait "$ML_PID" 2>/dev/null || true
        ML_PID=""
    fi

    # Stop stress-ng
    if [ -n "$STRESS_PID" ]; then
        echo "[!] Stopping stress-ng..."
        kill -INT "$STRESS_PID" 2>/dev/null || true
        wait "$STRESS_PID" 2>/dev/null || true
        STRESS_PID=""
    fi

    # Stop pinner
    if [ -n "$PINNER_PID" ]; then
        echo "[!] Stopping pinner..."
        sudo kill -INT "$PINNER_PID" 2>/dev/null || true
        wait "$PINNER_PID" 2>/dev/null || true
        PINNER_PID=""
    fi

    # Save current session
    if [ -n "$CURRENT_LABEL" ]; then
        echo "[!] Saving data for session: $CURRENT_LABEL"
        $READER_CMD --label "$CURRENT_LABEL" --append
    fi

    kill "$SUDO_KEEPALIVE_PID" 2>/dev/null || true

    echo "[!] Exit complete"
    exit 0
}

trap cleanup INT TERM
trap 'kill "$SUDO_KEEPALIVE_PID" 2>/dev/null || true' EXIT

# ===== HELPERS =====

start_pinner() {
    echo "[+] Starting pinner..."
    $PINNER_CMD &
    PINNER_PID=$!
    sleep 1
}

stop_pinner() {
    echo "[+] Stopping pinner..."
    sudo kill -INT "$PINNER_PID"
    wait "$PINNER_PID" 2>/dev/null || true
    PINNER_PID=""
}

run_reader() {
    local label=$1
    echo "[+] Running reader for label: $label"
    $READER_CMD --label "$label" --append
}

# ===== MAIN LOOP =====

while true; do

    for entry in "${SESSIONS[@]}"; do

        IFS="|" read -r label ml_cmd stress_cmd <<< "$entry"

        CURRENT_LABEL="$label"

        echo "======================================="
        echo "[SESSION] $label"
        echo "======================================="

        # Start ML first
        echo "[+] Starting ML workload..."
        timeout ${DURATION}s bash -c "$ml_cmd" &
        ML_PID=$!

        # Give ML time to initialize
        echo "[+] Waiting 10 seconds before starting stress-ng..."
        sleep 10

        # Start stress-ng
        echo "[+] Starting stress-ng..."
        timeout ${DURATION}s bash -c "$stress_cmd" &
        STRESS_PID=$!

        # Start pinner immediately after stress-ng
        sleep 5
        start_pinner

        # Wait for workloads
        wait "$ML_PID" 2>/dev/null || true
        ML_PID=""

        wait "$STRESS_PID" 2>/dev/null || true
        STRESS_PID=""

        # Stop pinner
        stop_pinner
        sleep 1

        # Save data
        run_reader "$label"
        CURRENT_LABEL=""

        echo "[✓] Completed: $label"
        echo

        if [ "$INTERRUPTED" = true ]; then
            echo "[!] Stopping further sessions"
            break 2
        fi

        sleep 5

    done

done

kill "$SUDO_KEEPALIVE_PID" 2>/dev/null || true

echo "All sessions complete!"