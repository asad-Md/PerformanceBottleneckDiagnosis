#!/usr/bin/env bash

set -e

# ===== CONFIG =====
CURRENT_LABEL=""
STRESS_PID=""
PINNER_PID=""
INTERRUPTED=false

DURATION=300   # seconds per session — longer than stress-ng runs since ML startup
               # (tokenizer/model/dataset download+load) eats into the window.
               # Bump this up if you want deeper stalls recorded.

PINNER_CMD="sudo ./pinner --obj perf_monitor.bpf.o"
READER_CMD="sudo ./reader"

# label|command
# Each command is run under `timeout ${DURATION}s`, same as stress-ng was.
SESSIONS=(
"llm_train_cpu_saturation|python3 ml_stress.py --mode llm --model gpt2-medium --batch-size 32 --seq-len 512 --workers 16"
"cnn_train_cpu_saturation|python3 ml_stress.py --mode cnn --batch-size 512 --workers 16"
"combined_llm_cnn_saturation|python3 ml_stress.py --mode combined --workers 16"
"bigdata_dataframe_flood|python3 ml_stress.py --mode bigdata --workers 16"
)

# ===== SUDO KEEP-ALIVE =====
echo "[+] Requesting sudo access..."
sudo -v

( while true; do sudo -v; sleep 60; done ) &
SUDO_KEEPALIVE_PID=$!

# ===== CTRL+C HANDLER =====
cleanup() {
  echo
  echo "[!] Ctrl+C detected — finishing current session safely..."

INTERRUPTED=true

# 1. stop stress/training first
if [ -n "$STRESS_PID" ]; then
    echo "[!] Stopping training workload..."
    kill -INT $STRESS_PID 2>/dev/null || true
    pkill -INT -P $STRESS_PID 2>/dev/null || true
    wait $STRESS_PID 2>/dev/null || true
STRESS_PID=""
fi

# 2. stop pinner
if [ -n "$PINNER_PID" ]; then
    echo "[!] Stopping pinner..."
    sudo kill -INT $PINNER_PID 2>/dev/null || true
    wait $PINNER_PID 2>/dev/null || true
PINNER_PID=""
fi

# 3. save current session
if [ -n "$CURRENT_LABEL" ]; then
    echo "[!] Saving data for session: $CURRENT_LABEL"
$READER_CMD --label "$CURRENT_LABEL" --append
fi

# stop sudo keepalive
  kill $SUDO_KEEPALIVE_PID 2>/dev/null || true

  echo "[!] Exit complete"
  exit 0
}

trap cleanup INT
trap 'kill $SUDO_KEEPALIVE_PID 2>/dev/null || true' EXIT

# ===== HELPERS =====

start_pinner() {
  echo "[+] Starting pinner..."
$PINNER_CMD &
PINNER_PID=$!
  sleep 1
}

stop_pinner() {
  echo "[+] Stopping pinner (SIGINT)..."
  sudo kill -INT $PINNER_PID
  wait $PINNER_PID 2>/dev/null || true
PINNER_PID=""
}

run_reader() {
local label=$1
  echo "[+] Running reader for label: $label"
$READER_CMD --label "$label" --append
}

# ===== MAIN LOOP =====

for entry in "${SESSIONS[@]}"; do
IFS="|" read -r label cmd <<< "$entry"

CURRENT_LABEL="$label"

echo "======================================="
echo "[SESSION] $label"
echo "======================================="

# 1. start training workload FIRST
echo "[+] Starting workload: $cmd"
timeout ${DURATION}s bash -c "$cmd" &
STRESS_PID=$!

# 2. wait 2 sec, then start pinner
sleep 2
start_pinner

# 3. wait until workload finishes (or times out)
wait $STRESS_PID
STRESS_PID=""

# 4. immediately stop pinner
stop_pinner

# 5. run reader
run_reader "$label"

echo "[✓] Completed: $label"
echo

# stop further sessions if interrupted
if [ "$INTERRUPTED" = true ]; then
echo "[!] Stopping further sessions"
break
fi

sleep 5
done

# final cleanup
kill $SUDO_KEEPALIVE_PID 2>/dev/null || true

echo "All sessions complete!"