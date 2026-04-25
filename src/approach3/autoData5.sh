#!/usr/bin/env bash

set -e

# ===== CONFIG =====
CURRENT_LABEL=""
STRESS_PID=""
PINNER_PID=""
INTERRUPTED=false

DURATION=75
PINNER_CMD="sudo ./pinner --obj perf_monitor.bpf.o"
READER_CMD="sudo ./reader"

SESSIONS=(
  "io_async_saturation|--io-uring 16 --iomix 8 --hdd 4 --hdd-bytes 2G"
  "sys_context_switch_flood|--context 128 --pipe 64 --sock 64 --yield 128 --sched-mix 16"
  "cpu_avx_vector_load|--vecmath 16 --fp-error 8 --bsearch 8 --crypt 8"
  "kernel_syscall_flood|--sysinfo 32 --fstat 32 --dup 32 --mmap 16"
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

  # 1. stop stress first
  if [ -n "$STRESS_PID" ]; then
    echo "[!] Stopping stress-ng..."
    kill -INT $STRESS_PID 2>/dev/null || true
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
  IFS="|" read -r label stress_args <<< "$entry"

  CURRENT_LABEL="$label"

  echo "======================================="
  echo "[SESSION] $label"
  echo "======================================="

  # 1. start stress FIRST
  if [ -z "$stress_args" ]; then
    echo "[+] Idle session (no stress-ng)"
    sleep $DURATION &
    STRESS_PID=$!
  else
    echo "[+] Starting stress-ng: $stress_args"
    stress-ng $stress_args --timeout ${DURATION}s &
    STRESS_PID=$!
  fi

  # 2. wait 2 sec, then start pinner
  sleep 2
  start_pinner

  # 3. wait until stress finishes
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
