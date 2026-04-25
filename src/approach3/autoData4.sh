#!/usr/bin/env bash

set -e

# ===== CONFIG =====
CURRENT_LABEL=""
PINNER_PID=""
INTERRUPTED=false

DURATION=300   # longer = more contention

PINNER_CMD="sudo ./pinner --obj perf_monitor.bpf.o"
READER_CMD="sudo ./reader"

# ===== SUDO KEEP-ALIVE =====
echo "[+] Requesting sudo..."
sudo -v
( while true; do sudo -v; sleep 60; done ) &
SUDO_KEEPALIVE_PID=$!

# ===== CTRL+C HANDLER =====
cleanup() {
  echo
  echo "[!] Ctrl+C → graceful shutdown..."

  INTERRUPTED=true

  pkill -INT stress-ng 2>/dev/null || true
  pkill -INT sysbench 2>/dev/null || true
  pkill -INT blender 2>/dev/null || true
  pkill -INT make 2>/dev/null || true
  pkill -INT docker 2>/dev/null || true

  if [ -n "$PINNER_PID" ]; then
    sudo kill -INT $PINNER_PID 2>/dev/null || true
    wait $PINNER_PID 2>/dev/null || true
  fi

  if [ -n "$CURRENT_LABEL" ]; then
    $READER_CMD --label "$CURRENT_LABEL" --append
  fi

  kill $SUDO_KEEPALIVE_PID 2>/dev/null || true
  exit 0
}

trap cleanup INT
trap 'kill $SUDO_KEEPALIVE_PID 2>/dev/null || true' EXIT

# ===== WORKLOAD FUNCTIONS =====

run_kernel_build() {
  if [ ! -d "linux" ]; then
    git clone https://github.com/torvalds/linux.git
  fi
  cd linux
  make -j$(nproc)*2 &
  cd ..
}

run_sysbench() {
  sysbench cpu --threads=$(nproc)*2 run &
}

run_blender() {
  blender -b -noaudio -f 1 &
}

run_docker_chaos() {
  for i in {1..4}; do
    docker run -d ubuntu bash -c "while true; do sha1sum /dev/zero; done"
  done
}

stop_docker() {
  docker kill $(docker ps -q) 2>/dev/null || true
}

# ===== MAIN =====

SESSIONS=(
  "real_compile"
  "compile_sysbench"
  "compile_blender"
  "full_system"
)

for label in "${SESSIONS[@]}"; do
  CURRENT_LABEL="$label"

  echo "======================================="
  echo "[SESSION] $label"
  echo "======================================="

  # start workloads
  case $label in
    real_compile)
      run_kernel_build
      ;;
    compile_sysbench)
      run_kernel_build
      run_sysbench
      ;;
    compile_blender)
      run_kernel_build
      run_blender
      ;;
    full_system)
      run_kernel_build
      run_sysbench
      run_blender
      run_docker_chaos
      ;;
  esac

  # wait 2 sec → start pinner
  sleep 2
  $PINNER_CMD &
  PINNER_PID=$!

  # run for duration
  sleep $DURATION

  # stop everything
  pkill -INT make 2>/dev/null || true
  pkill -INT sysbench 2>/dev/null || true
  pkill -INT blender 2>/dev/null || true
  stop_docker

  # stop pinner
  sudo kill -INT $PINNER_PID
  wait $PINNER_PID 2>/dev/null || true
  PINNER_PID=""

  # read data
  $READER_CMD --label "$label" --append

  echo "[✓] Completed: $label"
  echo

  if [ "$INTERRUPTED" = true ]; then
    break
  fi

  sleep 10
done

kill $SUDO_KEEPALIVE_PID 2>/dev/null || true

echo "All sessions complete!"