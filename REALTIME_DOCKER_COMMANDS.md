# Realtime dashboard: command-only runbook

```bash
# Terminal 1 — install host build prerequisites (choose your distribution)
# Debian/Ubuntu:
sudo apt-get update
sudo apt-get install -y clang gcc libbpf-dev libelf-dev zlib1g-dev bpftool

# Fedora:
sudo dnf install -y clang llvm gcc bpftool libbpf-devel elfutils-libelf-devel zlib-devel

# Move to the root of a complete project folder containing Dockerfile and src/.
cd /path/to/project
git lfs pull

# Ensure bpffs is mounted on the host.
mountpoint -q /sys/fs/bpf || sudo mount -t bpf bpf /sys/fs/bpf

# Build the host-side collector and pinner (x86-64).
clang -O2 -g -target bpf -D__TARGET_ARCH_x86 \
  -I/usr/include/bpf \
  -c src/realtime/realtime_monitor.bpf.c \
  -o src/realtime/realtime_monitor.bpf.o
gcc -O2 -Wall -Wextra \
  -o src/realtime/realtime_pinner \
  src/realtime/pinner.c -lbpf -lelf -lz

# Start the host collector and leave this terminal running.
# Stop any previous collector first; this replaces stale map pins.
sudo ./src/realtime/realtime_pinner
```

```bash
# Terminal 2 — from the same project root, build the dashboard image.
cd /path/to/project
docker build -t performance-bottleneck-dashboard .

# Export the image to deployment/ so it can be loaded on another Docker host.
mkdir -p deployment
docker save --output deployment/performance-bottleneck-dashboard.tar \
  performance-bottleneck-dashboard

# Run the dashboard with the host's writable bpffs mount.
# SYS_ADMIN is needed on hosts where CAP_BPF alone cannot read map BTF.
docker run --rm \
  --name performance-bottleneck-dashboard \
  --publish 8000:8000 \
  --mount type=bind,source=/sys/fs/bpf,target=/sys/fs/bpf \
  --cap-add=BPF \
  --cap-add=SYS_ADMIN \
  --env REALTIME_PIN_BASE=/sys/fs/bpf/pbd_test \
  performance-bottleneck-dashboard
```

```bash
# Terminal 3 — check readiness and the live API.
curl --fail-with-body --show-error http://localhost:8000/api/health
curl --fail-with-body --show-error http://localhost:8000/api/live
```

```bash
# Optional — inspect the pinned map and verify container BTF access.
docker exec performance-bottleneck-dashboard \
  bpftool -j map show pinned /sys/fs/bpf/pbd_test/sched_map

# Replace 245 with the btf_id returned by the preceding command.
docker exec performance-bottleneck-dashboard \
  bpftool btf dump id 245 format raw
```

```bash
# Stop the dashboard with Ctrl+C in Terminal 2, then stop the host collector
# with Ctrl+C in Terminal 1.
```
