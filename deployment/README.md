# Performance Bottleneck Dashboard deployment bundle

This folder contains the prebuilt `linux/amd64` dashboard image and the host-side
realtime eBPF collector. It is intended for a Linux x86-64 machine with Docker
and eBPF tracing support.

## Bundle contents

- `performance-bottleneck-dashboard.tar` — Docker image archive. Load it with
  `docker load`; the image is tagged `performance-bottleneck-dashboard`.
- `realtime_pinner` — host executable that loads and attaches the collector.
- `realtime_monitor.bpf.o` — matching eBPF object required by the pinner. Keep
  it beside the executable.
- `REALTIME_DOCKER_COMMANDS.md` — copy of the project command-only runbook.
- `SHA256SUMS` — checksums for the image archive and collector files.

## Host prerequisites

- Linux x86-64, Docker, and permission to load/attach eBPF programs.
- bpffs mounted at `/sys/fs/bpf`.
- The pinner's runtime libraries: libbpf, libelf, zlib, and zstd. On
  Debian/Ubuntu, install them with:

  ```bash
  sudo apt-get update
  sudo apt-get install -y libbpf1 libelf1 zlib1g libzstd1
  ```

If bpffs is not mounted, mount it once for this host:

```bash
mountpoint -q /sys/fs/bpf || sudo mount -t bpf bpf /sys/fs/bpf
```

## Run

From this directory, load the dashboard image:

```bash
docker load --input performance-bottleneck-dashboard.tar
```

In one terminal, start the host collector and leave it running:

```bash
sudo ./realtime_pinner
```

In another terminal, run the dashboard container. The bpffs bind mount must be
writable because the backend atomically consumes map entries:

```bash
docker run --rm \
  --name performance-bottleneck-dashboard \
  --publish 8000:8000 \
  --mount type=bind,source=/sys/fs/bpf,target=/sys/fs/bpf \
  --cap-add=BPF \
  --cap-add=SYS_ADMIN \
  --env REALTIME_PIN_BASE=/sys/fs/bpf/pbd_test \
  performance-bottleneck-dashboard
```

Open <http://localhost:8000/>. Check API readiness at
<http://localhost:8000/api/health>. Stop the dashboard and collector with
Ctrl+C in their respective terminals.

The collector must start before the dashboard. Stopping it detaches its probes
but leaves its maps pinned; starting it again replaces stale map pins. The
dashboard container is configured for a local trusted deployment; `SYS_ADMIN`
is a broad capability and should not be granted on untrusted hosts.
