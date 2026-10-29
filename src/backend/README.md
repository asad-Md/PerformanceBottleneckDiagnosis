# Backend and dashboard container

The root `Dockerfile` builds the React frontend and packages it with the FastAPI
backend in one image. The backend serves the UI at `/` and the API under `/api`
on port 8000. The image does not start the eBPF collector: run the collector on
the Linux host first, then mount its pinned maps into the container.

## Linux host prerequisites

- Linux with the required eBPF tracing support and `/sys/fs/bpf` mounted as
  bpffs.
- The host-built `src/realtime/realtime_pinner` executable and its matching
  `src/realtime/realtime_monitor.bpf.o` object.
- The full model files checked out, including the Git LFS-managed champion
  model; run `git lfs pull` if the checkout contains an LFS pointer instead.
- Permission to load/attach the BPF programs on the host.
- Docker with permission to bind-mount `/sys/fs/bpf` and grant the backend
  container the BPF capabilities needed to access the pinned maps.

The pinner runs on the host, not in Docker. Build the matching collector and
pinner as described in [`src/realtime/README.md`](../realtime/README.md). Keep
the executable beside `realtime_monitor.bpf.o`; the pinner resolves that
default object path relative to itself, so it can be launched from any working
directory. Start it from the repository root and leave it running:

```bash
sudo chmod +x ./src/realtime/realtime_pinner
sudo ./src/realtime/realtime_pinner
```

This dedicated `src/realtime` collector is the one compatible with the
backend's map BTF/ABI checks and model feature pipeline. Do not use
`src/approach3/perf_monitor.bpf.c` for this backend: although it has similarly
named maps, its BTF type names and memory/lock map layouts differ.

The collector pins the four maps under `/sys/fs/bpf/pbd_test`. Keep this
terminal running while using the dashboard. The pinner must have completed
startup before the backend container starts. These host-side collector files
are excluded from the Docker image.

## Build and run

From the repository root, build the single image:

```bash
docker build -t performance-bottleneck-dashboard .
```

In another terminal, run the image with the host's bpffs mounted at the same
path. This bind mount must remain writable: the backend atomically consumes
map entries while forming each snapshot. On the host where the BTF access
failure was observed, `SYS_ADMIN` was required in addition to `BPF`:

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

The container runs as root so its `bpftool`, `sudo -n`, and libbpf map
operations can use the granted capabilities. Kernel versions or host security
profiles may still block BTF-by-ID reads. To diagnose that separately from
missing pins, first inspect the map inside the container:

```bash
docker exec performance-bottleneck-dashboard \
  bpftool -j map show pinned /sys/fs/bpf/pbd_test/sched_map
```

Then check access using the numeric `btf_id` from that output (the example uses
`245`; replace it with the ID from your map output):

```bash
docker exec performance-bottleneck-dashboard bpftool btf dump id 245 format raw
```

If the BTF dump still fails with `Operation not permitted`, the host's Docker
security profile or rootless mode may be blocking access. `SYS_ADMIN` is broad;
use it only for a trusted local deployment when needed. Do not use
`--privileged` as a workaround.

Open <http://localhost:8000/> for the dashboard. The API remains available at
`/api/live` and `/api/health`. The backend consumes values from the shared maps;
do not run another reader against those same maps concurrently.

## Local frontend development

For development without Docker, run the backend on port 8000 and start Vite
from `src/frontend`. Vite proxies `/api` requests to the local backend. The
combined image leaves `VITE_API_BASE_URL` unset and uses the same-origin API
path; for a separately hosted frontend, set `VITE_API_BASE_URL` at build time
to the backend origin.

For a condensed, command-only setup from a fresh project folder, see
[`REALTIME_DOCKER_COMMANDS.md`](../../REALTIME_DOCKER_COMMANDS.md).
