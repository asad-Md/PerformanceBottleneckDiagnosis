# Backend and dashboard container

The root `Dockerfile` builds the React frontend and packages it with the FastAPI
backend in one image. The backend serves the UI at `/` and the API under `/api`
on port 8000. The image does not start the eBPF collector: run the collector on
the Linux host first, then mount its pinned maps into the container.

## Linux host prerequisites

- Linux with the required eBPF tracing support and `/sys/fs/bpf` mounted as
  bpffs.
- The `realtime_pinner` executable and matching `realtime_monitor.bpf.o`
  object in `src/backend/artifacts/`.
- The full model files checked out, including the Git LFS-managed champion
  model; run `git lfs pull` if the checkout contains an LFS pointer instead.
- Permission to load/attach the BPF programs on the host.
- Docker with permission to bind-mount `/sys/fs/bpf` and grant the backend
  container the BPF capabilities needed to access the pinned maps.

The pinner runs on the host, not in Docker. Start it from the artifact
directory and leave it running:

```bash
cd src/backend/artifacts
sudo chmod +x ./realtime_pinner
sudo ./realtime_pinner --obj ./realtime_monitor.bpf.o
```

The collector pins the four maps under `/sys/fs/bpf/pbd_test`. Keep this
terminal running while using the dashboard. The pinner must have completed
startup before the backend container starts. These two host-side collector
files are excluded from the Docker image.

## Build and run

From the repository root, build the single image:

```bash
docker build -t performance-bottleneck-dashboard .
```

In another terminal, run the image with the host's bpffs mounted at the same
path. This bind mount must remain writable: the backend atomically consumes
map entries while forming each snapshot.

```bash
docker run --rm \
  --name performance-bottleneck-dashboard \
  --publish 8000:8000 \
  --mount type=bind,source=/sys/fs/bpf,target=/sys/fs/bpf \
  --cap-add=BPF \
  --env REALTIME_PIN_BASE=/sys/fs/bpf/pbd_test \
  performance-bottleneck-dashboard
```

The container runs as root so its `bpftool`, `sudo -n`, and libbpf map
operations can use the granted capabilities. Kernel versions or host security
policies may require `SYS_ADMIN` instead of the split BPF capability; grant
only what the host requires. Rootless Docker or security
profiles that block BPF syscalls may prevent map access.

Open <http://localhost:8000/> for the dashboard. The API remains available at
`/api/live` and `/api/health`. The backend consumes values from the shared maps;
do not run another reader against those same maps concurrently.

## Local frontend development

For development without Docker, run the backend on port 8000 and start Vite
from `src/frontend`. Vite proxies `/api` requests to the local backend. The
combined image leaves `VITE_API_BASE_URL` unset and uses the same-origin API
path; for a separately hosted frontend, set `VITE_API_BASE_URL` at build time
to the backend origin.
