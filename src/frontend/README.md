# Realtime Bottleneck Diagnosis Dashboard

The React dashboard polls the live eBPF API at `http://127.0.0.1:8000/api/live`.
It displays the selected `(pid, cpu)` row, Champion V7 severity, MLP attribution,
observed metrics, and sample freshness. API failures are shown as a disconnected
state; the adapter does not substitute mock predictions.

## Run locally

Start the backend as a normal user if it is not already running:

```sh
venv/bin/python -m uvicorn src.backend.app:app --host 127.0.0.1 --port 8000
```

In another terminal, start the frontend:

```sh
cd src/frontend
npm run dev
```

Open <http://127.0.0.1:4173/>. The Vite development server listens on port 4173;
the backend must be available on port 8000. The dashboard polls every five
seconds and allows up to 60 seconds for a live map snapshot. Stale rows and
unavailable model predictions remain explicitly marked.

## Build

```sh
cd src/frontend
npm run build
```
