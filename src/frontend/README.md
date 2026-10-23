# Realtime Bottleneck Diagnosis Dashboard

The React dashboard polls the live eBPF API at
`http://127.0.0.1:8000/api/live`. It displays the selected `(pid, cpu)` row,
Champion V7 severity, MLP attribution, observed metrics, and sample freshness.
API failures are shown as a disconnected state; the adapter does not substitute
mock predictions.

## Run locally on Windows

From the repository root, start the backend as a normal user:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.backend.app:app --host 127.0.0.1 --port 8000
```

If FastAPI or Uvicorn is missing or too old for the Private Network Access
middleware option, install or upgrade them with:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade fastapi uvicorn
```

In another PowerShell window, start the frontend:

```powershell
Set-Location .\src\frontend
npm ci
npm run dev
```

`npm ci` is only needed if frontend dependencies have not been installed yet.
Open <http://127.0.0.1:4173/>. The Vite development server listens on port
4173; the backend must be available on port 8000. The dashboard polls every
five seconds and allows up to 60 seconds for a live map snapshot. Stale rows
and unavailable model predictions remain explicitly marked.

The live reader depends on Linux eBPF maps pinned under `/sys/fs/bpf` and
`bpftool`; a native Windows backend may start but cannot provide those live
eBPF readings. Run the backend in the Linux environment that owns the pinned
maps (for example, WSL2). From the repository root in that environment, the
startup command is:

```sh
.venv/bin/python -m uvicorn src.backend.app:app --host 127.0.0.1 --port 8000
```

If FastAPI or Uvicorn is missing in that Linux environment, install or upgrade
them with:

```sh
.venv/bin/python -m pip install --upgrade fastapi uvicorn
```

When using WSL2, confirm Windows can reach the backend at
`http://127.0.0.1:8000`; this depends on WSL localhost forwarding.

## Classroom demo: deployed frontend and local backend

The frontend request is hardcoded to `http://127.0.0.1:8000/api/live`. In a
browser, `127.0.0.1` means the computer running that browser, not Vercel's
servers. This demo therefore requires opening the deployed site on the same
laptop where the backend is running. The backend's existing CORS middleware
allows `https://performance-bottleneck-diagnosis-kappa.vercel.app`; edit the
`allow_origins` list in `src/backend/app.py` if the deployed frontend origin
changes. CORS credentials are disabled because the frontend does not use
cookies or credentialed requests. The middleware also opts into Private
Network Access for the listed origins when the browser requests it.

### Test the backend locally

With the backend running, from another PowerShell window check the health
endpoint and the live endpoint:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
Invoke-RestMethod http://127.0.0.1:8000/api/live
```

The backend also defines `POST /api/predict`; the dashboard currently calls
only `GET /api/live`.

### Test from the deployed site

1. Start the local backend on the demo laptop.
2. Open `https://performance-bottleneck-diagnosis-kappa.vercel.app` on that
   same laptop.
3. Open the browser developer tools' Network and Console tabs and check the
   `/api/live` request. A healthy local API response is not proof that the
   deployed page can reach it; verify the request in the browser.

An HTTPS page requesting an HTTP API can be blocked as mixed content. Browsers
may also restrict a public HTTPS origin from reaching a private/local network
address (Private Network Access or related local-network protections). CORS
configuration alone may not bypass these browser restrictions, and changing
the browser's insecure-content setting is not guaranteed to resolve them.

If the browser blocks the request, use an HTTPS tunnel such as Cloudflare
Tunnel or ngrok. Then change the frontend request URL to the tunnel's HTTPS
API URL and add the deployed Vercel origin to the backend's `allow_origins`.
No tunnel has been configured for this project.

## Build and test

From `src\frontend`, run the adapter tests with:

```powershell
node .\tests\realtimeAdapter.test.mjs
```

Build the frontend with:

```powershell
npm run build
```
