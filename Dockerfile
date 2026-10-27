FROM node:22-bookworm-slim AS frontend-build

WORKDIR /frontend
COPY src/frontend/package*.json ./
RUN npm ci
COPY src/frontend/ ./
RUN npm run build

FROM python:3.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    REALTIME_PIN_BASE=/sys/fs/bpf/pbd_test

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends bpftool libbpf1 sudo \
    && rm -rf /var/lib/apt/lists/*

COPY src/backend/requirements.txt ./src/backend/requirements.txt
RUN python -m pip install --no-cache-dir -r ./src/backend/requirements.txt

COPY src/backend/app.py src/backend/live_reader.py ./src/backend/
COPY src/backend/artifacts/ ./src/backend/artifacts/
COPY src/realtime/*.py ./src/realtime/
COPY --from=frontend-build /frontend/dist ./src/backend/static/

EXPOSE 8000

CMD ["uvicorn", "src.backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
