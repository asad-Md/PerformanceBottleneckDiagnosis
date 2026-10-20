import type {
  AttributionPrediction,
  ModelStatuses,
  RealtimeEventItem,
  RealtimeFeatureMetric,
  RealtimeObservation,
  RealtimePredictionResult,
  RealtimeSnapshot,
  SeverityPrediction,
} from '../types/realtime';

const API_BASE = 'http://127.0.0.1:8000';
const STALE_INFERENCE_REASON = 'Map snapshot is stale; no live inference was run.';
let pendingSnapshot: Promise<RealtimeSnapshot> | null = null;

export function formatUnavailablePrediction(
  prediction: { available: boolean; status?: string; error?: string | null } | null | undefined,
  dataStatus: 'live' | 'stale',
): string {
  if (prediction?.available) return 'Available';
  if (
    dataStatus === 'stale'
    && prediction?.status === 'unavailable'
    && prediction.error === STALE_INFERENCE_REASON
  ) {
    return 'Skipped — stale data';
  }
  if (prediction?.status === 'error') {
    return prediction.error ? `Error — ${prediction.error}` : 'Inference error';
  }
  return prediction?.error ? `Unavailable — ${prediction.error}` : 'Unavailable';
}

function modelStatuses(value: any): ModelStatuses {
  const status = (model: any, name: string) => ({
    available: Boolean(model?.available),
    status: model?.status ?? 'unavailable',
    checkpoint: model?.checkpoint ?? name,
    error: model?.error ?? null,
  });

  return {
    champion_v7: status(value?.champion_v7, 'Champion V7'),
    attribution_mlp: status(value?.attribution_mlp, 'baseline_original_best.pt'),
  };
}

function sampleDetails(process: any) {
  const value = process.sample_timestamp_ns;
  const isValidTimestamp = typeof value === 'string'
    && /^\d+$/.test(value)
    && BigInt(value) > 0n;
  const status = process.sample_timestamp_status === 'invalid'
    ? 'invalid'
    : process.sample_timestamp_status === 'available' && isValidTimestamp
      ? 'available'
      : 'unavailable';
  const age = process.data_age_seconds;

  return {
    timestampNs: status === 'available' ? value as string : null,
    timestampStatus: status as 'available' | 'unavailable' | 'invalid',
    ageSeconds: status === 'available'
      && typeof age === 'number'
      && Number.isFinite(age)
      && age >= 0
      ? age
      : null,
  };
}

function makeMetrics(process: any): RealtimeFeatureMetric[] {
  const values = process.metrics ?? process;
  const missing = new Set<string>(process.missing_metrics ?? []);
  const metric = (
    id: string,
    label: string,
    source: string,
    unit: string,
    scale = 1,
    derived?: number,
  ): RealtimeFeatureMetric => {
    const raw = derived ?? values[source];
    return {
      id,
      label,
      value: missing.has(source) || typeof raw !== 'number' || !Number.isFinite(raw)
        ? null
        : raw / scale,
      unit,
      direction: 'flat',
      description: missing.has(source) ? 'Not present in the current eBPF snapshot' : `Live eBPF metric: ${source}`,
      source,
    };
  };
  const lockFields = ['mutex_contentions', 'rwsem_read_contentions', 'rwsem_write_contentions'];
  const lockPressure = lockFields.some((field) => missing.has(field))
    ? undefined
    : lockFields.reduce((total, field) => total + Number(values[field] ?? 0), 0);

  return [
    metric('runtime', 'Total Runtime', 'total_runtime_ns', 'ms', 1e6),
    metric('ctxSwitches', 'Context Switches', 'ctx_switches', 'count'),
    metric('voluntarySwitches', 'Voluntary Switches', 'voluntary_switches', 'count'),
    metric('involuntarySwitches', 'Involuntary Switches', 'involuntary_switches', 'count'),
    metric('cpuMigrations', 'CPU Migrations', 'cpu_migrations', 'count'),
    metric('minorFaults', 'Minor Page Faults', 'minor_faults', 'faults'),
    metric('majorFaults', 'Major Page Faults', 'major_faults', 'faults'),
    metric('allocBytes', 'Allocated Bytes', 'total_alloc_bytes', 'bytes'),
    metric('freeCount', 'Free Operations', 'kfree_count', 'count'),
    metric('syscallCount', 'Syscall Count', 'syscall_count', 'count'),
    metric('syscallLatency', 'Average Syscall Latency', 'avg_syscall_latency_ns', 'ms', 1e6),
    metric('maxSyscallLatency', 'Maximum Syscall Latency', 'max_syscall_latency_ns', 'ms', 1e6),
    metric('readBytes', 'Read Bytes', 'read_bytes', 'bytes'),
    metric('writeBytes', 'Write Bytes', 'write_bytes', 'bytes'),
    metric('lockContention', 'Lock Contentions', 'lock_pressure', 'count', 1, lockPressure),
    metric('mutexWait', 'Average Mutex Wait', 'avg_mutex_wait_ns', 'ms', 1e6),
    metric('rwsemRead', 'RW Semaphore Read Contention', 'rwsem_read_contentions', 'count'),
    metric('rwsemWrite', 'RW Semaphore Write Contention', 'rwsem_write_contentions', 'count'),
  ];
}

export function makeSnapshot(data: any): RealtimeSnapshot {
  const processes = Array.isArray(data.processes) ? data.processes : [];
  const models = modelStatuses(data.models);
  const now = new Date().toISOString();
  const isFresh = (process: any) => {
    const sample = sampleDetails(process);
    return data.source_status === 'live'
      && process.data_status === 'live'
      && sample.timestampStatus === 'available'
      && sample.ageSeconds !== null;
  };
  const freshRowCount = processes.filter(isFresh).length;
  const staleRowCount = processes.length - freshRowCount;
  const selected = processes.find((process: any) =>
    isFresh(process)
      && process.severity_prediction?.available
      && process.attribution_prediction?.available,
  ) ?? processes.find((process: any) =>
    isFresh(process)
      && (process.severity_prediction?.available || process.attribution_prediction?.available),
  ) ?? processes.find(isFresh) ?? processes[0];

  if (!selected) {
    return {
      currentTimestamp: now,
      lastPolled: now,
      sampleTimestampNs: null,
      sampleTimestampClock: 'CLOCK_MONOTONIC',
      sampleTimestampStatus: 'unavailable',
      sampleAgeSeconds: null,
      connectionStatus: 'live',
      statusLabel: 'LIVE eBPF — NO PROCESSES',
      summary: data.source_error ?? 'The live API responded, but the eBPF maps returned no process metrics.',
      processCount: 0,
      freshRowCount,
      staleRowCount,
      prediction: null,
      metrics: [],
      observations: [],
      observationCount: 0,
      events: [],
      models,
    };
  }

  const attribution = selected.attribution_prediction as AttributionPrediction;
  const severity = selected.severity_prediction as SeverityPrediction;
  const sample = sampleDetails(selected);
  const prediction: RealtimePredictionResult = {
    pid: Number(selected.pid),
    cpu: Number(selected.cpu ?? 0),
    comm: String(selected.comm || 'unknown'),
    attribution_prediction: attribution,
    severity_prediction: severity,
    missing_metrics: selected.missing_metrics ?? [],
  };

  const observationProcesses = processes
    .filter((process: any) => process !== selected)
    .sort((left: any, right: any) => {
      const leftTimestamp = sampleDetails(left).timestampNs;
      const rightTimestamp = sampleDetails(right).timestampNs;
      if (leftTimestamp && rightTimestamp) {
        const leftNs = BigInt(leftTimestamp);
        const rightNs = BigInt(rightTimestamp);
        return leftNs > rightNs ? -1 : leftNs < rightNs ? 1 : 0;
      }
      if (leftTimestamp) return -1;
      if (rightTimestamp) return 1;
      return 0;
    })
    .slice(0, 20);
  const observations: RealtimeObservation[] = observationProcesses.map((process: any) => {
    const processSample = sampleDetails(process);
    return {
      sample_timestamp_ns: processSample.timestampNs,
      sample_timestamp_clock: 'CLOCK_MONOTONIC',
      sample_timestamp_status: processSample.timestampStatus,
      sample_age_seconds: processSample.ageSeconds,
      data_status: isFresh(process) ? 'live' : 'stale',
      comm: String(process.comm || 'unknown'),
      pid: Number(process.pid),
      cpu: Number(process.cpu ?? 0),
      attribution_labels: process.attribution_prediction?.available
        ? process.attribution_prediction.labels
        : null,
      attribution_status: process.attribution_prediction?.status ?? 'unavailable',
      attribution_prediction: process.attribution_prediction ?? null,
      severity_prediction: process.severity_prediction,
    };
  });

  const labels = attribution?.available ? attribution.labels ?? [] : [];
  const events: RealtimeEventItem[] = [
    {
      time: new Date().toLocaleTimeString(),
      text: `${selected.data_status === 'stale' ? 'Stale' : 'Live'} eBPF metrics for PID ${selected.pid} CPU ${selected.cpu} (${selected.comm}).`,
      level: 'info' as const,
    },
  ];
  if (!attribution?.available) {
    events.push({
      time: new Date().toLocaleTimeString(),
      text: `Attribution unavailable: ${attribution?.error ?? models.attribution_mlp.error ?? 'no attribution result returned'}`,
      level: 'warning' as const,
    });
  } else if (!labels.length) {
    events.push({
      time: new Date().toLocaleTimeString(),
      text: 'The attribution model returned no labels above its decision threshold.',
      level: 'info' as const,
    });
  } else {
    events.push({
      time: new Date().toLocaleTimeString(),
      text: `Attribution labels: ${labels.join(', ')}.`,
      level: 'warning' as const,
    });
  }

  const attributionNote = models.attribution_mlp.available
    ? 'Attribution model loaded.'
    : `Attribution unavailable: ${models.attribution_mlp.error ?? 'model failed to load'}.`;

  return {
    currentTimestamp: now,
    lastPolled: now,
    sampleTimestampNs: sample.timestampNs,
    sampleTimestampClock: 'CLOCK_MONOTONIC',
    sampleTimestampStatus: sample.timestampStatus,
    sampleAgeSeconds: sample.ageSeconds,
    connectionStatus: !isFresh(selected) ? 'stale' : 'live',
    statusLabel: !isFresh(selected) ? 'STALE eBPF' : 'LIVE eBPF',
    summary: `${!isFresh(selected) ? 'Stale' : 'Live'} eBPF data for ${processes.length} PID/CPU rows. Showing PID ${selected.pid} CPU ${selected.cpu} (${selected.comm}); counters are accumulated observations, not instantaneous rates. ${attributionNote}`,
    processCount: processes.length,
    freshRowCount,
    staleRowCount,
    prediction,
    metrics: makeMetrics(selected),
    observations,
    observationCount: Math.max(0, processes.length - 1),
    events,
    models,
  };
}

export function createRealtimeErrorSnapshot(error: unknown): RealtimeSnapshot {
  const now = new Date().toISOString();
  const message = error instanceof Error ? error.message : String(error);
  const unavailable = (checkpoint: string) => ({
    available: false,
    status: 'unavailable' as const,
    checkpoint,
    error: message,
  });

  return {
    currentTimestamp: now,
    lastPolled: now,
    sampleTimestampNs: null,
    sampleTimestampClock: 'CLOCK_MONOTONIC',
    sampleTimestampStatus: 'unavailable',
    sampleAgeSeconds: null,
    connectionStatus: 'disconnected',
    statusLabel: 'BACKEND UNAVAILABLE',
    summary: `Live data request failed: ${message}`,
    processCount: 0,
    freshRowCount: 0,
    staleRowCount: 0,
    prediction: null,
    metrics: [],
    observations: [],
    observationCount: 0,
    events: [],
    models: {
      champion_v7: unavailable('Champion V7'),
      attribution_mlp: unavailable('baseline_original_best.pt'),
    },
  };
}

export function createRealtimeAdapter() {
  return {
    loadSnapshot(): Promise<RealtimeSnapshot> {
      if (!pendingSnapshot) {
        pendingSnapshot = (async () => {
          const response = await fetch(`${API_BASE}/api/live`, {
            signal: AbortSignal.timeout(60000),
          });
          if (!response.ok) {
            let detail = `Live API returned HTTP ${response.status}`;
            try {
              const body = await response.json();
              if (typeof body.detail === 'string') detail = body.detail;
            } catch {
              // Keep the HTTP status when the error response has no JSON body.
            }
            throw new Error(detail);
          }
          return makeSnapshot(await response.json());
        })();
      }

      const currentRequest = pendingSnapshot;
      return currentRequest.finally(() => {
        if (pendingSnapshot === currentRequest) {
          pendingSnapshot = null;
        }
      });
    },
  };
}
