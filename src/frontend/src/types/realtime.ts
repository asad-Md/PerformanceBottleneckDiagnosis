export type ConnectionStatus = 'live' | 'stale' | 'disconnected' | 'reconnecting';
export type SeverityLabel = 'High' | 'Not-High';

export interface ModelStatus {
  available: boolean;
  status: 'available' | 'unavailable' | 'error';
  checkpoint: string;
  error?: string | null;
}

export interface ModelStatuses {
  champion_v7: ModelStatus;
  attribution_mlp: ModelStatus;
}

export interface SeverityPrediction {
  available: boolean;
  status: 'available' | 'unavailable' | 'error';
  prediction?: SeverityLabel;
  is_bottleneck?: boolean;
  confidence?: number;
  high_probability?: number;
  class_probabilities?: Record<SeverityLabel, number>;
  threshold?: number;
  zero_filled_inputs?: string[];
  error?: string | null;
}

export interface AttributionPrediction {
  available: boolean;
  status: 'available' | 'unavailable' | 'error';
  model: string;
  labels: string[] | null;
  probabilities: Record<string, number> | null;
  threshold: number | null;
  error?: string | null;
}

export interface RealtimePredictionResult {
  pid: number;
  cpu: number;
  comm: string;
  attribution_prediction: AttributionPrediction;
  severity_prediction: SeverityPrediction;
  missing_metrics: string[];
}

export interface RealtimeFeatureMetric {
  id: string;
  label: string;
  value: number | null;
  unit: string;
  direction: 'up' | 'down' | 'flat';
  description: string;
  source?: string;
}

export interface RealtimeHistoryItem {
  sample_timestamp_ns: string | null;
  sample_timestamp_clock: 'CLOCK_MONOTONIC';
  sample_timestamp_status: 'available' | 'unavailable' | 'invalid';
  sample_age_seconds: number | null;
  data_status: 'live' | 'stale';
  comm: string;
  pid: number;
  cpu: number;
  attribution_labels: string[] | null;
  attribution_status: string;
  severity_prediction?: SeverityPrediction;
}

export interface RealtimeEventItem {
  time: string;
  text: string;
  level: 'info' | 'warning' | 'critical';
}

export interface RealtimeSnapshot {
  currentTimestamp: string;
  lastPolled: string;
  sampleTimestampNs: string | null;
  sampleTimestampClock: 'CLOCK_MONOTONIC';
  sampleTimestampStatus: 'available' | 'unavailable' | 'invalid';
  sampleAgeSeconds: number | null;
  connectionStatus: ConnectionStatus;
  statusLabel: 'LIVE eBPF' | 'STALE eBPF' | 'LIVE eBPF — NO PROCESSES' | 'BACKEND UNAVAILABLE';
  summary: string;
  processCount: number;
  prediction: RealtimePredictionResult | null;
  metrics: RealtimeFeatureMetric[];
  history: RealtimeHistoryItem[];
  events: RealtimeEventItem[];
  models: ModelStatuses;
}
