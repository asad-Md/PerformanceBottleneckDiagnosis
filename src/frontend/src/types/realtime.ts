export type ConnectionStatus = 'live' | 'disconnected' | 'reconnecting';
export type BottleneckPrediction = 'Normal' | 'Low' | 'Medium' | 'High';
export type SeverityLabel = 'LOW' | 'MEDIUM' | 'HIGH' | 'NORMAL';

export interface RealtimePredictionResult {
  pid: number;
  cpu: number;
  comm: string;
  pred_class: number;
  prediction: BottleneckPrediction;
  is_bottleneck: boolean;
  confidence: number;
}

export interface RealtimeFeatureRow {
  timestamp_ns: number;
  pid: number;
  cpu: number;
  comm: string;
  ctx_switches: number;
  voluntary_switches: number;
  involuntary_switches: number;
  cpu_migrations: number;
  total_runtime_ns: number;
  stall_ns: number;
  avg_stall_ns: number;
  max_stall_ns: number;
  latency_count: number;
  avg_runq_ratio: number;
  minor_faults: number;
  major_faults: number;
  kmalloc_count: number;
  kfree_count: number;
  total_alloc_bytes: number;
  total_free_bytes: number;
  large_page_allocs: number;
  syscall_count: number;
  avg_syscall_latency_ns: number;
  max_syscall_latency_ns: number;
  read_count: number;
  write_count: number;
  read_bytes: number;
  write_bytes: number;
  mmap_count: number;
  futex_count: number;
  avg_futex_latency_ns: number;
  epoll_count: number;
  avg_epoll_latency_ns: number;
  poll_count: number;
  syscall_error_count: number;
  mutex_contentions: number;
  avg_mutex_wait_ns: number;
  max_mutex_wait_ns: number;
  rwsem_read_contentions: number;
  avg_rwsem_read_wait_ns: number;
  rwsem_write_contentions: number;
  avg_rwsem_write_wait_ns: number;
  max_rwsem_write_wait_ns: number;
  session_label?: string;
}

export interface RealtimeFeatureMetric {
  id: string;
  label: string;
  value: number;
  unit: string;
  delta: number;
  direction: 'up' | 'down';
  description: string;
  source?: string;
}

export interface RealtimeHistoryItem {
  timestamp: string;
  comm: string;
  pid: number;
  cpu: number;
  prediction: BottleneckPrediction;
  pred_class: number;
  is_bottleneck: boolean;
  confidence: number;
}

export interface RealtimeEventItem {
  time: string;
  text: string;
  level: 'info' | 'warning' | 'critical';
}

export interface RealtimeSnapshot {
  currentTimestamp: string;
  lastUpdated: string;
  connectionStatus: ConnectionStatus;
  statusLabel: 'HEALTHY' | 'WARNING' | 'BOTTLENECK DETECTED' | 'CRITICAL';
  summary: string;
  bottleneckCount: number;
  prediction: RealtimePredictionResult | null;
  metrics: RealtimeFeatureMetric[];
  featureRow: RealtimeFeatureRow | null;
  history: RealtimeHistoryItem[];
  events: RealtimeEventItem[];
}
