import type { RealtimeSnapshot, RealtimePredictionResult, RealtimeFeatureRow, RealtimeHistoryItem, RealtimeEventItem, RealtimeFeatureMetric } from '../types/realtime';

const featureRow: RealtimeFeatureRow = {
  timestamp_ns: 1724692953000000000,
  pid: 3219,
  cpu: 3,
  comm: 'stress-ng',
  ctx_switches: 18400,
  voluntary_switches: 9500,
  involuntary_switches: 6200,
  cpu_migrations: 142,
  total_runtime_ns: 960000000,
  stall_ns: 2450000,
  avg_stall_ns: 2450000,
  max_stall_ns: 4500000,
  latency_count: 15,
  avg_runq_ratio: 0.68,
  minor_faults: 312,
  major_faults: 212,
  kmalloc_count: 851,
  kfree_count: 824,
  total_alloc_bytes: 781000000,
  total_free_bytes: 702000000,
  large_page_allocs: 58,
  syscall_count: 4238,
  avg_syscall_latency_ns: 2960000,
  max_syscall_latency_ns: 5200000,
  read_count: 1100,
  write_count: 890,
  read_bytes: 1820000,
  write_bytes: 880000,
  mmap_count: 224,
  futex_count: 631,
  avg_futex_latency_ns: 640000,
  epoll_count: 42,
  avg_epoll_latency_ns: 240000,
  poll_count: 90,
  syscall_error_count: 18,
  mutex_contentions: 842,
  avg_mutex_wait_ns: 1720000,
  max_mutex_wait_ns: 3900000,
  rwsem_read_contentions: 398,
  avg_rwsem_read_wait_ns: 990000,
  rwsem_write_contentions: 400,
  avg_rwsem_write_wait_ns: 1100000,
  max_rwsem_write_wait_ns: 3000000,
  session_label: 'cpu_stress',
};

const prediction: RealtimePredictionResult = {
  pid: 3219,
  cpu: 3,
  comm: 'stress-ng',
  pred_class: 3,
  prediction: 'High',
  is_bottleneck: true,
  confidence: 94,
};

const metrics: RealtimeFeatureMetric[] = [
  { id: 'runtime', label: 'Runtime', value: 960, unit: 'ms', delta: 18, direction: 'up', description: 'Observed runtime', source: 'total_runtime_ns' },
  { id: 'ctxSwitches', label: 'Context Switches', value: 18400, unit: 'count', delta: 21, direction: 'up', description: 'Observed scheduling activity', source: 'ctx_switches' },
  { id: 'stall', label: 'Average Stall Time', value: 2.45, unit: 'ms', delta: 12, direction: 'up', description: 'Observed stall duration', source: 'avg_stall_ns' },
  { id: 'syscallLatency', label: 'Average Syscall Latency', value: 2.96, unit: 'ms', delta: 15, direction: 'up', description: 'Observed syscall latency', source: 'avg_syscall_latency_ns' },
  { id: 'lockContention', label: 'Lock Contentions', value: 1240, unit: 'ops', delta: 18, direction: 'up', description: 'Observed lock activity', source: 'lock_pressure' },
  { id: 'memoryPressure', label: 'Major Page Faults', value: 212, unit: 'faults', delta: 9, direction: 'up', description: 'Observed major page faults', source: 'major_faults' },
];

const history: RealtimeHistoryItem[] = [
  {
    timestamp: '21:42:33',
    comm: 'stress-ng',
    pid: 3219,
    cpu: 3,
    prediction: 'High',
    pred_class: 3,
    is_bottleneck: true,
    confidence: 94,
  },
  {
    timestamp: '21:42:15',
    comm: 'python',
    pid: 512,
    cpu: 1,
    prediction: 'Medium',
    pred_class: 2,
    is_bottleneck: true,
    confidence: 82,
  },
  {
    timestamp: '21:41:56',
    comm: 'kworker',
    pid: 1842,
    cpu: 5,
    prediction: 'Low',
    pred_class: 1,
    is_bottleneck: false,
    confidence: 61,
  },
];

const events: RealtimeEventItem[] = [
  { time: '21:42:33', text: 'PID 3219 CPU 3 stress-ng bottleneck detected at 94% confidence.', level: 'critical' },
  { time: '21:42:31', text: 'Context-switch and runqueue activity rose above the recent baseline.', level: 'warning' },
  { time: '21:42:28', text: 'System call latency increased and lock contention remained elevated.', level: 'warning' },
  { time: '21:42:11', text: 'Normal baseline established prior to the active contention window.', level: 'info' },
];

export const mockRealtimeSnapshot: RealtimeSnapshot = {
  currentTimestamp: '2026-08-26T21:42:33Z',
  lastUpdated: '2026-08-26T21:42:33Z',
  connectionStatus: 'live',
  statusLabel: 'BOTTLENECK DETECTED',
  summary: 'Realtime classification is currently identifying a CPU bottleneck for the active process.',
  bottleneckCount: 1,
  prediction,
  metrics,
  featureRow,
  history,
  events,
};

export function createRealtimeAdapter() {
  return {
    async loadSnapshot(): Promise<RealtimeSnapshot> {
      return mockRealtimeSnapshot;
    },
  };
}
