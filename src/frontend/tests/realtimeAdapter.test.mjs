import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { test } from 'node:test';
import { transform } from 'esbuild';

const adapterSource = new URL('../src/services/realtimeAdapter.ts', import.meta.url);
const source = await readFile(adapterSource, 'utf8');
const { code } = await transform(source, { loader: 'ts', format: 'esm' });
const { formatUnavailablePrediction, makeSnapshot } = await import(
  `data:text/javascript;base64,${Buffer.from(code).toString('base64')}`
);
const processTableSource = new URL('../src/services/processTable.ts', import.meta.url);
const processTableText = await readFile(processTableSource, 'utf8');
const { code: processTableCode } = await transform(processTableText, { loader: 'ts', format: 'esm' });
const {
  filterAndSortProcesses,
  formatProbabilityPercent,
  highClassProbability,
  normalizeProbability,
  predictedClassConfidence,
  summarizeVisibleProcesses,
} = await import(`data:text/javascript;base64,${Buffer.from(processTableCode).toString('base64')}`);

const staleSkipReason = 'Map snapshot is stale; no live inference was run.';

function row({ pid, cpu = 0, timestamp, dataStatus, severity, attribution, missing = [] }) {
  return {
    pid,
    cpu,
    comm: `process-${pid}`,
    session_label: 'live_ebpf',
    sample_timestamp_ns: String(timestamp),
    sample_timestamp_status: 'available',
    data_age_seconds: dataStatus === 'live' ? 0.2 : 2765,
    data_status: dataStatus,
    missing_metrics: missing,
    metrics: {},
    severity_prediction: severity,
    attribution_prediction: attribution,
  };
}

test('snapshot retains selected and supplementary rows for the all-process table', () => {
  const selected = row({
    pid: 3,
    timestamp: 900,
    dataStatus: 'live',
    severity: { available: true, status: 'available', prediction: 'Not-High' },
    attribution: { available: true, status: 'available', model: 'mlp', labels: [], probabilities: {}, threshold: 0.5 },
  });
  const freshUnavailable = row({
    pid: 2,
    timestamp: 850,
    dataStatus: 'live',
    missing: ['avg_mutex_wait_ns'],
    severity: {
      available: false,
      status: 'unavailable',
      error: 'Live collector does not provide required Champion V7 feature(s): avg_mutex_wait_ns',
      missing_features: ['avg_mutex_wait_ns'],
    },
    attribution: {
      available: false,
      status: 'unavailable',
      model: 'mlp',
      labels: null,
      probabilities: null,
      threshold: null,
      error: 'Required live attribution features are unavailable: avg_mutex_wait_ns',
      missing_features: ['avg_mutex_wait_ns'],
    },
  });
  const stale = row({
    pid: 1,
    timestamp: 800,
    dataStatus: 'stale',
    severity: { available: false, status: 'unavailable', error: staleSkipReason },
    attribution: {
      available: false,
      status: 'unavailable',
      model: 'mlp',
      labels: null,
      probabilities: null,
      threshold: null,
      error: staleSkipReason,
    },
  });

  const snapshot = makeSnapshot({
    source_status: 'live',
    processes: [stale, freshUnavailable, selected],
    models: {
      champion_v7: { available: true, status: 'available' },
      attribution_mlp: { available: true, status: 'available' },
    },
  });

  assert.equal(snapshot.prediction.pid, 3);
  assert.equal(snapshot.processCount, 2);
  assert.equal(snapshot.freshRowCount, 2);
  assert.equal(snapshot.staleRowCount, 1);
  assert.equal(snapshot.observationCount, 2);
  assert.deepEqual(snapshot.observations.map((item) => item.pid), [3, 2]);
  assert.equal(snapshot.observations.some((item) => item.pid === snapshot.prediction.pid), true);
  assert.equal(snapshot.observations[1].attribution_prediction.error, freshUnavailable.attribution_prediction.error);
});

test('snapshot does not select or expose stale rows when no fresh process remains', () => {
  const stale = row({
    pid: 7,
    timestamp: 800,
    dataStatus: 'stale',
    severity: { available: false, status: 'unavailable', error: staleSkipReason },
    attribution: { available: false, status: 'unavailable', model: 'mlp', labels: null, probabilities: null, threshold: null, error: staleSkipReason },
  });
  const snapshot = makeSnapshot({ source_status: 'stale', processes: [stale], models: {} });

  assert.equal(snapshot.prediction, null);
  assert.equal(snapshot.connectionStatus, 'stale');
  assert.equal(snapshot.processCount, 0);
  assert.equal(snapshot.freshRowCount, 0);
  assert.equal(snapshot.staleRowCount, 1);
  assert.equal(snapshot.observationCount, 0);
  assert.deepEqual(snapshot.observations, []);
});

test('snapshot keeps every fresh row visible even when the input contains more than fifty rows', () => {
  const available = {
    severity: { available: true, status: 'available', prediction: 'Not-High' },
    attribution: { available: true, status: 'available', model: 'mlp', labels: [], probabilities: {}, threshold: 0.5 },
  };
  const processes = Array.from({ length: 55 }, (_, index) => row({
    pid: index + 1,
    timestamp: 1000 + index,
    dataStatus: index === 54 ? 'stale' : 'live',
    ...available,
  }));
  const snapshot = makeSnapshot({ source_status: 'live', processes, models: {} });
  const visible = filterAndSortProcesses(snapshot.observations, {
    search: '',
    stressOnly: false,
    sort: 'pid',
  });

  assert.equal(snapshot.freshRowCount, 54);
  assert.equal(snapshot.observations.length, 54);
  assert.equal(visible.length, 54);
  assert.equal(visible.some((item) => item.data_status === 'stale'), false);
});

test('prediction status preserves stale skip, fresh backend reasons, and inference errors', () => {
  assert.equal(
    formatUnavailablePrediction({ available: false, status: 'unavailable', error: staleSkipReason }, 'stale'),
    'Skipped — stale data',
  );
  assert.equal(
    formatUnavailablePrediction({ available: false, status: 'unavailable', error: staleSkipReason }, 'live'),
    `Unavailable — ${staleSkipReason}`,
  );
  assert.equal(
    formatUnavailablePrediction({ available: false, status: 'unavailable', error: 'Missing avg_mutex_wait_ns' }, 'live'),
    'Unavailable — Missing avg_mutex_wait_ns',
  );
  assert.equal(
    formatUnavailablePrediction({ available: false, status: 'error', error: 'ValueError: model failure' }, 'live'),
    'Error — ValueError: model failure',
  );
  assert.equal(formatUnavailablePrediction(undefined, 'live'), 'Unavailable');
});

test('supplementary rows with valid monotonic timestamps are sorted newest first', () => {
  const common = {
    severity: { available: true, status: 'available', prediction: 'Not-High' },
    attribution: { available: true, status: 'available', model: 'mlp', labels: [], probabilities: {}, threshold: 0.5 },
  };
  const selected = row({ pid: 99, timestamp: '90071992547409930', dataStatus: 'live', ...common });
  const older = row({ pid: 1, timestamp: '90071992547409920', dataStatus: 'live', ...common });
  const newer = row({ pid: 2, timestamp: '90071992547409925', dataStatus: 'live', ...common });
  const snapshot = makeSnapshot({ source_status: 'live', processes: [selected, older, newer], models: {} });

  assert.deepEqual(snapshot.observations.map((item) => item.pid), [99, 2, 1]);
});

test('probability formatting validates values and reports percentages to one decimal', () => {
  assert.equal(normalizeProbability(0.824), 0.824);
  assert.equal(formatProbabilityPercent(0.824), '82.4%');
  for (const value of [undefined, null, Number.NaN, Infinity, -Infinity, -0.01, 1.01, '0.5']) {
    assert.equal(normalizeProbability(value), null);
    assert.equal(formatProbabilityPercent(value), 'N/A');
  }
});

test('predicted-class confidence differs from High-class probability', () => {
  const notHigh = {
    available: true,
    status: 'available',
    prediction: 'Not-High',
    confidence: 82,
    high_probability: 0.18,
    class_probabilities: { High: 0.18, 'Not-High': 0.82 },
  };
  assert.equal(predictedClassConfidence(notHigh), 0.82);
  assert.equal(highClassProbability(notHigh), 0.18);

  const high = {
    available: true,
    status: 'available',
    prediction: 'High',
    confidence: 91,
    high_probability: 0.91,
    class_probabilities: { High: 0.91, 'Not-High': 0.09 },
  };
  assert.equal(predictedClassConfidence(high), 0.91);
  assert.equal(highClassProbability(high), 0.91);
});

test('confidence uses backend percentage only when class-specific probabilities are absent', () => {
  assert.equal(predictedClassConfidence({
    available: true,
    status: 'available',
    prediction: 'Not-High',
    confidence: 76.5,
  }), 0.765);
  assert.equal(predictedClassConfidence({
    available: true,
    status: 'available',
    prediction: 'Not-High',
    confidence: 101,
  }), null);
  assert.equal(predictedClassConfidence({
    available: true,
    status: 'available',
    confidence: 93,
  }), null);
  assert.equal(predictedClassConfidence({ available: false, status: 'unavailable', confidence: 93 }), null);
});

test('process filtering, severity sorting, and counts exclude stale rows', () => {
  const process = (pid, comm, { high, predicted = 'High', stale = false, available = true } = {}) => ({
    pid,
    cpu: 0,
    comm,
    data_status: stale ? 'stale' : 'live',
    metrics: { runtimeMs: null, contextSwitches: null, cpuMigrations: null, allocationBytes: null, majorFaults: null, minorFaults: null },
    severity_prediction: available ? {
      available: true,
      status: 'available',
      prediction: predicted,
      high_probability: high,
      class_probabilities: { High: high, 'Not-High': high === undefined ? undefined : 1 - high },
    } : { available: false, status: 'unavailable' },
    attribution_prediction: null,
    attribution_labels: null,
  });
  const items = [
    process(20, 'stress-ng', { high: 0.82, predicted: 'Not-High' }),
    process(10, 'worker', { high: 0.91 }),
    process(30, 'stress', { high: 0.65, stale: true }),
    process(40, 'unknown', { available: false }),
  ];
  const defaults = { search: '', stressOnly: false, sort: 'high-probability' };
  assert.deepEqual(filterAndSortProcesses(items, defaults).map((item) => item.pid), [10, 20, 40]);
  assert.deepEqual(filterAndSortProcesses(items, { ...defaults, search: 'STRESS' }).map((item) => item.pid), [20]);
  assert.deepEqual(filterAndSortProcesses(items, { ...defaults, search: '20' }).map((item) => item.pid), [20]);
  assert.deepEqual(filterAndSortProcesses(items, { ...defaults, stressOnly: true }).map((item) => item.pid), [20]);
  assert.deepEqual(summarizeVisibleProcesses(filterAndSortProcesses(items, defaults)), {
    visible: 3,
    high: 1,
    notHigh: 1,
    confidenceAvailable: 2,
  });
});
