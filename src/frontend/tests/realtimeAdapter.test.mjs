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

test('snapshot separates selected diagnosis and summarizes fresh/stale observations', () => {
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
  assert.equal(snapshot.freshRowCount, 2);
  assert.equal(snapshot.staleRowCount, 1);
  assert.equal(snapshot.observationCount, 2);
  assert.deepEqual(snapshot.observations.map((item) => item.pid), [2, 1]);
  assert.equal(snapshot.observations.some((item) => item.pid === snapshot.prediction.pid), false);
  assert.equal(snapshot.observations[0].attribution_prediction.error, freshUnavailable.attribution_prediction.error);
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

  assert.deepEqual(snapshot.observations.map((item) => item.pid), [2, 1]);
});
