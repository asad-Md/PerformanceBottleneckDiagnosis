# Realtime Diagnosis Runtime Benchmark Report

This report is generated from the raw cycle/resource CSV files and run JSON metadata. It contains no simulated measurements.

## Methodology and metric boundaries

- The predictor snapshots and deletes captured pinned-map entries. Polling targets the configured start-to-start interval; overruns start the next cycle immediately.
- Detection-path latency measures map reads through the existing classifier prediction; full cycle duration also includes diagnostic formatting and console output.
- Resource samples use `/proc/stat`, `/proc/meminfo`, and selected `/proc/<pid>/stat` fields. Host CPU is 0-100%; process CPU is percent of one logical CPU; memory values are resident/used bytes.
- Clearing map entries is not atomic with kernel-side updates, so a racing update can be lost and snapshots are not guaranteed to be non-overlapping.
- The benchmark's MLP attribution stage is conditional on qualifying rows. Empty fields are unavailable or indicate that no row qualified, not zero.
- Detection delay is not measured: no independently verified bottleneck onset timestamp is associated with a diagnosed process.

## Run status

- completed: 2 run(s)

Failed, incomplete, and interrupted runs remain in `run_summary.csv`; interpret them separately from completed repetitions.

## Per-condition results

### cpu / medium / ebpf_plus_diagnosis / configured interval 5000.0 ms

- Runs: 1; completed: 1; incomplete/failed: 0.
- Diagnosis cycles: 60.
- Pooled cycle latency median/p95: 106.495499 / 109.80345725 ms.
- Median of per-run cycle medians: 106.495499 ms; between-run standard deviation of run medians: not measured ms.
- Median of per-run p95 cycle latency: 109.80345725 ms.
- Actual start-to-start interval median/p95: 5000.084627 / 5000.1469130000005 ms.
- Median records processed per cycle: 229.0.
- Missed-deadline rate: not measured (not measured means no deadline was defined).

### cpu / medium / workload_only / configured interval nan ms

- Runs: 1; completed: 1; incomplete/failed: 0.
- Diagnosis cycles: 0.
- Pooled cycle latency median/p95: not measured / not measured ms.
- Median of per-run cycle medians: not measured ms; between-run standard deviation of run medians: not measured ms.
- Median of per-run p95 cycle latency: not measured ms.
- Actual start-to-start interval median/p95: not measured / not measured ms.
- Median records processed per cycle: not measured.
- Missed-deadline rate: not measured (not measured means no deadline was defined).

## Matched monitoring overhead

### cpu / medium

- Matched repetitions: 1.
- Mean host CPU change: 0.0468066833229237 percentage points (between-repetition SD: not measured).
- Mean host memory change: 1000439808.0 bytes (between-repetition SD: not measured).
- Mean monitoring-process CPU: 0.0% of one logical CPU.
- Mean monitoring-process RSS: 1193455616.0 bytes.

