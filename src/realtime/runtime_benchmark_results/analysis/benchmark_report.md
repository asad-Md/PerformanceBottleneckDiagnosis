# Realtime Diagnosis Runtime Benchmark Report

This report is generated from the raw cycle/resource CSV files and run JSON metadata. It contains no simulated measurements.

## Methodology and metric boundaries

- The existing predictor repeatedly reads pinned maps and waits after each cycle; observed start-to-start cadence includes work plus the fixed post-cycle interval.
- Detection-path latency measures map reads through the existing classifier prediction; full cycle duration also includes diagnostic formatting and console output.
- Resource samples use `/proc/stat`, `/proc/meminfo`, and selected `/proc/<pid>/stat` fields. Host CPU is 0-100%; process CPU is percent of one logical CPU; memory values are resident/used bytes.
- Map snapshots are neither cleared nor differenced by the reader; do not treat cycles as independent windows until map update semantics are verified.
- A cycle deadline and MLP attribution stage are not defined in the current active realtime path. Empty fields are unavailable, not zero.
- Detection delay is not measured: no independently verified bottleneck onset timestamp is associated with a diagnosed process.

## Run status

- completed: 2 run(s)

Failed, incomplete, and interrupted runs remain in `run_summary.csv`; interpret them separately from completed repetitions.

## Per-condition results

### cpu / medium / ebpf_plus_diagnosis / configured interval 5000.0 ms

- Runs: 1; completed: 1; incomplete/failed: 0.
- Diagnosis cycles: 55.
- Pooled cycle latency median/p95: 388.161561 / 667.6391091999999 ms.
- Median of per-run cycle medians: 388.161561 ms; between-run standard deviation of run medians: not measured ms.
- Median of per-run p95 cycle latency: 667.6391091999999 ms.
- Actual start-to-start interval median/p95: 5357.169118 / 5667.738280699999 ms.
- Median records processed per cycle: 3496.0.
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
- Mean host CPU change: 0.03136704119850009 percentage points (between-repetition SD: not measured).
- Mean host memory change: 772595712.0 bytes (between-repetition SD: not measured).
- Mean monitoring-process CPU: 0.0% of one logical CPU.
- Mean monitoring-process RSS: 1152294912.0 bytes.

