# Realtime diagnosis runtime benchmark

This benchmark measures the existing `Predictor` loop. It does not load eBPF programs, start workloads, clear maps, change model inputs, or run a second diagnosis scheduler. Run it manually on the Linux host where the existing pinned maps and model environment are available.

Results are written to `src/realtime/runtime_benchmark_results/` by default. Run the analysis notebook with `src/realtime/` as its working directory so its default `runtime_benchmark_results/` path resolves to that folder.

## What the measurements mean

- `cycle_start_ns` and `cycle_end_ns` use Python's monotonic `perf_counter_ns()` clock.
- The existing loop waits for the configured polling interval **after** each cycle. Thus start-to-start cadence includes work plus the fixed post-cycle wait; this runner does not add another scheduler.
- `detection_latency_ms` covers map reads through prediction completion. `rf_inference_ms` measures the active classifier's scaling, `predict()`, and optional `predict_proba()` calls together. `total_cycle_ms` additionally includes diagnostic rendering and console output. CSV queue writes happen after `cycle_end_ns`, outside that cycle duration.
- The active real-time path has no separate MLP attribution call. `mlp_inference_ms` is empty, not zero.
- `aggregation_ms` measures reconstruction and merging of rows by `(pid, cpu)`; `feature_ms` measures the existing feature engineering and model-column selection. No transformations, feature order, missing-value behavior, or thresholds are changed.
- `deadline_missed` is empty because the existing loop defines no cycle deadline. Its scheduling policy is fixed delay, not a fixed-deadline schedule.
- `actual_cycle_interval_ms` is the interval between consecutive cycle starts, including a prior warm-up cycle where applicable. The first cycle has no preceding interval.
- Maps are read as repeated snapshots and are not cleared or differenced by this code. Until map update semantics are independently established, cycles must not be presented as non-overlapping measurement windows.
- Each run JSON records its status. A configuration left with `status: running` (for example, after abrupt termination) is classified as incomplete by the summarizer. Failed cycles are retained with null unavailable stages and an error string.
- Cycle CSV records are written on a background thread and flushed per row. Queue overflow or writer failures raise an explicit error rather than silently dropping rows. Logging is not part of measured cycle duration, but it can affect scheduling and system-resource measurements.
- Host CPU is aggregate `/proc/stat` utilization normalized to 0–100%. Per-process CPU is a percentage of one logical CPU and may exceed 100% for multi-threaded processes. RSS and host-used memory are in bytes. Resource sampling is performed by a separate process, and should be run in both conditions.

## Prerequisites and safety

Run these checks yourself on the Linux target:

```bash
command -v bpftool
command -v stress-ng
test -r /proc/stat && test -r /proc/meminfo
```

The existing pinner and pinned maps need their normal setup and privileges. Follow the project's current pinner procedure; the benchmark runner itself does not call the pinner. It intentionally skips the predictor's normal shutdown unpin operation so a benchmark exit does not remove pins needed by another process. Do not manually clear maps for this experiment. Use the established pinner/session lifecycle for a fresh capture where needed.

`stress-ng` is optional. If it is unavailable, use an existing project workload and record its exact command. CPU, memory, and mixed examples below are commands for the user to launch manually; the benchmark never launches or terminates them.

Choose memory workload size only after checking available RAM and system policy. Keep it comfortably below available memory (for example, no more than one quarter of currently available RAM), do not use `--vm-keep` unless its memory cost is safe for the host, and stop if swapping or memory pressure becomes material. All examples use bounded durations.

## Workload-only baseline

Use the same workload command, parameters, duration, warm-up, sampling interval, and repetition index for its matched monitored run. Start each repetition with an appropriate fresh workload/capture session; do not pool cycles from unrelated run IDs.

Example CPU workload with two workers, 30 seconds of warm-up, and 300 seconds measured:

```bash
cd /path/to/PerformanceBottleneckDiagnosis
RUN_ID=cpu-medium-baseline-r1
PAIR_ID=cpu-medium-r1
WORKLOAD_COMMAND='stress-ng --cpu 2 --cpu-method matrixprod --timeout 330s --metrics-brief'
stress-ng --cpu 2 --cpu-method matrixprod --timeout 330s --metrics-brief &
WORKLOAD_PID=$!
python -m src.realtime.runtime_benchmark sample-resources \
  --run-id "$RUN_ID" --comparison-id "$PAIR_ID" --repetition-index 1 \
  --condition workload_only --workload-type cpu --intensity medium --workload-workers 2 \
  --workload-duration-s 330 --workload-command "$WORKLOAD_COMMAND" \
  --warmup-s 30 --duration-s 300 --sample-interval-s 1 \
  --workload-pid "$WORKLOAD_PID"
wait "$WORKLOAD_PID"
```

To exercise other patterns, use bounded equivalents and record the exact command:

```bash
# CPU intensity is controlled by worker count; keep the same count in the matched pair.
stress-ng --cpu 4 --cpu-method matrixprod --timeout 330s --metrics-brief

# Memory intensity is controlled by the bounded byte count; check available RAM first.
stress-ng --vm 1 --vm-bytes 512M --timeout 330s --metrics-brief

# Mixed CPU and bounded memory pressure.
stress-ng --cpu 2 --vm 1 --vm-bytes 256M --timeout 330s --metrics-brief
```

Record meaningful workload output such as the `stress-ng` metrics in a separate experiment log. This code does not infer workload throughput from diagnosis or resource samples.

## eBPF plus diagnosis condition

Start the existing pinner using its documented project procedure in a separate terminal. Launch the workload, run the current periodic diagnosis loop in benchmark mode, and run resource collection in a third process. The example uses a 1-second diagnosis interval; omit `--interval-s` to use the existing configured interval (five seconds by default), or pass `2` or `5`.

```bash
cd /path/to/PerformanceBottleneckDiagnosis
RUN_ID=cpu-medium-monitored-r1
PAIR_ID=cpu-medium-r1
WORKLOAD_COMMAND='stress-ng --cpu 2 --cpu-method matrixprod --timeout 330s --metrics-brief'
stress-ng --cpu 2 --cpu-method matrixprod --timeout 330s --metrics-brief &
WORKLOAD_PID=$!

sudo -E "$(command -v python)" -m src.realtime.runtime_benchmark run \
  --run-id "$RUN_ID" --comparison-id "$PAIR_ID" --repetition-index 1 \
  --workload-type cpu --intensity medium --workload-workers 2 \
  --workload-duration-s 330 --workload-command "$WORKLOAD_COMMAND" \
  --interval-s 1 --warmup-s 30 --duration-s 300 \
  > "src/realtime/runtime_benchmark_results/${RUN_ID}.log" 2>&1 &
MONITOR_PID=$!

python -m src.realtime.runtime_benchmark sample-resources \
  --run-id "${RUN_ID}-resources" --parent-run-id "$RUN_ID" \
  --comparison-id "$PAIR_ID" --repetition-index 1 \
  --condition ebpf_plus_diagnosis --workload-type cpu --intensity medium --workload-workers 2 \
  --workload-duration-s 330 --workload-command "$WORKLOAD_COMMAND" \
  --interval-s 1 --warmup-s 30 --duration-s 300 --sample-interval-s 1 \
  --workload-pid "$WORKLOAD_PID" --monitor-pid "$MONITOR_PID"

wait "$MONITOR_PID"
wait "$WORKLOAD_PID"
```

For each repetition, create a distinct `run-id`; use the same `comparison-id` and repetition index for the workload-only and monitored runs. Use at least five matched repetitions where practical. Keep the workload command and arguments identical between conditions. The resource sampler process is present in both conditions; its sampling interval and command should also be held constant. If `/proc/<pid>/stat` is not readable from the unprivileged sampler, use the host's approved permissions mechanism and ensure result-file ownership still permits subsequent analysis.

Only the `workload_only` and `ebpf_plus_diagnosis` conditions are implemented. Do not label a run "instrumentation only" unless you have independently established a valid way to run the existing eBPF instrumentation without userspace diagnosis.

## Commands and outputs

`run` supports 1-, 2-, and 5-second intervals while retaining the existing config value when `--interval-s` is omitted. Its duration is the measured window after warm-up; a cycle that starts before the end is allowed to finish. The current post-cycle wait remains the scheduling policy.

After collecting real runs:

```bash
cd /path/to/PerformanceBottleneckDiagnosis
python -m src.realtime.summarize_runtime_benchmark \
  --results-dir src/realtime/runtime_benchmark_results
python -m src.realtime.plot_runtime_benchmark \
  --results-dir src/realtime/runtime_benchmark_results
```

Artifacts:

- `cycle_metrics.csv`: one successful or failed row for each attempted diagnosis cycle, including warm-up rows.
- `system_resource_metrics.csv`: timestamped host and selected-process samples. The baseline process should pass `--workload-pid`; monitored runs should pass both process IDs.
- `run_<run_id>.json`: run configuration, environment facts, artifact hashes, and completion status.
- `run_summary.csv`: one row per recorded run, including incomplete/failed status and per-run cycle statistics.
- `analysis/group_summary.csv`: workload/condition summaries with both pooled cycle statistics and across-run variability.
- `analysis/overhead_summary.csv`: paired host CPU/memory and workload CPU deltas where matching resource samples exist.
- `analysis/overhead_group_summary.csv`: mean and between-repetition standard deviation for paired overhead metrics.
- `analysis/benchmark_report.md`: generated methodology, run status, actual available results, and explicit limitations; it reports no results when none have been collected.
- `resource_sampler_status` in `run_summary.csv`: completion state of the separate resource sampler attached with `--parent-run-id`.
- `analysis/figures/*.png` and `*.pdf`: figures generated only when the relevant real CSV measurements exist.

`run_summary.csv` is generated by the summary command, not pre-populated with example results. Keep any software-test fixtures in temporary directories; do not place them in the experiment results directory.

## Interpretation and limitations

Report diagnosis-path latency, full cycle duration, observed start-to-start interval, and resource overhead as separate measurements. Neither inference time nor cycle cadence measures bottleneck detection delay. No trustworthy independent bottleneck-onset timestamp or process-ground-truth association is implemented here, so detection delay is not estimated.

The inspected reader does not clear maps or compute interval deltas. Repeated model inputs may therefore reflect overlapping/cumulative map state. Map definitions and pinner behavior must be verified against their source before claiming that cycles represent fresh windows. A lower cycle latency alone is not evidence that monitoring has low overhead; use matched workload-only and monitored repetitions and report the process and host measurements.
