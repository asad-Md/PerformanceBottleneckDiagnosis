# Realtime diagnosis runtime benchmark

This benchmark measures the `Predictor` loop with the v7.2 SHAP-reduced classifier and matching attribution MLP. It does not load eBPF programs or start workloads; it uses the predictor's map snapshot/reset behavior and polling schedule. Run it manually on the Linux host where the realtime collector's pinned maps and model artifacts are available.

Results are written to `src/realtime/runtime_benchmark_results/` by default. Run the analysis notebook with `src/realtime/` as its working directory so its default `runtime_benchmark_results/` path resolves to that folder.

## What the measurements mean

- `cycle_start_ns` and `cycle_end_ns` use Python's monotonic `perf_counter_ns()` clock.
- The configured polling interval targets the start-to-start cadence. A cycle that overruns starts its successor immediately; otherwise the predictor waits only for the remaining interval.
- `map_read_ms` covers the four sequential pinned-map dumps and immediate deletion of the captured entries. Each dump starts a separate `bpftool` process and parses the full map as JSON, which can dominate this stage. Map deletion is not atomic with kernel-side updates; an update racing with deletion can be lost.
- `aggregation_ms` measures reconstruction and merging of map rows by `(pid, cpu)`. `feature_ms` separately measures feature engineering and model-column selection.
- `detection_latency_ms` covers map I/O through diagnosis and any eligible attribution inference. `rf_inference_ms` measures the active classifier's scaling, `predict()`, and optional `predict_proba()` calls together. `mlp_inference_ms` measures attribution feature selection, scaling, and MLP inference; it is empty when no row qualifies. `output_ms` measures diagnostic rendering and console output. `total_cycle_ms` includes those stages; CSV queue writes happen after `cycle_end_ns`, outside that cycle duration.
- The configured polling interval (default 5 seconds) is a target interval between cycle starts. If the work in a cycle takes longer than the interval, the next cycle starts immediately. Attribution timing is only populated in benchmark mode and when a row qualifies.
- Attribution runs only in benchmark mode, for rows predicted as bottlenecks with classifier confidence strictly above 80%. Its results are recorded in `attribution_results_json` in `cycle_metrics.csv`; each JSON array entry includes `pid`, `cpu`, `comm`, labels predicted at the notebook's 0.50 sigmoid threshold, and probabilities for all six attribution targets. No attribution is added to the live console output.
- `aggregation_ms` measures reconstruction and merging of rows by `(pid, cpu)`; `feature_ms` measures the existing feature engineering and model-column selection. No transformations, feature order, missing-value behavior, or thresholds are changed.
- `deadline_missed` is empty because the benchmark does not evaluate a separate cycle deadline. The polling interval is a cadence target, not a guarantee that every cycle will finish before its next start.
- `actual_cycle_interval_ms` is the interval between consecutive cycle starts, including a prior warm-up cycle where applicable. The first cycle has no preceding interval.
- Each cycle snapshots the pinned maps and then deletes the captured entries. This is not atomic with kernel-side updates, so updates that race with deletion can be lost; the snapshots are intended as short windows, not guaranteed non-overlapping measurements.
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

The benchmark also requires PyTorch, the `baseline_original_best.pt` checkpoint, the v2 notebook-generated attribution feature/target column JSON files, and the matching `feature_scaler.pkl` saved by `bottleneck_attrition_analysis_v2.ipynb`. The checkpoint and column files default to `src/attributionAnalysis/pipeline_comparison_artifacts/`; the scaler is expected at the same location. The notebook trains the attribution MLP on the same ordered 20 inputs as the SHAP-reduced classifier. The classifier, feature-list, and threshold defaults are `src/realtime/champion_model_v7_shap.pkl`, `feature_cols_v7_shap.json`, and `label_thresholds_v7_shap.json`. The default collector object is `src/realtime/realtime_monitor.bpf.o`. See [README.md](./README.md) for the BPF object and pinner build/setup steps. Missing or incompatible artifacts stop the benchmark with an explicit error rather than silently skipping either model.

The reduced realtime pinner and pinned maps need their normal setup and privileges. Start `realtime_pinner` with the matching `realtime_monitor.bpf.o` before diagnosis benchmarking; the benchmark runner itself does not call the pinner. It intentionally skips the predictor's normal shutdown unpin operation so a benchmark exit does not remove pins needed by another process. The predictor deletes captured entries after each snapshot; do not manually clear maps while the benchmark is running. Use the established pinner/session lifecycle for a fresh capture where needed.

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

Start `realtime_pinner` with `realtime_monitor.bpf.o` as described in [README.md](./README.md) before diagnosis. Launch the workload, run the periodic diagnosis loop in benchmark mode, and run resource collection in a third process. The monitored example uses a 5-second post-cycle diagnosis wait. The resource sampler continues sampling independently every 1 second.

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
  --interval-s 5 --warmup-s 30 --duration-s 300 \
  > "src/realtime/runtime_benchmark_results/${RUN_ID}.log" 2>&1 &
MONITOR_PID=$!

python -m src.realtime.runtime_benchmark sample-resources \
  --run-id "${RUN_ID}-resources" --parent-run-id "$RUN_ID" \
  --comparison-id "$PAIR_ID" --repetition-index 1 \
  --condition ebpf_plus_diagnosis --workload-type cpu --intensity medium --workload-workers 2 \
  --workload-duration-s 330 --workload-command "$WORKLOAD_COMMAND" \
  --interval-s 5 --warmup-s 30 --duration-s 300 --sample-interval-s 1 \
  --workload-pid "$WORKLOAD_PID" --monitor-pid "$MONITOR_PID"

wait "$MONITOR_PID"
wait "$WORKLOAD_PID"
```

For each repetition, create a distinct `run-id`; use the same `comparison-id` and repetition index for the workload-only and monitored runs. Use at least five matched repetitions where practical. Keep the workload command and arguments identical between conditions. The resource sampler process is present in both conditions; its sampling interval and command should also be held constant. If `/proc/<pid>/stat` is not readable from the unprivileged sampler, use the host's approved permissions mechanism and ensure result-file ownership still permits subsequent analysis.

Only the `workload_only` and `ebpf_plus_diagnosis` conditions are implemented. Do not label a run "instrumentation only" unless you have independently established a valid way to run the existing eBPF instrumentation without userspace diagnosis.

## Commands and outputs

`run` supports 1-, 2-, and 5-second intervals while retaining the existing config value when `--interval-s` is omitted. Its duration is the measured window after warm-up; a cycle that starts before the end is allowed to finish. The configured interval is a start-to-start target; if diagnosis exceeds it, the next cycle starts immediately. The one-second `--sample-interval-s` controls resource sampling independently of diagnosis scheduling.

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
  When an existing file predates attribution, its header is extended atomically and prior rows are preserved with empty attribution cells.
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

The reader deletes captured entries after each snapshot, but reset is not atomic with kernel-side writes. Updates racing with deletion can be lost, and keys introduced after a map's snapshot can remain until the next cycle. Treat these as intended short windows, not perfectly isolated measurements. A lower cycle latency alone is not evidence that monitoring has low overhead; use matched workload-only and monitored repetitions and report the process and host measurements.




###################################################################


Yes. For **one quick CPU comparison**, run the same workload twice: once without diagnosis (baseline), then once with diagnosis (monitored). Each run is about **3 minutes**, so the total is about **6 minutes**, plus setup. This is a pilot, not enough repetitions for a strong paper claim.

Use **two Linux terminal windows**. The commands below assume your Linux project folder is `/home/asad/Code/FYP/PerformanceBottleneckDiagnosis`, as shown in your instructions.

## First: prepare Terminal 2

Open Terminal 2 and enter:

```bash
cd /home/asad/Code/FYP/PerformanceBottleneckDiagnosis
source venv/bin/activate
command -v stress-ng
command -v bpftool
```

Both `command -v` checks should print a path. If `stress-ng` is missing, stop before running the workload. Also make sure the project’s usual Python dependencies are available in the activated `venv`.

## Run 1: baseline — Terminal 2 only

Do **not** start the pinner for this run. Paste this whole block into Terminal 2:

```bash
cd /home/asad/Code/FYP/PerformanceBottleneckDiagnosis
source venv/bin/activate

RUN_ID=cpu-baseline-r1
PAIR_ID=cpu-pilot-r1
WORKLOAD_COMMAND='stress-ng --cpu 2 --cpu-method matrixprod --timeout 180s --metrics-brief'

stress-ng --cpu 2 --cpu-method matrixprod --timeout 180s --metrics-brief &
WORKLOAD_PID=$!

python -m src.realtime.runtime_benchmark sample-resources \
  --run-id "$RUN_ID" --comparison-id "$PAIR_ID" --repetition-index 1 \
  --condition workload_only --workload-type cpu --intensity 2-workers --workload-workers 2 \
  --workload-duration-s 180 --workload-command "$WORKLOAD_COMMAND" \
  --warmup-s 20 --duration-s 160 --sample-interval-s 1 \
  --workload-pid "$WORKLOAD_PID"

wait "$WORKLOAD_PID"
```

This starts a two-worker CPU workload and samples the computer while it runs. The workload is capped at 180 seconds. The sampler treats the first 20 seconds as warm-up and measures the remaining 160 seconds. Keep Terminal 2 open until it finishes.

## Run 2: monitored — start the pinner, then use both terminals

**Terminal 1:** Open a second terminal, then enter:

```bash
cd /home/asad/Code/FYP/PerformanceBottleneckDiagnosis/src/realtime
sudo ./realtime_pinner --obj ./realtime_monitor.bpf.o
```

Leave Terminal 1 open with the pinner running. It’s needed for the monitored run, not for the baseline.

**Terminal 2:** After Run 1 has finished and the pinner is running in Terminal 1, paste this block into Terminal 2:

```bash
cd /home/asad/Code/FYP/PerformanceBottleneckDiagnosis
source venv/bin/activate

# Ask for sudo authentication now, while this command is in the foreground.
sudo -v

mkdir -p src/realtime/runtime_benchmark_results

RUN_ID=cpu-monitored-r1
PAIR_ID=cpu-pilot-r1
WORKLOAD_COMMAND='stress-ng --cpu 2 --cpu-method matrixprod --timeout 180s --metrics-brief'
PID_FILE="/tmp/${RUN_ID}.pid"

stress-ng --cpu 2 --cpu-method matrixprod --timeout 180s --metrics-brief &
WORKLOAD_PID=$!

# Start diagnosis with elevated permissions and save the Python process ID for sampling.
sudo -E sh -c 'echo $$ > "$1"; shift; exec "$@"' sh "$PID_FILE" \
  "$(command -v python)" -m src.realtime.runtime_benchmark run \
  --run-id "$RUN_ID" --comparison-id "$PAIR_ID" --repetition-index 1 \
  --workload-type cpu --intensity 2-workers --workload-workers 2 \
  --workload-duration-s 180 --workload-command "$WORKLOAD_COMMAND" \
  --interval-s 5 --warmup-s 20 --duration-s 160 \
  > "src/realtime/runtime_benchmark_results/${RUN_ID}.log" 2>&1 &

# Wait until the diagnosis process has written its process ID.
while [ ! -s "$PID_FILE" ]; do sleep 0.2; done
MONITOR_PID=$(cat "$PID_FILE")

# Sample workload, diagnosis process, and system resources while they run.
python -m src.realtime.runtime_benchmark sample-resources \
  --run-id "${RUN_ID}-resources" --parent-run-id "$RUN_ID" \
  --comparison-id "$PAIR_ID" --repetition-index 1 \
  --condition ebpf_plus_diagnosis --workload-type cpu --intensity 2-workers --workload-workers 2 \
  --workload-duration-s 180 --workload-command "$WORKLOAD_COMMAND" \
  --interval-s 5 --warmup-s 20 --duration-s 160 --sample-interval-s 1 \
  --workload-pid "$WORKLOAD_PID" --monitor-pid "$MONITOR_PID"

wait "$WORKLOAD_PID"
wait
```

Leave `realtime_pinner` running until the monitored block completes. Then stop it with Ctrl+C; don’t manually clear or remove maps.

## After both runs

In Terminal 2, run:

```bash
cd /home/asad/Code/FYP/PerformanceBottleneckDiagnosis
source venv/bin/activate

python -m src.realtime.summarize_runtime_benchmark \
  --results-dir src/realtime/runtime_benchmark_results

python -m src.realtime.plot_runtime_benchmark \
  --results-dir src/realtime/runtime_benchmark_results
```

Your CSVs and report will be under `src/realtime/runtime_benchmark_results/`. Open `realtime_performance_analysis.ipynb` and run it with `src/realtime` as its working directory.

This compares **one CPU workload** in two conditions. To keep this short, skip additional workload types for now. A limitation: this single matched pair is useful for checking that collection and analysis work, but not for claiming repeatable overhead measurements.

**Task completed:**

Explained the whole one-time Linux CPU pilot with separate Terminal 1 (pinner) and Terminal 2 (setup, baseline, monitored run, and analysis), including explicit commands and expected wait/order. Kept each workload bounded to 180 seconds with a 20-second warm-up, matched baseline/monitored identifiers, and clarified that a single pair is only a collection pilot—not robust publication evidence.