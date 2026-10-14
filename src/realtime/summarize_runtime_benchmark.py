from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .runtime_benchmark import RESULTS_DIR, RUN_FIELDS


GROUP_KEYS = [
    "workload_type",
    "intensity",
    "configured_interval_ms",
    "condition",
]
RESOURCE_METRICS = [
    "system_cpu_percent",
    "system_memory_used_bytes",
    "system_memory_used_percent",
    "workload_process_cpu_percent_one_core",
    "workload_process_rss_bytes",
    "monitor_process_cpu_percent_one_core",
    "monitor_process_rss_bytes",
]


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").dropna()


def _median(series: pd.Series) -> float | None:
    values = _numeric(series)
    return float(values.median()) if not values.empty else None


def _quantile(series: pd.Series, q: float) -> float | None:
    values = _numeric(series)
    return float(values.quantile(q)) if not values.empty else None


def _mean(series: pd.Series) -> float | None:
    values = _numeric(series)
    return float(values.mean()) if not values.empty else None


def _std(series: pd.Series) -> float | None:
    values = _numeric(series)
    return float(values.std(ddof=1)) if len(values) > 1 else None


def _display_value(value: Any) -> str:
    try:
        if pd.isna(value):
            return "not measured"
    except (TypeError, ValueError):
        pass
    return str(value)


def _load_run_configs(results_dir: Path) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("run_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Could not read run configuration {path}: {exc}") from exc
        if not isinstance(payload, dict) or not payload.get("run_id"):
            raise RuntimeError(f"Malformed run configuration: {path}")
        payload["_config_path"] = str(path)
        configs.append(payload)
    return configs


def _resources_by_run(resources: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    if resources.empty or "run_id" not in resources:
        return {}
    measured = resources
    if "phase" in resources.columns:
        measured = resources[resources["phase"].eq("measurement")]

    by_run: dict[str, dict[str, float | None]] = {}
    for run_id, group in measured.groupby("run_id", dropna=False):
        values: dict[str, float | None] = {}
        for metric in RESOURCE_METRICS:
            values[f"median_{metric}"] = _median(group[metric]) if metric in group else None
        values["n_resource_samples"] = int(len(group))
        by_run[str(run_id)] = values
    return by_run


def _summarize_run(
    config: dict[str, Any],
    cycles: pd.DataFrame,
    resources: dict[str, dict[str, float | None]],
    resource_sampler_status: str | None = None,
) -> dict[str, Any]:
    run_id = str(config["run_id"])
    run_cycles = cycles[cycles["run_id"].astype(str).eq(run_id)] if "run_id" in cycles else pd.DataFrame()
    measured = run_cycles
    if "phase" in run_cycles:
        measured = measured[run_cycles["phase"].eq("measurement")]
    if "cycle_status" in measured:
        successful = measured[measured["cycle_status"].eq("completed")]
    else:
        successful = measured

    warmup_count = int(run_cycles["phase"].eq("warmup").sum()) if "phase" in run_cycles else 0
    failed_count = int(run_cycles["cycle_status"].eq("failed").sum()) if "cycle_status" in run_cycles else 0
    status = str(config.get("status", "unknown"))
    if status == "running":
        status = "incomplete"

    interval_ms = config.get("configured_interval_ms")
    cycle_interval = successful["actual_cycle_interval_ms"] if "actual_cycle_interval_ms" in successful else pd.Series(dtype=float)
    missed = successful["deadline_missed"] if "deadline_missed" in successful else pd.Series(dtype=object)
    missed_values = missed.dropna().astype(str).str.lower().isin({"true", "1", "yes"})
    deadline_rate = float(missed_values.mean()) if len(missed_values) else None
    row: dict[str, Any] = {
        "run_id": run_id,
        "comparison_id": config.get("comparison_id"),
        "repetition_index": config.get("repetition_index"),
        "condition": config.get("condition"),
        "workload_type": config.get("workload_type"),
        "intensity": config.get("intensity"),
        "workload_workers": config.get("workload_workers"),
        "workload_duration_s": config.get("workload_duration_s"),
        "workload_command": config.get("workload_command"),
        "configured_interval_ms": interval_ms,
        "warmup_duration_s": config.get("warmup_duration_s"),
        "measurement_duration_s": config.get("measurement_duration_s"),
        "status": status,
        "started_at": config.get("started_at"),
        "ended_at": config.get("ended_at"),
        "n_cycles": int(len(successful)),
        "n_warmup_cycles": warmup_count,
        "n_failed_cycles": failed_count,
        "median_total_cycle_ms": _median(successful["total_cycle_ms"]) if "total_cycle_ms" in successful else None,
        "p95_total_cycle_ms": _quantile(successful["total_cycle_ms"], 0.95) if "total_cycle_ms" in successful else None,
        "mean_total_cycle_ms": _mean(successful["total_cycle_ms"]) if "total_cycle_ms" in successful else None,
        "std_total_cycle_ms": _std(successful["total_cycle_ms"]) if "total_cycle_ms" in successful else None,
        "median_detection_latency_ms": (
            _median(successful["detection_latency_ms"]) if "detection_latency_ms" in successful else None
        ),
        "p95_detection_latency_ms": (
            _quantile(successful["detection_latency_ms"], 0.95) if "detection_latency_ms" in successful else None
        ),
        "median_actual_cycle_interval_ms": _median(cycle_interval),
        "p95_actual_cycle_interval_ms": _quantile(cycle_interval, 0.95),
        "min_actual_cycle_interval_ms": (
            float(_numeric(cycle_interval).min()) if not _numeric(cycle_interval).empty else None
        ),
        "max_actual_cycle_interval_ms": (
            float(_numeric(cycle_interval).max()) if not _numeric(cycle_interval).empty else None
        ),
        "median_records_processed": (
            _median(successful["records_processed"]) if "records_processed" in successful else None
        ),
        "mean_records_processed": (
            _mean(successful["records_processed"]) if "records_processed" in successful else None
        ),
        "deadline_missed_rate": deadline_rate,
        "error": config.get("error"),
        "resource_sampler_status": resource_sampler_status,
        "config_path": config.get("_config_path"),
    }
    row.update(resources.get(run_id, {}))
    return row


def _group_summary(run_summary: pd.DataFrame, cycles: pd.DataFrame) -> pd.DataFrame:
    if run_summary.empty:
        return pd.DataFrame()

    output: list[dict[str, Any]] = []
    for keys, run_group in run_summary.groupby(GROUP_KEYS, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        group_values = dict(zip(GROUP_KEYS, keys))
        run_ids = run_group["run_id"].astype(str).tolist()
        raw = cycles[cycles["run_id"].astype(str).isin(run_ids)] if "run_id" in cycles else pd.DataFrame()
        if "phase" in raw:
            raw = raw[raw["phase"].eq("measurement")]
        if "cycle_status" in raw:
            raw = raw[raw["cycle_status"].eq("completed")]

        row: dict[str, Any] = {
            **group_values,
            "n_runs": int(run_group["run_id"].nunique()),
            "n_completed_runs": int(run_group["status"].eq("completed").sum()),
            "n_incomplete_or_failed_runs": int((~run_group["status"].eq("completed")).sum()),
            "n_cycles": int(len(raw)),
            "pooled_cycle_median_total_cycle_ms": (
                _median(raw["total_cycle_ms"]) if "total_cycle_ms" in raw else None
            ),
            "pooled_cycle_mean_total_cycle_ms": (
                _mean(raw["total_cycle_ms"]) if "total_cycle_ms" in raw else None
            ),
            "pooled_cycle_std_total_cycle_ms": (
                _std(raw["total_cycle_ms"]) if "total_cycle_ms" in raw else None
            ),
            "pooled_cycle_p95_total_cycle_ms": (
                _quantile(raw["total_cycle_ms"], 0.95) if "total_cycle_ms" in raw else None
            ),
            "median_run_median_total_cycle_ms": _median(run_group["median_total_cycle_ms"]),
            "between_run_std_median_total_cycle_ms": _std(run_group["median_total_cycle_ms"]),
            "median_run_p95_total_cycle_ms": _median(run_group["p95_total_cycle_ms"]),
            "median_detection_latency_ms": _median(run_group["median_detection_latency_ms"]),
            "median_run_detection_p95_ms": _median(run_group["p95_detection_latency_ms"]),
            "median_actual_cycle_interval_ms": (
                _median(raw["actual_cycle_interval_ms"]) if "actual_cycle_interval_ms" in raw else None
            ),
            "mean_actual_cycle_interval_ms": (
                _mean(raw["actual_cycle_interval_ms"]) if "actual_cycle_interval_ms" in raw else None
            ),
            "std_actual_cycle_interval_ms": (
                _std(raw["actual_cycle_interval_ms"]) if "actual_cycle_interval_ms" in raw else None
            ),
            "p95_actual_cycle_interval_ms": (
                _quantile(raw["actual_cycle_interval_ms"], 0.95) if "actual_cycle_interval_ms" in raw else None
            ),
            "median_records_processed": (
                _median(raw["records_processed"]) if "records_processed" in raw else None
            ),
            "deadline_missed_rate": (
                _mean(raw["deadline_missed"].dropna().astype(str).str.lower().isin({"true", "1", "yes"}).astype(float))
                if "deadline_missed" in raw and raw["deadline_missed"].notna().any()
                else None
            ),
        }
        for metric in RESOURCE_METRICS:
            column = f"median_{metric}"
            if column in run_group:
                row[f"between_run_mean_{metric}"] = _mean(run_group[column])
                row[f"between_run_std_{metric}"] = _std(run_group[column])
        output.append(row)
    return pd.DataFrame(output)


def _paired_overhead(resources: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "comparison_id",
        "workload_type",
        "intensity",
        "repetition_index",
        "workload_only_system_cpu_percent",
        "monitored_system_cpu_percent",
        "system_cpu_delta_percentage_points",
        "system_cpu_overhead_percent",
        "workload_only_system_memory_used_bytes",
        "monitored_system_memory_used_bytes",
        "system_memory_delta_bytes",
        "system_memory_overhead_percent",
        "workload_only_workload_process_cpu_percent_one_core",
        "monitored_workload_process_cpu_percent_one_core",
        "workload_process_cpu_delta_percentage_points",
        "monitor_process_cpu_percent_one_core",
        "monitor_process_rss_bytes",
    ]
    if resources.empty or not {"condition", "comparison_id", "run_id"}.issubset(resources.columns):
        return pd.DataFrame(columns=columns)
    frame = resources
    frame = frame[frame["comparison_id"].notna()]
    if "phase" in frame:
        frame = frame[frame["phase"].eq("measurement")]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    metadata_columns = ["comparison_id", "condition", "workload_type", "intensity", "repetition_index", "run_id"]
    medians: list[dict[str, Any]] = []
    for keys, group in frame.groupby(metadata_columns, dropna=False):
        entry = dict(zip(metadata_columns, keys))
        for metric in RESOURCE_METRICS:
            entry[metric] = _median(group[metric]) if metric in group else None
        medians.append(entry)
    by_run = pd.DataFrame(medians)
    paired: list[dict[str, Any]] = []
    for comparison_id, compare_group in by_run.groupby("comparison_id", dropna=False):
        base = compare_group[compare_group["condition"].eq("workload_only")]
        monitored = compare_group[compare_group["condition"].eq("ebpf_plus_diagnosis")]
        if base.empty or monitored.empty:
            continue
        for _, baseline in base.iterrows():
            candidates = monitored
            if pd.notna(baseline.get("repetition_index")):
                candidates = candidates[candidates["repetition_index"].eq(baseline["repetition_index"])]
            if baseline.get("workload_type") is not None:
                candidates = candidates[candidates["workload_type"].eq(baseline["workload_type"])]
            if baseline.get("intensity") is not None:
                candidates = candidates[candidates["intensity"].eq(baseline["intensity"])]
            if candidates.empty:
                continue
            current = candidates.iloc[0]
            base_cpu = baseline.get("system_cpu_percent")
            monitored_cpu = current.get("system_cpu_percent")
            base_mem = baseline.get("system_memory_used_bytes")
            monitored_mem = current.get("system_memory_used_bytes")
            base_work_cpu = baseline.get("workload_process_cpu_percent_one_core")
            monitored_work_cpu = current.get("workload_process_cpu_percent_one_core")
            entry = {
                "comparison_id": comparison_id,
                "workload_type": baseline.get("workload_type"),
                "intensity": baseline.get("intensity"),
                "repetition_index": baseline.get("repetition_index"),
                "workload_only_system_cpu_percent": base_cpu,
                "monitored_system_cpu_percent": monitored_cpu,
                "system_cpu_delta_percentage_points": (
                    monitored_cpu - base_cpu if pd.notna(base_cpu) and pd.notna(monitored_cpu) else None
                ),
                "system_cpu_overhead_percent": (
                    (monitored_cpu - base_cpu) / base_cpu * 100
                    if pd.notna(base_cpu) and pd.notna(monitored_cpu) and base_cpu != 0
                    else None
                ),
                "workload_only_system_memory_used_bytes": base_mem,
                "monitored_system_memory_used_bytes": monitored_mem,
                "system_memory_delta_bytes": (
                    monitored_mem - base_mem if pd.notna(base_mem) and pd.notna(monitored_mem) else None
                ),
                "system_memory_overhead_percent": (
                    (monitored_mem - base_mem) / base_mem * 100
                    if pd.notna(base_mem) and pd.notna(monitored_mem) and base_mem != 0
                    else None
                ),
                "workload_only_workload_process_cpu_percent_one_core": base_work_cpu,
                "monitored_workload_process_cpu_percent_one_core": monitored_work_cpu,
                "workload_process_cpu_delta_percentage_points": (
                    monitored_work_cpu - base_work_cpu
                    if pd.notna(base_work_cpu) and pd.notna(monitored_work_cpu)
                    else None
                ),
                "monitor_process_cpu_percent_one_core": current.get("monitor_process_cpu_percent_one_core"),
                "monitor_process_rss_bytes": current.get("monitor_process_rss_bytes"),
            }
            paired.append(entry)
    return pd.DataFrame(paired, columns=columns)


def _overhead_group_summary(overhead: pd.DataFrame) -> pd.DataFrame:
    if overhead.empty:
        return pd.DataFrame()
    keys = ["workload_type", "intensity"]
    metrics = [
        "system_cpu_delta_percentage_points",
        "system_cpu_overhead_percent",
        "system_memory_delta_bytes",
        "system_memory_overhead_percent",
        "workload_process_cpu_delta_percentage_points",
        "monitor_process_cpu_percent_one_core",
        "monitor_process_rss_bytes",
    ]
    rows: list[dict[str, Any]] = []
    for values, group in overhead.groupby(keys, dropna=False):
        values = values if isinstance(values, tuple) else (values,)
        row: dict[str, Any] = {
            **dict(zip(keys, values)),
            "n_matched_repetitions": int(group["comparison_id"].count()),
        }
        for metric in metrics:
            if metric in group:
                row[f"mean_{metric}"] = _mean(group[metric])
                row[f"std_{metric}"] = _std(group[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def _write_report(
    path: Path,
    run_summary: pd.DataFrame,
    group_summary: pd.DataFrame,
    overhead_groups: pd.DataFrame,
) -> None:
    lines = [
        "# Realtime Diagnosis Runtime Benchmark Report",
        "",
        "This report is generated from the raw cycle/resource CSV files and run JSON metadata. It contains no simulated measurements.",
        "",
        "## Methodology and metric boundaries",
        "",
        "- The predictor snapshots and deletes captured pinned-map entries. Polling targets the configured start-to-start interval; overruns start the next cycle immediately.",
        "- Detection-path latency measures map reads through the existing classifier prediction; full cycle duration also includes diagnostic formatting and console output.",
        "- Resource samples use `/proc/stat`, `/proc/meminfo`, and selected `/proc/<pid>/stat` fields. Host CPU is 0-100%; process CPU is percent of one logical CPU; memory values are resident/used bytes.",
        "- Clearing map entries is not atomic with kernel-side updates, so a racing update can be lost and snapshots are not guaranteed to be non-overlapping.",
        "- The benchmark's MLP attribution stage is conditional on qualifying rows. Empty fields are unavailable or indicate that no row qualified, not zero.",
        "- Detection delay is not measured: no independently verified bottleneck onset timestamp is associated with a diagnosed process.",
        "",
        "## Run status",
        "",
    ]
    if run_summary.empty:
        lines.append("No run metadata or measurements have been collected.")
    else:
        for status, count in run_summary["status"].value_counts(dropna=False).items():
            lines.append(f"- {status}: {int(count)} run(s)")
        lines.extend([
            "",
            "Failed, incomplete, and interrupted runs remain in `run_summary.csv`; interpret them separately from completed repetitions.",
        ])
    lines.extend(["", "## Per-condition results", ""])
    if group_summary.empty:
        lines.append("No cycle run groups with measurements are available yet; no latency results are reported.")
    else:
        for _, row in group_summary.iterrows():
            label = (
                f"{row.get('workload_type')} / {row.get('intensity')} / "
                f"{row.get('condition')} / configured interval {row.get('configured_interval_ms')} ms"
            )
            lines.extend([
                f"### {label}",
                "",
                f"- Runs: {int(row['n_runs'])}; completed: {int(row['n_completed_runs'])}; incomplete/failed: {int(row['n_incomplete_or_failed_runs'])}.",
                f"- Diagnosis cycles: {int(row['n_cycles'])}.",
                f"- Pooled cycle latency median/p95: {_display_value(row.get('pooled_cycle_median_total_cycle_ms'))} / {_display_value(row.get('pooled_cycle_p95_total_cycle_ms'))} ms.",
                f"- Median of per-run cycle medians: {_display_value(row.get('median_run_median_total_cycle_ms'))} ms; between-run standard deviation of run medians: {_display_value(row.get('between_run_std_median_total_cycle_ms'))} ms.",
                f"- Median of per-run p95 cycle latency: {_display_value(row.get('median_run_p95_total_cycle_ms'))} ms.",
                f"- Actual start-to-start interval median/p95: {_display_value(row.get('median_actual_cycle_interval_ms'))} / {_display_value(row.get('p95_actual_cycle_interval_ms'))} ms.",
                f"- Median records processed per cycle: {_display_value(row.get('median_records_processed'))}.",
                f"- Missed-deadline rate: {_display_value(row.get('deadline_missed_rate'))} (not measured means no deadline was defined).",
                "",
            ])
    lines.extend(["## Matched monitoring overhead", ""])
    if overhead_groups.empty:
        lines.append(
            "No paired workload-only and eBPF-plus-diagnosis resource samples were available. "
            "No monitoring overhead claim can be made from cycle latency alone."
        )
    else:
        for _, row in overhead_groups.iterrows():
            lines.extend([
                f"### {row.get('workload_type')} / {row.get('intensity')}",
                "",
                f"- Matched repetitions: {int(row['n_matched_repetitions'])}.",
                f"- Mean host CPU change: {_display_value(row.get('mean_system_cpu_delta_percentage_points'))} percentage points "
                f"(between-repetition SD: {_display_value(row.get('std_system_cpu_delta_percentage_points'))}).",
                f"- Mean host memory change: {_display_value(row.get('mean_system_memory_delta_bytes'))} bytes "
                f"(between-repetition SD: {_display_value(row.get('std_system_memory_delta_bytes'))}).",
                f"- Mean monitoring-process CPU: {_display_value(row.get('mean_monitor_process_cpu_percent_one_core'))}% of one logical CPU.",
                f"- Mean monitoring-process RSS: {_display_value(row.get('mean_monitor_process_rss_bytes'))} bytes.",
                "",
            ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cycles = _read_csv(results_dir / "cycle_metrics.csv")
    resources = _read_csv(results_dir / "system_resource_metrics.csv")
    if not cycles.empty:
        for column in ("total_cycle_ms", "actual_cycle_interval_ms", "detection_latency_ms", "records_processed"):
            if column in cycles:
                cycles[column] = pd.to_numeric(cycles[column], errors="coerce")
    if not resources.empty:
        for column in RESOURCE_METRICS:
            if column in resources:
                resources[column] = pd.to_numeric(resources[column], errors="coerce")

    all_configs = _load_run_configs(results_dir)
    experiment_configs = [
        config for config in all_configs if config.get("measurement_role") != "resource_sampler"
    ]
    sampler_status = {
        str(config["parent_run_id"]): str(config.get("status", "unknown"))
        for config in all_configs
        if config.get("measurement_role") == "resource_sampler" and config.get("parent_run_id")
    }
    configs = experiment_configs
    known_ids = {str(config["run_id"]) for config in configs}
    orphan_ids: set[str] = set()
    if not cycles.empty and "run_id" in cycles:
        orphan_ids.update(set(cycles["run_id"].dropna().astype(str)) - known_ids)
    if not resources.empty and "run_id" in resources:
        orphan_ids.update(set(resources["run_id"].dropna().astype(str)) - known_ids)
    for orphan_id in sorted(orphan_ids):
        configs.append({
            "run_id": orphan_id,
            "status": "unknown_orphaned",
            "condition": None,
            "workload_type": None,
            "intensity": None,
        })

    resources_by_run = _resources_by_run(resources)
    run_rows = [
        _summarize_run(config, cycles, resources_by_run, sampler_status.get(str(config["run_id"])))
        for config in configs
    ]
    run_summary = pd.DataFrame(run_rows)
    for field in RUN_FIELDS:
        if field not in run_summary:
            run_summary[field] = None
    run_summary = run_summary[RUN_FIELDS + [
        column for column in run_summary.columns
        if column not in RUN_FIELDS
    ]]
    group_summary = _group_summary(run_summary, cycles)
    overhead = _paired_overhead(resources)
    overhead_groups = _overhead_group_summary(overhead)

    results_dir.mkdir(parents=True, exist_ok=True)
    run_summary.to_csv(results_dir / "run_summary.csv", index=False)
    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    group_summary.to_csv(analysis_dir / "group_summary.csv", index=False)
    overhead.to_csv(analysis_dir / "overhead_summary.csv", index=False)
    overhead_groups.to_csv(analysis_dir / "overhead_group_summary.csv", index=False)
    _write_report(
        analysis_dir / "benchmark_report.md",
        run_summary,
        group_summary,
        overhead_groups,
    )
    return run_summary, group_summary, overhead


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize recorded realtime benchmark CSVs without simulating results.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    run_summary, group_summary, overhead = summarize(args.results_dir)
    print(f"Runs summarized: {len(run_summary)}")
    print(f"Condition groups summarized: {len(group_summary)}")
    print(f"Matched baseline/monitored comparisons: {len(overhead)}")
    print(f"Run summary: {args.results_dir / 'run_summary.csv'}")
    print(f"Analysis tables: {args.results_dir / 'analysis'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
