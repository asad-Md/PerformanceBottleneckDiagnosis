from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .runtime_benchmark import RESULTS_DIR


def _load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _save(fig: object, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")


def plot_latency(cycles: pd.DataFrame, output_dir: Path) -> bool:
    if cycles.empty or not {"total_cycle_ms", "detection_latency_ms"}.issubset(cycles.columns):
        print("Skipping latency figure: no cycle latency measurements are available.")
        return False
    frame = cycles.copy()
    if "phase" in frame:
        frame = frame[frame["phase"].eq("measurement")]
    if "cycle_status" in frame:
        frame = frame[frame["cycle_status"].eq("completed")]
    frame["total_cycle_ms"] = pd.to_numeric(frame["total_cycle_ms"], errors="coerce")
    frame["detection_latency_ms"] = pd.to_numeric(frame["detection_latency_ms"], errors="coerce")
    frame = frame.dropna(subset=["total_cycle_ms", "detection_latency_ms"])
    if frame.empty:
        print("Skipping latency figure: no completed measurement cycles have both latency values.")
        return False

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    group_columns = [
        column for column in ("workload_type", "intensity", "condition", "configured_interval_ms")
        if column in frame
    ]
    if not group_columns:
        frame["plot_group"] = "recorded run"
        group_columns = ["plot_group"]
    grouped = list(frame.groupby(group_columns, dropna=False, sort=True))
    centers = list(range(len(grouped)))
    width = 0.18
    detection_values: list[list[float]] = []
    total_values: list[list[float]] = []
    labels: list[str] = []
    for keys, group in grouped:
        keys = keys if isinstance(keys, tuple) else (keys,)
        labels.append("\n".join(f"{key}={value}" for key, value in zip(group_columns, keys)))
        detection_values.append(group["detection_latency_ms"].dropna().tolist())
        total_values.append(group["total_cycle_ms"].dropna().tolist())

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    det_positions = [center - width for center in centers]
    total_positions = [center + width for center in centers]
    ax.boxplot(detection_values, positions=det_positions, widths=width * 0.8, patch_artist=True,
               boxprops={"facecolor": "#4C78A8"}, medianprops={"color": "black"})
    ax.boxplot(total_values, positions=total_positions, widths=width * 0.8, patch_artist=True,
               boxprops={"facecolor": "#F58518"}, medianprops={"color": "black"})
    ax.set_xticks(centers)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Realtime Diagnosis Latency")
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend(
        handles=[
            Patch(facecolor="#4C78A8", label="Detection path (maps through prediction)"),
            Patch(facecolor="#F58518", label="Full cycle (includes diagnostic output)"),
        ],
        fontsize=7,
        loc="best",
    )
    fig.tight_layout()
    _save(fig, output_dir, "diagnosis_latency")
    plt.close(fig)
    return True


def plot_intervals(cycles: pd.DataFrame, output_dir: Path) -> bool:
    if cycles.empty or not {"actual_cycle_interval_ms", "configured_interval_ms"}.issubset(cycles.columns):
        print("Skipping interval figure: configured or actual interval data is unavailable.")
        return False
    frame = cycles.copy()
    if "phase" in frame:
        frame = frame[frame["phase"].eq("measurement")]
    frame["actual_cycle_interval_ms"] = pd.to_numeric(frame["actual_cycle_interval_ms"], errors="coerce")
    frame["configured_interval_ms"] = pd.to_numeric(frame["configured_interval_ms"], errors="coerce")
    frame = frame.dropna(subset=["actual_cycle_interval_ms", "configured_interval_ms"])
    if frame.empty:
        print("Skipping interval figure: no cycles have both configured and observed intervals.")
        return False

    import matplotlib.pyplot as plt

    group_columns = [
        column for column in ("workload_type", "intensity", "condition", "configured_interval_ms")
        if column in frame
    ]
    grouped = list(frame.groupby(group_columns, dropna=False, sort=True))
    values: list[list[float]] = []
    targets: list[float] = []
    labels: list[str] = []
    for keys, group in grouped:
        keys = keys if isinstance(keys, tuple) else (keys,)
        labels.append("\n".join(f"{key}={value}" for key, value in zip(group_columns, keys)))
        values.append(group["actual_cycle_interval_ms"].dropna().tolist())
        targets.append(float(group["configured_interval_ms"].median()))

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    positions = list(range(len(grouped)))
    ax.boxplot(values, positions=positions, widths=0.55, patch_artist=True,
               boxprops={"facecolor": "#72B7B2"}, medianprops={"color": "black"})
    ax.scatter(positions, targets, color="#E45756", marker="D", s=20, label="Configured interval")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("Start-to-start interval (ms)")
    ax.set_title("Configured and Observed Monitoring Interval")
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, output_dir, "monitoring_interval")
    plt.close(fig)
    return True


def plot_overhead(overhead: pd.DataFrame, output_dir: Path) -> bool:
    required = {
        "system_cpu_delta_percentage_points",
        "monitor_process_rss_bytes",
        "workload_type",
        "intensity",
    }
    if overhead.empty or not required.issubset(overhead.columns):
        print("Skipping overhead figure: no matched resource measurements are available.")
        return False
    frame = overhead.copy()
    frame["system_cpu_delta_percentage_points"] = pd.to_numeric(
        frame["system_cpu_delta_percentage_points"], errors="coerce"
    )
    frame["monitor_process_rss_bytes"] = pd.to_numeric(frame["monitor_process_rss_bytes"], errors="coerce")
    frame["workload_group"] = frame["workload_type"].astype(str) + "\n" + frame["intensity"].astype(str)
    groups = list(frame.groupby("workload_group", sort=True))
    if not any(group["system_cpu_delta_percentage_points"].notna().any() for _, group in groups):
        print("Skipping overhead figure: paired system CPU measurements are unavailable.")
        return False

    import matplotlib.pyplot as plt

    labels = [name for name, _ in groups]
    cpu_means = [
        pd.to_numeric(group["system_cpu_delta_percentage_points"], errors="coerce").mean()
        for _, group in groups
    ]
    cpu_errors = [
        pd.to_numeric(group["system_cpu_delta_percentage_points"], errors="coerce").std(ddof=1)
        for _, group in groups
    ]
    rss_means = [
        pd.to_numeric(group["monitor_process_rss_bytes"], errors="coerce").mean() / (1024**2)
        for _, group in groups
    ]
    rss_errors = [
        pd.to_numeric(group["monitor_process_rss_bytes"], errors="coerce").std(ddof=1) / (1024**2)
        for _, group in groups
    ]

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.6))
    positions = list(range(len(groups)))
    axes[0].bar(positions, cpu_means, yerr=cpu_errors, color="#4C78A8", capsize=3)
    axes[0].axhline(0, color="black", linewidth=0.7)
    axes[0].set_ylabel("Host CPU change (percentage points)")
    axes[0].set_title("Matched workload-only vs monitored")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[1].bar(positions, rss_means, yerr=rss_errors, color="#F58518", capsize=3)
    axes[1].set_ylabel("Diagnosis process RSS (MiB)")
    axes[1].set_title("Monitoring process memory")
    axes[1].grid(True, axis="y", alpha=0.25)
    for axis in axes:
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
    fig.tight_layout()
    _save(fig, output_dir, "monitoring_overhead")
    plt.close(fig)
    return True


def generate_plots(results_dir: Path) -> list[Path]:
    cycles = _load(results_dir / "cycle_metrics.csv")
    overhead = _load(results_dir / "analysis" / "overhead_summary.csv")
    figures_dir = results_dir / "analysis" / "figures"
    generated: list[Path] = []
    for stem, function, data in (
        ("diagnosis_latency", plot_latency, cycles),
        ("monitoring_interval", plot_intervals, cycles),
        ("monitoring_overhead", plot_overhead, overhead),
    ):
        if function(data, figures_dir):
            generated.extend((figures_dir / f"{stem}.png", figures_dir / f"{stem}.pdf"))
    return generated


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot actual runtime benchmark CSV measurements.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    generated = generate_plots(args.results_dir)
    if generated:
        for path in generated:
            print(path)
    else:
        print("No figures generated; collect real benchmark measurements first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
