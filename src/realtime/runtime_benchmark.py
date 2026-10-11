from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import queue
import re
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from .config import RealtimeConfig, load_config
    from .predictor import Predictor
except ImportError:  # pragma: no cover - direct-script fallback
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from realtime.config import RealtimeConfig, load_config
    from realtime.predictor import Predictor


RESULTS_DIR = Path(__file__).resolve().parent / "runtime_benchmark_results"
CYCLE_FIELDS = [
    "run_id",
    "comparison_id",
    "repetition_index",
    "condition",
    "workload_type",
    "intensity",
    "workload_workers",
    "cycle_id",
    "phase",
    "cycle_start_ns",
    "cycle_end_ns",
    "map_read_ms",
    "aggregation_ms",
    "feature_ms",
    "rf_inference_ms",
    "classifier_inference_ms",
    "mlp_inference_ms",
    "model_class",
    "output_ms",
    "detection_latency_ms",
    "total_cycle_ms",
    "records_processed",
    "high_predictions",
    "configured_interval_ms",
    "actual_cycle_interval_ms",
    "deadline_missed",
    "cycle_status",
    "error",
]
RESOURCE_FIELDS = [
    "run_id",
    "resource_run_id",
    "comparison_id",
    "repetition_index",
    "condition",
    "workload_type",
    "intensity",
    "workload_workers",
    "timestamp_ns",
    "elapsed_s",
    "phase",
    "system_cpu_percent",
    "system_memory_used_bytes",
    "system_memory_used_percent",
    "workload_pid",
    "workload_process_cpu_percent_one_core",
    "workload_process_rss_bytes",
    "monitor_pid",
    "monitor_process_cpu_percent_one_core",
    "monitor_process_rss_bytes",
]
RUN_FIELDS = [
    "run_id",
    "comparison_id",
    "repetition_index",
    "condition",
    "workload_type",
    "intensity",
    "workload_workers",
    "workload_duration_s",
    "workload_command",
    "configured_interval_ms",
    "warmup_duration_s",
    "measurement_duration_s",
    "status",
    "started_at",
    "ended_at",
    "n_cycles",
    "n_warmup_cycles",
    "n_failed_cycles",
    "median_total_cycle_ms",
    "p95_total_cycle_ms",
    "median_actual_cycle_interval_ms",
    "p95_actual_cycle_interval_ms",
    "median_records_processed",
    "error",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_run_id(value: str | None) -> str:
    run_id = value or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    if not re.fullmatch(r"[A-Za-z0-9._-]+", run_id):
        raise ValueError("run_id may contain only letters, digits, dot, underscore, and hyphen")
    return run_id


def _read_first_line(path: Path, prefix: str) -> str | None:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith(prefix):
                    return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def _memory_info() -> dict[str, int] | None:
    try:
        values: dict[str, int] = {}
        with Path("/proc/meminfo").open("r", encoding="utf-8") as handle:
            for line in handle:
                key, value = line.split(":", 1)
                if key in {"MemTotal", "MemAvailable"}:
                    values[key] = int(value.strip().split()[0]) * 1024
        if "MemTotal" not in values:
            return None
        return values
    except (OSError, ValueError, IndexError):
        return None


def _file_sha256(path: str | None) -> str | None:
    if not path:
        return None
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def capture_environment(config: RealtimeConfig, workload_command: str | None) -> dict[str, Any]:
    memory = _memory_info()
    os_release: dict[str, str] = {}
    try:
        for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                os_release[key] = value.strip().strip('"')
    except OSError:
        pass

    cpu_model = _read_first_line(Path("/proc/cpuinfo"), "model name")
    if cpu_model is None:
        cpu_model = _read_first_line(Path("/proc/cpuinfo"), "Hardware")

    return {
        "captured_at": _now_iso(),
        "platform": platform.platform(),
        "linux_distribution": os_release.get("PRETTY_NAME"),
        "kernel_version": platform.release(),
        "cpu_model": cpu_model,
        "logical_cpu_count": os.cpu_count(),
        "ram_total_bytes": memory.get("MemTotal") if memory else None,
        "python_version": platform.python_version(),
        "effective_uid": os.geteuid() if hasattr(os, "geteuid") else None,
        "libraries": {
            name: _package_version(name)
            for name in ("numpy", "pandas", "scikit-learn", "joblib", "matplotlib")
        },
        "bpf_map_paths": {
            "sched": config.sched_map_path,
            "mem": config.mem_map_path,
            "syscall": config.syscall_map_path,
            "lock": config.lock_map_path,
        },
        "model_filename": Path(config.model_path).name if config.model_path else None,
        "model_sha256": _file_sha256(config.model_path),
        "feature_columns_filename": Path(config.feature_cols_path).name if config.feature_cols_path else None,
        "feature_columns_sha256": _file_sha256(config.feature_cols_path),
        "workload_command": workload_command,
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


class AsyncCsvWriter:
    """Write cycle rows off the diagnosis thread; fail explicitly if the queue cannot keep up."""

    _STOP = object()

    def __init__(self, path: Path, fields: list[str], queue_size: int = 256) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not path.exists() or path.stat().st_size == 0
        self._handle = path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=fields, extrasaction="ignore")
        if is_new:
            self._writer.writeheader()
            self._handle.flush()
        self._queue: queue.Queue[object] = queue.Queue(maxsize=queue_size)
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._write_rows, name="cycle-csv-writer", daemon=True)
        self._thread.start()

    def submit(self, row: dict[str, Any]) -> None:
        self._raise_writer_error()
        try:
            self._queue.put(row, timeout=1)
        except queue.Full as exc:
            self._raise_writer_error()
            raise RuntimeError("Cycle CSV writer fell behind; refusing to drop benchmark rows") from exc
        self._raise_writer_error()

    def _write_rows(self) -> None:
        try:
            while True:
                item = self._queue.get()
                if item is self._STOP:
                    return
                self._writer.writerow(item)  # type: ignore[arg-type]
                self._handle.flush()
        except BaseException as exc:
            self._error = exc

    def _raise_writer_error(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"Cycle CSV writer failed: {self._error}") from self._error

    def close(self) -> None:
        if self._thread.is_alive():
            try:
                self._queue.put(self._STOP, timeout=1)
            except queue.Full as exc:
                self._raise_writer_error()
                raise RuntimeError("Cycle CSV writer could not be drained safely") from exc
            self._thread.join()
        self._handle.close()
        self._raise_writer_error()


def _append_csv_row(path: Path, fields: list[str], row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def _common_run_metadata(args: argparse.Namespace, condition: str) -> dict[str, Any]:
    return {
        "run_id": _safe_run_id(args.run_id),
        "comparison_id": args.comparison_id,
        "repetition_index": args.repetition_index,
        "condition": condition,
        "workload_type": args.workload_type,
        "intensity": args.intensity,
        "workload_workers": args.workload_workers,
        "workload_duration_s": args.workload_duration_s,
        "workload_command": args.workload_command,
        "parent_run_id": args.parent_run_id,
    }


def run_diagnosis(args: argparse.Namespace) -> int:
    config = load_config()
    if args.interval_s is not None:
        config.polling_interval_s = args.interval_s
    if config.polling_interval_s <= 0:
        raise ValueError("The configured polling interval must be positive")

    metadata = _common_run_metadata(args, "ebpf_plus_diagnosis")
    run_id = metadata["run_id"]
    output_dir: Path = args.output_dir
    run_path = output_dir / f"run_{run_id}.json"
    if run_path.exists():
        raise FileExistsError(f"Run metadata already exists: {run_path}")

    started_at = _now_iso()
    run_config: dict[str, Any] = {
        **metadata,
        "measurement_role": "experiment",
        "status": "running",
        "started_at": started_at,
        "ended_at": None,
        "configured_interval_ms": config.polling_interval_s * 1000,
        "warmup_duration_s": args.warmup_s,
        "measurement_duration_s": args.duration_s,
        "monitor_pid": os.getpid(),
        "cycle_timing_convention": (
            "cycle timestamps use perf_counter_ns; total_cycle_ms ends after console output and excludes "
            "asynchronous CSV writing; actual_cycle_interval_ms is consecutive cycle starts"
        ),
        "map_snapshot_convention": (
            "Each cycle dumps current pinned maps without clearing or differencing them; map values may "
            "overlap or be cumulative."
        ),
        "deadline_policy": "No cycle deadline is defined; deadline_missed is null.",
        "environment": capture_environment(config, args.workload_command),
    }
    _write_json_atomic(run_path, run_config)

    cycle_writer: AsyncCsvWriter | None = None
    status = "completed"
    error_text: str | None = None
    try:
        cycle_writer = AsyncCsvWriter(output_dir / "cycle_metrics.csv", CYCLE_FIELDS)
        predictor = Predictor(config, benchmark_mode=True, cleanup_maps_on_exit=False)
        run_config["active_model_class"] = type(predictor.model).__name__
        _write_json_atomic(run_path, run_config)

        def record_cycle(measurement: dict[str, object]) -> None:
            row = {**metadata, **measurement}
            row["configured_interval_ms"] = config.polling_interval_s * 1000
            model_class = type(predictor.model).__name__
            row["model_class"] = model_class
            if "randomforest" not in model_class.lower():
                row["rf_inference_ms"] = None
            cycle_writer.submit(row)

        predictor.run_benchmark(
            record_cycle,
            duration_s=args.duration_s,
            warmup_s=args.warmup_s,
        )
    except KeyboardInterrupt:
        status = "interrupted"
        print(f"\n[benchmark] interrupted; completed cycle rows are retained for run {run_id}")
    except Exception as exc:
        status = "failed"
        error_text = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if cycle_writer is not None:
            try:
                cycle_writer.close()
            except Exception as exc:
                status = "failed"
                error_text = error_text or f"{type(exc).__name__}: {exc}"
                raise
            finally:
                run_config["status"] = status
                run_config["ended_at"] = _now_iso()
                run_config["error"] = error_text
                _write_json_atomic(run_path, run_config)
        else:
            run_config["status"] = status
            run_config["ended_at"] = _now_iso()
            run_config["error"] = error_text
            _write_json_atomic(run_path, run_config)

    print(f"[benchmark] status={status} run_id={run_id}")
    print(f"[benchmark] cycle metrics: {output_dir / 'cycle_metrics.csv'}")
    print(f"[benchmark] run configuration: {run_path}")
    return 130 if status == "interrupted" else 0


def _read_system_cpu() -> tuple[int, int]:
    with Path("/proc/stat").open("r", encoding="utf-8") as handle:
        fields = handle.readline().split()
    if not fields or fields[0] != "cpu":
        raise RuntimeError("Could not parse aggregate CPU counters from /proc/stat")
    counters = [int(item) for item in fields[1:]]
    if len(counters) < 5:
        raise RuntimeError("Insufficient aggregate CPU counters in /proc/stat")
    return sum(counters), counters[3] + counters[4]


def _read_process_counters(pid: int) -> tuple[int, int] | None:
    try:
        content = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    closing_paren = content.rfind(")")
    if closing_paren < 0:
        raise RuntimeError(f"Malformed process stat file for PID {pid}")
    fields = content[closing_paren + 2 :].split()
    if len(fields) <= 21:
        raise RuntimeError(f"Incomplete process stat file for PID {pid}")
    cpu_ticks = int(fields[11]) + int(fields[12])
    rss_pages = int(fields[21])
    return cpu_ticks, rss_pages * os.sysconf("SC_PAGE_SIZE")


def _cpu_percent(previous: tuple[int, int], current: tuple[int, int]) -> float | None:
    delta_total = current[0] - previous[0]
    if delta_total <= 0:
        return None
    delta_idle = current[1] - previous[1]
    return max(0.0, min(100.0, (delta_total - delta_idle) / delta_total * 100.0))


def _process_cpu_percent(
    previous_ticks: int | None,
    current_ticks: int | None,
    elapsed_s: float,
) -> float | None:
    if previous_ticks is None or current_ticks is None or elapsed_s <= 0:
        return None
    return max(0.0, (current_ticks - previous_ticks) / os.sysconf("SC_CLK_TCK") / elapsed_s * 100.0)


def sample_resources(args: argparse.Namespace) -> int:
    if not Path("/proc/stat").exists() or not Path("/proc/meminfo").exists():
        raise RuntimeError("Resource sampling requires Linux /proc/stat and /proc/meminfo")
    if args.workload_pid is None and args.monitor_pid is None:
        raise ValueError("Provide --workload-pid, --monitor-pid, or both")

    metadata = _common_run_metadata(args, args.condition)
    run_id = metadata["run_id"]
    measurement_run_id = args.parent_run_id or run_id
    output_dir: Path = args.output_dir
    run_path = output_dir / f"run_{run_id}.json"
    if run_path.exists():
        raise FileExistsError(f"Run metadata already exists: {run_path}")

    started_at = _now_iso()
    start_ns = time.perf_counter_ns()
    warmup_ns = int(args.warmup_s * 1_000_000_000)
    end_ns = start_ns + warmup_ns + int(args.duration_s * 1_000_000_000)
    run_config: dict[str, Any] = {
        **metadata,
        "measurement_role": "resource_sampler" if args.parent_run_id else "experiment",
        "status": "running",
        "started_at": started_at,
        "ended_at": None,
        "configured_interval_ms": args.interval_s * 1000 if args.interval_s else None,
        "warmup_duration_s": args.warmup_s,
        "measurement_duration_s": args.duration_s,
        "workload_pid": args.workload_pid,
        "monitor_pid": args.monitor_pid,
        "resource_sampling_interval_s": args.sample_interval_s,
        "resource_measurement_convention": (
            "system_cpu_percent is aggregate host utilization normalized to 0-100 from /proc/stat; "
            "per-process CPU may exceed 100 and is percent of one logical CPU; RSS is resident bytes; "
            "system memory is MemTotal-MemAvailable. The external sampler runs in both comparison conditions."
        ),
        "environment": capture_environment(load_config(), args.workload_command),
    }
    _write_json_atomic(run_path, run_config)

    output_path = output_dir / "system_resource_metrics.csv"
    status = "completed"
    error_text: str | None = None
    try:
        previous_system = _read_system_cpu()
        previous_workload = _read_process_counters(args.workload_pid) if args.workload_pid else None
        previous_monitor = _read_process_counters(args.monitor_pid) if args.monitor_pid else None
        previous_sample_ns = time.perf_counter_ns()
        while time.perf_counter_ns() < end_ns:
            sample_ns = time.perf_counter_ns()
            sample_elapsed_s = (sample_ns - start_ns) / 1_000_000_000
            current_system = _read_system_cpu()
            current_workload = _read_process_counters(args.workload_pid) if args.workload_pid else None
            current_monitor = _read_process_counters(args.monitor_pid) if args.monitor_pid else None
            memory = _memory_info()
            if memory is None or "MemAvailable" not in memory:
                raise RuntimeError("Could not read MemTotal and MemAvailable from /proc/meminfo")

            sample_interval_s = (sample_ns - previous_sample_ns) / 1_000_000_000
            workload_sample_ok = current_workload is not None
            monitor_sample_ok = current_monitor is not None
            row = {
                **metadata,
                "run_id": measurement_run_id,
                "resource_run_id": run_id,
                "timestamp_ns": sample_ns,
                "elapsed_s": sample_elapsed_s,
                "phase": "warmup" if sample_ns - start_ns < warmup_ns else "measurement",
                "system_cpu_percent": _cpu_percent(previous_system, current_system),
                "system_memory_used_bytes": memory["MemTotal"] - memory["MemAvailable"],
                "system_memory_used_percent": (
                    (memory["MemTotal"] - memory["MemAvailable"]) / memory["MemTotal"] * 100
                    if memory["MemTotal"]
                    else None
                ),
                "workload_pid": args.workload_pid,
                "workload_process_cpu_percent_one_core": (
                    _process_cpu_percent(
                        previous_workload[0] if previous_workload else None,
                        current_workload[0] if workload_sample_ok else None,
                        sample_interval_s,
                    )
                ),
                "workload_process_rss_bytes": current_workload[1] if workload_sample_ok else None,
                "monitor_pid": args.monitor_pid,
                "monitor_process_cpu_percent_one_core": (
                    _process_cpu_percent(
                        previous_monitor[0] if previous_monitor else None,
                        current_monitor[0] if monitor_sample_ok else None,
                        sample_interval_s,
                    )
                ),
                "monitor_process_rss_bytes": current_monitor[1] if monitor_sample_ok else None,
            }
            _append_csv_row(output_path, RESOURCE_FIELDS, row)
            previous_system = current_system
            previous_workload = current_workload
            previous_monitor = current_monitor
            previous_sample_ns = sample_ns

            remaining_s = (end_ns - time.perf_counter_ns()) / 1_000_000_000
            if remaining_s > 0:
                time.sleep(min(args.sample_interval_s, remaining_s))
    except KeyboardInterrupt:
        status = "interrupted"
        print(f"\n[benchmark] resource sampling interrupted; samples for run {run_id} are retained")
    except Exception as exc:
        status = "failed"
        error_text = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        run_config["status"] = status
        run_config["ended_at"] = _now_iso()
        run_config["error"] = error_text
        _write_json_atomic(run_path, run_config)

    print(f"[benchmark] resource sampling status={status} run_id={run_id}")
    print(f"[benchmark] resource metrics: {output_path}")
    return 130 if status == "interrupted" else 0


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id", help="Stable unique identifier; generated if omitted")
    parser.add_argument("--comparison-id", help="Shared ID for matched workload-only and monitored repetitions")
    parser.add_argument("--repetition-index", type=int)
    parser.add_argument("--workload-type", required=True, help="For example: cpu, memory, or mixed")
    parser.add_argument("--intensity", default="unspecified", help="Human-readable workload intensity/configuration")
    parser.add_argument("--workload-workers", type=int)
    parser.add_argument("--workload-duration-s", type=float)
    parser.add_argument("--workload-command", default="", help="Exact manually launched workload command")
    parser.add_argument(
        "--parent-run-id",
        help="Attach an out-of-process resource sampler to an existing diagnosis run without counting it as another repetition",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the existing realtime predictor or collect Linux resource samples; workloads are never launched."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the existing periodic predictor with per-cycle timing")
    _add_common_arguments(run_parser)
    run_parser.add_argument("--interval-s", type=float, choices=(1.0, 2.0, 5.0))
    run_parser.add_argument("--duration-s", type=float, default=300.0, help="Measured runtime after warm-up")
    run_parser.add_argument("--warmup-s", type=float, default=30.0)
    run_parser.set_defaults(handler=run_diagnosis)

    resource_parser = subparsers.add_parser("sample-resources", help="Sample Linux /proc resources out of process")
    _add_common_arguments(resource_parser)
    resource_parser.add_argument(
        "--condition",
        choices=("workload_only", "ebpf_plus_diagnosis"),
        required=True,
    )
    resource_parser.add_argument("--interval-s", type=float, choices=(1.0, 2.0, 5.0))
    resource_parser.add_argument("--duration-s", type=float, default=300.0, help="Measured runtime after warm-up")
    resource_parser.add_argument("--warmup-s", type=float, default=30.0)
    resource_parser.add_argument("--sample-interval-s", type=float, default=1.0)
    resource_parser.add_argument("--workload-pid", type=int, help="PID of the workload process to sample")
    resource_parser.add_argument("--monitor-pid", type=int, help="PID of the diagnosis process to sample")
    resource_parser.set_defaults(handler=sample_resources)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("duration_s", "warmup_s"):
        value = getattr(args, name)
        if value < 0 or (name == "duration_s" and value == 0):
            parser.error(f"--{name.replace('_', '-')} must be positive" if name == "duration_s" else "--warmup-s cannot be negative")
    if hasattr(args, "sample_interval_s") and args.sample_interval_s <= 0:
        parser.error("--sample-interval-s must be positive")
    if args.parent_run_id is not None:
        try:
            args.parent_run_id = _safe_run_id(args.parent_run_id)
        except ValueError as exc:
            parser.error(str(exc))
    try:
        return args.handler(args)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
