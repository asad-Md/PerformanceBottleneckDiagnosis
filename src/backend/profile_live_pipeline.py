"""Synthetic, read-only stage profiler for the /api/live implementation.

This harness never calls bpftool, reads pinned maps, opens a server, or writes
benchmark output unless --output is explicitly supplied. Fixture values are
synthetic and must not be interpreted as kernel measurements.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from fastapi.responses import JSONResponse

from src.backend import app, live_reader


def _fixture_maps(row_count: int, fresh_fraction: float) -> dict[str, str]:
    now_ns = time.monotonic_ns()
    fresh_count = int(row_count * fresh_fraction)
    snapshots: dict[str, list[dict]] = {name: [] for name in live_reader.MAPS}
    for index in range(row_count):
        pid, cpu = 1000 + index, index % 16
        last_update = now_ns - (1_000_000 if index < fresh_count else 3_600_000_000_000)
        key = {"pid": pid, "cpu": cpu}
        values = {
            "sched": {
                "ctx_switches": index + 10, "voluntary_switches": index + 5,
                "involuntary_switches": index + 5, "cpu_migrations": index % 7,
                "total_runtime_ns": (index + 1) * 1_000_000,
                "stall_ns": (index + 1) * 100, "max_stall_ns": (index + 1) * 10,
                "latency_count": index + 1, "runq_ratio_sum": (index + 1) * 20,
                "runq_ratio_count": index + 1, "last_update_ns": last_update,
                "comm": f"synthetic-{index % 13}",
            },
            "mem": {
                "minor_faults": index + 1, "major_faults": index % 3,
                "kmalloc_count": index + 1, "kfree_count": index,
                "total_alloc_bytes": (index + 1) * 64,
                "last_update_ns": last_update,
            },
            "syscall": {
                "total_count": index + 1, "latency_sum_ns": (index + 1) * 200,
                "latency_max_ns": 200, "read_count": index + 1,
                "write_count": index + 1, "read_bytes": (index + 1) * 128,
                "write_bytes": (index + 1) * 64, "mmap_count": index % 5,
                "futex_count": index + 1, "futex_latency_sum_ns": (index + 1) * 50,
                "epoll_count": index + 1, "epoll_latency_sum_ns": (index + 1) * 70,
                "poll_count": index % 4, "error_count": index % 2,
                "last_update_ns": last_update,
            },
            "lock": {
                "mutex_contentions": index + 1, "mutex_wait_sum_ns": (index + 1) * 90,
                "mutex_wait_max_ns": 90, "rwsem_read_contentions": index + 1,
                "rwsem_read_wait_sum_ns": (index + 1) * 60,
                "rwsem_write_contentions": index + 1,
                "last_update_ns": last_update,
            },
        }
        for name, value in values.items():
            snapshots[name].append({"formatted": {"key": key, "value": value}})
    # Store encoded fixture JSON so the measured dump substitute includes the
    # production reader's JSON decoding step, without invoking bpftool.
    return {name: json.dumps(entries, separators=(",", ":")) for name, entries in snapshots.items()}


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "median_ms": round(statistics.median(values) * 1000, 3),
        "min_ms": round(min(values) * 1000, 3),
        "max_ms": round(max(values) * 1000, 3),
        "stdev_ms": round(statistics.pstdev(values) * 1000, 3),
    }


def _compare_scalar_for_sample(response: dict, rows: list[dict], limit: int = 5) -> None:
    fresh = [row for row in rows if row.get("data_status") == "live"][:limit]
    by_identity = {(row["pid"], row["cpu"]): row for row in response["processes"]}
    for raw in fresh:
        output = by_identity[(raw["pid"], raw["cpu"])]
        scalar_champion = app.predict_champion(raw)
        scalar_mlp = app.predict_mlp(raw, raw.get("missing_metrics", []))
        batched_champion = output["severity_prediction"]
        batched_mlp = output["attribution_prediction"]
        if scalar_champion.get("available") != batched_champion.get("available"):
            raise AssertionError("Champion availability differs from scalar inference")
        if scalar_champion.get("available"):
            if scalar_champion["prediction"] != batched_champion["prediction"]:
                raise AssertionError("Champion class differs from scalar inference")
            if abs(scalar_champion["high_probability"] - batched_champion["high_probability"]) > 1e-6:
                raise AssertionError("Champion probability differs from scalar inference")
        if scalar_mlp.get("available") != batched_mlp.get("available"):
            raise AssertionError("MLP availability differs from scalar inference")
        if scalar_mlp.get("available"):
            if scalar_mlp["labels"] != batched_mlp["labels"]:
                raise AssertionError("MLP labels differ from scalar inference")
            for target, score in scalar_mlp["probabilities"].items():
                difference = abs(score - batched_mlp["probabilities"][target])
                if difference > 1e-5:
                    raise AssertionError(
                        f"MLP probability differs for {target}: delta={difference:g}"
                    )


def profile(row_count: int, repeats: int, fresh_fraction: float) -> dict:
    encoded_maps = _fixture_maps(row_count, fresh_fraction)
    map_decode_seconds: dict[str, list[float]] = defaultdict(list)
    map_calls = 0
    read_seconds: list[float] = []
    champion_seconds: list[float] = []
    mlp_seconds: list[float] = []
    champion_inference_seconds: list[float] = []
    mlp_inference_seconds: list[float] = []
    route_seconds: list[float] = []
    app_missing_seconds: list[float] = []
    response_seconds: list[float] = []
    serialized_bytes: list[int] = []
    representative_rows: list[dict] = []
    representative_response: dict | None = None
    original_dump = live_reader.dump_map
    original_reader = app.read_live_processes
    original_champion_batch = app.predict_champion_batch
    original_mlp_batch = app.predict_mlp_batch
    original_champion_predict = app.champion_model.predict_proba if app.champion_model else None
    original_mlp_predict = app.attributor.predict if app.attributor else None

    def fixture_dump(name: str) -> list[dict]:
        nonlocal map_calls
        started = time.perf_counter()
        try:
            return json.loads(encoded_maps[name])
        finally:
            map_decode_seconds[name].append(time.perf_counter() - started)
            map_calls += 1

    def timed_reader() -> list[dict]:
        started = time.perf_counter()
        try:
            return original_reader()
        finally:
            read_seconds.append(time.perf_counter() - started)

    def timed_champion(rows: list[dict]) -> list[dict]:
        started = time.perf_counter()
        try:
            return original_champion_batch(rows)
        finally:
            champion_seconds.append(time.perf_counter() - started)

    def timed_mlp(rows: list[dict], missing: list[list[str]]) -> list[dict]:
        started = time.perf_counter()
        try:
            return original_mlp_batch(rows, missing)
        finally:
            mlp_seconds.append(time.perf_counter() - started)

    def timed_champion_predict(features):
        started = time.perf_counter()
        try:
            return original_champion_predict(features)
        finally:
            champion_inference_seconds.append(time.perf_counter() - started)

    def timed_mlp_predict(features):
        started = time.perf_counter()
        try:
            return original_mlp_predict(features)
        finally:
            mlp_inference_seconds.append(time.perf_counter() - started)

    live_reader.dump_map = fixture_dump
    app.read_live_processes = timed_reader
    app.predict_champion_batch = timed_champion
    app.predict_mlp_batch = timed_mlp
    if app.champion_model:
        app.champion_model.predict_proba = timed_champion_predict
    if app.attributor:
        app.attributor.predict = timed_mlp_predict

    try:
        # Untimed warm-up to initialize pandas/sklearn/torch execution paths.
        warmup_response = app.live()
        if warmup_response["count"] != row_count:
            raise AssertionError("Synthetic reader did not reconstruct the requested row count")
        # Rebuild expected reader output independently for semantic checks.
        expected_rows = original_reader()
        expected_ids = [(row["pid"], row["cpu"]) for row in expected_rows]
        map_decode_seconds.clear()
        read_seconds.clear()
        champion_seconds.clear()
        mlp_seconds.clear()
        champion_inference_seconds.clear()
        mlp_inference_seconds.clear()
        route_seconds.clear()
        response_seconds.clear()
        serialized_bytes.clear()
        map_calls = 0

        for run_index in range(repeats):
            route_started = time.perf_counter()
            response = app.live()
            route_seconds.append(time.perf_counter() - route_started)
            if response["count"] != row_count:
                raise AssertionError("API row count changed for deterministic fixture")
            actual_ids = [(row["pid"], row["cpu"]) for row in response["processes"]]
            if actual_ids != expected_ids:
                raise AssertionError("API response row identity/order differs from reader output")
            for source, output in zip(expected_rows, response["processes"]):
                expected_metrics = {
                    key: value for key, value in source.items()
                    if key not in app.LIVE_ROW_METADATA_FIELDS
                }
                expected_missing = sorted(
                    set(source.get("missing_metrics", ())) | app.LIVE_RAW_METRIC_FIELDS.difference(source)
                )
                if output["metrics"] != expected_metrics:
                    raise AssertionError("API metrics differ from reader output")
                if output["missing_metrics"] != expected_missing:
                    raise AssertionError("API missing-feature list differs from reader output")
            if run_index == 0:
                representative_rows = expected_rows
                representative_response = response

            encoded_started = time.perf_counter()
            body = JSONResponse(content=response).body
            response_seconds.append(time.perf_counter() - encoded_started)
            serialized_bytes.append(len(body))
        _compare_scalar_for_sample(representative_response, representative_rows)
        for _ in range(repeats):
            started = time.perf_counter()
            [
                sorted(
                    set(row.get("missing_metrics", ()))
                    | app.LIVE_RAW_METRIC_FIELDS.difference(row)
                )
                for row in representative_rows
            ]
            app_missing_seconds.append(time.perf_counter() - started)
    finally:
        live_reader.dump_map = original_dump
        app.read_live_processes = original_reader
        app.predict_champion_batch = original_champion_batch
        app.predict_mlp_batch = original_mlp_batch
        if app.champion_model and original_champion_predict:
            app.champion_model.predict_proba = original_champion_predict
        if app.attributor and original_mlp_predict:
            app.attributor.predict = original_mlp_predict

    # `live()` was timed through individual wrappers. Its total can be timed
    # without altering the API by a second deterministic fixture pass below.
    # The profile loop above captures all internal stage timings; app orchestration
    # and response dictionary creation are reported as the residual after those.
    # Re-run once after restoring wrappers is intentionally avoided; route total
    # is measured with a lightweight wrapper around the existing function below.
    # (See timed route invocation above: response construction is represented by
    # route time minus reader and model helper stage times.)
    if representative_response is None:
        raise AssertionError("No profile response was produced")

    map_medians = {
        name: _summary(map_decode_seconds[name][:repeats]) for name in live_reader.MAPS
    }
    measured_map_decode_total = [
        sum(map_decode_seconds[name][i] for name in live_reader.MAPS)
        for i in range(repeats)
    ]
    aggregate_and_freshness = [
        max(0.0, read_seconds[i] - measured_map_decode_total[i])
        for i in range(repeats)
    ]
    return {
        "fixture_notice": "All metrics and timestamps are deterministic synthetic fixtures, not live observations.",
        "rows_requested": row_count,
        "rows_reconstructed": len(representative_rows),
        "fresh_rows": sum(row["data_status"] == "live" for row in representative_rows),
        "stale_rows": sum(row["data_status"] != "live" for row in representative_rows),
        "map_entry_count_per_map": row_count,
        "map_fixture_json_bytes": {name: len(value.encode("utf-8")) for name, value in encoded_maps.items()},
        "runs": repeats,
        "bpftool_wall_time": "NOT MEASURED: live map access is intentionally disabled; fixture decoding is measured instead.",
        "map_fixture_json_decode": map_medians,
        "map_decode_all_maps_median_ms": round(statistics.median(
            [sum(map_decode_seconds[name][i] for name in live_reader.MAPS) for i in range(repeats)]
        ) * 1000, 3),
        "reader_aggregation_derivation_freshness_sort": _summary(aggregate_and_freshness),
        "champion_batch_total_prep_plus_inference": _summary(champion_seconds[:repeats]),
        "champion_model_predict_proba_only": _summary(champion_inference_seconds[:repeats]),
        "mlp_batch_total_prep_plus_inference": _summary(mlp_seconds[:repeats]),
        "mlp_scaler_and_torch_predict_only": _summary(mlp_inference_seconds[:repeats]),
        "app_missing_feature_calculation_isolated": _summary(app_missing_seconds),
        "route_python_total": _summary(route_seconds),
        "route_orchestration_and_row_construction_residual": _summary([
            max(0.0, route_seconds[i] - read_seconds[i] - champion_seconds[i] - mlp_seconds[i])
            for i in range(repeats)
        ]),
        "json_response_encoding": _summary(response_seconds),
        "serialized_response_bytes": {
            "median": int(statistics.median(serialized_bytes)),
            "min": min(serialized_bytes),
            "max": max(serialized_bytes),
        },
        "map_dump_calls": map_calls,
        "validation": {
            "pid_cpu_identity_and_order_match_reader": True,
            "metrics_and_missing_feature_lists_match_reader": True,
            "scalar_prediction_equivalence_checked_on_fresh_sample_rows": min(5, sum(
                row["data_status"] == "live" for row in representative_rows
            )),
        },
        "note": "Variability is population standard deviation. Fixture JSON decode excludes subprocess and kernel-map transfer costs.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--fresh-fraction", type=float, default=0.02)
    parser.add_argument("--output", type=Path, help="Optional separate JSON output file")
    args = parser.parse_args()
    if args.rows <= 0 or args.repeats < 2 or not 0 <= args.fresh_fraction <= 1:
        parser.error("rows must be positive, repeats at least 2, and fresh fraction between 0 and 1")
    result = profile(args.rows, args.repeats, args.fresh_fraction)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote profile summary to {args.output}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
