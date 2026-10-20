"""Offline profiler for previously captured bpftool map-dump JSON fixtures.

This command reads only the named fixture files. It never invokes bpftool,
queries the live API, or writes to the source repository unless --output is
explicitly pointed there by the caller. Model calls in the diagnostic model
benchmark can include stale captured rows; those outputs are discarded and
must not be treated as live predictions.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from src.backend import app, live_reader


MAP_NAMES = tuple(live_reader.MAPS)
MAP_FILES = {name: f"{map_name}.json" for name, map_name in live_reader.MAPS.items()}


def _summary(seconds: list[float]) -> dict[str, float] | None:
    if not seconds:
        return None
    return {
        "median_ms": round(statistics.median(seconds) * 1000, 3),
        "min_ms": round(min(seconds) * 1000, 3),
        "max_ms": round(max(seconds) * 1000, 3),
        "stdev_ms": round(statistics.pstdev(seconds) * 1000, 3),
    }


def _capture_durations(capture_dir: Path) -> dict[str, dict[str, str | float]]:
    events: dict[str, dict[str, str]] = defaultdict(dict)
    for line in (capture_dir / "timestamps.tsv").read_text(encoding="utf-8").splitlines():
        map_name, timestamp, event = line.split("\t")
        events[map_name][event] = timestamp
    durations = {}
    for name in MAP_NAMES:
        start = events[name].get("start")
        end = events[name].get("end")
        if not start or not end:
            durations[name] = {"status": "unavailable", "start_utc": start, "end_utc": end}
            continue
        start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
        durations[name] = {
            "status": "measured_from_capture_timestamps",
            "start_utc": start,
            "end_utc": end,
            "duration_ms": round((end_dt - start_dt).total_seconds() * 1000, 3),
        }
    return durations


def _freshness_recheck(rows: list[dict]) -> list[str]:
    now_ns = time.monotonic_ns()
    stale_after = float(os.environ.get("REALTIME_STALE_AFTER_SECONDS", "30"))
    states = []
    for row in rows:
        sample_ns = row.get("sample_timestamp_ns")
        age = (now_ns - sample_ns) / 1e9 if sample_ns is not None else None
        states.append("stale" if age is None or age > stale_after else "live")
    return states


def _missing_feature_rows(rows: list[dict]) -> list[list[str]]:
    return [
        sorted(set(row.get("missing_metrics", ())) | app.LIVE_RAW_METRIC_FIELDS.difference(row))
        for row in rows
    ]


def _assert_reader_response_equivalent(rows: list[dict], response: dict) -> None:
    api_rows = response.get("processes", [])
    if response.get("count") != len(rows) or len(api_rows) != len(rows):
        raise AssertionError("Reader and API response row counts differ")
    reader_ids = [(row["pid"], row["cpu"]) for row in rows]
    api_ids = [(row["pid"], row["cpu"]) for row in api_rows]
    if api_ids != reader_ids:
        raise AssertionError("Reader and API response PID/CPU identity or ordering differs")
    expected_missing = _missing_feature_rows(rows)
    for source, api_row, missing in zip(rows, api_rows, expected_missing):
        metrics = {
            key: value for key, value in source.items()
            if key not in app.LIVE_ROW_METADATA_FIELDS
        }
        if api_row.get("metrics") != metrics:
            raise AssertionError(f"Metric values differ for pid/cpu {(source['pid'], source['cpu'])}")
        if api_row.get("missing_metrics") != missing:
            raise AssertionError(f"Missing-feature semantics differ for pid/cpu {(source['pid'], source['cpu'])}")


def _assert_json_native(value, location: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AssertionError(f"Non-finite float at {location}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _assert_json_native(child, f"{location}[{index}]")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise AssertionError(f"Non-string JSON object key at {location}: {key!r}")
            _assert_json_native(child, f"{location}.{key}")
        return
    raise AssertionError(f"Non-JSON-native value at {location}: {type(value).__name__}")


def _prediction_equivalence(
    champion_rows: list[dict], champion_batch: list[dict],
    mlp_rows: list[dict], mlp_missing: list[list[str]], mlp_batch: list[dict],
) -> dict:
    champion_checked = 0
    for row, batched in zip(champion_rows, champion_batch):
        scalar = app.predict_champion(row)
        if scalar.get("available") != batched.get("available"):
            raise AssertionError("Champion scalar/batch availability differs")
        if scalar.get("available"):
            if scalar["prediction"] != batched["prediction"]:
                raise AssertionError("Champion scalar/batch class differs")
            if abs(scalar["high_probability"] - batched["high_probability"]) > 1e-6:
                raise AssertionError("Champion scalar/batch probability differs")
        champion_checked += 1

    mlp_checked = 0
    for row, missing, batched in zip(mlp_rows, mlp_missing, mlp_batch):
        scalar = app.predict_mlp(row, missing)
        if scalar.get("available") != batched.get("available"):
            raise AssertionError("MLP scalar/batch availability differs")
        if scalar.get("available"):
            if scalar["labels"] != batched["labels"]:
                raise AssertionError("MLP scalar/batch labels differ")
            for target, value in scalar["probabilities"].items():
                if abs(value - batched["probabilities"][target]) > 1e-5:
                    raise AssertionError(f"MLP scalar/batch probability differs for {target}")
        mlp_checked += 1
    return {"champion_rows_checked": champion_checked, "mlp_rows_checked": mlp_checked}


def profile(capture_dir: Path, repeats: int) -> dict:
    capture_dir = capture_dir.resolve(strict=True)
    fixture_bytes: dict[str, bytes] = {}
    fixture_entries: dict[str, int] = {}
    fixture_hashes: dict[str, str] = {}
    fixture_decode_times: dict[str, list[float]] = defaultdict(list)
    for name, filename in MAP_FILES.items():
        path = capture_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required captured map fixture is missing: {path}")
        encoded = path.read_bytes()
        fixture_bytes[name] = encoded
        fixture_hashes[name] = hashlib.sha256(encoded).hexdigest()
        decoded = json.loads(encoded)
        if not isinstance(decoded, list):
            raise ValueError(f"Captured map fixture must contain a JSON list: {path}")
        fixture_entries[name] = len(decoded)
        del decoded

    decode_times: dict[str, list[float]] = defaultdict(list)
    reader_times: list[float] = []
    route_times: list[float] = []
    route_champion_batch_times: list[float] = []
    route_mlp_batch_times: list[float] = []
    route_champion_predict_times: list[float] = []
    route_mlp_predict_times: list[float] = []
    champion_batch_times: list[float] = []
    champion_predict_times: list[float] = []
    mlp_batch_times: list[float] = []
    mlp_predict_times: list[float] = []
    response_encode_times: list[float] = []
    direct_render_times: list[float] = []
    direct_total_times: list[float] = []
    response_bytes: list[int] = []
    direct_response_bytes: list[int] = []
    missing_calc_times: list[float] = []
    freshness_calc_times: list[float] = []
    framework_encode_times: list[float] = []
    equivalence_bodies: tuple[bytes, bytes] | None = None
    phase = ["route"]

    original_dump = live_reader.dump_map
    original_read = app.read_live_processes
    original_live_payload = app._live_payload
    original_champion_batch = app.predict_champion_batch
    original_mlp_batch = app.predict_mlp_batch
    champion_model = app.champion_model
    attributor = app.attributor
    original_champion_predict = champion_model.predict_proba if champion_model else None
    original_mlp_predict = attributor.predict if attributor else None

    def fixture_dump(name: str) -> list[dict]:
        started = time.perf_counter()
        try:
            return json.loads(fixture_bytes[name])
        finally:
            decode_times[name].append(time.perf_counter() - started)

    def timed_read() -> list[dict]:
        started = time.perf_counter()
        try:
            return original_read()
        finally:
            reader_times.append(time.perf_counter() - started)

    def timed_live() -> dict:
        started = time.perf_counter()
        try:
            return original_live_payload()
        finally:
            route_times.append(time.perf_counter() - started)

    def timed_champion_batch(rows: list[dict]) -> list[dict]:
        started = time.perf_counter()
        try:
            result = original_champion_batch(rows)
            return result
        finally:
            target = route_champion_batch_times if phase[0] == "route" else champion_batch_times
            target.append(time.perf_counter() - started)

    def timed_mlp_batch(rows: list[dict], missing: list[list[str]]) -> list[dict]:
        started = time.perf_counter()
        try:
            return original_mlp_batch(rows, missing)
        finally:
            target = route_mlp_batch_times if phase[0] == "route" else mlp_batch_times
            target.append(time.perf_counter() - started)

    def timed_champion_predict(features):
        started = time.perf_counter()
        try:
            return original_champion_predict(features)
        finally:
            target = route_champion_predict_times if phase[0] == "route" else champion_predict_times
            target.append(time.perf_counter() - started)

    def timed_mlp_predict(features):
        started = time.perf_counter()
        try:
            return original_mlp_predict(features)
        finally:
            target = route_mlp_predict_times if phase[0] == "route" else mlp_predict_times
            target.append(time.perf_counter() - started)

    live_reader.dump_map = fixture_dump
    app.read_live_processes = timed_read
    app._live_payload = timed_live
    app.predict_champion_batch = timed_champion_batch
    app.predict_mlp_batch = timed_mlp_batch
    if champion_model:
        champion_model.predict_proba = timed_champion_predict
    if attributor:
        attributor.predict = timed_mlp_predict

    try:
        # One reference reader pass establishes expected output for the exact
        # captured bytes. It is excluded from reported repeated timings.
        reference_rows = original_read()
        reference_ids = [(row["pid"], row["cpu"]) for row in reference_rows]
        reference_missing = _missing_feature_rows(reference_rows)
        reference_freshness = _freshness_recheck(reference_rows)
        if reference_freshness != [row["data_status"] for row in reference_rows]:
            raise AssertionError("Freshness calculation differs from reader status")

        # Required untimed warm-up; discard its stage timings.
        phase[0] = "route"
        warmup = app._live_payload()
        _assert_reader_response_equivalent(reference_rows, warmup)
        del warmup
        for series in (
            decode_times, reader_times, route_times, route_champion_batch_times,
            route_mlp_batch_times, route_champion_predict_times, route_mlp_predict_times,
            champion_batch_times, champion_predict_times,
            mlp_batch_times, mlp_predict_times,
        ):
            if isinstance(series, dict):
                series.clear()
            else:
                series.clear()

        route_all_predictions_unavailable = True
        representative = None
        for _ in range(repeats):
            gc.collect()
            phase[0] = "route"
            response = app._live_payload()
            _assert_reader_response_equivalent(reference_rows, response)
            representative = response
            route_all_predictions_unavailable = route_all_predictions_unavailable and all(
                not row["severity_prediction"].get("available")
                and not row["attribution_prediction"].get("available")
                for row in response["processes"]
            )
            encoder_started = time.perf_counter()
            response_content = jsonable_encoder(response)
            conversion_elapsed = time.perf_counter() - encoder_started
            framework_encode_times.append(conversion_elapsed)
            body_started = time.perf_counter()
            legacy_body = JSONResponse(content=response_content).body
            legacy_render_elapsed = time.perf_counter() - body_started
            response_encode_times.append(legacy_render_elapsed)

            direct_started = time.perf_counter()
            direct_body = JSONResponse(content=response).body
            direct_elapsed = time.perf_counter() - direct_started
            direct_render_times.append(direct_elapsed)
            direct_total_times.append(direct_elapsed)
            response_bytes.append(len(legacy_body))
            direct_response_bytes.append(len(direct_body))
            if equivalence_bodies is None:
                equivalence_bodies = (legacy_body, direct_body)

        # These exact application calculations are timed separately on the
        # reader output; they overlap with the route total and are not added to it.
        for _ in range(repeats):
            started = time.perf_counter()
            missing = _missing_feature_rows(reference_rows)
            missing_calc_times.append(time.perf_counter() - started)
            if missing != reference_missing:
                raise AssertionError("Missing-feature calculation changed across fixture runs")
            started = time.perf_counter()
            freshness = _freshness_recheck(reference_rows)
            freshness_calc_times.append(time.perf_counter() - started)
            if freshness != [row["data_status"] for row in reference_rows]:
                raise AssertionError("Freshness recomputation differs from reader status")

        # Model stage timings are measured separately only on rows with actual
        # observed required inputs. Captured rows may be stale; this diagnostic
        # benchmark explicitly bypasses the route's freshness gate, discards all
        # predictions, and never presents them as live results.
        champion_indices = [
            i for i, row in enumerate(reference_rows)
            if not (app.CHAMPION_RAW_REQUIRED - set(row))
        ]
        mlp_indices = [
            i for i, row in enumerate(reference_rows)
            if not (set(attributor.feature_columns) - set(row))
            and not (set(reference_missing[i]) & set(attributor.feature_columns))
        ] if attributor else []
        champion_rows = [reference_rows[i] for i in champion_indices]
        mlp_rows = [reference_rows[i] for i in mlp_indices]
        mlp_missing = [reference_missing[i] for i in mlp_indices]
        offline_validation: dict[str, int] = {"champion_rows_checked": 0, "mlp_rows_checked": 0}
        if champion_rows or mlp_rows:
            phase[0] = "offline_model"
            champion_output = []
            mlp_output = []
            for _ in range(repeats):
                champion_output = app.predict_champion_batch(champion_rows) if champion_rows else []
                mlp_output = app.predict_mlp_batch(mlp_rows, mlp_missing) if mlp_rows else []
            sample_count = min(5, len(champion_rows), len(mlp_rows))
            if sample_count:
                offline_validation = _prediction_equivalence(
                    champion_rows[:sample_count], champion_output[:sample_count],
                    mlp_rows[:sample_count], mlp_missing[:sample_count], mlp_output[:sample_count],
                )
            elif champion_rows:
                for raw, result in zip(champion_rows[:5], champion_output[:5]):
                    scalar = app.predict_champion(raw)
                    if scalar.get("available") != result.get("available"):
                        raise AssertionError("Champion scalar/batch availability differs")
                    if scalar.get("available") and (
                        scalar["prediction"] != result["prediction"]
                        or abs(scalar["high_probability"] - result["high_probability"]) > 1e-6
                    ):
                        raise AssertionError("Champion scalar/batch output differs")
                    offline_validation["champion_rows_checked"] += 1
            elif mlp_rows:
                for raw, missing_row, result in zip(mlp_rows[:5], mlp_missing[:5], mlp_output[:5]):
                    scalar = app.predict_mlp(raw, missing_row)
                    if scalar.get("available") != result.get("available"):
                        raise AssertionError("MLP scalar/batch availability differs")
                    if scalar.get("available") and scalar["labels"] != result["labels"]:
                        raise AssertionError("MLP scalar/batch labels differ")
                    offline_validation["mlp_rows_checked"] += 1
        else:
            phase[0] = "offline_model"
    finally:
        live_reader.dump_map = original_dump
        app.read_live_processes = original_read
        app._live_payload = original_live_payload
        app.predict_champion_batch = original_champion_batch
        app.predict_mlp_batch = original_mlp_batch
        if champion_model and original_champion_predict:
            champion_model.predict_proba = original_champion_predict
        if attributor and original_mlp_predict:
            attributor.predict = original_mlp_predict

    if equivalence_bodies is None:
        raise AssertionError("No response bodies were produced for serializer comparison")
    _assert_json_native(response)
    legacy_parsed = json.loads(equivalence_bodies[0])
    direct_parsed = json.loads(equivalence_bodies[1])
    parsed_equivalent = legacy_parsed == direct_parsed
    if not parsed_equivalent:
        raise AssertionError("Legacy and direct serializers produced different parsed JSON objects")
    byte_equivalent = equivalence_bodies[0] == equivalence_bodies[1]
    if not byte_equivalent:
        raise AssertionError("Legacy and direct serializers produced different JSON bytes")
    del legacy_parsed, direct_parsed, equivalence_bodies

    # The reader's inclusive time contains JSON decoding. Subtract the measured
    # per-map loads to expose its aggregation/derivation/freshness/sort remainder.
    decode_sum_by_run = [sum(decode_times[name][i] for name in MAP_NAMES) for i in range(repeats)]
    reader_processing = [max(0.0, reader_times[i] - decode_sum_by_run[i]) for i in range(repeats)]
    response_construction = [
        max(0.0, route_times[i] - reader_times[i] - route_champion_batch_times[i] - route_mlp_batch_times[i])
        for i in range(repeats)
    ]
    count = len(reference_rows)
    fresh_count = sum(row["data_status"] == "live" for row in reference_rows)
    missing_distribution = Counter(feature for row in reference_missing for feature in row)
    missing_rows = sum(bool(row) for row in reference_missing)
    model_route_calls = {
        "champion_batch_calls": len(route_champion_batch_times),
        "mlp_batch_calls": len(route_mlp_batch_times),
        "champion_predict_proba_calls": len(route_champion_predict_times),
        "mlp_predict_calls": len(route_mlp_predict_times),
        "route_predictions_all_unavailable": route_all_predictions_unavailable,
    }
    return {
        "capture_directory": str(capture_dir),
        "capture_timestamp_intervals": _capture_durations(capture_dir),
        "capture_map_dump_durations_are_single_capture_samples": True,
        "fixture_files": {
            name: {
                "path": str(capture_dir / MAP_FILES[name]),
                "bytes": len(fixture_bytes[name]),
                "sha256": fixture_hashes[name],
                "entries": fixture_entries[name],
            }
            for name in MAP_NAMES
        },
        "total_map_entries": sum(fixture_entries.values()),
        "total_fixture_bytes": sum(map(len, fixture_bytes.values())),
        "reader_rows": count,
        "fresh_rows_at_profile_time": fresh_count,
        "stale_rows_at_profile_time": count - fresh_count,
        "rows_with_missing_metrics": missing_rows,
        "missing_feature_row_occurrences": dict(sorted(missing_distribution.items())),
        "warmup_runs": 1,
        "measured_repeats": repeats,
        "actual_bpftool_dump_durations": _capture_durations(capture_dir),
        "fixture_json_decode_by_map": {name: _summary(decode_times[name]) for name in MAP_NAMES},
        "reader_aggregation_derivation_freshness_and_sort": _summary(reader_processing),
        "freshness_recalculation_microbenchmark": _summary(freshness_calc_times),
        "missing_feature_recalculation_microbenchmark": _summary(missing_calc_times),
        "champion_eligible_rows_for_offline_model_profile": len(champion_rows),
        "champion_batch_preparation_plus_inference": _summary(champion_batch_times),
        "champion_predict_proba_only": _summary(champion_predict_times),
        "mlp_eligible_rows_for_offline_model_profile": len(mlp_rows),
        "mlp_batch_preparation_plus_inference": _summary(mlp_batch_times),
        "mlp_scaler_and_torch_predict_only": _summary(mlp_predict_times),
        "offline_model_profile_bypassed_freshness_gate": bool(champion_rows or mlp_rows),
        "offline_model_profile_prediction_outputs_discarded": True,
        "route_model_call_counts": model_route_calls,
        "response_construction_residual": _summary(response_construction),
        "python_route_total": _summary(route_times),
        "legacy_jsonable_encoder_conversion": _summary(framework_encode_times),
        "legacy_json_rendering": _summary(response_encode_times),
        "legacy_total_encoding": _summary([
            framework_encode_times[i] + response_encode_times[i]
            for i in range(repeats)
        ]),
        "direct_json_rendering": _summary(direct_render_times),
        "direct_total_encoding": _summary(direct_total_times),
        "serialized_response_bytes": {
            "median": int(statistics.median(response_bytes)),
            "min": min(response_bytes),
            "max": max(response_bytes),
        },
        "direct_serialized_response_bytes": {
            "median": int(statistics.median(direct_response_bytes)),
            "min": min(direct_response_bytes),
            "max": max(direct_response_bytes),
        },
        "validation": {
            "reader_vs_endpoint_pid_cpu_identity_and_order_match": True,
            "reader_vs_endpoint_metric_values_match": True,
            "reader_vs_endpoint_missing_feature_lists_match": True,
            "freshness_recalculation_matches_reader": True,
            "offline_scalar_vs_batch_prediction_checks": offline_validation,
            "old_and_direct_parsed_json_objects_equal": parsed_equivalent,
            "old_and_direct_json_bytes_equal": byte_equivalent,
        },
        "timing_note": "All repeated results use preloaded captured JSON bytes. JSON decoding is timed; fixture file I/O is excluded. Reader stage is inclusive except measured JSON decode. Freshness/missing microbenchmarks overlap the reader stage and are not additive.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True, help="Directory containing the four captured map JSON files and timestamps.tsv")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, help="Optional output JSON path, preferably under /tmp")
    args = parser.parse_args()
    if args.repeats < 5:
        parser.error("repeats must be at least 5")
    result = profile(args.fixtures, args.repeats)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote offline profile summary to {args.output}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
