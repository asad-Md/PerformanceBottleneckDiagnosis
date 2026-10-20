import json
import logging
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

try:
    from .live_reader import ATTRIBUTION_FEATURES, read_live_processes
except ImportError:
    from live_reader import ATTRIBUTION_FEATURES, read_live_processes

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.realtime.realtime_reader import engineer_features_v5
from src.realtime.attribution_model import BottleneckAttributor
from src.realtime.config import RealtimeConfig

CHAMPION_MODEL_PATH = ROOT / "src" / "approach3" / "champion_model_v7.pkl"
CHAMPION_FEATURES_PATH = ROOT / "src" / "approach3" / "feature_cols_v7.json"
CHAMPION_THRESHOLDS_PATH = ROOT / "src" / "approach3" / "label_thresholds_v7.json"
ATTRIBUTION_DIR = ROOT / "src" / "attributionAnalysis" / "pipeline_comparison_artifacts"
ATTRIBUTION_MODEL_PATH = ATTRIBUTION_DIR / "baseline_original_best.pt"
ATTRIBUTION_SCALER_PATH = ATTRIBUTION_DIR / "feature_scaler.pkl"
ATTRIBUTION_FEATURES_PATH = ATTRIBUTION_DIR / "attribution_feature_cols.json"
ATTRIBUTION_TARGETS_PATH = ATTRIBUTION_DIR / "attribution_target_cols.json"
ATTRIBUTION_THRESHOLD = 0.5

CHAMPION_FEATURES_REQUIRED = [
    "ctx_switches", "cpu_migrations", "involuntary_switches", "voluntary_switches",
    "involuntary_ratio", "voluntary_ratio", "avg_runq_ratio", "runtime_per_switch",
    "minor_faults", "major_faults", "kmalloc_count", "kfree_count", "total_alloc_bytes",
    "alloc_pressure", "read_count", "write_count", "read_bytes", "write_bytes",
    "read_write_ratio", "io_bytes_total", "mutex_contentions", "avg_mutex_wait_ns",
    "rwsem_read_contentions", "rwsem_write_contentions", "lock_pressure", "syscall_count",
    "avg_syscall_latency_ns", "futex_count", "log_total_runtime_ns",
    "log_avg_syscall_latency_ns", "log_max_syscall_latency_ns", "log_io_bytes_total",
    "log_total_alloc_bytes",
]

attributor: BottleneckAttributor | None = None
attribution_error: str | None = None
try:
    attributor = BottleneckAttributor(RealtimeConfig(
        attribution_model_path=str(ATTRIBUTION_MODEL_PATH),
        attribution_scaler_path=str(ATTRIBUTION_SCALER_PATH),
        attribution_feature_cols_path=str(ATTRIBUTION_FEATURES_PATH),
        attribution_target_cols_path=str(ATTRIBUTION_TARGETS_PATH),
    ))
except Exception as exc:
    attribution_error = f"{type(exc).__name__}: {exc}"
    logging.exception("Baseline attribution model is unavailable")

def load_champion_artifacts(
    model_path: Path,
    features_path: Path,
    thresholds_path: Path,
):
    with open(features_path, encoding="utf-8") as f:
        feature_columns = json.load(f)
    with open(thresholds_path, encoding="utf-8") as f:
        threshold = float(json.load(f)["decision_probability_threshold"])
    if feature_columns != CHAMPION_FEATURES_REQUIRED:
        raise ValueError("Champion V7 feature metadata does not match the required 33-feature order")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Champion V7 decision probability threshold must be between 0 and 1")

    model = joblib.load(model_path)
    model_feature_names = list(getattr(model, "feature_names_in_", []))
    if model_feature_names != feature_columns:
        raise ValueError("Champion V7 feature metadata does not match the model's training order")
    if set(getattr(model, "classes_", [])) != {0, 1}:
        raise ValueError("Champion V7 must be a binary classifier with classes 0 and 1")
    return model, feature_columns, threshold


champion_model = None
CHAMPION_FEATURE_COLS = CHAMPION_FEATURES_REQUIRED
CHAMPION_THRESHOLD = 0.5
champion_error: str | None = None
try:
    champion_model, CHAMPION_FEATURE_COLS, CHAMPION_THRESHOLD = load_champion_artifacts(
        CHAMPION_MODEL_PATH,
        CHAMPION_FEATURES_PATH,
        CHAMPION_THRESHOLDS_PATH,
    )
except Exception as exc:
    champion_error = f"{type(exc).__name__}: {exc}"
    logging.exception("Champion V7 severity model is unavailable")

CHAMPION_RAW_REQUIRED = {
    "ctx_switches", "cpu_migrations", "involuntary_switches", "voluntary_switches",
    "avg_runq_ratio", "total_runtime_ns", "minor_faults", "major_faults",
    "kmalloc_count", "kfree_count", "total_alloc_bytes", "read_count", "write_count",
    "read_bytes", "write_bytes", "mutex_contentions", "avg_mutex_wait_ns",
    "rwsem_read_contentions", "rwsem_write_contentions", "syscall_count",
    "avg_syscall_latency_ns", "max_syscall_latency_ns", "futex_count",
}

LIVE_RAW_METRIC_FIELDS = ATTRIBUTION_FEATURES
LIVE_ROW_METADATA_FIELDS = {
    "pid", "cpu", "comm", "session_label",
    "sample_timestamp_ns", "sample_timestamp_clock", "sample_timestamp_status",
    "data_age_seconds", "data_status", "missing_metrics",
}
app = FastAPI(title="Performance Bottleneck Diagnosis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProcessMetrics(BaseModel):
    pid: int = 0
    cpu: float = 0
    comm: str = "unknown"
    session_label: str = "submitted_data"

    total_runtime_ns: float = 0
    ctx_switches: float = 0
    avg_runq_ratio: float = 0
    involuntary_switches: float = 0
    voluntary_switches: float = 0
    kmalloc_count: float = 0
    read_bytes: float = 0
    write_bytes: float = 0
    read_count: float = 0
    write_count: float = 0
    futex_count: float = 0
    total_alloc_bytes: float = 0
    mutex_contentions: float = 0
    rwsem_read_contentions: float = 0
    rwsem_write_contentions: float = 0
    cpu_migrations: float = 0
    minor_faults: float = 0
    syscall_count: float = 0
    avg_mutex_wait_ns: float = 0
    kfree_count: float = 0
    max_syscall_latency_ns: float = 0
    avg_syscall_latency_ns: float = 0
    major_faults: float = 0


def model_statuses() -> dict:
    return {
        "champion_v7": {
            "available": champion_model is not None,
            "status": "available" if champion_model is not None else "unavailable",
            "checkpoint": str(CHAMPION_MODEL_PATH),
            "error": champion_error,
        },
        "attribution_mlp": {
            "available": attributor is not None,
            "status": "available" if attributor is not None else "unavailable",
            "checkpoint": str(ATTRIBUTION_MODEL_PATH),
            "error": attribution_error,
        },
    }


@app.get("/api/health")
def health():
    statuses = model_statuses()
    return {
        "status": "ok" if all(model["available"] for model in statuses.values()) else "degraded",
        "model_loaded": any(model["available"] for model in statuses.values()),
        "models": statuses,
    }


def _mlp_unavailable(error: str, missing_features: list[str] | None = None) -> dict:
    result = {
        "available": False,
        "status": "unavailable",
        "model": "baseline_original_best.pt",
        "labels": None,
        "probabilities": None,
        "threshold": None,
        "error": error,
    }
    if missing_features is not None:
        result["missing_features"] = missing_features
    return result


def predict_mlp_batch(
    raw_rows: list[dict],
    missing_metrics_rows: list[list[str]] | None = None,
) -> list[dict]:
    """Run attribution on complete fresh rows in one model call."""
    if not raw_rows:
        return []

    if attributor is None:
        return [
            _mlp_unavailable(attribution_error or "Baseline attribution model did not load")
            for _ in raw_rows
        ]

    if missing_metrics_rows is None:
        missing_metrics_rows = [[] for _ in raw_rows]
    if len(missing_metrics_rows) != len(raw_rows):
        raise ValueError("missing_metrics_rows must have one entry per attribution row")

    outputs: list[dict | None] = [None] * len(raw_rows)
    eligible_indices: list[int] = []
    for index, (raw, missing_metrics) in enumerate(zip(raw_rows, missing_metrics_rows)):
        absent = sorted(
            (set(attributor.feature_columns) - set(raw))
            | set(missing_metrics).intersection(attributor.feature_columns)
        )
        if absent:
            outputs[index] = _mlp_unavailable(
                "Required live attribution features are unavailable: " + ", ".join(absent),
                absent,
            )
            continue
        try:
            values = np.asarray([raw[column] for column in attributor.feature_columns], dtype=np.float32)
        except (TypeError, ValueError, OverflowError):
            outputs[index] = {
                **_mlp_unavailable("RuntimeError: Realtime attribution features must be numeric"),
                "status": "error",
                "missing_features": [],
            }
            continue
        if not np.isfinite(values).all():
            outputs[index] = {
                **_mlp_unavailable("RuntimeError: Realtime attribution features contain NaN or infinite values"),
                "status": "error",
                "missing_features": [],
            }
            continue
        eligible_indices.append(index)

    if not eligible_indices:
        return [output for output in outputs if output is not None]

    features = pd.DataFrame(
        [{column: raw_rows[index][column] for column in attributor.feature_columns}
         for index in eligible_indices],
        columns=attributor.feature_columns,
    )
    try:
        probabilities_batch = attributor.predict(features)
        if len(probabilities_batch) != len(eligible_indices):
            raise RuntimeError("Attribution MLP returned a different number of results than input rows")
        for index, probabilities in zip(eligible_indices, probabilities_batch):
            scores = {
                name: round(float(score), 6)
                for name, score in zip(attributor.target_columns, probabilities)
            }
            selected = [name for name, score in scores.items() if score >= attributor.THRESHOLD]
            outputs[index] = {
                "available": True,
                "status": "available",
                "model": "baseline_original_best.pt",
                "labels": selected,
                "probabilities": scores,
                "threshold": attributor.THRESHOLD,
                "error": None,
                "missing_features": [],
            }
    except Exception as exc:
        logging.exception("Baseline attribution batch inference failed")
        for index in eligible_indices:
            outputs[index] = {
                **_mlp_unavailable(f"{type(exc).__name__}: {exc}"),
                "status": "error",
                "missing_features": [],
            }

    return [output for output in outputs if output is not None]


def predict_mlp(raw: dict, missing_metrics: list[str] | None = None) -> dict:
    """Scalar compatibility wrapper for submitted single-row predictions."""
    return predict_mlp_batch([raw], [missing_metrics or []])[0]


def predict_champion(raw: dict) -> dict:
    if champion_model is None:
        return {
            "available": False,
            "status": "unavailable",
            "error": champion_error or "Champion V7 model did not load",
        }
    absent = sorted(CHAMPION_RAW_REQUIRED - set(raw))
    if absent:
        return {
            "available": False,
            "status": "unavailable",
            "error": "Required Champion V7 live features are unavailable: " + ", ".join(absent),
            "missing_features": absent,
        }
    try:
        featured = engineer_features_v5(pd.DataFrame([raw]))
        feature_frame = featured.reindex(columns=CHAMPION_FEATURE_COLS)
        if not np.isfinite(feature_frame.to_numpy(dtype=float)).all():
            return {
                "available": False,
                "status": "unavailable",
                "error": "Required Champion V7 live features are non-finite.",
                "missing_features": [
                    feature for feature, value in feature_frame.iloc[0].items()
                    if not np.isfinite(float(value))
                ],
            }
        probabilities = champion_model.predict_proba(feature_frame)[0]
        classes = list(champion_model.classes_)
        high_probability = float(probabilities[classes.index(1)])
        predicted_class = int(high_probability >= CHAMPION_THRESHOLD)
        class_probabilities = {
            ("High" if int(label) == 1 else "Not-High"): round(float(probability), 6)
            for label, probability in zip(classes, probabilities)
        }
        return {
            "available": True,
            "status": "available",
            "pred_class": predicted_class,
            "prediction": "High" if predicted_class else "Not-High",
            "is_bottleneck": bool(predicted_class),
            "confidence": round(max(class_probabilities.values()) * 100, 2),
            "high_probability": round(high_probability, 6),
            "class_probabilities": class_probabilities,
            "threshold": CHAMPION_THRESHOLD,
            "zero_filled_inputs": [],
        }
    except Exception as exc:
        logging.exception("Champion V7 inference failed")
        return {
            "available": False,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def predict_champion_batch(raw_rows: list[dict]) -> list[dict]:
    """Run Champion V7 inference on a batch while preserving per-row output."""
    if not raw_rows:
        return []

    if champion_model is None:
        return [
            {
                "available": False,
                "status": "unavailable",
                "error": champion_error or "Champion V7 model did not load",
            }
            for _ in raw_rows
        ]

    outputs: list[dict | None] = [None] * len(raw_rows)
    eligible_indices: list[int] = []
    for index, row in enumerate(raw_rows):
        absent = sorted(CHAMPION_RAW_REQUIRED - set(row))
        if absent:
            outputs[index] = {
                "available": False,
                "status": "unavailable",
                "error": "Live collector does not provide required Champion V7 feature(s): " + ", ".join(absent),
                "missing_features": absent,
            }
            continue
        try:
            raw_values = np.asarray([row[column] for column in CHAMPION_RAW_REQUIRED], dtype=float)
        except (TypeError, ValueError, OverflowError):
            outputs[index] = {
                "available": False,
                "status": "unavailable",
                "error": "Required Champion V7 live inputs are not numeric.",
                "missing_features": [],
            }
            continue
        if not np.isfinite(raw_values).all():
            outputs[index] = {
                "available": False,
                "status": "unavailable",
                "error": "Required Champion V7 live inputs contain NaN or infinite values.",
                "missing_features": [],
            }
            continue
        eligible_indices.append(index)

    if not eligible_indices:
        return [output for output in outputs if output is not None]

    eligible_rows = [raw_rows[index] for index in eligible_indices]
    try:
        featured = engineer_features_v5(pd.DataFrame(eligible_rows))
        feature_frame = featured.reindex(columns=CHAMPION_FEATURE_COLS)
        feature_values = feature_frame.to_numpy(dtype=float)
        invalid_rows = ~np.isfinite(feature_values).all(axis=1)
        valid_batch_positions = []
        for batch_index, is_invalid in enumerate(invalid_rows):
            original_index = eligible_indices[batch_index]
            if is_invalid:
                outputs[original_index] = {
                    "available": False,
                    "status": "unavailable",
                    "error": "Required Champion V7 features are non-finite in the live snapshot.",
                    "missing_features": [
                        feature for feature, value in feature_frame.iloc[batch_index].items()
                        if not np.isfinite(float(value))
                    ],
                }
            else:
                valid_batch_positions.append(batch_index)

        if not valid_batch_positions:
            return [output for output in outputs if output is not None]

        probabilities_batch = champion_model.predict_proba(
            feature_frame.iloc[valid_batch_positions]
        )
        if len(probabilities_batch) != len(valid_batch_positions):
            raise RuntimeError("Champion V7 returned a different number of results than input rows")
        classes = list(champion_model.classes_)
        high_index = classes.index(1)
        for batch_position, probabilities in zip(valid_batch_positions, probabilities_batch):
            high_probability = float(probabilities[high_index])
            predicted_class = int(high_probability >= CHAMPION_THRESHOLD)
            class_probabilities = {
                ("High" if int(label) == 1 else "Not-High"): round(
                    float(probability), 6
                )
                for label, probability in zip(classes, probabilities)
            }
            outputs[eligible_indices[batch_position]] = {
                "available": True,
                "status": "available",
                "pred_class": predicted_class,
                "prediction": "High" if predicted_class else "Not-High",
                "is_bottleneck": bool(predicted_class),
                "confidence": round(
                    max(class_probabilities.values()) * 100, 2
                ),
                "high_probability": round(high_probability, 6),
                "class_probabilities": class_probabilities,
                "threshold": CHAMPION_THRESHOLD,
                "zero_filled_inputs": [],
            }

    except Exception as exc:
        logging.exception("Champion V7 batch inference failed")
        for index in eligible_indices:
            if outputs[index] is None:
                outputs[index] = {
                    "available": False,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }

    return [output for output in outputs if output is not None]


@app.post("/api/predict")
def predict(data: ProcessMetrics):
    raw = data.model_dump(exclude_unset=True)
    return {
        "process": {
            "pid": data.pid,
            "cpu": data.cpu,
            "comm": data.comm,
            "session_label": data.session_label,
        },
        "attribution_prediction": predict_mlp(raw),
        "severity_prediction": predict_champion(raw),
        "models": model_statuses(),
    }


@app.get("/api/live")
def live():
    """Read live eBPF metrics and run the severity and attribution models."""
    return JSONResponse(content=_live_payload())


def _live_payload() -> dict:
    """Build the existing /api/live JSON payload before response rendering."""
    try:
        processes = read_live_processes()
        if not processes:
            return {
                "source": "live_ebpf",
                "source_status": "empty",
                "source_error": "The four pinned maps are readable but contain no process entries.",
                "count": 0,
                "models": model_statuses(),
                "processes": [],
            }
        missing_by_index = [
            sorted(
                set(row.get("missing_metrics", ())) | LIVE_RAW_METRIC_FIELDS.difference(row)
            )
            for row in processes
        ]
        severity_predictions = [
            {"available": False, "status": "unavailable", "error": "Map snapshot is stale; no live inference was run."}
            for _ in processes
        ]
        attribution_predictions = [
            {"available": False, "status": "unavailable", "model": "baseline_original_best.pt",
             "labels": None, "probabilities": None, "threshold": None,
             "error": "Map snapshot is stale; no live inference was run.", "missing_features": []}
            for _ in processes
        ]
        fresh_indices = [
            index for index, row in enumerate(processes)
            if row.get("data_status") == "live"
        ]
        fresh_rows = [processes[index] for index in fresh_indices]
        fresh_severity = predict_champion_batch(fresh_rows)
        fresh_attribution = predict_mlp_batch(
            fresh_rows,
            [missing_by_index[index] for index in fresh_indices],
        )
        for index, severity, attribution in zip(
            fresh_indices, fresh_severity, fresh_attribution
        ):
            severity_predictions[index] = severity
            attribution_predictions[index] = attribution

        results = []
        for index, row in enumerate(processes):
            results.append({
                "pid": row["pid"],
                "cpu": row["cpu"],
                "comm": row["comm"],
                "session_label": row["session_label"],
                # Emit the monotonic nanosecond counter as decimal text so
                # JavaScript never rounds its integer representation.
                "sample_timestamp_ns": (
                    str(row["sample_timestamp_ns"])
                    if row.get("sample_timestamp_ns") is not None else None
                ),
                "sample_timestamp_clock": row.get("sample_timestamp_clock"),
                "sample_timestamp_status": row.get("sample_timestamp_status", "unavailable"),
                "data_age_seconds": row.get("data_age_seconds"),
                "data_status": row.get("data_status", "stale"),
                "process": {
                    "pid": row["pid"],
                    "cpu": row["cpu"],
                    "comm": row["comm"],
                    "session_label": row["session_label"],
                },
                "attribution_prediction": attribution_predictions[index],
                "severity_prediction": severity_predictions[index],
                "missing_metrics": missing_by_index[index],
                "metrics": {
                    key: value for key, value in row.items()
                    if key not in LIVE_ROW_METADATA_FIELDS
                },
            })

        return {
            "source": "live_ebpf",
            "source_status": "stale" if all(row.get("data_status") != "live" for row in processes) else "live",
            "source_error": None,
            "count": len(results),
            "models": model_statuses(),
            "processes": results,
        }
    except Exception as exc:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=503,
            detail=f"Unable to read live eBPF metrics: {exc}",
        ) from exc
