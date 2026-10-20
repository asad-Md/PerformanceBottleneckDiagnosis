import json
import logging
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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


def predict_mlp(raw: dict, missing_metrics: list[str] | None = None) -> dict:
    if attributor is None:
        return {
            "available": False,
            "status": "unavailable",
            "model": "baseline_original_best.pt",
            "labels": None,
            "probabilities": None,
            "threshold": None,
            "error": attribution_error or "Baseline attribution model did not load",
        }

    absent = sorted(
        (set(attributor.feature_columns) - set(raw))
        | set(missing_metrics or ()).intersection(attributor.feature_columns)
    )
    if absent:
        return {
            "available": False,
            "status": "unavailable",
            "model": "baseline_original_best.pt",
            "labels": None,
            "probabilities": None,
            "threshold": None,
            "error": "Required live attribution features are unavailable: " + ", ".join(absent),
            "missing_features": absent,
        }

    try:
        features = pd.DataFrame([raw], columns=attributor.feature_columns)
        probabilities = attributor.predict(features)[0]
        labels = attributor.target_columns
        scores = {name: round(float(score), 6) for name, score in zip(labels, probabilities)}
        selected = [name for name, score in scores.items() if score >= attributor.THRESHOLD]
        return {
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
        logging.exception("Baseline attribution inference failed")
        return {
            "available": False,
            "status": "error",
            "model": "baseline_original_best.pt",
            "labels": None,
            "probabilities": None,
            "threshold": None,
            "error": f"{type(exc).__name__}: {exc}",
            "missing_features": [],
        }


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

    # Do not let request-model defaults or NaN filling turn absent counters
    # into apparently observed zeros.
    unavailable_inputs = [sorted(CHAMPION_RAW_REQUIRED - set(row)) for row in raw_rows]
    if any(unavailable_inputs):
        return [
            {
                "available": False,
                "status": "unavailable",
                "error": "Live collector does not provide required Champion V7 feature(s): " + ", ".join(missing),
                "missing_features": missing,
            }
            for missing in unavailable_inputs
        ]

    try:
        featured = engineer_features_v5(pd.DataFrame(raw_rows))
        feature_frame = featured.reindex(columns=CHAMPION_FEATURE_COLS)
        invalid_rows = ~np.isfinite(feature_frame.to_numpy(dtype=float)).all(axis=1)
        if invalid_rows.any():
            return [
                ({"available": False, "status": "unavailable",
                  "error": "Required Champion V7 features are non-finite in the live snapshot.",
                  "missing_features": [
                      feature for feature, value in feature_frame.iloc[index].items()
                      if not np.isfinite(float(value))
                  ]}
                 if invalid_rows[index] else {
                     "available": False, "status": "unavailable",
                     "error": "Champion V7 batch rejected because another row has invalid features.",
                     "missing_features": [],
                 })
                for index in range(len(raw_rows))
            ]

        probabilities_batch = champion_model.predict_proba(feature_frame)
        classes = list(champion_model.classes_)
        high_index = classes.index(1)
        predictions = []

        for probabilities in probabilities_batch:
            high_probability = float(probabilities[high_index])
            predicted_class = int(high_probability >= CHAMPION_THRESHOLD)
            class_probabilities = {
                ("High" if int(label) == 1 else "Not-High"): round(
                    float(probability), 6
                )
                for label, probability in zip(classes, probabilities)
            }
            predictions.append({
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
            })

        return predictions

    except Exception as exc:
        logging.exception("Champion V7 batch inference failed")
        return [
            {
                "available": False,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            for _ in raw_rows
        ]


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
        results = []
        severity_predictions = [
            ({"available": False, "status": "unavailable", "error": "Map snapshot is stale; no live inference was run."}
             if row.get("data_status") != "live" else predict_champion_batch([row])[0])
            for row in processes
        ]

        for row, severity in zip(processes, severity_predictions):
            missing_metrics = sorted(
                set(row.get("missing_metrics", ())) | LIVE_RAW_METRIC_FIELDS.difference(row)
            )
            attribution = (
                {"available": False, "status": "unavailable", "model": "baseline_original_best.pt",
                 "labels": None, "probabilities": None, "threshold": None,
                 "error": "Map snapshot is stale; no live inference was run.", "missing_features": []}
                if row.get("data_status") != "live" else predict_mlp(row, missing_metrics)
            )
            results.append({
                **row,
                # Emit the monotonic nanosecond counter as decimal text so
                # JavaScript never rounds its integer representation.
                "sample_timestamp_ns": (
                    str(row["sample_timestamp_ns"])
                    if row.get("sample_timestamp_ns") is not None else None
                ),
                "process": {
                    "pid": row["pid"],
                    "cpu": row["cpu"],
                    "comm": row["comm"],
                    "session_label": row["session_label"],
                },
                "attribution_prediction": attribution,
                "severity_prediction": severity,
                "missing_metrics": missing_metrics,
                "metrics": {
                    key: value for key, value in row.items()
                    if key not in {
                        "pid", "cpu", "comm", "session_label",
                        "sample_timestamp_ns", "sample_timestamp_clock",
                        "sample_timestamp_status",
                    }
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
