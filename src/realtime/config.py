from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RealtimeConfig:
    polling_interval_s: float = 5.0
    pin_base: str = "/sys/fs/bpf"
    bpf_object_path: str = ""
    sched_map_path: str = "/sys/fs/bpf/sched_map"
    mem_map_path: str = "/sys/fs/bpf/mem_map"
    syscall_map_path: str = "/sys/fs/bpf/syscall_map"
    lock_map_path: str = "/sys/fs/bpf/lock_map"

    model_path: str = ""
    scaler_path: str = ""
    feature_cols_path: str = ""
    label_thresholds_path: str = ""
    attribution_model_path: str = ""
    attribution_scaler_path: str = ""
    attribution_feature_cols_path: str = ""
    attribution_target_cols_path: str = ""


def load_config() -> RealtimeConfig:
    cfg = RealtimeConfig()
    root = Path(__file__).resolve().parent.parent
    local_dir = Path(__file__).resolve().parent

    cfg.polling_interval_s = float(os.getenv("REALTIME_POLLING_INTERVAL_S", "5.0"))
    cfg.pin_base = os.getenv("REALTIME_PIN_BASE", "/sys/fs/bpf")
    cfg.bpf_object_path = os.getenv(
        "REALTIME_BPF_OBJECT_PATH",
        str(local_dir / "realtime_monitor.bpf.o"),
    )
    cfg.sched_map_path = os.getenv("REALTIME_SCHED_MAP", f"{cfg.pin_base}/sched_map")
    cfg.mem_map_path = os.getenv("REALTIME_MEM_MAP", f"{cfg.pin_base}/mem_map")
    cfg.syscall_map_path = os.getenv("REALTIME_SYSCALL_MAP", f"{cfg.pin_base}/syscall_map")
    cfg.lock_map_path = os.getenv("REALTIME_LOCK_MAP", f"{cfg.pin_base}/lock_map")

    cfg.model_path = os.getenv(
        "REALTIME_MODEL_PATH",
        str(local_dir / "champion_model_v7_shap.pkl"),
    )
    cfg.scaler_path = os.getenv("REALTIME_SCALER_PATH", "")
    cfg.feature_cols_path = os.getenv(
        "REALTIME_FEATURE_COLS_PATH",
        str(local_dir / "feature_cols_v7_shap.json"),
    )
    cfg.label_thresholds_path = os.getenv(
        "REALTIME_LABEL_THRESHOLDS_PATH",
        str(local_dir / "label_thresholds_v7_shap.json"),
    )

    attribution_dir = root / "attributionAnalysis" / "pipeline_comparison_artifacts"
    cfg.attribution_model_path = os.getenv(
        "REALTIME_ATTRIBUTION_MODEL_PATH",
        str(attribution_dir / "baseline_original_best.pt"),
    )
    cfg.attribution_scaler_path = os.getenv(
        "REALTIME_ATTRIBUTION_SCALER_PATH",
        str(attribution_dir / "feature_scaler.pkl"),
    )
    cfg.attribution_feature_cols_path = os.getenv(
        "REALTIME_ATTRIBUTION_FEATURE_COLS_PATH",
        str(attribution_dir / "attribution_feature_cols.json"),
    )
    cfg.attribution_target_cols_path = os.getenv(
        "REALTIME_ATTRIBUTION_TARGET_COLS_PATH",
        str(attribution_dir / "attribution_target_cols.json"),
    )

    return cfg