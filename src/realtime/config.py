from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RealtimeConfig:
    polling_interval_s: float = 5.0
    pin_base: str = "/sys/fs/bpf"
    sched_map_path: str = "/sys/fs/bpf/sched_map"
    mem_map_path: str = "/sys/fs/bpf/mem_map"
    syscall_map_path: str = "/sys/fs/bpf/syscall_map"
    lock_map_path: str = "/sys/fs/bpf/lock_map"

    # v5 artifacts (bottleneck_diagnosis_v5.ipynb, Section 8)
    model_path: str = ""
    scaler_path: str = ""
    feature_cols_path: str = ""
    label_thresholds_path: str = ""


def load_config() -> RealtimeConfig:
    cfg = RealtimeConfig()
    root = Path(__file__).resolve().parent.parent
    # Same directory as this config.py / predictor.py — where the user keeps the v5 artifacts.
    local_dir = Path(__file__).resolve().parent

    cfg.polling_interval_s = float(os.getenv("REALTIME_POLLING_INTERVAL_S", "5.0"))
    cfg.pin_base = os.getenv("REALTIME_PIN_BASE", "/sys/fs/bpf")
    cfg.sched_map_path = os.getenv("REALTIME_SCHED_MAP", f"{cfg.pin_base}/sched_map")
    cfg.mem_map_path = os.getenv("REALTIME_MEM_MAP", f"{cfg.pin_base}/mem_map")
    cfg.syscall_map_path = os.getenv("REALTIME_SYSCALL_MAP", f"{cfg.pin_base}/syscall_map")
    cfg.lock_map_path = os.getenv("REALTIME_LOCK_MAP", f"{cfg.pin_base}/lock_map")

    model_candidates = [
        os.getenv("REALTIME_MODEL_PATH", ""),
        str(local_dir / "champion_model_v5.pkl"),
        str(root / "ml" / "artifacts" / "champion_model_v5.pkl"),
    ]
    for candidate in model_candidates:
        if candidate and os.path.exists(candidate):
            cfg.model_path = candidate
            break

    scaler_candidates = [
        os.getenv("REALTIME_SCALER_PATH", ""),
        str(local_dir / "feature_scaler_v5.pkl"),
        str(root / "ml" / "artifacts" / "feature_scaler_v5.pkl"),
    ]
    for candidate in scaler_candidates:
        if candidate and os.path.exists(candidate):
            cfg.scaler_path = candidate
            break

    feature_candidates = [
        os.getenv("REALTIME_FEATURE_COLS_PATH", ""),
        str(local_dir / "feature_cols_v5.json"),
        str(root / "ml" / "artifacts" / "feature_cols_v5.json"),
    ]
    for candidate in feature_candidates:
        if candidate and os.path.exists(candidate):
            cfg.feature_cols_path = candidate
            break

    threshold_candidates = [
        os.getenv("REALTIME_LABEL_THRESHOLDS_PATH", ""),
        str(local_dir / "label_thresholds_v5.json"),
        str(root / "ml" / "artifacts" / "label_thresholds_v5.json"),
    ]
    for candidate in threshold_candidates:
        if candidate and os.path.exists(candidate):
            cfg.label_thresholds_path = candidate
            break

    return cfg