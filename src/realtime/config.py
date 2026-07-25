from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RealtimeConfig:
    polling_interval_s: float = 1.0
    pin_base: str = "/sys/fs/bpf"
    sched_map_path: str = "/sys/fs/bpf/sched_map"
    mem_map_path: str = "/sys/fs/bpf/mem_map"
    syscall_map_path: str = "/sys/fs/bpf/syscall_map"
    lock_map_path: str = "/sys/fs/bpf/lock_map"
    model_path: str = ""
    feature_cols_path: str = ""


def load_config() -> RealtimeConfig:
    cfg = RealtimeConfig()
    root = Path(__file__).resolve().parent.parent
    approach3 = root / "approach3"

    cfg.polling_interval_s = float(os.getenv("REALTIME_POLLING_INTERVAL_S", "1.0"))
    cfg.pin_base = os.getenv("REALTIME_PIN_BASE", "/sys/fs/bpf")
    cfg.sched_map_path = os.getenv("REALTIME_SCHED_MAP", f"{cfg.pin_base}/sched_map")
    cfg.mem_map_path = os.getenv("REALTIME_MEM_MAP", f"{cfg.pin_base}/mem_map")
    cfg.syscall_map_path = os.getenv("REALTIME_SYSCALL_MAP", f"{cfg.pin_base}/syscall_map")
    cfg.lock_map_path = os.getenv("REALTIME_LOCK_MAP", f"{cfg.pin_base}/lock_map")

    model_candidates = [
        os.getenv("REALTIME_MODEL_PATH", ""),
        str(approach3 / "rf_bottleneck_classifier.pkl"),
        str(approach3 / "xgb_bottleneck_classifier.pkl"),
        str(root / "ml" / "artifacts" / "rf_bottleneck_classifier.pkl"),
        str(root / "ml" / "artifacts" / "xgb_bottleneck_classifier.pkl"),
    ]
    for candidate in model_candidates:
        if candidate and os.path.exists(candidate):
            cfg.model_path = candidate
            break

    feature_candidates = [
        os.getenv("REALTIME_FEATURE_COLS_PATH", ""),
        str(root / "ml" / "artifacts" / "feature_cols.json"),
        str(approach3 / "feature_cols.json"),
    ]
    for candidate in feature_candidates:
        if candidate and os.path.exists(candidate):
            cfg.feature_cols_path = candidate
            break

    return cfg
