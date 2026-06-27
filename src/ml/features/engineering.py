"""Feature engineering for CPU, memory, syscall, lock, and cross-domain signals."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"


ENGINEERED_FEATURES = [
    "runtime_per_switch",
    "voluntary_ratio",
    "involuntary_ratio",
    "migration_ratio",
    "switches_per_runtime",
    "fault_rate",
    "major_fault_ratio",
    "alloc_pressure",
    "free_pressure",
    "large_page_ratio",
    "syscall_rate",
    "bytes_per_syscall",
    "futex_ratio",
    "io_intensity",
    "syscall_error_ratio",
    "lock_pressure",
    "mutex_wait_per_contention",
    "rwsem_pressure",
    "rwsem_write_ratio",
    "cpu_memory_ratio",
    "io_cpu_ratio",
    "lock_cpu_ratio",
    "memory_lock_ratio",
]


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0)
    return pd.Series(0, index=df.index, dtype="float64")


def _safe_div(numerator: pd.Series, denominator: pd.Series | int | float) -> pd.Series:
    den = denominator if isinstance(denominator, pd.Series) else pd.Series(denominator, index=numerator.index)
    den = pd.to_numeric(den, errors="coerce").replace(0, np.nan)
    return (pd.to_numeric(numerator, errors="coerce") / den).replace([np.inf, -np.inf], np.nan).fillna(0)


def engineer_features(
    df: pd.DataFrame,
    artifact_dir: Optional[Path | str] = DEFAULT_ARTIFACT_DIR,
) -> pd.DataFrame:
    """Return a copy of df with engineered feature columns added."""

    out = df.copy()

    ctx = _num(out, "ctx_switches")
    runtime = _num(out, "total_runtime_ns")
    voluntary = _num(out, "voluntary_switches")
    involuntary = _num(out, "involuntary_switches")
    migrations = _num(out, "cpu_migrations")

    minor_faults = _num(out, "minor_faults")
    major_faults = _num(out, "major_faults")
    total_faults = minor_faults + major_faults
    kmalloc = _num(out, "kmalloc_count")
    kfree = _num(out, "kfree_count")
    alloc_bytes = _num(out, "total_alloc_bytes")
    free_bytes = _num(out, "total_free_bytes")
    large_pages = _num(out, "large_page_allocs")

    syscalls = _num(out, "syscall_count")
    reads = _num(out, "read_count")
    writes = _num(out, "write_count")
    read_bytes = _num(out, "read_bytes")
    write_bytes = _num(out, "write_bytes")
    io_bytes = read_bytes + write_bytes
    futex = _num(out, "futex_count")
    syscall_errors = _num(out, "syscall_error_count")

    mutex_contentions = _num(out, "mutex_contentions")
    avg_mutex_wait = _num(out, "avg_mutex_wait_ns")
    rwsem_read = _num(out, "rwsem_read_contentions")
    rwsem_write = _num(out, "rwsem_write_contentions")
    rwsem_total = rwsem_read + rwsem_write

    out["runtime_per_switch"] = _safe_div(runtime, ctx)
    out["voluntary_ratio"] = _safe_div(voluntary, ctx)
    out["involuntary_ratio"] = _safe_div(involuntary, ctx)
    out["migration_ratio"] = _safe_div(migrations, ctx)
    out["switches_per_runtime"] = _safe_div(ctx, runtime)

    out["fault_rate"] = _safe_div(total_faults, runtime)
    out["major_fault_ratio"] = _safe_div(major_faults, total_faults)
    out["alloc_pressure"] = _safe_div(alloc_bytes, ctx)
    out["free_pressure"] = _safe_div(free_bytes, ctx)
    out["large_page_ratio"] = _safe_div(large_pages, kmalloc)

    out["syscall_rate"] = _safe_div(syscalls, runtime)
    out["bytes_per_syscall"] = _safe_div(io_bytes, syscalls)
    out["futex_ratio"] = _safe_div(futex, syscalls)
    out["io_intensity"] = _safe_div(reads + writes, syscalls)
    out["syscall_error_ratio"] = _safe_div(syscall_errors, syscalls)

    out["lock_pressure"] = mutex_contentions + rwsem_total
    out["mutex_wait_per_contention"] = _safe_div(avg_mutex_wait * mutex_contentions, mutex_contentions)
    out["rwsem_pressure"] = rwsem_total
    out["rwsem_write_ratio"] = _safe_div(rwsem_write, rwsem_total)

    cpu_pressure = out["switches_per_runtime"] + out["involuntary_ratio"] + out["migration_ratio"]
    memory_pressure = out["fault_rate"] + out["alloc_pressure"] + out["large_page_ratio"]
    io_pressure = out["syscall_rate"] + out["io_intensity"] + out["bytes_per_syscall"]
    lock_pressure = out["lock_pressure"] + out["mutex_wait_per_contention"] + out["rwsem_pressure"]

    out["cpu_memory_ratio"] = _safe_div(cpu_pressure, memory_pressure)
    out["io_cpu_ratio"] = _safe_div(io_pressure, cpu_pressure)
    out["lock_cpu_ratio"] = _safe_div(lock_pressure, cpu_pressure)
    out["memory_lock_ratio"] = _safe_div(memory_pressure, lock_pressure)

    for col in ENGINEERED_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)

    if artifact_dir is not None:
        out_dir = Path(artifact_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "feature_cols.json").write_text(
            json.dumps(ENGINEERED_FEATURES, indent=2),
            encoding="utf-8",
        )

    return out

