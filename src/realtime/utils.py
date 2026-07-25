from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any


def run_bpftool_dump(map_path: str) -> list[dict[str, Any]]:
    """Read a pinned BPF map via bpftool and return a list of entries."""
    if not map_path:
        raise RuntimeError("No pinned BPF map path was provided")
    if not os.path.exists(map_path):
        raise RuntimeError(f"Missing pinned BPF map: {map_path}")

    cmd = ["bpftool", "-j", "map", "dump", "pinned", map_path]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError(f"bpftool executable not found while reading pinned map '{map_path}'") from exc

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip() or "no output"
        raise RuntimeError(f"bpftool execution failed for pinned map '{map_path}': {detail}")

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Malformed bpftool JSON for pinned map '{map_path}': {exc}") from exc

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if "entries" in payload:
            return payload.get("entries", [])
        if "error" in payload:
            raise RuntimeError(f"bpftool reported an error for pinned map '{map_path}': {payload['error']}")
    raise RuntimeError(f"Unexpected bpftool payload for pinned map '{map_path}': {type(payload).__name__}")


def make_map_key(pid: int, cpu: int) -> tuple[int, int]:
    return (pid, cpu)


def load_feature_columns(path: str | Path | None) -> list[str]:
    if not path:
        raise RuntimeError("Missing feature_cols.json path")
    feature_path = Path(path)
    if not feature_path.exists():
        raise RuntimeError(f"Missing feature_cols.json: {feature_path}")
    with feature_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or not data:
        raise RuntimeError(f"feature_cols.json is empty or malformed: {feature_path}")
    return [str(item) for item in data]
