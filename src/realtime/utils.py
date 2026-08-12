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


def unpin_maps(map_paths: list[str]) -> None:
    """Unpin BPF maps by removing their bpffs pin files."""
    for map_path in map_paths:
        if not map_path:
            continue
        try:
            if os.path.exists(map_path):
                os.remove(map_path)
                print(f"[cleanup] unpinned {map_path}")
            else:
                print(f"[cleanup] already gone: {map_path}")
        except PermissionError:
            print(f"[cleanup] permission denied unpinning {map_path} (needs root/sudo)")
        except OSError as exc:
            print(f"[cleanup] failed to unpin {map_path}: {exc}")


def load_json(path: str | Path | None, description: str) -> Any:
    """Generic JSON loader for realtime artifacts."""
    if not path:
        raise RuntimeError(f"Missing path for {description}")

    json_path = Path(path)
    if not json_path.exists():
        raise RuntimeError(f"Missing {description}: {json_path}")

    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_feature_columns(path: str | Path | None) -> list[str]:
    data = load_json(path, description="feature columns JSON")
    if not isinstance(data, list) or not data:
        raise RuntimeError(f"Feature columns file is empty or malformed: {path}")
    return [str(item) for item in data]


def load_label_thresholds(path: str | Path | None) -> dict[str, float]:
    """Loads label thresholds JSON for informational inspection."""
    data = load_json(path, description="label thresholds JSON")
    if not isinstance(data, dict):
        raise RuntimeError(f"Label thresholds file is malformed: {path}")
    return {str(k): float(v) for k, v in data.items()}


