from __future__ import annotations

import ctypes
import ctypes.util
import errno
import json
import os
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Any


_MAP_VALUE_STRUCTS = {
    "sched": (struct.Struct("=11Q16s"), (
        "ctx_switches", "voluntary_switches", "involuntary_switches",
        "cpu_migrations", "total_runtime_ns", "stall_ns", "max_stall_ns",
        "latency_count", "runq_ratio_sum", "runq_ratio_count", "last_update_ns",
        "comm",
    )),
    "mem": (struct.Struct("=6Q"), (
        "minor_faults", "major_faults", "kmalloc_count", "kfree_count",
        "total_alloc_bytes", "last_update_ns",
    )),
    "syscall": (struct.Struct("=15Q"), (
        "total_count", "latency_sum_ns", "latency_max_ns", "read_count",
        "write_count", "read_bytes", "write_bytes", "mmap_count", "futex_count",
        "futex_latency_sum_ns", "epoll_count", "epoll_latency_sum_ns",
        "poll_count", "error_count", "last_update_ns",
    )),
    "lock": (struct.Struct("=7Q"), (
        "mutex_contentions", "mutex_wait_sum_ns", "mutex_wait_max_ns",
        "rwsem_read_contentions", "rwsem_read_wait_sum_ns",
        "rwsem_write_contentions", "last_update_ns",
    )),
}
_MAP_KEY_SIZE = 8
_BPF_MAP_TYPE_HASH = 1


class _BpfMapInfoPrefix(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("id", ctypes.c_uint32),
        ("key_size", ctypes.c_uint32),
        ("value_size", ctypes.c_uint32),
        ("max_entries", ctypes.c_uint32),
        ("map_flags", ctypes.c_uint32),
        ("name", ctypes.c_char * 16),
    ]


class AtomicMapConsumeError(RuntimeError):
    def __init__(self, message: str, partial_snapshot: dict[str, list[dict[str, Any]]]):
        super().__init__(message)
        self.partial_snapshot = partial_snapshot


def _load_libbpf(api=None):
    if api is not None:
        return api
    library = ctypes.util.find_library("bpf")
    if not library:
        raise RuntimeError("libbpf is required for atomic BPF map consumption")
    api = ctypes.CDLL(library)
    api.bpf_obj_get.argtypes = [ctypes.c_char_p]
    api.bpf_obj_get.restype = ctypes.c_int
    api.bpf_map_lookup_and_delete_elem.argtypes = [
        ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
    ]
    api.bpf_map_lookup_and_delete_elem.restype = ctypes.c_int
    api.bpf_obj_get_info_by_fd.argtypes = [
        ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32),
    ]
    api.bpf_obj_get_info_by_fd.restype = ctypes.c_int
    return api


def _raw_map_key(entry: dict[str, Any], map_path: str) -> tuple[bytes, tuple[int, int]]:
    formatted = entry.get("formatted") if isinstance(entry, dict) else None
    key = entry.get("key") if isinstance(entry, dict) else None
    if key is None and isinstance(formatted, dict):
        key = formatted.get("key")
    if isinstance(key, dict) and "pid" in key and "cpu" in key:
        pid, cpu = int(key["pid"]), int(key["cpu"])
        return struct.pack("=II", pid, cpu), (pid, cpu)
    if not isinstance(key, list) or len(key) != 8:
        raise RuntimeError(
            f"Cannot atomically consume pinned map {map_path!r}: expected an 8-byte raw key"
        )

    key_bytes = bytearray()
    for value in key:
        try:
            byte = int(value, 0) if isinstance(value, str) else int(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid raw key byte in pinned map {map_path!r}") from exc
        if not 0 <= byte <= 0xFF:
            raise RuntimeError(f"Invalid raw key byte in pinned map {map_path!r}")
        key_bytes.append(byte)
    pid, cpu = struct.unpack("=II", key_bytes)
    return bytes(key_bytes), (pid, cpu)


def _decode_map_value(map_name: str, value: bytes) -> dict[str, Any]:
    value_struct, fields = _MAP_VALUE_STRUCTS[map_name]
    if len(value) != value_struct.size:
        raise RuntimeError(
            f"Atomic {map_name!r} map lookup returned {len(value)} bytes; "
            f"expected {value_struct.size}"
        )
    decoded = value_struct.unpack(value)
    result = dict(zip(fields, decoded))
    if map_name == "sched":
        result["comm"] = result["comm"].split(b"\0", 1)[0].decode(errors="replace")
    return result


def consume_map_snapshot(
    map_entries: list[tuple[str, str, list[dict[str, Any]]]],
    *,
    expected_metadata: dict[str, dict[str, Any]] | None = None,
    api=None,
) -> dict[str, list[dict[str, Any]]]:
    """Atomically retrieve and delete each captured key's current value.

    The input dump supplies candidate keys only. Value retrieval and deletion
    are one kernel operation, so updates before it are included and updates
    after it remain in a newly created entry for the next inference interval.
    """
    libbpf = _load_libbpf(api)
    snapshots: dict[str, list[dict[str, Any]]] = {}
    for map_name, map_path, entries in map_entries:
        if map_name not in _MAP_VALUE_STRUCTS:
            raise ValueError(f"Unsupported inference map {map_name!r}")
        map_fd = libbpf.bpf_obj_get(os.fsencode(map_path))
        if map_fd < 0:
            raise AtomicMapConsumeError(
                f"Could not open pinned map {map_path!r} (error {-map_fd})",
                snapshots,
            )
        consumed: list[dict[str, Any]] = []
        try:
            info = _BpfMapInfoPrefix()
            info_size = ctypes.c_uint32(ctypes.sizeof(info))
            info_result = libbpf.bpf_obj_get_info_by_fd(
                map_fd, ctypes.byref(info), ctypes.byref(info_size),
            )
            expected_value_size = _MAP_VALUE_STRUCTS[map_name][0].size
            expected_id = (expected_metadata or {}).get(map_name, {}).get("id")
            if (info_result != 0 or info.type != _BPF_MAP_TYPE_HASH
                    or info.key_size != _MAP_KEY_SIZE
                    or info.value_size != expected_value_size
                    or (expected_id is not None and info.id != int(expected_id))):
                raise AtomicMapConsumeError(
                    f"Pinned map {map_path!r} changed or has incompatible ABI "
                    f"before atomic consume (info_result={info_result}, id={info.id}, "
                    f"type={info.type}, key_size={info.key_size}, value_size={info.value_size})",
                    snapshots,
                )
            value_buffer = ctypes.create_string_buffer(_MAP_VALUE_STRUCTS[map_name][0].size)
            for entry in entries:
                try:
                    key_bytes, key = _raw_map_key(entry, map_path)
                    key_buffer = ctypes.create_string_buffer(key_bytes, len(key_bytes))
                    result = libbpf.bpf_map_lookup_and_delete_elem(
                        map_fd, key_buffer, value_buffer,
                    )
                    if result == -errno.ENOENT:
                        # Exit cleanup or another consumer already retired it.
                        continue
                    if result != 0:
                        raise RuntimeError(f"atomic map operation returned {result}")
                    consumed.append({
                        "key": {"pid": key[0], "cpu": key[1]},
                        "value": _decode_map_value(
                            map_name,
                            value_buffer.raw[:_MAP_VALUE_STRUCTS[map_name][0].size],
                        ),
                    })
                except Exception as exc:
                    snapshots[map_name] = consumed
                    raise AtomicMapConsumeError(
                        f"Atomic consume failed for {map_name!r} map {map_path!r} "
                        f"after {len(consumed)} consumed entries: {exc}",
                        snapshots,
                    ) from exc
        finally:
            os.close(map_fd)
        snapshots[map_name] = consumed
    return snapshots


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


def run_bpftool_clear(map_entries: list[tuple[str, list[dict[str, Any]]]]) -> None:
    """Delete captured map entries in one bpftool batch, using their raw key bytes."""
    commands: list[str] = []
    seen_commands: set[str] = set()
    for map_path, entries in map_entries:
        if not map_path:
            raise RuntimeError("No pinned BPF map path was provided")
        if not os.path.exists(map_path):
            raise RuntimeError(f"Missing pinned BPF map: {map_path}")
        if any(character.isspace() for character in map_path):
            raise RuntimeError(f"Cannot clear pinned BPF map with whitespace in its path: {map_path!r}")

        for index, entry in enumerate(entries):
            key = entry.get("key") if isinstance(entry, dict) else None
            if not isinstance(key, list) or not key:
                raise RuntimeError(
                    f"Cannot clear pinned BPF map '{map_path}': entry {index} has no raw byte-array key"
                )

            key_bytes: list[str] = []
            for byte in key:
                if isinstance(byte, bool):
                    raise RuntimeError(
                        f"Cannot clear pinned BPF map '{map_path}': entry {index} has an invalid key byte"
                    )
                try:
                    if isinstance(byte, int):
                        byte_value = byte
                    elif isinstance(byte, str):
                        try:
                            byte_value = int(byte, 0)
                        except ValueError:
                            byte_value = int(byte, 10)
                    else:
                        raise ValueError("key byte is not an integer")
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"Cannot clear pinned BPF map '{map_path}': entry {index} has an invalid key byte"
                    ) from exc
                if not 0 <= byte_value <= 0xFF:
                    raise RuntimeError(
                        f"Cannot clear pinned BPF map '{map_path}': entry {index} has an invalid key byte"
                    )
                key_bytes.append(f"0x{byte_value:02x}")

            command = f"map delete pinned {map_path} key hex {' '.join(key_bytes)}"
            if command not in seen_commands:
                seen_commands.add(command)
                commands.append(command)

    if not commands:
        return

    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".bpftool", delete=True) as batch_file:
            batch_file.write("\n".join(commands) + "\n")
            batch_file.flush()
            proc = subprocess.run(
                ["bpftool", "batch", "file", batch_file.name],
                capture_output=True,
                text=True,
                check=False,
            )
    except FileNotFoundError as exc:
        raise RuntimeError("bpftool executable not found while clearing pinned maps") from exc

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip() or "no output"
        raise RuntimeError(f"bpftool failed to clear pinned maps: {detail}")


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
