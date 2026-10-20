import json
import os
import subprocess
import time
import shutil
from collections import defaultdict
from pathlib import Path

DEFAULT_PIN_DIR = Path("/sys/fs/bpf/pbd_test")
MAPS = {
    "sched": "sched_map",
    "mem": "mem_map",
    "syscall": "syscall_map",
    "lock": "lock_map",
}

METRIC_SOURCE_KEYS = {
    "sched": {
        "ctx_switches": ("ctx_switches",),
        "voluntary_switches": ("voluntary_switches",),
        "involuntary_switches": ("involuntary_switches",),
        "cpu_migrations": ("cpu_migrations",),
        "total_runtime_ns": ("total_runtime_ns",),
        "stall_ns": ("stall_ns",),
        "max_stall_ns": ("max_stall_ns",),
        "latency_count": ("latency_count",),
        "runq_ratio_sum": ("runq_ratio_sum",),
        "runq_ratio_count": ("runq_ratio_count",),
    },
    "mem": {
        "minor_faults": ("minor_faults",),
        "major_faults": ("major_faults",),
        "kmalloc_count": ("kmalloc_count",),
        "kfree_count": ("kfree_count",),
        "total_alloc_bytes": ("total_alloc_bytes",),
    },
    "syscall": {
        "syscall_count": ("total_count",),
        "max_syscall_latency_ns": ("latency_max_ns",),
        "read_count": ("read_count",),
        "write_count": ("write_count",),
        "read_bytes": ("read_bytes",),
        "write_bytes": ("write_bytes",),
        "mmap_count": ("mmap_count",),
        "futex_count": ("futex_count",),
        "epoll_count": ("epoll_count",),
        "poll_count": ("poll_count",),
        "syscall_error_count": ("error_count",),
    },
    "lock": {
        "mutex_contentions": ("mutex_contentions",),
        "max_mutex_wait_ns": ("mutex_wait_max_ns",),
        "rwsem_read_contentions": ("rwsem_read_contentions",),
        "rwsem_write_contentions": ("rwsem_write_contentions",),
    },
}

DERIVED_METRIC_SOURCE_KEYS = {
    "avg_runq_ratio": ("runq_ratio_sum", "runq_ratio_count"),
    "avg_stall_ns": ("stall_ns", "latency_count"),
    "avg_syscall_latency_ns": ("latency_sum_ns", "total_count"),
    "avg_futex_latency_ns": ("futex_latency_sum_ns", "futex_count"),
    "avg_epoll_latency_ns": ("epoll_latency_sum_ns", "epoll_count"),
    "avg_mutex_wait_ns": ("mutex_wait_sum_ns", "mutex_contentions"),
    "avg_rwsem_read_wait_ns": ("rwsem_read_wait_sum_ns", "rwsem_read_contentions"),
}

ATTRIBUTION_FEATURES = {
    "ctx_switches", "voluntary_switches", "involuntary_switches", "cpu_migrations",
    "total_runtime_ns", "stall_ns", "avg_stall_ns", "max_stall_ns", "latency_count",
    "minor_faults", "major_faults", "kmalloc_count", "kfree_count", "total_alloc_bytes",
    "syscall_count", "avg_syscall_latency_ns", "max_syscall_latency_ns", "read_count",
    "write_count", "read_bytes", "write_bytes", "mmap_count", "futex_count",
    "avg_futex_latency_ns", "epoll_count", "avg_epoll_latency_ns", "poll_count",
    "syscall_error_count", "mutex_contentions", "avg_mutex_wait_ns", "max_mutex_wait_ns",
    "rwsem_read_contentions", "avg_rwsem_read_wait_ns",
}

REQUIRED_LIVE_METRICS = ATTRIBUTION_FEATURES


def get_pin_dir() -> Path:
    configured = Path(os.environ.get("REALTIME_PIN_BASE") or DEFAULT_PIN_DIR)
    # Restrict privileged map reads to a direct child of bpffs. Resolve the
    # parent (not the map pin itself, which is a bpffs handle).
    root = Path("/sys/fs/bpf").resolve()
    parent = configured.parent.resolve()
    if parent != root or configured.name in {"", ".", ".."}:
        raise ValueError("REALTIME_PIN_BASE must be a direct directory under /sys/fs/bpf")
    return root / configured.name


def _as_int(value):
    if isinstance(value, bool):
        raise ValueError("boolean is not a map integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value, 0)
        except ValueError:
            try:
                return int(value, 16)
            except ValueError:
                return int(value, 10)
    return int(value)


def _decode_key(key):
    if isinstance(key, dict):
        return _as_int(key["pid"]), _as_int(key["cpu"])
    if isinstance(key, list):
        if len(key) == 2:
            return _as_int(key[0]), _as_int(key[1])
        if len(key) == 8:
            key_bytes = bytes(_as_int(byte) for byte in key)
            return int.from_bytes(key_bytes[:4], "little"), int.from_bytes(key_bytes[4:], "little")
    raise ValueError("unsupported bpftool map key")


def _add_sum(row, value, source, destination=None):
    destination = destination or source
    if source in value:
        row[destination] = row.get(destination, 0) + _as_int(value[source])


def _add_max(row, value, source, destination=None):
    destination = destination or source
    if source in value:
        row[destination] = max(row.get(destination, 0), _as_int(value[source]))


def _average(row, numerator, denominator, output):
    if numerator in row and denominator in row and row[denominator] > 0:
        # Match the training reader's uint64 integer-division feature values.
        row[output] = row[numerator] // row[denominator]
        row["_observed_metrics"].add(output)


def dump_map(name):
    if name not in MAPS:
        raise ValueError(f"Unsupported pinned map name: {name!r}")
    path = get_pin_dir() / MAPS[name]
    bpftool = shutil.which("bpftool") or "/usr/sbin/bpftool"
    result = subprocess.run(
        ["sudo", "-n", bpftool, "-j", "map", "dump", "pinned", str(path)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        if "password is required" in detail.lower() or "a password is required" in detail.lower():
            raise PermissionError(
                "Pinned-map read requires explicit authorization for the narrowly scoped "
                f"bpftool read command ({name} at {path}); sudo -n cannot prompt."
            )
        raise RuntimeError(f"bpftool failed for map {name!r} at {path} (exit={result.returncode}): {detail}")
    payload = json.loads(result.stdout)
    if not isinstance(payload, list):
        raise ValueError(f"bpftool returned a non-list payload for map {name!r}")
    return payload


def decode(entry):
    if not isinstance(entry, dict):
        return None
    try:
        formatted = entry.get("formatted", entry)
        if not isinstance(formatted, dict):
            return None
        key = _decode_key(formatted.get("key", entry.get("key")))
        value = formatted.get("value", entry.get("value"))
        if not isinstance(value, dict):
            return None
        return key, value
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def read_live_processes():
    snapshots = {name: dump_map(name) for name in MAPS}
    processes = defaultdict(lambda: {
        "pid": 0, "cpu": 0, "comm": "",
        "_observed_metrics": set(),
        "_observed_source_keys": set(),
        "_last_update_ns": 0,
        "_timestamp_missing": False,
        "_timestamp_invalid": False,
    })

    for map_name, entries in snapshots.items():
        for entry in entries:
            decoded = decode(entry)
            if not decoded:
                continue
            (pid, cpu), value = decoded
            if pid <= 0:
                continue
            # Champion training rows are per (pid, cpu); do not merge counters
            # from a migrating process into one differently distributed row.
            row = processes[(pid, cpu)]
            row["pid"] = pid
            row["cpu"] = cpu
            row["_observed_source_keys"].update(value)
            for metric, source_keys in METRIC_SOURCE_KEYS.get(map_name, {}).items():
                if all(source_key in value for source_key in source_keys):
                    row["_observed_metrics"].add(metric)

            if map_name == "sched":
                for field in (
                    "ctx_switches", "voluntary_switches", "involuntary_switches",
                    "cpu_migrations", "total_runtime_ns", "stall_ns", "latency_count",
                    "runq_ratio_sum", "runq_ratio_count",
                ):
                    _add_sum(row, value, field)
                for field in ("max_stall_ns",):
                    _add_max(row, value, field)
                comm = value.get("comm", "")
                if isinstance(comm, list):
                    try:
                        comm = bytes(_as_int(byte) for byte in comm).split(b"\0", 1)[0].decode(errors="replace")
                    except (ValueError, TypeError):
                        comm = ""
                row["comm"] = str(comm).strip() or row["comm"]
            elif map_name == "mem":
                for field in (
                    "minor_faults", "major_faults", "kmalloc_count", "kfree_count",
                    "total_alloc_bytes",
                ):
                    _add_sum(row, value, field)
            elif map_name == "syscall":
                for source, output in (
                    ("total_count", "syscall_count"), ("read_count", "read_count"),
                    ("write_count", "write_count"), ("latency_sum_ns", "latency_sum_ns"),
                    ("read_bytes", "read_bytes"), ("write_bytes", "write_bytes"),
                    ("mmap_count", "mmap_count"), ("futex_count", "futex_count"),
                    ("futex_latency_sum_ns", "futex_latency_sum_ns"),
                    ("epoll_count", "epoll_count"), ("epoll_latency_sum_ns", "epoll_latency_sum_ns"),
                    ("poll_count", "poll_count"), ("error_count", "syscall_error_count"),
                ):
                    _add_sum(row, value, source, output)
                _add_max(row, value, "latency_max_ns", "max_syscall_latency_ns")
            elif map_name == "lock":
                for field in (
                    "mutex_contentions", "mutex_wait_sum_ns", "rwsem_read_contentions",
                    "rwsem_read_wait_sum_ns", "rwsem_write_contentions",
                ):
                    _add_sum(row, value, field)
                _add_max(row, value, "mutex_wait_max_ns", "max_mutex_wait_ns")

            if "last_update_ns" not in value:
                row["_timestamp_missing"] = True
            else:
                try:
                    update_ns = _as_int(value["last_update_ns"])
                    if update_ns <= 0:
                        row["_timestamp_invalid"] = True
                    else:
                        row["_last_update_ns"] = max(
                            row["_last_update_ns"], update_ns
                        )
                except (TypeError, ValueError, OverflowError):
                    row["_timestamp_invalid"] = True

    output = []
    for row in processes.values():
        for numerator, denominator, average in (
            ("stall_ns", "latency_count", "avg_stall_ns"),
            ("latency_sum_ns", "syscall_count", "avg_syscall_latency_ns"),
            ("futex_latency_sum_ns", "futex_count", "avg_futex_latency_ns"),
            ("epoll_latency_sum_ns", "epoll_count", "avg_epoll_latency_ns"),
            ("mutex_wait_sum_ns", "mutex_contentions", "avg_mutex_wait_ns"),
            ("rwsem_read_wait_sum_ns", "rwsem_read_contentions", "avg_rwsem_read_wait_ns"),
        ):
            required_sources = DERIVED_METRIC_SOURCE_KEYS[average]
            if set(required_sources).issubset(row["_observed_source_keys"]):
                _average(row, numerator, denominator, average)

        # The v7 training reader used unsigned integer division for this
        # feature; keep its floor-mean semantics and require at least one sample.
        if (set(DERIVED_METRIC_SOURCE_KEYS["avg_runq_ratio"]).issubset(
                row["_observed_source_keys"]
        ) and row.get("runq_ratio_count", 0) > 0):
            row["avg_runq_ratio"] = row["runq_ratio_sum"] // row["runq_ratio_count"]
            row["_observed_metrics"].add("avg_runq_ratio")

        row["session_label"] = "live_ebpf"
        row["missing_metrics"] = sorted(ATTRIBUTION_FEATURES - row["_observed_metrics"])
        now_ns = time.monotonic_ns()
        sample_ns = row["_last_update_ns"]
        if row["_timestamp_invalid"] or (sample_ns and sample_ns > now_ns):
            timestamp_status = "invalid"
            sample_ns = None
        elif row["_timestamp_missing"] or not sample_ns:
            timestamp_status = "unavailable"
            sample_ns = None
        else:
            timestamp_status = "available"
        age_seconds = (now_ns - sample_ns) / 1e9 if sample_ns is not None else None
        row["sample_timestamp_ns"] = sample_ns
        row["sample_timestamp_clock"] = "CLOCK_MONOTONIC"
        row["sample_timestamp_status"] = timestamp_status
        row["data_age_seconds"] = age_seconds
        row["data_status"] = (
            "stale" if age_seconds is None or age_seconds > float(
                os.environ.get("REALTIME_STALE_AFTER_SECONDS", "30")
            ) else "live"
        )
        for field in (
            "latency_sum_ns", "futex_latency_sum_ns", "epoll_latency_sum_ns",
            "mutex_wait_sum_ns", "rwsem_read_wait_sum_ns", "mutex_wait_max_ns",
            "latency_max_ns", "runq_ratio_sum", "runq_ratio_count",
        ):
            row.pop(field, None)
        row.pop("_observed_metrics", None)
        row.pop("_observed_source_keys", None)
        row.pop("_last_update_ns", None)
        row.pop("_timestamp_missing", None)
        row.pop("_timestamp_invalid", None)
        output.append(row)

    return sorted(output, key=lambda item: item.get("total_runtime_ns", 0), reverse=True)
