import json
import os
import re
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

# Fields present in each current map value. These are used to reject decoded
# values from an unexpected map ABI; missing fields in a present value remain
# missing rather than being silently filled with zero.
MAP_VALUE_FIELDS = {
    "sched": {
        "ctx_switches", "voluntary_switches", "involuntary_switches",
        "cpu_migrations", "total_runtime_ns", "stall_ns", "max_stall_ns",
        "latency_count", "runq_ratio_sum", "runq_ratio_count",
        "last_update_ns", "comm",
    },
    "mem": {
        "minor_faults", "major_faults", "kmalloc_count", "kfree_count",
        "total_alloc_bytes", "last_update_ns",
    },
    "syscall": {
        "total_count", "latency_sum_ns", "latency_max_ns", "read_count",
        "write_count", "read_bytes", "write_bytes", "mmap_count",
        "futex_count", "futex_latency_sum_ns", "epoll_count",
        "epoll_latency_sum_ns", "poll_count", "error_count",
        "last_update_ns",
    },
    "lock": {
        "mutex_contentions", "mutex_wait_sum_ns", "mutex_wait_max_ns",
        "rwsem_read_contentions", "rwsem_read_wait_sum_ns",
        "rwsem_write_contentions", "last_update_ns",
    },
}

# Sizes from struct agg_key and the four value structs in
# src/realtime/realtime_monitor.bpf.c. bpftool's map dump omits ABI metadata
# when a hash map is empty, so map show must verify the pinned object's name,
# type, key size, and value size before an empty snapshot can be trusted.
MAP_ABI = {
    "sched": (8, 104),
    "mem": (8, 48),
    "syscall": (8, 120),
    "lock": (8, 56),
}

MAP_VALUE_STRUCTS = {
    "sched": ("sched_value", 104),
    "mem": ("mem_value", 48),
    "syscall": ("syscall_value", 120),
    "lock": ("lock_value", 56),
}
KEY_STRUCT = ("agg_key", 8, {"pid", "cpu"})

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
    raise ValueError("map integer must be an integer or integer-formatted string")


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
    if numerator in row and denominator in row:
        # Match the training reader's uint64 integer-division feature values.
        row[output] = row[numerator] // row[denominator] if row[denominator] > 0 else 0
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


def _parse_btf_integer(token, field):
    """Parse a decimal BTF attribute, allowing its trailing separator comma."""
    if not isinstance(token, str):
        raise ValueError(f"invalid BTF {field}: expected an integer token")
    match = re.fullmatch(r"([0-9]+),?", token)
    if match is None:
        raise ValueError(f"invalid BTF {field}: {token!r}")
    return int(match.group(1), 10)


def _parse_raw_btf_schema(raw_btf, map_name):
    """Resolve a map's key/value structs from bpftool's raw-text BTF dump."""
    if map_name not in MAPS or not isinstance(raw_btf, str):
        raise ValueError("invalid map name or raw BTF payload")

    header_re = re.compile(r"^\[(\d+)\]\s+(\S+)\s+'([^']*)'(.*)$")
    attr_re = re.compile(r"([a-zA-Z_]+)=([^\s]+)")
    member_re = re.compile(r"^\s+'([^']+)'\s+type_id=(\d+)\b")
    types = {}
    current = None
    for line in raw_btf.splitlines():
        header = header_re.match(line)
        if header:
            type_id = int(header.group(1))
            if type_id in types:
                raise ValueError(f"duplicate BTF type id {type_id}")
            kind, type_name, tail = header.group(2), header.group(3), header.group(4)
            item = {
                "id": type_id,
                "kind": kind,
                "name": type_name,
                "attrs": dict(attr_re.findall(tail)),
                "members": [],
                "body": [],
            }
            if "size" in item["attrs"]:
                item["size"] = _parse_btf_integer(item["attrs"]["size"], "size")
            if "vlen" in item["attrs"]:
                item["vlen"] = _parse_btf_integer(item["attrs"]["vlen"], "vlen")
            types[type_id] = item
            current = item
            continue
        if current is None:
            if line.strip():
                raise ValueError("malformed raw BTF output before first type")
            continue
        current["body"].append(line)
        member = member_re.match(line)
        if member:
            current["members"].append({"name": member.group(1), "type_id": int(member.group(2))})

    if not types:
        raise ValueError("raw BTF output contains no types")
    for item in types.values():
        if item["kind"] in {"STRUCT", "UNION"}:
            if item.get("vlen") != len(item["members"]):
                raise ValueError(f"malformed BTF members for {item['kind']} {item['name']!r}")

    def type_ref(type_id):
        seen = set()
        while True:
            if type_id in seen:
                raise ValueError("cyclic BTF type reference")
            seen.add(type_id)
            item = types.get(type_id)
            if item is None:
                raise ValueError(f"BTF references missing type id {type_id}")
            if item["kind"] in {
                "TYPEDEF", "CONST", "VOLATILE", "RESTRICT", "TYPE_TAG", "PTR"
            }:
                type_text = " ".join(item["body"])
                next_id = item["attrs"].get("type_id")
                if next_id is None:
                    match = re.search(r"type_id=(\d+)", type_text)
                    next_id = match.group(1) if match else None
                if next_id is None:
                    raise ValueError(f"BTF {item['kind']} type {type_id} lacks type_id")
                type_id = _parse_btf_integer(next_id, "type_id")
                continue
            return item

    map_symbol = MAPS[map_name]
    variables = [item for item in types.values()
                 if item["kind"] == "VAR" and item["name"] == map_symbol]
    if len(variables) != 1:
        raise ValueError(f"BTF does not uniquely identify map symbol {map_symbol!r}")
    variable = variables[0]
    map_sections = [item for item in types.values()
                    if item["kind"] == "DATASEC" and item["name"] == ".maps"]
    if len(map_sections) != 1 or not any(
        re.search(rf"\btype_id={variable['id']}\b", line)
        for line in map_sections[0]["body"]
    ):
        raise ValueError(f"BTF map symbol {map_symbol!r} is not uniquely in .maps")

    map_decl = type_ref(_parse_btf_integer(variable["attrs"].get("type_id", "0"), "type_id"))
    if map_decl["kind"] != "STRUCT":
        raise ValueError(f"BTF map symbol {map_symbol!r} is not a map struct")
    value_members = [member for member in map_decl["members"] if member["name"] == "value"]
    key_members = [member for member in map_decl["members"] if member["name"] == "key"]
    if len(value_members) != 1 or len(key_members) != 1:
        raise ValueError(f"BTF map declaration {map_symbol!r} lacks unique key/value members")
    value_type = type_ref(value_members[0]["type_id"])
    key_type = type_ref(key_members[0]["type_id"])
    expected_value_name, expected_value_size = MAP_VALUE_STRUCTS[map_name]
    if value_type["kind"] != "STRUCT" or value_type["name"] != expected_value_name:
        raise ValueError(
            f"BTF map {map_symbol!r} resolves to unexpected value struct "
            f"{value_type['name']!r}"
        )
    if value_type.get("size") != expected_value_size:
        raise ValueError(f"BTF value struct {expected_value_name!r} has incompatible size")
    expected_value_fields = MAP_VALUE_FIELDS[map_name]
    value_fields = [member["name"] for member in value_type["members"]]
    if len(value_fields) != len(expected_value_fields) or set(value_fields) != expected_value_fields:
        raise ValueError(f"BTF value struct {expected_value_name!r} has incompatible members")
    if (key_type["kind"] != "STRUCT" or key_type["name"] != KEY_STRUCT[0]
            or key_type.get("size") != KEY_STRUCT[1]
            or set(member["name"] for member in key_type["members"]) != KEY_STRUCT[2]
            or len(key_type["members"]) != len(KEY_STRUCT[2])):
        raise ValueError(f"BTF map declaration {map_symbol!r} has incompatible key struct")
    return {
        "map_symbol": map_symbol,
        "key_struct": key_type["name"],
        "key_fields": [member["name"] for member in key_type["members"]],
        "value_struct": value_type["name"],
        "value_struct_size": value_type["size"],
        "value_fields": value_fields,
    }


def get_map_metadata(name, map_path=None):
    if name not in MAPS:
        raise ValueError(f"Unsupported pinned map name: {name!r}")
    path = Path(map_path) if map_path is not None else get_pin_dir() / MAPS[name]
    bpftool = shutil.which("bpftool") or "/usr/sbin/bpftool"
    result = subprocess.run(
        ["sudo", "-n", bpftool, "-j", "map", "show", "pinned", str(path)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"bpftool metadata read failed for map {name!r} at {path} "
            f"(exit={result.returncode}): {detail}"
        )
    payload = json.loads(result.stdout)
    if isinstance(payload, list) and len(payload) == 1:
        payload = payload[0]
    if not isinstance(payload, dict):
        raise ValueError(f"bpftool returned invalid metadata for map {name!r}")
    btf_id = payload.get("btf_id")
    try:
        btf_id = _as_int(btf_id)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"map {name!r} has no verifiable BTF ID") from exc
    if btf_id <= 0:
        raise ValueError(f"map {name!r} has no verifiable BTF ID")
    btf_result = subprocess.run(
        ["sudo", "-n", bpftool, "btf", "dump", "id", str(btf_id), "format", "raw"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if btf_result.returncode != 0:
        detail = btf_result.stderr.strip() or btf_result.stdout.strip()
        raise RuntimeError(f"bpftool BTF read failed for map {name!r}: {detail}")
    payload["btf_id"] = btf_id
    payload.update(_parse_raw_btf_schema(btf_result.stdout, name))
    return payload


def decode(entry):
    if not isinstance(entry, dict):
        raise ValueError("malformed bpftool map entry: expected an object")
    try:
        formatted = entry.get("formatted", entry)
        if not isinstance(formatted, dict):
            raise ValueError("malformed bpftool map entry: formatted value must be an object")
        key = _decode_key(formatted.get("key", entry.get("key")))
        value = formatted.get("value", entry.get("value"))
        if not isinstance(value, dict):
            raise ValueError("malformed bpftool map entry: value must be an object")
        return key, value
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"malformed bpftool map entry: {exc}") from exc


def _validate_map_metadata(metadata):
    if not isinstance(metadata, dict) or set(metadata) != set(MAPS):
        raise ValueError("map metadata must identify exactly sched, mem, syscall, and lock maps")
    for map_name, info in metadata.items():
        if not isinstance(info, dict):
            raise ValueError(f"map {map_name!r} metadata must be an object")
        key_size, value_size = MAP_ABI[map_name]
        actual_name = info.get("name")
        actual_type = info.get("type")
        actual_key_size = info.get("bytes_key", info.get("key_size"))
        actual_value_size = info.get("bytes_value", info.get("value_size"))
        actual_fields = info.get("value_fields")
        actual_btf_id = info.get("btf_id")
        actual_key_fields = info.get("key_fields")
        expected_struct, expected_struct_size = MAP_VALUE_STRUCTS[map_name]
        try:
            actual_key_size = _as_int(actual_key_size)
            actual_value_size = _as_int(actual_value_size)
            actual_btf_id = _as_int(actual_btf_id)
            actual_struct_size = _as_int(info.get("value_struct_size"))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"map {map_name!r} metadata lacks verifiable ABI identity") from exc
        if (actual_name != MAPS[map_name] or str(actual_type).lower() != "hash"
                or (actual_key_size, actual_value_size) != (key_size, value_size)
                or actual_btf_id <= 0
                or info.get("map_symbol") != MAPS[map_name]
                or info.get("key_struct") != KEY_STRUCT[0]
                or not isinstance(actual_key_fields, list)
                or any(not isinstance(field, str) for field in actual_key_fields)
                or set(actual_key_fields) != KEY_STRUCT[2]
                or len(actual_key_fields) != len(KEY_STRUCT[2])
                or info.get("value_struct") != expected_struct
                or actual_struct_size != expected_struct_size
                or not isinstance(actual_fields, list)
                or len(actual_fields) != len(MAP_VALUE_FIELDS[map_name])
                or any(not isinstance(field, str) for field in actual_fields)
                or set(actual_fields) != MAP_VALUE_FIELDS[map_name]):
            raise ValueError(f"map {map_name!r} pinned identity or ABI is incompatible")


def _validate_map_snapshot(snapshots, metadata):
    if not isinstance(snapshots, dict) or set(snapshots) != set(MAPS):
        raise ValueError("map snapshot must contain exactly sched, mem, syscall, and lock maps")
    _validate_map_metadata(metadata)
    for map_name, entries in snapshots.items():
        if not isinstance(entries, list):
            raise ValueError(f"map {map_name!r} snapshot must be a list")
        allowed_fields = MAP_VALUE_FIELDS[map_name]
        if not entries:
            # The BTF member names plus map identity and sizes above verify
            # empty map ABI before its absent keys authorize zero defaults.
            continue
        for entry in entries:
            _key, value = decode(entry)
            missing = allowed_fields - set(value)
            if missing:
                raise ValueError(
                    f"incomplete {map_name!r} map value; missing field(s): "
                    + ", ".join(sorted(missing))
                )
            unsupported = set(value) - allowed_fields
            if unsupported:
                raise ValueError(
                    f"unsupported {map_name!r} map value field(s): "
                    + ", ".join(sorted(unsupported))
                )
            for field, raw_value in value.items():
                if field == "last_update_ns":
                    # Timestamp parse failures have their own explicit invalid
                    # status below; they do not invalidate metric structure.
                    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, str)):
                        raise ValueError(f"malformed {map_name!r} map timestamp field")
                    continue
                if field == "comm":
                    if not isinstance(raw_value, (str, list)):
                        raise ValueError("malformed sched map comm field")
                    if isinstance(raw_value, list):
                        try:
                            for byte in raw_value:
                                parsed = _as_int(byte)
                                if not 0 <= parsed <= 255:
                                    raise ValueError
                        except (TypeError, ValueError, OverflowError) as exc:
                            raise ValueError("malformed sched map comm byte array") from exc
                    continue
                try:
                    parsed_value = _as_int(raw_value)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(
                        f"malformed {map_name!r} map field {field!r}"
                    ) from exc
                if parsed_value < 0:
                    raise ValueError(f"negative unsigned {map_name!r} map field {field!r}")


def read_live_processes():
    metadata = {name: get_map_metadata(name) for name in MAPS}
    snapshots = {name: dump_map(name) for name in MAPS}
    _validate_map_snapshot(snapshots, metadata)
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
            row.setdefault("_observed_map_families", set()).add(map_name)
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
        # The baseline training reader built a complete row from the union of
        # map keys and treated a valid absent map-family entry as zero. Keep
        # missing fields in a present map value distinct: only absent families
        # receive these defaults.
        for map_name, fields in MAP_VALUE_FIELDS.items():
            if map_name in row["_observed_map_families"]:
                continue
            for metric in METRIC_SOURCE_KEYS[map_name]:
                row.setdefault(metric, 0)
                row["_observed_metrics"].add(metric)
            derived_fields = {
                field
                for sources in DERIVED_METRIC_SOURCE_KEYS.values()
                for field in sources
                if field in fields
            }
            for field in derived_fields:
                row.setdefault(field, 0)
                row["_observed_source_keys"].add(field)

        for numerator, denominator, average in (
            ("runq_ratio_sum", "runq_ratio_count", "avg_runq_ratio"),
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
        row.pop("_observed_map_families", None)
        row.pop("_last_update_ns", None)
        row.pop("_timestamp_missing", None)
        row.pop("_timestamp_invalid", None)
        output.append(row)

    return sorted(output, key=lambda item: item.get("total_runtime_ns", 0), reverse=True)
