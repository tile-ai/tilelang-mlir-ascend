"""Raw Ascend profiler records, independent of host launch delays."""

import struct
from pathlib import Path


def _resolve_kernel_names(prof_path: Path) -> dict[int, str]:
    """Resolve kernel names from CANN task-track and hash records."""
    hash_to_name: dict[int, str] = {}
    for path in prof_path.rglob("*hash_dic.slice_*"):
        if path.name.endswith(".done"):
            continue
        text = path.read_bytes().decode("utf-8", errors="replace")
        for line in text.strip().split("\n"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                try:
                    h = int(parts[0])
                    if h >= 2**63:
                        h -= 2**64
                    hash_to_name[h] = parts[1]
                except ValueError:
                    pass
    seq_to_name: dict[int, str] = {}
    for path in prof_path.rglob("*task_track.slice_*"):
        if path.name.endswith(".done"):
            continue
        data = path.read_bytes()
        for offset in range(0, len(data) - len(data) % 64, 64):
            values = struct.unpack_from("<8q", data, offset)
            seq = values[3] >> 32 & 65535
            name_hash = values[5]
            if name_hash != 0:
                seq_to_name[seq] = hash_to_name.get(name_hash, f"<unknown {name_hash}>")
    return seq_to_name


def _parse_ffts_profile(
    prof_path: Path,
    repeats: int,
    exclude_name_marker: str | None = None,
    exclude_names: frozenset[str] = frozenset(),
    require_marker_seen: bool = False,
) -> dict[str, float]:
    """Read the validated 128-byte ffts layout; reject missing device activity."""
    total_ns = 0
    intervals = []
    count = 0
    aic_cycles = 0
    aiv_cycles = 0
    excluded_ns = 0
    excluded_count = 0
    records_seen = 0
    needs_names = bool(exclude_name_marker or exclude_names)
    seq_to_name = _resolve_kernel_names(prof_path) if needs_names else {}
    flush_kernel_seen = False
    for path in prof_path.rglob("ffts_profile*"):
        if path.name.endswith(".done") or path.stat().st_size == 0:
            continue
        data = path.read_bytes()
        if len(data) % 128:
            raise RuntimeError(f"Truncated ffts_profile record: {path}")
        for offset in range(0, len(data) - len(data) % 128, 128):
            values = struct.unpack_from("<16q", data, offset)
            cycles = values[1]
            if cycles == 0:
                continue
            duration = values[15] - values[14]
            if duration < 0:
                raise RuntimeError(f"Negative kernel duration in {path}")
            records_seen += 1
            if needs_names:
                seq = values[0] >> 32 & 65535
                name = seq_to_name.get(seq, "")
                is_flush = bool(exclude_name_marker) and exclude_name_marker in name
                flush_kernel_seen = flush_kernel_seen or is_flush
                if is_flush or name in exclude_names:
                    excluded_ns += duration
                    excluded_count += 1
                    continue
            total_ns += duration
            intervals.append((values[14], values[15]))
            count += 1
            if values[2] >= 1 << 31:
                aiv_cycles += cycles
            else:
                aic_cycles += cycles
    if records_seen == 0:
        raise RuntimeError(f"torch_npu.profiler emitted no ffts_profile records under {prof_path}")
    if require_marker_seen and exclude_name_marker and (not flush_kernel_seen):
        raise RuntimeError(
            "cache-flush kernel never matched by name during profile parsing; check CANN hash_dic/task_track name resolution"
        )
    if count == 0:
        return {
            "latency_ms": 0.0,
            "profile_kernel_us": 0.0,
            "profile_kernel_count": 0.0,
            "profile_aic_cycles": 0.0,
            "profile_aiv_cycles": 0.0,
            "profile_excluded_kernel_count": excluded_count / repeats,
            "profile_excluded_kernel_us": excluded_ns / repeats / 1000.0,
        }
    intervals.sort()
    start, end = intervals[0]
    busy_ns = 0
    for next_start, next_end in intervals[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            busy_ns += end - start
            start, end = next_start, next_end
    busy_ns += end - start
    result = {
        "latency_ms": busy_ns / repeats / 1000000.0,
        "kernel_span_ms": (max(v[1] for v in intervals) - intervals[0][0]) / repeats / 1000000.0,
        "profile_kernel_us": total_ns / repeats / 1000.0,
        "profile_kernel_count": count / repeats,
        "profile_aic_cycles": aic_cycles / repeats,
        "profile_aiv_cycles": aiv_cycles / repeats,
    }
    if needs_names:
        result["profile_excluded_kernel_count"] = excluded_count / repeats
        result["profile_excluded_kernel_us"] = excluded_ns / repeats / 1000.0
    return result
