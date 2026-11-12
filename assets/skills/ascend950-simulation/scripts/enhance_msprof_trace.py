"""Preserve an msprof merged trace and supplement missing GET/RLS from its own dump."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

BUFFER_NAMES = {"GET_BUF", "RLS_BUF"}
BUFFER_FLOW_CATEGORY = "GET/RLS buffer scopes"
BUFFER_COLOR = "BUFFER:7"  # Insight palette hash: sunsetOrange, index 9.
LINE = re.compile(
    r"\[info\]\s+\[(\d+)\]\s+\(PC:\s*(0x[0-9a-fA-F]+)\).*?"
    r"\(ID:\s*(\d+)\)\s+([\w.]+)\s+(\{.*\})\s*$"
)


def read_dump(path, core):
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        match = LINE.fullmatch(line)
        if not match:
            raise ValueError(f"Unrecognized dump record: {path}:{number}")
        tick, pc, identifier, name, params = match.groups()
        records.append(
            dict(
                core=core,
                tick=int(tick),
                pc=int(pc, 16),
                instr_id=int(identifier),
                name=name,
                params=json.loads(params),
                source_file=path.name,
                source_line=number,
            )
        )
    return records


def record_key(record):
    # IDs alone can collide across instruction streams in one core.
    return record["core"], record["pc"], record["name"], record["tick"]


def native_records(events, ticks_per_us):
    endings = defaultdict(list)
    for event in events:
        if event.get("ph") == "E":
            identity = tuple(str(event.get(k, "")) for k in ("pid", "tid", "id", "name"))
            endings[identity].append(event["ts"])
    records = []
    for event in events:
        if event.get("ph") not in {"X", "B"} or "pc_addr" not in event.get("args", {}):
            continue
        if event["ph"] == "X":
            end = event["ts"] + event["dur"]
        else:
            identity = tuple(str(event.get(k, "")) for k in ("pid", "tid", "id", "name"))
            matches = endings.get(identity, [])
            if len(matches) != 1:
                raise ValueError(f"Expected one matching native E endpoint: {identity}")
            end = matches[0]
        if not math.isfinite(end) or end < event["ts"]:
            raise ValueError("Invalid native instruction interval")
        records.append(
            dict(core=str(event["pid"]), pc=int(event["args"]["pc_addr"], 16), name=event["name"], tick=round(end * ticks_per_us))
        )
    if not records:
        raise ValueError("No native instructions; use simulator/trace.json, not a per-core trace")
    return records


def pair_buffers(records):
    channels = defaultdict(list)
    unlinked = []
    for record in records:
        params = record["params"]
        pipe = params.get("target_pipe")
        values = [params.get(k) for k in ("buf_id", "buf_mode", "buf_cnt")]
        if (
            not isinstance(pipe, str)
            or not pipe
            or pipe == "UNKNOWN"
            or pipe != params.get("instr_type")
            or any(type(v) is not int or v < 0 for v in values)
        ):
            unlinked.append(
                dict(
                    source_file=record["source_file"],
                    source_line=record["source_line"],
                    reason="missing/invalid buffer PIPE, ID, mode or epoch",
                )
            )
            continue
        record["pipe"] = pipe
        channel = (record["core"], pipe, *values)
        channels[channel].append(record)
    pairs = []
    for channel, endpoints in sorted(channels.items()):
        gets = [r for r in endpoints if r["name"] == "GET_BUF"]
        releases = [r for r in endpoints if r["name"] == "RLS_BUF"]
        reason = None
        if len(gets) != 1 or len(releases) != 1:
            reason = "epoch requires exactly one GET and one RLS; no FIFO guessing"
        elif gets[0]["tick"] > releases[0]["tick"]:
            reason = "release completion precedes acquire completion"
        if reason:
            unlinked.extend(
                dict(source_file=r["source_file"], source_line=r["source_line"], channel=channel, reason=reason) for r in endpoints
            )
            continue
        flow_id = "viewer:buffers:" + ":".join(map(str, channel))
        for record in endpoints:
            record["flow_id"] = flow_id
        pairs.append((flow_id, gets[0], releases[0]))
    return pairs, unlinked


def supplement(profile, frequency_ghz):
    """Return enhanced JSON and audit without writing or modifying native artifacts."""
    if not math.isfinite(frequency_ghz) or frequency_ghz <= 0:
        raise ValueError("frequency_ghz must be finite and positive")
    profile = Path(profile).resolve()
    native_path = profile / "simulator" / "trace.json"
    dump_dir = profile / "dump"
    trace = json.loads(native_path.read_text(encoding="utf-8"))
    if trace.get("profilingType") != "op" or not isinstance(trace.get("traceEvents"), list):
        raise ValueError("Expected a native msprof operator trace")
    if "msprof_buffer_enhancement" in trace.get("otherData", {}):
        raise ValueError("Already enhanced; use the original native profile")
    events = trace["traceEvents"]
    native = native_records(events, frequency_ghz * 1000)
    if any(r["name"] in BUFFER_NAMES for r in native):
        raise ValueError("Native trace already contains GET/RLS; audit this version before adding compatibility anchors")
    cores = {r["core"] for r in native}
    files = {p.name.removesuffix(".instr_log.dump"): p for p in dump_dir.glob("*.instr_log.dump")}
    if cores - files.keys():
        raise ValueError(f"Missing completion dump files: {sorted(cores - files.keys())}")
    raw = [r for core in sorted(cores) for r in read_dump(files[core], core)]
    raw_counts = Counter(record_key(r) for r in raw)
    native_counts = Counter(record_key(r) for r in native)
    extra = native_counts - raw_counts
    missing = raw_counts - native_counts
    missing_names = Counter()
    for (_, _, name, _), count in missing.items():
        missing_names[name] += count
    if extra or any(name not in BUFFER_NAMES for name in missing_names):
        raise ValueError(
            "Native/raw coverage mismatch; check run, clock and unsupported omissions: "
            + json.dumps(dict(extra_trace_records=sum(extra.values()), missing_names=dict(missing_names)))
        )
    remaining = missing.copy()
    buffers = []
    for record in raw:
        key = record_key(record)
        if remaining[key]:
            buffers.append(dict(record))
            remaining[key] -= 1
    pairs, unlinked = pair_buffers(buffers)
    existing_ids = {str(e["id"]) for e in events if "id" in e}
    if any(flow_id in existing_ids for flow_id, _, _ in pairs):
        raise ValueError("Buffer flow ID collides with a native ID")
    additions, tracks = [], set()
    tick_us = 1 / (frequency_ghz * 1000)
    anchors = 0
    for record in buffers:
        core, name, tick = record["core"], record["name"], record["tick"]
        params = record["params"]
        pipe = record.get("pipe", params.get("instr_type", "FLOWCTRL"))
        flow_id = record.get("flow_id", f"viewer:unlinked:{core}:{record['source_line']}")
        args = dict(
            params,
            instr_id=record["instr_id"],
            pc=hex(record["pc"]),
            pc_addr=hex(record["pc"]),
            hardware_pipe=pipe,
            raw_completion_tick=tick,
            raw_dispatch_tick=None,
            start_tick=tick,
            end_tick=tick,
            sync_point_tick=tick,
            source_file=record["source_file"],
            source_line=record["source_line"],
            semantic_duration_cycles=0,
            display_duration_cycles=1,
            display_duration_only=True,
            sync_flow_id=flow_id,
            point_semantics="completion-only zero-cost endpoint; 1 tick is display width only",
        )
        additions.append(
            dict(
                name=name,
                ph="X",
                pid=core,
                tid=pipe,
                id=flow_id,
                ts=tick * tick_us,
                dur=tick_us,
                cat="Buffer synchronization",
                cname=BUFFER_COLOR,
                args=args,
            )
        )
        anchor_tid = f"{pipe} / {name} (compat)"
        record["anchor_tid"] = anchor_tid
        if (core, anchor_tid) not in tracks:
            tracks.add((core, anchor_tid))
            additions.append(dict(ph="M", name="thread_name", pid=core, tid=anchor_tid, args=dict(name=anchor_tid)))
        anchor_args = dict(
            args,
            viewer_compatibility_anchor=True,
            compatibility_kind="buffer",
            original_instruction_name=name,
            source_instr_id=record["instr_id"],
            source_display_tid=pipe,
            lane_semantics="viewer-only GET/RLS flow anchor; not a hardware PIPE",
            compatibility_purpose="flow linkage only; not an executed flag instruction",
        )
        anchor = dict(
            name="SET_FLAG" if name == "GET_BUF" else "WAIT_FLAG",
            ph="B",
            pid=core,
            tid=anchor_tid,
            id=flow_id,
            ts=tick * tick_us,
            cat="Viewer compatibility anchor (GET/RLS)",
            cname=BUFFER_COLOR,
            args=anchor_args,
        )
        additions.extend([anchor, dict(anchor, ph="E", ts=(tick + 1) * tick_us)])
        anchors += 1
    for flow_id, get, release in pairs:
        common = dict(
            id=flow_id,
            name="GET_BUF -> RLS_BUF",
            cat=BUFFER_FLOW_CATEGORY,
            cname=BUFFER_COLOR,
            args=dict(
                get_instr_id=get["instr_id"],
                release_instr_id=release["instr_id"],
                semantics="acquire/release scope; not instruction latency",
            ),
        )
        additions.append(dict(common, ph="s", pid=get["core"], tid=get["anchor_tid"], ts=get["tick"] * tick_us))
        finish = dict(common, ph="t", pid=release["core"], tid=release["anchor_tid"], ts=release["tick"] * tick_us, bp="e")
        additions.extend([finish, dict(finish, ph="f")])
    audit = dict(
        status="PASS" if not unlinked else "PASS_WITH_UNLINKED_BUFFER_ENDPOINTS",
        native_profile=str(profile),
        native_trace_sha256=hashlib.sha256(native_path.read_bytes()).hexdigest(),
        clock_frequency_ghz=frequency_ghz,
        timestamp_unit="microseconds",
        source_completion_records=len(raw),
        native_logical_instructions=len(native),
        enhanced_logical_instructions=len(native) + len(buffers),
        native_events_input=len(events),
        native_events_preserved=len(events),
        native_instruction_events_preserved=sum(e.get("ph") in {"X", "B", "E"} for e in events),
        added_buffer_events=len(buffers),
        missing_native_name_counts=dict(missing_names),
        unsupported_omissions=0,
        matched_native_completion_ticks=len(native),
        covered_cores=sorted(cores),
        excluded_dump_cores=sorted(files.keys() - cores),
        completion_dump_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for core, p in sorted(files.items()) if core in cores},
        coverage_scope="exact multiset of core, PC, name and completion tick in covered raw dumps; not proof of simulator callback completeness",
        buffer_pairs=len(pairs),
        compatibility_anchors=anchors,
        buffer_compatibility_anchors=anchors,
        native_flow_pairs=sum(e.get("ph") == "s" for e in events),
        total_flow_pairs=sum(e.get("ph") == "s" for e in events) + len(pairs),
        buffer_marker_width_cycles=1,
        semantic_buffer_duration_cycles=0,
        buffer_flow_category=BUFFER_FLOW_CATEGORY,
        unlinked=unlinked,
        compatibility_anchor_semantics="synthetic 1-tick SET_FLAG/WAIT_FLAG anchors for GET/RLS only; excluded from real instruction/performance counts",
        insight_import="NOT_RUN",
        gui_click_test="NOT_RUN",
    )
    # Keep every native event unchanged, including intra-block B/E and flow tracks.
    trace["traceEvents"] = events + additions
    trace.setdefault("otherData", {})["msprof_buffer_enhancement"] = audit
    return trace, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path, help="One native OPPROF directory containing dump/ and simulator/")
    parser.add_argument("output", type=Path, help="New enhanced trace path; native input is never overwritten")
    parser.add_argument("--frequency-ghz", type=float, required=True, help="Verified simulator clock, not host CPU frequency")
    args = parser.parse_args()
    report = args.output.with_suffix(".validation.json")
    if args.output.exists() or report.exists():
        parser.error("Output or validation report already exists; use a new directory")
    try:
        trace, audit = supplement(args.profile, args.frequency_ghz)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(trace, ensure_ascii=False), encoding="utf-8")
    report.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                k: audit[k]
                for k in (
                    "status",
                    "native_logical_instructions",
                    "added_buffer_events",
                    "enhanced_logical_instructions",
                    "buffer_pairs",
                    "compatibility_anchors",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
