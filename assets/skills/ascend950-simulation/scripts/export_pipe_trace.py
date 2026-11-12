"""Export CANN SQLite intervals to pipe-based MindStudio/Chrome Trace JSON."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3

PIPE_ORDER = ["SCALAR", "FLOWCTRL", "MTE2", "MTE1", "CUBE", "FIXP", "VECTOR", "MTE3", "PUSHQ"]
# Insight's simulation view hashes cname into this 12-entry theme palette.
# Verified against Ascend/msinsight: AscendUnit.tsx, units/utils.ts, colorUtils.ts.
COLOR_INDEX = {
    "SCALAR": 0,
    "FLOWCTRL": 4,
    "MTE2": 5,
    "MTE1": 3,
    "CUBE": 7,
    "FIXP": 9,
    "VECTOR": 2,
    "MTE3": 1,
    "PUSHQ": 10,
    "SYNC": 6,
    "BUFFER": 9,
    "UNKNOWN": 11,
}
# Insight hashes flow.cat, rather than flow.cname, for the line color.
# This readable category hashes to sunsetOrange (9), matching BUFFER slices.
BUFFER_FLOW_CATEGORY = "GET/RLS buffer scopes"
SET_NAMES = {"SET_FLAG", "SET_INTRA_BLOCK", "SET_INTRA_BLOCKI"}
WAIT_NAMES = {"WAIT_FLAG", "WAIT_INTRA_BLOCK", "WAIT_INTRA_BLOCKI"}
BUFFER_NAMES = {"GET_BUF", "RLS_BUF"}
SYNC_NAMES = SET_NAMES | WAIT_NAMES | BUFFER_NAMES


def normalize_pipe(pipe):
    return "VECTOR" if pipe in ("VEC", "VECTOR", "SIMD", "SIMT") else pipe


def sync_channel(record):
    """Resolve a directed flag channel; never match only by numeric flag ID."""
    name, args = record["name"], record["args"]
    core = record["core"]
    if name in ("SET_FLAG", "WAIT_FLAG"):
        if not all(key in args for key in ("flag_id", "target_pipe", "consumer_pipe")):
            return None
        return ("flags", core, normalize_pipe(args["target_pipe"]), normalize_pipe(args["consumer_pipe"]), int(args["flag_id"]))
    if name not in SET_NAMES | WAIT_NAMES or "sync_id" not in args:
        return None
    block, core_type = core.rsplit(".", 1)
    flag = int(args["sync_id"])
    is_set = name in SET_NAMES
    # Ascend950 intra-block ID mapping: AIC 0..15 <-> AIV0 0..15;
    # AIC 16..31 <-> AIV1 0..15. Signals are consumed FIFO per channel.
    if core_type == "AIC" and 0 <= flag < 32:
        vector = "AIV0" if flag < 16 else "AIV1"
        source, target = ("AIC", vector) if is_set else (vector, "AIC")
        flag %= 16
    elif core_type in ("AIV0", "AIV1") and 0 <= flag < 16:
        source, target = (core_type, "AIC") if is_set else ("AIC", core_type)
    else:
        return None
    return ("intra_blocks", block, source, target, flag)


def build_sync_pairs(records):
    channels = defaultdict(lambda: {"set": [], "wait": []})
    unlinked = []
    for record in records:
        if record["name"] not in SET_NAMES | WAIT_NAMES:
            continue
        channel = sync_channel(record)
        if channel is None:
            unlinked.append({"instr_id": record["args"]["instr_id"], "reason": "unsupported or incomplete sync metadata"})
            continue
        channels[channel]["set" if record["name"] in SET_NAMES else "wait"].append(record)
    pairs = []
    for channel, endpoints in sorted(channels.items()):
        sets = sorted(endpoints["set"], key=lambda item: (item["start"], item["args"]["instr_id"]))
        waits = sorted(endpoints["wait"], key=lambda item: (item["start"], item["args"]["instr_id"]))
        # Missing occurrences can shift FIFO rounds. Do not guess across gaps.
        if len(sets) != len(waits):
            unlinked.extend(
                {
                    "instr_id": item["args"]["instr_id"],
                    "channel": channel,
                    "reason": "set/wait counts differ; occurrence alignment ambiguous",
                }
                for item in sets + waits
            )
            continue
        for occurrence, (source, target) in enumerate(zip(sets, waits)):
            if source["start"] > target["start"]:
                unlinked.extend(
                    {
                        "instr_id": item["args"]["instr_id"],
                        "channel": channel,
                        "reason": "recorded set begins after paired wait; temporal ordering unconfirmed",
                    }
                    for item in (source, target)
                )
                continue
            identifier = "sync:" + ":".join(map(str, channel)) + f":round{occurrence}"
            source["sync_flow_id"] = target["sync_flow_id"] = identifier
            pairs.append({"id": identifier, "channel": channel, "set": source, "wait": target})
    return pairs, unlinked


def build_buffer_pairs(records):
    """Pair an acquire/release scope, not an assumed cross-PIPE dependency."""
    channels = defaultdict(lambda: {"GET_BUF": [], "RLS_BUF": []})
    unlinked = []
    for record in records:
        if record["name"] not in BUFFER_NAMES:
            continue
        args = record["args"]
        # buf_cnt distinguishes uses of the same physical buffer slot.
        values = [args.get(key) for key in ("buf_id", "buf_cnt", "buf_mode")]
        pipe = normalize_pipe(args.get("target_pipe", record["pipe"]))
        if (
            any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values)
            or pipe != record["pipe"]
            or pipe == "UNKNOWN"
            or record["core"].endswith(".UNKNOWN")
        ):
            unlinked.append({"instr_id": args["instr_id"], "reason": "missing or invalid buffer ID, epoch, mode, core or PIPE"})
            continue
        buffer_id, epoch, mode = values
        channel = ("buffers", record["core"], pipe, buffer_id, mode, epoch)
        channels[channel][record["name"]].append(record)
    pairs = []
    for channel, endpoints in sorted(channels.items()):
        gets, releases = endpoints["GET_BUF"], endpoints["RLS_BUF"]
        reason = None
        if len(gets) != 1 or len(releases) != 1:
            reason = "buffer epoch requires exactly one GET_BUF and one RLS_BUF"
        elif gets[0]["start"] > releases[0]["start"]:
            reason = "buffer release precedes acquire; temporal ordering unconfirmed"
        if reason:
            unlinked.extend({"instr_id": item["args"]["instr_id"], "channel": channel, "reason": reason} for item in gets + releases)
            continue
        source, target = gets[0], releases[0]
        identifier = "sync:" + ":".join(map(str, channel))
        source["sync_flow_id"] = target["sync_flow_id"] = identifier
        pairs.append({"id": identifier, "channel": channel, "set": source, "wait": target})
    return pairs, unlinked


def color_key(pipe):
    target = COLOR_INDEX.get(pipe, COLOR_INDEX["UNKNOWN"])
    for suffix in range(1000):
        key = f"{pipe}:{suffix}"
        if sum(31 * ord(char) for char in key) % 12 == target:
            return key
    raise ValueError(f"Cannot construct a color key for {pipe}")


def hardware_pipe(raw_pipe):
    if raw_pipe.startswith("RVEC"):
        return "VECTOR"
    return "FLOWCTRL" if raw_pipe == "FC" else raw_pipe


def union_intervals(records):
    """Merge recorded intervals only; preserve their union and source count."""
    groups = defaultdict(list)
    for record in records:
        groups[(record["core"], record["pipe"])].append(record)
    merged = []
    for (core, pipe), source in groups.items():
        active = None
        for record in sorted(source, key=lambda item: (item["start"], item["end"])):
            if active is not None and record["start"] <= active["end"]:
                active["end"] = max(active["end"], record["end"])
                active["count"] += 1
            else:
                active = dict(
                    core=core, pipe=pipe, start=record["start"], end=record["end"], count=1, name=f"{pipe} recorded interval union", args={}
                )
                merged.append(active)
    return sorted(merged, key=lambda item: (item["start"], item["core"], item["pipe"]))


def export(database, output, layout="pipe", overview=False):
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as db:
        info = dict(db.execute("SELECT * FROM KernelInfo"))
        frequency = float(info["frequency_ghz"])
        if frequency <= 0:
            raise ValueError("KernelInfo.frequency_ghz must be positive")
        # Requested display width only; GET/RLS have zero semantic duration.
        buffer_marker_ticks = 1
        rows = db.execute("""SELECT ExecInstrId, SourceInstrAddr,
            ExecInstrTickStart, ExecInstrTickEnd, ExecInstrName, ExecInstrCoreId,
            ExtendParams FROM ExecutedInstructions
            ORDER BY ExecInstrTickStart, ExecInstrId""").fetchall()
    if not rows:
        raise ValueError("ExecutedInstructions is empty")
    if layout not in ("pipe", "slots"):
        raise ValueError("layout must be pipe or slots")
    if overview and layout != "pipe":
        raise ValueError("overview requires pipe layout")
    records = []
    for identifier, pc, start, end, name, core_id, raw_params in rows:
        if end < start:
            raise ValueError(f"Instruction {identifier}: end precedes start")
        params = json.loads(raw_params or "{}")
        core = f"CORE{core_id}.{params.get('core_type', 'UNKNOWN')}"
        raw_pipe = params.get("instr_type", "UNKNOWN")
        pipe = hardware_pipe(raw_pipe)
        args = dict(params, instr_id=identifier, pc=hex(pc), start_tick=start, end_tick=end, raw_pipe=raw_pipe, hardware_pipe=pipe)
        if name in BUFFER_NAMES:
            # These are zero-cost synchronization endpoints. The helper can
            # synthesize [completion-1, completion] for completion-only events.
            # Display width is one tick; semantic duration and raw ticks remain.
            start = end
            args.update(
                sync_point_tick=end,
                semantic_duration_cycles=0,
                display_duration_cycles=buffer_marker_ticks,
                display_duration_only=True,
                point_semantics="zero-duration buffer synchronization; width is for viewer hit-testing",
            )
            end += buffer_marker_ticks
        records.append(dict(core=core, pipe=pipe, raw_pipe=raw_pipe, start=start, end=end, name=name, args=args, count=1))
    pairs, unlinked = build_sync_pairs(records)
    buffer_pairs, buffer_unlinked = build_buffer_pairs(records)
    pairs.extend(buffer_pairs)
    unlinked.extend(buffer_unlinked)
    if overview:
        # Keep every synchronization endpoint, including zero-duration GET/RLS.
        sync_records = [record for record in records if record["name"] in SYNC_NAMES]
        ordinary = [record for record in records if record["name"] not in SYNC_NAMES]
        records = sorted(union_intervals(ordinary) + sync_records, key=lambda item: (item["start"], item["core"], item["pipe"]))
    events, lanes, tracks = [], {}, set()
    compatibility_anchors = 0
    core_names = sorted({record["core"] for record in records})
    for order, core in enumerate(core_names):
        events.extend(
            [
                {"ph": "M", "name": "process_name", "pid": core, "tid": 0, "args": {"name": core}},
                {"ph": "M", "name": "process_sort_index", "pid": core, "tid": 0, "args": {"sort_index": order}},
            ]
        )
    for record in records:
        pid, pipe = record["core"], record["pipe"]
        start, end, args = record["start"], record["end"], record["args"]
        tid = pipe
        if layout == "slots":
            slots = lanes.setdefault((pid, pipe), [])
            slot = next((index for index, last_end in enumerate(slots) if last_end <= start), len(slots))
            if slot == len(slots):
                slots.append(end)
            else:
                slots[slot] = end
            tid = f"{pipe}/slot{slot}"
        track = (pid, tid)
        if track not in tracks:
            tracks.add(track)
            events.append({"ph": "M", "name": "thread_name", "pid": pid, "tid": tid, "args": {"name": tid}})
            events.append(
                {
                    "ph": "M",
                    "name": "thread_sort_index",
                    "pid": pid,
                    "tid": tid,
                    "args": {"sort_index": 3 * (PIPE_ORDER.index(pipe) if pipe in PIPE_ORDER else len(PIPE_ORDER))},
                }
            )
        is_sync_endpoint = record["name"] in SYNC_NAMES
        if overview and not is_sync_endpoint:
            args = dict(
                start_tick=start,
                end_tick=end,
                source_instruction_count=record["count"],
                interval_semantics="union of recorded intervals; may include waits",
            )
        elif overview:
            args = dict(args, source_instruction_count=1)
        color_role = (
            "BUFFER"
            if record["name"] in BUFFER_NAMES
            else "SYNC"
            if is_sync_endpoint or args.get("is_sync") or args.get("is_wait")
            else pipe
        )
        event = {
            "name": record["name"],
            "cat": record.get("raw_pipe", pipe),
            "pid": pid,
            "tid": tid,
            "ts": start / (frequency * 1000),
            "cname": color_key(color_role),
            "args": args,
        }
        record["display_tid"] = tid
        if is_sync_endpoint:
            # Insight only populates slice.flagId from B/E.id, not from X.id
            # or args.flag_id. The same ID must also be used by flow events.
            sync_id = record.get("sync_flow_id", f"unlinked:{pid}:{args['instr_id']}")
            event.update(ph="B", id=sync_id, args=dict(args, sync_flow_id=sync_id))
            if record["name"] not in {"SET_FLAG", "WAIT_FLAG"}:
                # Always preserve the real mnemonic. Stock Insight ignores
                # X.id, so GET/RLS also need an explicitly marked B/E anchor.
                events.append(dict(event, ph="X", dur=(end - start) / (frequency * 1000)))
                if record["name"] in BUFFER_NAMES:
                    # The stock simulation flow query chooses the first slice
                    # for (flow ID, track), ignoring the endpoint timestamp.
                    # Separate role tracks keep that lookup unambiguous and
                    # avoid stacking anchors under original/compute slices.
                    anchor_tid = f"{pipe} / {record['name']} (compat)"
                    record["flow_tid"] = anchor_tid
                    if (pid, anchor_tid) not in tracks:
                        tracks.add((pid, anchor_tid))
                        events.extend(
                            [
                                {"ph": "M", "name": "thread_name", "pid": pid, "tid": anchor_tid, "args": {"name": anchor_tid}},
                                {
                                    "ph": "M",
                                    "name": "thread_sort_index",
                                    "pid": pid,
                                    "tid": anchor_tid,
                                    "args": {
                                        "sort_index": 3 * (PIPE_ORDER.index(pipe) if pipe in PIPE_ORDER else len(PIPE_ORDER))
                                        + (1 if record["name"] == "GET_BUF" else 2)
                                    },
                                },
                            ]
                        )
                    anchor_args = dict(
                        event["args"],
                        viewer_compatibility_anchor=True,
                        original_instruction_name=record["name"],
                        source_instr_id=args["instr_id"],
                        source_display_tid=tid,
                        lane_semantics="viewer-only GET/RLS flow anchor; not a hardware PIPE",
                        compatibility_purpose="MindStudio flow linkage only; not an executed flag instruction",
                    )
                    anchor = dict(
                        event,
                        tid=anchor_tid,
                        name="SET_FLAG" if record["name"] == "GET_BUF" else "WAIT_FLAG",
                        cat="Viewer compatibility anchor (GET/RLS)",
                        args=anchor_args,
                    )
                    events.append(anchor)
                    events.append(dict(anchor, ph="E", ts=end / (frequency * 1000)))
                    compatibility_anchors += 1
                continue
            events.append(event)
            events.append(dict(event, ph="E", ts=end / (frequency * 1000)))
        else:
            event.update(ph="X", dur=(end - start) / (frequency * 1000))
            events.append(event)
    for pair in pairs:
        source, target = pair["set"], pair["wait"]
        category = {"flags": "Pipe dependency (flags)", "intra_blocks": "Pipe dependency (intra blocks)", "buffers": BUFFER_FLOW_CATEGORY}[
            pair["channel"][0]
        ]
        identities = (
            {
                "get_instr_id": source["args"]["instr_id"],
                "release_instr_id": target["args"]["instr_id"],
                "semantics": "acquire/release correspondence; not instruction latency",
            }
            if pair["channel"][0] == "buffers"
            else {"set_instr_id": source["args"]["instr_id"], "wait_instr_id": target["args"]["instr_id"]}
        )
        common = {
            "id": pair["id"],
            "name": f"{source['name']} -> {target['name']}",
            "cat": category,
            "cname": color_key("BUFFER" if pair["channel"][0] == "buffers" else "SYNC"),
            "args": {"channel": pair["channel"], **identities},
        }
        events.append(
            dict(
                common,
                ph="s",
                pid=source["core"],
                tid=source.get("flow_tid", source["display_tid"]),
                ts=source["start"] / (frequency * 1000),
            )
        )
        finish = dict(
            common,
            ph="t",
            pid=target["core"],
            tid=target.get("flow_tid", target["display_tid"]),
            ts=target["start"] / (frequency * 1000),
            bp="e",
        )
        events.append(finish)
        # Insight's simulation parser consumes s/t but ignores f. Also close
        # the chain for ordinary Chrome Trace viewers with an f at that anchor.
        events.append(dict(finish, ph="f"))
    output.write_text(
        json.dumps(
            {
                "profilingType": "op",
                "schemaVersion": 1,
                "traceEvents": events,
                "displayTimeUnit": "ns",
                "otherData": {
                    "source_database": str(database),
                    "kernel_info": info,
                    "timestamp_unit": "microseconds",
                    "timestamp_origin": "original simulator tick zero",
                    "interval_semantics": "recorded start-to-end; may include queue/dependency waits",
                    "lane_semantics": "one hardware pipe per core, plus explicitly marked viewer-only GET/RLS compatibility lanes"
                    if layout == "pipe"
                    else "virtual visual slots; not physical units",
                    "view": "recorded_interval_union" if overview else "individual_instructions",
                    "source_instruction_count": len(rows),
                    "exported_interval_count": len(records),
                    "synchronization": {
                        "pairs": len(pairs),
                        "flag_pairs": sum(pair["channel"][0] == "flags" for pair in pairs),
                        "intra_block_pairs": sum(pair["channel"][0] == "intra_blocks" for pair in pairs),
                        "buffer_pairs": len(buffer_pairs),
                        "buffer_pair_semantics": "GET/RLS acquire/release scope within one core, PIPE, buffer ID, mode and epoch",
                        "zero_duration_buffer_events": sum(row[4] in BUFFER_NAMES for row in rows),
                        "buffer_marker_width_cycles": buffer_marker_ticks,
                        "buffer_marker_semantics": "display/hit-test width only; semantic duration is zero",
                        "compatibility_anchors": compatibility_anchors,
                        "compatibility_anchor_width_cycles": 1,
                        "compatibility_anchor_lanes": "separate GET_BUF and RLS_BUF viewer tracks per hardware PIPE; flow endpoints use the anchor tracks",
                        "buffer_flow_category": BUFFER_FLOW_CATEGORY,
                        "buffer_color": "sunsetOrange; cname for slices and category hash for lines",
                        "compatibility_anchor_semantics": "synthetic SET_FLAG/WAIT_FLAG for GET/RLS flow linkage only; excluded from instruction and cycle counts",
                        "buffer_encoding": "original GET/RLS X slices plus 1-tick B/E SET_FLAG/WAIT_FLAG compatibility anchors",
                        "insight_linking": "Stock Insight links the marked SET_FLAG/WAIT_FLAG anchors; original X slices require parser support for X.id. One tick can quantize to zero integer nanoseconds.",
                        "buffer_point_anchor": "raw completion/end_tick; original start_tick and end_tick retained in args",
                        "unlinked": unlinked,
                        "flow_phases": "s/t for Insight plus f for Chrome Trace",
                    },
                    "cname_palette_indices": COLOR_INDEX,
                },
            }
        ),
        encoding="utf-8",
    )
    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layout", choices=("pipe", "slots"), default="pipe")
    parser.add_argument("--overview", action="store_true", help="Merge overlapping recorded intervals within each PIPE")
    arguments = parser.parse_args()
    count = export(arguments.database, arguments.output, arguments.layout, arguments.overview)
    print(f"Read {count} recorded instructions; wrote {arguments.output}")
