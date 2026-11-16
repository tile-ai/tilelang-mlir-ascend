"""Summarize SIMT probe profiles: geometry, occupancy, and isolated per-op cost."""

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROFILES = ROOT / "profiles"
TILE = 4096
MATH_OPS = {"exp", "expdif", "expm", "exps", "ln", "sqrt", "rsqrt", "rcp", "sigmoid"}
TAG = re.compile(r"(?P<op>[a-z]+)-r(?P<rounds>\d+)(?:-t(?P<tiles>\d+))?(?:-(?P<rep>[a-z]))?")


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return [row for row in csv.DictReader(stream) if row.get("block_id") not in (None, "")]


def table(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return [row for row in csv.DictReader(stream) if any(row.values())]


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean(source, key):
    values = [number(row[key]) for row in source]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def summarize(name):
    found = sorted((PROFILES / name).glob("OPPROF_*"))
    if len(found) != 1:
        raise SystemExit(f"{name}: expected one OPPROF directory, found {len(found)}")
    directory = found[0]
    basic = table(directory / "OpBasicInfo.csv")
    arithmetic = rows(directory / "ArithmeticUtilization.csv")
    pipe = rows(directory / "PipeUtilization.csv")
    if len(basic) != 1:
        raise SystemExit(f"{name}: expected one OpBasicInfo row")
    blocks = int(basic[0]["Block Dim"])
    if len(arithmetic) != blocks or len(pipe) != blocks:
        raise SystemExit(f"{name}: expected {blocks} AIV rows, got {len(arithmetic)}/{len(pipe)}")
    duration = number(basic[0]["Task Duration(us)"])
    if not duration or duration <= 0:
        raise SystemExit(f"{name}: bad duration")
    cycles = [number(row["aiv_total_cycles"]) for row in arithmetic]
    cycles = [value for value in cycles if value]
    if not cycles:
        raise SystemExit(f"{name}: no cycle counters")

    return {
        "config": name,
        "task_duration_us": duration,
        "blocks": blocks,
        "mean_aiv_cycles": sum(cycles) / len(cycles),
        "min_aiv_cycles": min(cycles),
        "max_aiv_cycles": max(cycles),
        "aiv_vec_ratio": mean(arithmetic, "aiv_vec_ratio"),
        "aiv_vec_simt_vf_ratio": mean(arithmetic, "aiv_vec_simt_vf_ratio"),
        "aiv_vec_sfu_ratio": mean(arithmetic, "aiv_vec_sfu_ratio"),
        "aiv_scalar_ratio": mean(pipe, "aiv_scalar_ratio"),
        "aiv_mte2_ratio": mean(pipe, "aiv_mte2_ratio"),
        "frequency_mhz": number(basic[0]["Current Freq"]),
        "rated_frequency_mhz": number(basic[0]["Rated Freq"]),
    }


def main():
    captures = {}
    tags = {}
    for path in sorted(PROFILES.iterdir()):
        match = TAG.fullmatch(path.name)
        if path.is_dir() and match:
            tags[path.name] = match.groupdict()
            captures[path.name] = summarize(path.name)
    if not captures:
        raise SystemExit("no profiles found")

    for name, data in captures.items():
        tag = tags[name]
        tiles = int(tag["tiles"] or 4)
        rounds = int(tag["rounds"])
        data.update(op=tag["op"], rounds=rounds, tiles=tiles, repetition=tag["rep"] or "a")
        data["elements_per_block"] = tiles * TILE
        data["element_rounds_per_block"] = data["elements_per_block"] * rounds
        data["element_ops_per_cycle"] = data["element_rounds_per_block"] / data["mean_aiv_cycles"]
        data["vector_instruction_cycles"] = 128 / data["element_ops_per_cycle"]
        if tag["op"] == "fma":
            seconds = data["mean_aiv_cycles"] / (data["frequency_mhz"] * 1e6)
            data["achieved_tflops"] = (
                2 * data["elements_per_block"] * rounds * data["blocks"] / seconds / 1e12
            )

    # Baseline: control capture with the same rounds, tiles and repetition.
    controls = {
        (data["rounds"], data["tiles"], data["repetition"]): data
        for data in captures.values() if data["op"] == "control"
    }
    isolated = {}
    for name, data in captures.items():
        if data["op"] not in MATH_OPS:
            continue
        base = controls.get((data["rounds"], data["tiles"], data["repetition"]))
        if not base:
            continue
        delta_cycles = data["mean_aiv_cycles"] - base["mean_aiv_cycles"]
        per_element_round = delta_cycles / data["element_rounds_per_block"]
        isolated[name] = {
            "op": data["op"],
            "rounds": data["rounds"],
            "tiles": data["tiles"],
            "control_cycles": base["mean_aiv_cycles"],
            "measured_cycles": data["mean_aiv_cycles"],
            "delta_cycles": delta_cycles,
            "cycles_per_op": per_element_round,
            "ops_per_cycle_per_aiv": 1 / per_element_round if per_element_round > 0 else None,
            "vector_instruction_cycles": per_element_round * 128,
            "total_ops_per_s": None,
        }
        if per_element_round > 0:
            seconds = data["mean_aiv_cycles"] / (data["frequency_mhz"] * 1e6)
            isolated[name]["total_ops_per_s"] = (
                data["element_rounds_per_block"] * data["blocks"] / seconds
            )

    report = {
        "geometry": {"tile": TILE, "blocks_default": 72, "aiv_lanes_fp32": 128},
        "captures": captures,
        "isolated": isolated,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (ROOT / "results.json").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

