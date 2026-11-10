#!/usr/bin/env python3
"""Cube-only useful-operation utilization. Python 3.9+, standard library only."""

import argparse
import json
import math
from pathlib import Path


CATALOG = Path(__file__).resolve().parents[1] / "references" / "cube-peaks.json"
DTYPES = ("fp16", "bf16", "tf32", "fp8", "mxfp8", "hif8", "mxfp4", "int8")


def integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def causal_pairs(m, n, offset=0):
    """Count (i,j) satisfying j <= i + offset without allocating an M*N mask."""
    lo = max(0, -offset)
    hi = min(m, n - offset - 1)
    count = max(0, hi - lo)
    ramp = count * (lo + hi - 1 + 2 * (offset + 1)) // 2
    full_rows = max(0, m - max(0, n - offset - 1))
    return ramp + full_rows * n


def gemm_ops(spec):
    if not isinstance(spec, dict):
        raise ValueError("each GEMM must be an object")
    allowed = {"name", "m", "n", "k", "batch", "mask", "causal_offset", "valid_pairs", "valid_macs"}
    if set(spec) - allowed:
        raise ValueError(f"unknown GEMM fields: {sorted(set(spec) - allowed)}")
    for key in ("m", "n", "k"):
        if key not in spec:
            raise ValueError(f"missing GEMM dimension: {key}")
    m, n, k = (integer(spec[key], key, 1) for key in ("m", "n", "k"))
    batch = integer(spec.get("batch", 1), "batch", 1)
    mask = spec.get("mask", "dense")
    if mask not in ("dense", "causal"):
        raise ValueError("mask must be dense or causal")
    selectors = sum(key in spec for key in ("valid_pairs", "valid_macs")) + (mask == "causal")
    if selectors > 1:
        raise ValueError("causal mask, valid_pairs, and valid_macs are mutually exclusive")
    if "causal_offset" in spec and mask != "causal":
        raise ValueError("causal_offset requires mask=causal")
    if "valid_macs" in spec:
        macs = integer(spec["valid_macs"], "valid_macs")
    else:
        if mask == "causal":
            offset = spec.get("causal_offset", 0)
            if type(offset) is not int:
                raise ValueError("causal_offset must be an integer")
            pairs = causal_pairs(m, n, offset)
        else:
            pairs = integer(spec.get("valid_pairs", m * n), "valid_pairs")
        if pairs > m * n:
            raise ValueError("valid_pairs exceeds M*N")
        macs = pairs * k
    if macs > m * n * k:
        raise ValueError("valid_macs exceeds M*N*K")
    return {
        "spec": spec,
        "dense_ops": 2 * batch * m * n * k,
        "useful_ops": 2 * batch * macs,
        "excluded_ops": 2 * batch * (m * n * k - macs),
    }


def calculate(device, device_cores, cores, dtype, time, unit, iterations, specs):
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    integer(device_cores, "device_cores", 1)
    integer(cores, "cores", 1)
    integer(iterations, "iterations", 1)
    if cores > device_cores:
        raise ValueError("used Cube cores exceed device Cube cores")
    if isinstance(time, bool) or not isinstance(time, (int, float)) or not math.isfinite(time) or time <= 0:
        raise ValueError("time must be finite and > 0")
    if unit not in ("s", "ms", "us", "ns"):
        raise ValueError("unsupported time unit")
    try:
        peak = catalog["devices"][device][str(device_cores)][dtype]
    except KeyError:
        raise ValueError("unverified device/core-count/dtype combination; use --list-devices") from None
    if not isinstance(specs, list) or not specs:
        raise ValueError("GEMM input must be a nonempty JSON array")
    details = [gemm_ops(spec) for spec in specs]
    seconds = time * {"s": 1, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}[unit] / iterations
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("normalized duration is out of range")
    effective_peak = peak * cores / device_cores
    useful = sum(item["useful_ops"] for item in details)
    achieved = useful / seconds / 1e12
    utilization = achieved / effective_peak * 100
    if not math.isfinite(utilization):
        raise ValueError("calculated utilization is out of range")
    return {
        "metric": "Cube integer operation utilization" if dtype == "int8" else "Cube MFU",
        "device": device, "dtype": dtype, "device_cube_cores": device_cores, "used_cube_cores": cores,
        "time_input": {"value": time, "unit": unit, "iterations": iterations},
        "seconds_per_iteration": seconds,
        "dense_ops_per_iteration": sum(item["dense_ops"] for item in details),
        "useful_ops_per_iteration": useful,
        "excluded_ops_per_iteration": sum(item["excluded_ops"] for item in details),
        "peak_unit": "TOPS" if dtype == "int8" else "TFLOPS",
        "device_cube_peak": peak, "used_cube_peak": effective_peak,
        "achieved_teraops_per_second": achieved,
        "utilization_percent": utilization,
        "mfu_percent": None if dtype == "int8" else utilization,
        "gemms": details,
        "source_url": catalog["source_url"], "source_table": catalog["source_table"],
        "source_checked_on": catalog["checked_on"],
        "warnings": ["Above 100%: check timing scope, iterations, dtype, device variant, cores and mask counts."] if utilization > 100 else [],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--device", choices=("950pr", "950dt"), type=str.lower)
    parser.add_argument("--device-cores", type=int, help="total Cube cores of the verified physical variant")
    parser.add_argument("--cores", type=int, help="participating physical Cube cores, not AIVs or logical blocks")
    parser.add_argument("--dtype", choices=DTYPES, type=str.lower, help="actual Cube instruction precision")
    parser.add_argument("--time", type=float, help="total measured time for --iterations repetitions")
    parser.add_argument("--unit", choices=("s", "ms", "us", "ns"))
    parser.add_argument("--iterations", type=int, default=1)
    for dim in ("m", "n", "k", "batch", "valid-pairs", "valid-macs", "causal-offset"):
        parser.add_argument("--" + dim, type=int)
    parser.add_argument("--mask", choices=("dense", "causal"))
    parser.add_argument("--gemms-json", type=Path, help="JSON array of GEMMs executed once per iteration; one common dtype")
    parser.add_argument("--json", action="store_true", help="emit full audit details as JSON")
    args = parser.parse_args()
    if args.list_devices:
        print(CATALOG.read_text(encoding="utf-8"))
        return
    for key in ("device", "device_cores", "cores", "dtype", "time", "unit"):
        if getattr(args, key) is None:
            parser.error("required argument: --" + key.replace("_", "-"))
    spec = {key: getattr(args, key) for key in
            ("m", "n", "k", "batch", "mask", "valid_pairs", "valid_macs", "causal_offset")
            if getattr(args, key) is not None}
    try:
        if args.gemms_json:
            if spec:
                raise ValueError("--gemms-json cannot be combined with single-GEMM parameters")
            specs = json.loads(args.gemms_json.read_text(encoding="utf-8-sig"))
        else:
            specs = [spec]
        result = calculate(args.device, args.device_cores, args.cores, args.dtype,
                           args.time, args.unit, args.iterations, specs)
    except (ValueError, OSError, OverflowError) as error:
        parser.error(str(error))
    if args.json:
        print(json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False))
    else:
        print(f"{result['metric']}: {result['utilization_percent']:.6f}%")
        print(f"Device: {args.device}, Cube cores: {args.cores}/{args.device_cores}, dtype: {args.dtype}")
        print(f"Useful ops/iteration: {result['useful_ops_per_iteration']}; excluded: {result['excluded_ops_per_iteration']}")
        print(f"Time/iteration: {result['seconds_per_iteration']:.9g} s")
        print(f"Achieved: {result['achieved_teraops_per_second']:.9g}; Cube peak: {result['used_cube_peak']:.9g} {result['peak_unit']}")
        print(f"Source: {result['source_url']} ({result['source_checked_on']})")
        for warning in result["warnings"]:
            print("WARNING: " + warning)


if __name__ == "__main__":
    main()
