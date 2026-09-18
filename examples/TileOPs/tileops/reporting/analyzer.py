"""Normalize correctness and performance records into report-ready data."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import PurePosixPath
from typing import Any


def _profile_validity(result: dict[str, Any]) -> tuple[bool, str | None]:
    """Return whether one TileOps timing record is valid for reporting."""
    if result.get("prof_mode") != "msprof":
        return True, None
    metadata = result.get("msprof") or {}
    requested = metadata.get("kernel_name_requested")
    resolved = metadata.get("kernel_name_resolved")
    if requested and resolved is None:
        return False, f"kernel filter {requested!r} fell back to an unfiltered capture"
    captured_count = metadata.get("captured_op_count")
    if isinstance(captured_count, int) and captured_count > 1:
        return False, f"unfiltered capture contains {captured_count} distinct ops"
    return True, None


def _analyze_performance(benchmark: dict[str, Any]) -> tuple[list[dict], dict, list[str]]:
    requested_mode = benchmark.get("profiling_mode_requested")
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in benchmark.get("records", []):
        grouped[(record.get("operator", "unknown"), record.get("case_id", "unknown"))].append(
            record
        )

    cases = []
    warnings = []
    fallback_count = 0
    kernel_filter_fallback_count = 0
    invalid_profile_count = 0

    for (operator, case_id), records in grouped.items():
        candidate = next(
            (record for record in records if str(record.get("tag", "")).startswith("tileops")),
            None,
        )
        if candidate is None:
            warnings.append(f"{operator}/{case_id}: missing tileops candidate record")
            continue

        result = candidate.get("result", {})
        actual_mode = result.get("prof_mode")
        candidate_valid, candidate_invalid_reason = _profile_validity(result)
        if not candidate_valid:
            invalid_profile_count += 1
        if requested_mode == "msprof" and actual_mode != "msprof":
            fallback_count += 1
            warnings.append(
                f"{operator}/{case_id}: requested msprof but recorded {actual_mode or 'unknown'}"
            )

        msprof_metadata = result.get("msprof") or {}
        if (
            actual_mode == "msprof"
            and msprof_metadata.get("kernel_name_requested")
            and msprof_metadata.get("kernel_name_resolved") is None
        ):
            kernel_filter_fallback_count += 1
            warnings.append(
                f"{operator}/{case_id}: requested kernel filter "
                f"{msprof_metadata['kernel_name_requested']!r} produced no data; "
                "msprof retried without a kernel filter"
            )
        elif not candidate_valid:
            warnings.append(
                f"{operator}/{case_id}: invalid tileops profile: {candidate_invalid_reason}"
            )

        roofline_ratio = result.get("Ratio(%)")

        cases.append(
            {
                "operator": operator,
                "case_id": case_id,
                "label": candidate.get("label") or candidate.get("params", {}).get("label"),
                "params": candidate.get("params", {}),
                "config": candidate.get("config"),
                "tag": candidate.get("tag", "tileops"),
                "latency_us": result.get("latency_us"),
                "prof_mode": actual_mode,
                "profiler_valid": candidate_valid,
                "invalid_reason": candidate_invalid_reason,
                "roofline_utilization_percent": roofline_ratio,
                "bandwidth_tb_s": result.get("Bandwidth(TB/s)", result.get("bandwidth_tbs")),
                "arithmetic_intensity_ops_per_byte": result.get("AI(Ops/Byte)"),
                "performance_tops": result.get("Perf(TOps/s)", result.get("tflops")),
                "computility_tops": result.get("Computility(TOps/s)"),
                "artifact_dir": result.get("artifact_dir"),
                "msprof": msprof_metadata or None,
            }
        )

    summary = {
        "case_count": len(cases),
        "msprof_case_count": sum(1 for case in cases if case["prof_mode"] == "msprof"),
        "profiler_fallback_count": fallback_count,
        "kernel_filter_fallback_count": kernel_filter_fallback_count,
        "invalid_profile_count": invalid_profile_count,
    }
    return cases, summary, warnings


def _test_key(path: str | None) -> str:
    if not path:
        return ""
    return PurePosixPath(path.replace("\\", "/")).stem.lower()


def _case_operator(
    case: dict[str, Any], operator_catalog: list[dict[str, Any]], default: str | None
) -> str | None:
    """Map a JUnit case to its manifest operator through the test filename."""
    declared = (case.get("properties") or {}).get("op")
    if declared:
        return str(declared)
    haystack = f"{case.get('classname', '')} {case.get('nodeid', '')}".lower()
    matches = [
        entry.get("name")
        for entry in operator_catalog
        if _test_key(entry.get("test")) and _test_key(entry.get("test")) in haystack
    ]
    return matches[0] if len(matches) == 1 else default


def _range(values: list[float]) -> dict[str, float] | None:
    return {"min": min(values), "max": max(values)} if values else None


def _numeric_property(case: dict[str, Any], name: str) -> float | None:
    try:
        value = float((case.get("properties") or {}).get(name))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _analyze_operators(
    *,
    requested_operator: str,
    correctness: dict[str, Any],
    perf_cases: list[dict[str, Any]],
    operator_catalog: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    perf_names = {str(case.get("operator")) for case in perf_cases if case.get("operator")}
    correctness_names = {
        str((case.get("properties") or {}).get("op"))
        for case in correctness.get("cases", [])
        if (case.get("properties") or {}).get("op")
    }
    if requested_operator == "all":
        names = [str(entry["name"]) for entry in operator_catalog if entry.get("name")]
        names.extend(sorted((perf_names | correctness_names).difference(names)))
    else:
        names = [requested_operator]

    catalog_by_name = {str(entry.get("name")): entry for entry in operator_catalog}
    correctness_by_operator: dict[str, list[dict[str, Any]]] = defaultdict(list)
    default = names[0] if len(names) == 1 else None
    for case in correctness.get("cases", []):
        name = _case_operator(case, operator_catalog, default)
        if name:
            correctness_by_operator[name].append(case)

    results = []
    for name in names:
        correct_cases = correctness_by_operator.get(name, [])
        if len(names) == 1 and not correct_cases and not correctness.get("cases"):
            total = int(correctness.get("tests", 0) or 0)
            passed = int(correctness.get("passed", 0) or 0)
            failed = int(correctness.get("failed", 0) or 0) + int(correctness.get("errors", 0) or 0)
        else:
            total = len(correct_cases)
            passed = sum(1 for case in correct_cases if case.get("outcome") == "passed")
            failed = sum(1 for case in correct_cases if case.get("outcome") in {"failed", "error"})

        op_perf_cases = [case for case in perf_cases if case.get("operator") == name]
        ratios = [
            float(case["roofline_utilization_percent"])
            for case in op_perf_cases
            if isinstance(case.get("roofline_utilization_percent"), (int, float))
        ]
        pass_rate = passed / total if total else None
        max_abs_errors = [
            value
            for case in correct_cases
            if (value := _numeric_property(case, "max_abs_err")) is not None
        ]
        entry = catalog_by_name.get(name, {})
        results.append(
            {
                "operator": name,
                "family": entry.get("family"),
                "cases": total,
                "passed": passed,
                "failed": failed,
                "pass_rate": pass_rate,
                "avg_max_abs_err": (
                    sum(max_abs_errors) / len(max_abs_errors) if max_abs_errors else None
                ),
                "max_abs_err": max(max_abs_errors) if max_abs_errors else None,
                "performance_cases": len(op_perf_cases),
                "ratio_range": _range(ratios),
            }
        )
    return results


def analyze_run(
    *,
    operator: str,
    correctness: dict[str, Any],
    correctness_exit_code: int,
    benchmark: dict[str, Any] | None = None,
    benchmark_exit_code: int | None = None,
    benchmark_tests: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    setup: dict[str, Any] | None = None,
    operator_catalog: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the canonical run result consumed by all report renderers."""
    benchmark = benchmark or {"records": [], "status": "skipped"}
    benchmark_tests = benchmark_tests or {
        "status": "skipped",
        "tests": 0,
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "cases": [],
    }
    perf_cases, perf_summary, warnings = _analyze_performance(benchmark)
    operators = _analyze_operators(
        requested_operator=operator,
        correctness=correctness,
        perf_cases=perf_cases,
        operator_catalog=operator_catalog or [],
    )

    correctness_passed = correctness_exit_code == 0 and correctness.get("status") == "passed"
    benchmark_requested = benchmark_exit_code is not None
    benchmark_failed = int(benchmark_tests.get("failed", 0) or 0) + int(
        benchmark_tests.get("errors", 0) or 0
    )
    benchmark_passed = (
        benchmark_requested
        and benchmark_exit_code == 0
        and benchmark.get("status") == "present"
        and bool(perf_cases)
        and perf_summary["profiler_fallback_count"] == 0
        and perf_summary["kernel_filter_fallback_count"] == 0
        and perf_summary["invalid_profile_count"] == 0
        and benchmark_failed == 0
    )
    if correctness_passed and benchmark_requested and not benchmark_passed:
        warnings.append("correctness passed, but the benchmark stage did not produce valid records")
    if benchmark_failed:
        warnings.append(f"benchmark pytest reported {benchmark_failed} failed/error cases")

    total_cases = int(correctness.get("tests", 0) or 0)
    passed_cases = int(correctness.get("passed", 0) or 0)
    failed_cases = int(correctness.get("failed", 0) or 0) + int(correctness.get("errors", 0) or 0)

    return {
        "schema_version": 2,
        "operator": operator,
        "status": (
            "failed" if not correctness_passed else "passed" if benchmark_passed else "partial"
        ),
        "metadata": metadata or {},
        "setup": setup or {},
        "summary": {
            "correctness_passed": correctness_passed,
            "correctness_tests": correctness.get("tests", 0),
            "correctness_failed": correctness.get("failed", 0) + correctness.get("errors", 0),
            "operator_count": len(operators),
            "total_cases": total_cases,
            "passed_cases": passed_cases,
            "failed_cases": failed_cases,
            "pass_rate": passed_cases / total_cases if total_cases else None,
            "benchmark_requested": benchmark_requested,
            "benchmark_passed": benchmark_passed,
            "benchmark_tests": benchmark_tests.get("tests", 0),
            "benchmark_failed": benchmark_failed,
            **perf_summary,
        },
        "correctness": correctness,
        "benchmark_tests": benchmark_tests,
        "performance": {
            "source": benchmark.get("path"),
            "requested_prof_mode": benchmark.get("profiling_mode_requested"),
            "cases": perf_cases,
        },
        "operators": operators,
        "warnings": warnings,
    }
