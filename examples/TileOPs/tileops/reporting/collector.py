"""Collectors for pytest JUnit XML and TileOPs benchmark JSON."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def _number(value: str | None) -> int:
    try:
        return int(value or 0)
    except ValueError:
        return 0


def parse_junit_report(path: str | Path) -> dict[str, Any]:
    """Parse a pytest JUnit report into a small stable result schema."""
    report_path = Path(path)
    if not report_path.exists():
        return {
            "path": str(report_path),
            "status": "missing",
            "tests": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "skipped": 0,
            "cases": [],
        }

    root = ET.parse(report_path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    cases: list[dict[str, Any]] = []

    for case in root.iter("testcase"):
        outcome = "passed"
        detail = None
        for tag in ("failure", "error", "skipped"):
            node = case.find(tag)
            if node is not None:
                outcome = "failed" if tag == "failure" else tag
                detail = node.get("message") or (node.text or "").strip() or None
                break
        properties = {
            prop.get("name", ""): prop.get("value", "")
            for prop in case.findall("./properties/property")
            if prop.get("name")
        }
        cases.append(
            {
                "nodeid": "::".join(
                    part for part in (case.get("classname"), case.get("name")) if part
                ),
                "name": case.get("name", ""),
                "classname": case.get("classname", ""),
                "time_s": float(case.get("time", "0") or 0),
                "outcome": outcome,
                "message": detail,
                "properties": properties,
            }
        )

    reported_tests = sum(_number(suite.get("tests")) for suite in suites)
    reported_failed = sum(_number(suite.get("failures")) for suite in suites)
    reported_errors = sum(_number(suite.get("errors")) for suite in suites)
    reported_skipped = sum(_number(suite.get("skipped")) for suite in suites)
    tests = max(reported_tests, len(cases))
    failed = max(reported_failed, sum(1 for case in cases if case["outcome"] == "failed"))
    errors = max(reported_errors, sum(1 for case in cases if case["outcome"] == "error"))
    skipped = max(reported_skipped, sum(1 for case in cases if case["outcome"] == "skipped"))
    passed = max(tests - failed - errors - skipped, 0)
    return {
        "path": str(report_path),
        "status": "passed" if failed == 0 and errors == 0 else "failed",
        "tests": tests,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
        "cases": cases,
    }


def load_benchmark_report(path: str | Path) -> dict[str, Any]:
    """Load the JSON emitted by ``BenchmarkReport.dump_json``."""
    report_path = Path(path)
    if not report_path.exists():
        return {
            "path": str(report_path),
            "status": "missing",
            "schema_version": 1,
            "records": [],
        }
    with report_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data["path"] = str(report_path)
    data["status"] = "present"
    return data
