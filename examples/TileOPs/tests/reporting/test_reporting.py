import json
from pathlib import Path

from tileops.benchmark.benchmark_base import BenchmarkReport
from tileops.benchmark.msprof import _parse_op_basic_info
from tileops.reporting.analyzer import analyze_run
from tileops.reporting.collector import parse_junit_report
from tileops.reporting.report import write_reports
from tileops.reporting.runner import resolve_operator, run_operator


def test_parse_junit_report_preserves_properties(tmp_path):
    report = tmp_path / "pytest.xml"
    report.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite tests="2" failures="1" errors="0" skipped="0">
  <testcase classname="tests.ops.test_x" name="test_ok" time="0.1">
    <properties><property name="max_abs_err" value="1.00e-04"/></properties>
  </testcase>
  <testcase classname="tests.ops.test_x" name="test_bad" time="0.2">
    <failure message="not close">details</failure>
  </testcase>
</testsuite></testsuites>""",
        encoding="utf-8",
    )

    parsed = parse_junit_report(report)

    assert parsed["tests"] == 2
    assert parsed["passed"] == 1
    assert parsed["failed"] == 1
    assert parsed["cases"][0]["properties"]["max_abs_err"] == "1.00e-04"


def test_parse_junit_report_counts_suite_level_errors(tmp_path):
    report = tmp_path / "collection_error.xml"
    report.write_text(
        '<testsuites><testsuite tests="1" failures="0" errors="1" skipped="0"/></testsuites>',
        encoding="utf-8",
    )

    parsed = parse_junit_report(report)

    assert parsed["status"] == "failed"
    assert parsed["tests"] == 1
    assert parsed["errors"] == 1
    assert parsed["passed"] == 0


def test_msprof_csv_parser_reports_distinct_captured_ops(tmp_path):
    first = tmp_path / "OPPROF_demo" / "Main" / "0"
    second = tmp_path / "OPPROF_demo" / "Aux" / "0"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "OpBasicInfo_0.csv").write_text(
        "Op Name,Task Duration(us)\nMain,8.5\n", encoding="utf-8"
    )
    (second / "OpBasicInfo_1.csv").write_text(
        "Op Name,Task Duration(us)\nAux,2.0\n", encoding="utf-8"
    )

    durations, first_name, names = _parse_op_basic_info(tmp_path)

    assert durations == [2.0, 8.5]
    assert first_name == "Aux"
    assert names == ["Aux", "Main"]


def test_analyzer_preserves_per_shape_roofline_ratio():
    benchmark = {
        "status": "present",
        "path": "benchmark.json",
        "profiling_mode_requested": "msprof",
        "records": [
            {
                "operator": "DemoOp",
                "case_id": "case-1",
                "label": "smoke-16",
                "tag": "tileops",
                "params": {"shape": [16], "dtype": "float16"},
                "result": {
                    "latency_us": 10.0,
                    "prof_mode": "msprof",
                    "Ratio(%)": 62.5,
                },
            },
        ],
    }
    correctness = {
        "status": "passed",
        "tests": 1,
        "passed": 1,
        "failed": 0,
        "errors": 0,
        "cases": [],
    }

    run = analyze_run(
        operator="DemoOp",
        correctness=correctness,
        correctness_exit_code=0,
        benchmark=benchmark,
        benchmark_exit_code=0,
    )

    case = run["performance"]["cases"][0]
    assert case["roofline_utilization_percent"] == 62.5
    assert case["label"] == "smoke-16"
    assert "baselines" not in case
    assert "average_roofline_utilization_percent" not in run["summary"]
    assert "speedup_range" not in run["operators"][0]
    assert run["operators"][0]["ratio_range"] == {"min": 62.5, "max": 62.5}


def test_analyzer_rejects_candidate_profiler_fallback():
    benchmark = {
        "status": "present",
        "profiling_mode_requested": "msprof",
        "records": [
            {
                "operator": "DemoOp",
                "case_id": "case-1",
                "tag": "tileops",
                "params": {},
                "result": {"latency_us": 10.0, "prof_mode": "events"},
            },
        ],
    }
    correctness = {
        "status": "passed",
        "tests": 1,
        "failed": 0,
        "errors": 0,
        "cases": [],
    }

    run = analyze_run(
        operator="DemoOp",
        correctness=correctness,
        correctness_exit_code=0,
        benchmark=benchmark,
        benchmark_exit_code=0,
    )

    assert run["status"] == "partial"
    assert run["summary"]["profiler_fallback_count"] == 1
    assert "baselines" not in run["performance"]["cases"][0]


def test_benchmark_report_writes_structured_json(tmp_path, monkeypatch):
    BenchmarkReport.clear()
    BenchmarkReport.set_prof_mode("events")
    monkeypatch.setenv(
        "PYTEST_CURRENT_TEST",
        "benchmarks/ops/bench_demo.py::test_demo[smoke-8x16-float16] (call)",
    )
    BenchmarkReport.record(
        "DemoOp",
        {"shape": (8, 16), "dtype": "float16"},
        {"latency_us": 12.5, "Ratio(%)": 62.5, "prof_mode": "events"},
        tag="tileops",
    )

    output = BenchmarkReport.dump_json(tmp_path / "benchmark.json")
    data = json.loads(output.read_text(encoding="utf-8"))

    assert data["schema_version"] == 1
    assert data["profiling_mode_requested"] == "events"
    assert data["records"][0]["operator"] == "DemoOp"
    assert data["records"][0]["case_id"].startswith("DemoOp-")
    assert data["records"][0]["label"] == "smoke-8x16"

    markdown = tmp_path / "session.md"
    monkeypatch.setenv("TILEOPS_BENCHMARK_REPORT_PATH", str(markdown))
    BenchmarkReport.dump()
    assert markdown.is_file()
    assert markdown.with_suffix(".json").is_file()
    header = next(
        line
        for line in markdown.read_text(encoding="utf-8").splitlines()
        if line.startswith("| Label |")
    )
    assert header.startswith("| Label | Latency (us) | Ratio (%) |")
    BenchmarkReport.clear()


def test_write_reports_creates_all_formats(tmp_path):
    run = {
        "operator": "DemoOp",
        "status": "passed",
        "summary": {
            "correctness_passed": True,
            "correctness_tests": 1,
            "correctness_failed": 0,
            "operator_count": 1,
            "total_cases": 1,
            "passed_cases": 1,
            "failed_cases": 0,
            "pass_rate": 1.0,
            "benchmark_passed": True,
            "case_count": 1,
            "msprof_case_count": 1,
            "profiler_fallback_count": 0,
            "kernel_filter_fallback_count": 0,
        },
        "warnings": [],
        "setup": {
            "metadata": {
                "framework": "TileOPs Reporting 0.1.0",
                "date": "2026-09-17T10:00:00+08:00",
                "evaluation_scope": "DemoOp",
                "profiler": "msprof",
            },
            "environment": {
                "npu": "Ascend NPU x 1",
                "cann": "9.0.0",
                "pytorch": "2.6.0",
                "python": "3.11.0",
                "os": "Linux",
            },
        },
        "correctness": {"cases": []},
        "operators": [
            {
                "operator": "DemoOp",
                "cases": 1,
                "passed": 1,
                "failed": 0,
                "pass_rate": 1.0,
                "avg_max_abs_err": 1e-4,
                "max_abs_err": 1e-4,
                "performance_cases": 1,
                "ratio_range": {"min": 52.25, "max": 52.25},
            }
        ],
        "performance": {
            "cases": [
                {
                    "operator": "DemoOp",
                    "case_id": "DemoOp-0123456789ab",
                    "label": "smoke-16x32",
                    "params": {"shape": [16, 32], "dtype": "float16"},
                    "prof_mode": "msprof",
                    "latency_us": 8.5,
                    "roofline_utilization_percent": 52.25,
                    "bandwidth_tb_s": 1.25,
                    "arithmetic_intensity_ops_per_byte": 4.0,
                    "performance_tops": 5.0,
                    "artifact_dir": "/tmp/artifacts/main_abc",
                    "msprof": {
                        "kernel_name_resolved": "main",
                        "sample_count": 10,
                        "min_latency_us": 8.0,
                        "latency_us": 8.5,
                        "max_latency_us": 9.0,
                    },
                }
            ]
        },
    }

    paths = write_reports(run, tmp_path)

    assert all(path.exists() for path in paths.values())
    markdown = paths["markdown"].read_text(encoding="utf-8")
    html = paths["html"].read_text(encoding="utf-8")
    assert "DemoOp" in markdown
    assert "[16, 32]" in markdown
    assert "| DemoOp | 100.0% (1/1) | 1.00e-04 |" in markdown
    assert "| Label | Latency (us) | Ratio (%) |" in markdown
    assert "N/Ax" not in markdown
    assert "Shape / Parameters" in html
    assert "<th>Label</th><th>Latency (us)</th><th>Ratio</th>" in html
    assert "smoke-16x32" in html
    assert "[16, 32]" in html
    assert "<th>Operator</th><th>Correctness</th><th>Avg Max Abs Error</th>" in html
    assert "100.0% (1/1)" in html
    assert "Metadata / 元信息" in html
    assert "Environment / 运行环境" in html
    assert "Operator Analysis / 算子分析" in html
    assert "Avg Max Abs Error" in html
    assert "1.00e-04" in html
    assert "Performance Shapes" in html
    assert "Speedup" not in html
    assert "52.25%" in html
    assert "Avg Roofline" not in html
    assert "geometric mean" not in html
    assert "DemoOp-0123456789ab" in html
    assert "main_abc" in html
    assert "{{" not in html


def test_runner_gates_benchmark_and_writes_report(tmp_path, monkeypatch):
    (tmp_path / "test_demo.py").write_text("", encoding="utf-8")
    (tmp_path / "bench_demo.py").write_text("", encoding="utf-8")
    calls = []

    def fake_run_pytest(**kwargs):
        target = kwargs["target"]
        calls.append(target)
        Path(kwargs["junit_path"]).write_text(
            '<testsuite tests="1"><testcase classname="demo" name="ok"/></testsuite>',
            encoding="utf-8",
        )
        if target == "bench_demo.py":
            benchmark_json = Path(kwargs["env"]["TILEOPS_BENCHMARK_REPORT_PATH"]).with_suffix(
                ".json"
            )
            benchmark_json.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "profiling_mode_requested": "events",
                        "records": [
                            {
                                "operator": "ExternalOp",
                                "case_id": "case-1",
                                "tag": "tileops",
                                "params": {"shape": [16]},
                                "result": {
                                    "latency_us": 5.0,
                                    "prof_mode": "events",
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
        return 0, ["pytest", target]

    monkeypatch.setattr("tileops.reporting.runner._run_pytest", fake_run_pytest)

    run, run_dir, exit_code = run_operator(
        "ExternalOp",
        test_file="test_demo.py",
        benchmark_file="bench_demo.py",
        prof_mode="events",
        reports_dir="reports",
        root=tmp_path,
    )

    assert calls == ["test_demo.py", "bench_demo.py"]
    assert exit_code == 0
    assert run["status"] == "passed"
    assert run["summary"]["benchmark_tests"] == 1
    assert run["summary"]["benchmark_failed"] == 0
    assert (run_dir / "run.json").is_file()
    assert (run_dir / "report.md").is_file()


def test_runner_skips_benchmark_after_correctness_failure(tmp_path, monkeypatch):
    (tmp_path / "test_demo.py").write_text("", encoding="utf-8")
    (tmp_path / "bench_demo.py").write_text("", encoding="utf-8")
    calls = []

    def fake_run_pytest(**kwargs):
        calls.append(kwargs["target"])
        Path(kwargs["junit_path"]).write_text(
            '<testsuite tests="1" failures="1"><testcase classname="demo" '
            'name="bad"><failure message="not close"/></testcase></testsuite>',
            encoding="utf-8",
        )
        return 1, ["pytest", kwargs["target"]]

    monkeypatch.setattr("tileops.reporting.runner._run_pytest", fake_run_pytest)

    run, _run_dir, exit_code = run_operator(
        "ExternalOp",
        test_file="test_demo.py",
        benchmark_file="bench_demo.py",
        reports_dir="reports",
        root=tmp_path,
    )

    assert calls == ["test_demo.py"]
    assert exit_code == 1
    assert run["status"] == "failed"
    assert run["summary"]["benchmark_requested"] is False


def test_all_mode_resolves_pytest_directories():
    assert resolve_operator("all") == ("tests/ops", "benchmarks/ops")


def test_runner_rejects_xdist_for_benchmark_collection(tmp_path):
    (tmp_path / "test_demo.py").write_text("", encoding="utf-8")
    (tmp_path / "bench_demo.py").write_text("", encoding="utf-8")

    try:
        run_operator(
            "ExternalOp",
            test_file="test_demo.py",
            benchmark_file="bench_demo.py",
            pytest_args=["-n", "2"],
            root=tmp_path,
        )
    except ValueError as exc:
        assert "pytest-xdist" in str(exc)
    else:
        raise AssertionError("xdist arguments must be rejected")


def test_analyzer_builds_multi_operator_summary_without_cross_shape_averages():
    correctness = {
        "status": "failed",
        "tests": 3,
        "passed": 2,
        "failed": 1,
        "errors": 0,
        "cases": [
            {
                "classname": "shared.tests",
                "nodeid": "shared.tests::a",
                "outcome": "passed",
                "properties": {"op": "AlphaOp", "max_abs_err": "1.0e-4"},
            },
            {
                "classname": "tests.ops.test_alpha",
                "nodeid": "tests.ops.test_alpha::b",
                "outcome": "failed",
            },
            {
                "classname": "shared.tests",
                "nodeid": "shared.tests::b",
                "outcome": "passed",
                "properties": {"op": "BetaOp", "max_abs_err": "2.0e-4"},
            },
        ],
    }
    benchmark = {
        "status": "present",
        "profiling_mode_requested": "msprof",
        "records": [
            {
                "operator": "AlphaOp",
                "case_id": "alpha-1",
                "tag": "tileops",
                "params": {"shape": [16]},
                "result": {"latency_us": 10.0, "prof_mode": "msprof", "Ratio(%)": 40.0},
            },
            {
                "operator": "BetaOp",
                "case_id": "beta-1",
                "tag": "tileops",
                "params": {"shape": [32]},
                "result": {"latency_us": 10.0, "prof_mode": "msprof", "Ratio(%)": 70.0},
            },
        ],
    }
    catalog = [
        {"name": "AlphaOp", "family": "demo", "test": "tests/ops/test_alpha.py"},
        {"name": "BetaOp", "family": "demo", "test": "tests/ops/test_beta.py"},
    ]

    run = analyze_run(
        operator="all",
        correctness=correctness,
        correctness_exit_code=1,
        benchmark=benchmark,
        benchmark_exit_code=0,
        operator_catalog=catalog,
    )

    assert run["summary"]["operator_count"] == 2
    assert run["summary"]["pass_rate"] == 2 / 3
    assert run["operators"][0]["cases"] == 2
    assert run["operators"][0]["pass_rate"] == 0.5
    assert run["operators"][0]["avg_max_abs_err"] == 1e-4
    assert run["operators"][0]["performance_cases"] == 1
    assert "speedup_range" not in run["operators"][0]
    assert run["operators"][1]["ratio_range"] == {"min": 70.0, "max": 70.0}


def test_render_upgrades_legacy_run_json(tmp_path):
    legacy = {
        "schema_version": 1,
        "operator": "DemoOp",
        "status": "passed",
        "metadata": {
            "generated_at": "2026-09-17T10:00:00+08:00",
            "python": "3.11.0",
            "platform": "Linux",
            "prof_mode_requested": "events",
        },
        "summary": {"correctness_tests": 1, "correctness_failed": 0, "case_count": 1},
        "correctness": {"tests": 1, "passed": 1, "failed": 0, "errors": 0, "cases": []},
        "performance": {
            "cases": [
                {
                    "operator": "DemoOp",
                    "params": {"shape": [16]},
                    "latency_us": 5.0,
                    "prof_mode": "events",
                }
            ]
        },
        "warnings": [],
    }

    paths = write_reports(legacy, tmp_path)
    html = paths["html"].read_text(encoding="utf-8")

    assert "DemoOp" in html
    assert "[16]" in html
    assert "Python" in html
    assert "No operator records" not in html


def test_benchmark_failures_are_preserved_in_diagnostics(tmp_path):
    correctness = {
        "status": "passed",
        "tests": 1,
        "passed": 1,
        "failed": 0,
        "errors": 0,
        "cases": [],
    }
    benchmark_tests = {
        "status": "failed",
        "tests": 1,
        "passed": 0,
        "failed": 1,
        "errors": 0,
        "skipped": 0,
        "cases": [
            {
                "nodeid": "benchmarks.ops.test_demo::test_shape",
                "outcome": "failed",
                "message": "device error",
            }
        ],
    }

    run = analyze_run(
        operator="DemoOp",
        correctness=correctness,
        correctness_exit_code=0,
        benchmark={"status": "present", "records": []},
        benchmark_exit_code=1,
        benchmark_tests=benchmark_tests,
    )
    paths = write_reports(run, tmp_path)
    html = paths["html"].read_text(encoding="utf-8")

    assert run["summary"]["benchmark_failed"] == 1
    assert run["status"] == "partial"
    assert "Benchmark" in html
    assert "device error" in html
