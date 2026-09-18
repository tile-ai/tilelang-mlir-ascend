"""Thin stage-gated runner around the existing pytest suites."""

from __future__ import annotations

import json
import os
import platform
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from tileops.manifest import load_manifest
from tileops.reporting.analyzer import analyze_run
from tileops.reporting.collector import load_benchmark_report, parse_junit_report
from tileops.reporting.report import write_reports
from tileops.reporting.setup_info import collect_setup_info


def project_root() -> Path:
    """Return the standalone ``examples/TileOPs`` project root."""
    return Path(__file__).resolve().parents[2]


def list_operators() -> list[dict[str, Any]]:
    """Return reportable operators and their declared test/benchmark paths."""
    operators = []
    for name, entry in sorted(load_manifest().items()):
        source = entry.get("source") or {}
        operators.append(
            {
                "name": name,
                "family": entry.get("family"),
                "status": entry.get("status"),
                "test": source.get("test"),
                "benchmark": source.get("bench"),
            }
        )
    return operators


def resolve_operator(
    operator: str,
    *,
    test_file: str | None = None,
    benchmark_file: str | None = None,
) -> tuple[str, str]:
    """Resolve test paths from the manifest, allowing explicit overrides."""
    if operator == "all" and not test_file and not benchmark_file:
        return "tests/ops", "benchmarks/ops"

    entry = load_manifest().get(operator)
    source = (entry or {}).get("source") or {}
    resolved_test = test_file or source.get("test")
    resolved_benchmark = benchmark_file or source.get("bench")
    if not resolved_test or not resolved_benchmark:
        raise ValueError(
            f"operator {operator!r} must declare source.test/source.bench in the manifest "
            "or receive --test-file and --benchmark-file"
        )
    return str(resolved_test), str(resolved_benchmark)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "operator"


def _reject_xdist(pytest_args: list[str], pytest_addopts: str | None) -> None:
    """Reject process-parallel benchmark collection until worker merging exists."""
    args = [*pytest_args, *shlex.split(pytest_addopts or "")]
    for index, arg in enumerate(args):
        if arg == "-n":
            value = args[index + 1] if index + 1 < len(args) else ""
            if value not in {"", "0"}:
                raise ValueError(
                    "pytest-xdist is not supported by tileops-report: benchmark "
                    "workers would overwrite benchmark.json. Remove '-n'."
                )
        elif arg == "--numprocesses":
            value = args[index + 1] if index + 1 < len(args) else ""
            if value not in {"", "0"}:
                raise ValueError(
                    "pytest-xdist is not supported by tileops-report. Remove "
                    "'--numprocesses' or set it to 0."
                )
        elif (arg.startswith("-n") and arg not in {"-n", "-n0"}) or (
            arg.startswith("--numprocesses=") and arg != "--numprocesses=0"
        ):
            raise ValueError(
                "pytest-xdist is not supported by tileops-report: benchmark workers "
                "would overwrite benchmark.json. Run the report serially."
            )


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _run_pytest(
    *,
    root: Path,
    target: str,
    junit_path: Path,
    log_path: Path,
    env: dict[str, str],
    pytest_args: list[str],
    timeout: int | None,
) -> tuple[int, list[str]]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        target,
        "-sv",
        f"--junitxml={junit_path}",
        *pytest_args,
    ]
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"command: {' '.join(command)}\n\n")
        log.flush()
        try:
            result = subprocess.run(
                command,
                cwd=root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=timeout,
            )
            return result.returncode, command
        except subprocess.TimeoutExpired:
            log.write(f"\nERROR: pytest timed out after {timeout} seconds\n")
            return 124, command


def run_operator(
    operator: str,
    *,
    test_file: str | None = None,
    benchmark_file: str | None = None,
    prof_mode: str = "msprof",
    kernel_name: str | None = None,
    reports_dir: str | Path = "reports/tileops",
    pytest_args: list[str] | None = None,
    timeout: int | None = None,
    root: str | Path | None = None,
) -> tuple[dict[str, Any], Path, int]:
    """Run correctness then benchmark, and emit one self-contained report directory."""
    repo_root = Path(root).resolve() if root else project_root()
    resolved_test, resolved_benchmark = resolve_operator(
        operator, test_file=test_file, benchmark_file=benchmark_file
    )
    for target in (resolved_test, resolved_benchmark):
        if not (repo_root / target).exists():
            raise FileNotFoundError(f"pytest target does not exist: {repo_root / target}")

    args = list(pytest_args or [])
    base_env = dict(os.environ)
    _reject_xdist(args, base_env.get("PYTEST_ADDOPTS"))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = (repo_root / reports_dir / f"{stamp}_{_safe_name(operator)}").resolve()
    pytest_dir = run_dir / "pytest"
    artifact_dir = run_dir / "artifacts" / "msprof"
    pytest_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    correctness_xml = pytest_dir / "correctness.xml"
    correctness_log = pytest_dir / "correctness.log"
    print(f"[tileops-report] correctness: {resolved_test}")
    correctness_code, correctness_command = _run_pytest(
        root=repo_root,
        target=resolved_test,
        junit_path=correctness_xml,
        log_path=correctness_log,
        env=base_env,
        pytest_args=args,
        timeout=timeout,
    )
    correctness = parse_junit_report(correctness_xml)

    benchmark_code: int | None = None
    benchmark_command: list[str] | None = None
    benchmark_json = run_dir / "benchmark.json"
    benchmark: dict[str, Any] = {"status": "skipped", "records": []}
    benchmark_tests: dict[str, Any] = {
        "status": "skipped",
        "tests": 0,
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "cases": [],
    }

    if correctness_code == 0 and correctness.get("status") == "passed":
        benchmark_env = dict(base_env)
        benchmark_env.update(
            {
                "TILEOPS_PROF_MODE": prof_mode,
                "TILEOPS_MSPROF_KEEP_OUTPUT": "1",
                "TILEOPS_MSPROF_ARTIFACT_ROOT": str(artifact_dir),
                "TILEOPS_BENCHMARK_REPORT_PATH": str(run_dir / "benchmark.md"),
            }
        )
        if kernel_name:
            benchmark_env["TILEOPS_MSPROF_KERNEL_NAME"] = kernel_name
        else:
            benchmark_env.pop("TILEOPS_MSPROF_KERNEL_NAME", None)

        print(f"[tileops-report] benchmark: {resolved_benchmark} ({prof_mode})")
        benchmark_xml = pytest_dir / "benchmark.xml"
        benchmark_code, benchmark_command = _run_pytest(
            root=repo_root,
            target=resolved_benchmark,
            junit_path=benchmark_xml,
            log_path=pytest_dir / "benchmark.log",
            env=benchmark_env,
            pytest_args=args,
            timeout=timeout,
        )
        benchmark_tests = parse_junit_report(benchmark_xml)
        benchmark = load_benchmark_report(benchmark_json)
    else:
        print("[tileops-report] benchmark skipped because correctness did not pass")

    generated_at = datetime.now().astimezone().isoformat()
    git_commit = _git_commit(repo_root)
    metadata = {
        "run_id": run_dir.name,
        "generated_at": generated_at,
        "project_root": str(repo_root),
        "git_commit": git_commit,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "test_file": resolved_test,
        "benchmark_file": resolved_benchmark,
        "prof_mode_requested": prof_mode,
        "kernel_name_requested": kernel_name,
        "commands": {
            "correctness": correctness_command,
            "benchmark": benchmark_command,
        },
        "exit_codes": {
            "correctness": correctness_code,
            "benchmark": benchmark_code,
        },
    }
    operator_catalog = list_operators()
    setup = collect_setup_info(
        scope=operator,
        run_id=run_dir.name,
        generated_at=generated_at,
        profiler=prof_mode,
        git_commit=git_commit,
        test_target=resolved_test,
        benchmark_target=resolved_benchmark,
    )
    run = analyze_run(
        operator=operator,
        correctness=correctness,
        correctness_exit_code=correctness_code,
        benchmark=benchmark,
        benchmark_exit_code=benchmark_code,
        benchmark_tests=benchmark_tests,
        metadata=metadata,
        setup=setup,
        operator_catalog=operator_catalog,
    )
    write_reports(run, run_dir)

    # Keep a compact machine-readable pointer for simple automation.
    latest = run_dir.parent / "latest.json"
    latest.write_text(
        json.dumps({"run_id": run_dir.name, "run_json": str(run_dir / "run.json")}, indent=2),
        encoding="utf-8",
    )

    exit_code = 0 if run["status"] == "passed" else 1
    return run, run_dir, exit_code
