"""Command line interface for TileOPs evaluation reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tileops.reporting.report import write_reports
from tileops.reporting.runner import list_operators, run_operator


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tileops-report",
        description="Run existing TileOPs pytest gates and generate structured reports.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="list operators declared in the manifest")

    run = subparsers.add_parser("run", help="run correctness, then benchmark and report")
    selection = run.add_mutually_exclusive_group(required=True)
    selection.add_argument("--op", help="manifest operator name")
    selection.add_argument(
        "--all",
        action="store_true",
        help="run pytest tests/ops followed by pytest benchmarks/ops",
    )
    run.add_argument("--test-file", help="override manifest source.test")
    run.add_argument("--benchmark-file", help="override manifest source.bench")
    run.add_argument("--prof-mode", choices=("msprof", "events"), default="msprof")
    run.add_argument("--kernel-name", help="explicit TILEOPS_MSPROF_KERNEL_NAME")
    run.add_argument("--reports-dir", default="reports/tileops")
    run.add_argument("--timeout", type=int, help="timeout in seconds for each pytest stage")
    run.add_argument(
        "--pytest-arg",
        action="append",
        default=[],
        help="additional pytest argument; repeat this option for multiple arguments",
    )

    render = subparsers.add_parser("render", help="regenerate Markdown/HTML from run.json")
    render.add_argument("run_json")
    render.add_argument("--output-dir")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "list":
        for operator in list_operators():
            print(
                f"{operator['name']:<28} family={operator['family'] or '-':<12} "
                f"test={operator['test'] or '-'} bench={operator['benchmark'] or '-'}"
            )
        return 0

    if args.command == "render":
        source = Path(args.run_json).resolve()
        run = json.loads(source.read_text(encoding="utf-8"))
        output = Path(args.output_dir).resolve() if args.output_dir else source.parent
        paths = write_reports(run, output)
        print(f"Reports written to {paths['markdown']} and {paths['html']}")
        return 0

    operator = "all" if args.all else args.op
    run, run_dir, exit_code = run_operator(
        operator,
        test_file=args.test_file,
        benchmark_file=args.benchmark_file,
        prof_mode=args.prof_mode,
        kernel_name=args.kernel_name,
        reports_dir=args.reports_dir,
        pytest_args=args.pytest_arg,
        timeout=args.timeout,
    )
    print(f"[tileops-report] status={run['status']} report={run_dir / 'report.md'}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
