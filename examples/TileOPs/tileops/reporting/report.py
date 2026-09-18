"""Markdown and template-backed HTML renderers for analyzed TileOPs runs."""

from __future__ import annotations

import html
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

_HTML_TEMPLATE = Path(__file__).with_name("report_template.html")


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _fmt_with_suffix(value: Any, suffix: str, digits: int = 4) -> str:
    return "N/A" if value is None else f"{_fmt(value, digits)}{suffix}"


def _fmt_rate(value: Any) -> str:
    return "N/A" if not isinstance(value, (int, float)) else f"{value:.1%}"


def _fmt_correctness(operator: dict[str, Any]) -> str:
    return (
        f"{_fmt_rate(operator.get('pass_rate'))} "
        f"({operator.get('passed', 0)}/{operator.get('cases', 0)})"
    )


def _fmt_scientific(value: Any) -> str:
    return "N/A" if not isinstance(value, (int, float)) else f"{value:.2e}"


def _fmt_range(value: dict[str, Any] | None, suffix: str, digits: int) -> str:
    if not value:
        return "N/A"
    low = value.get("min")
    high = value.get("max")
    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        return "N/A"
    if low == high:
        return f"{low:.{digits}f}{suffix}"
    return f"{low:.{digits}f}{suffix} – {high:.{digits}f}{suffix}"


def _display_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
    return "N/A" if value is None else str(value)


def _shape_or_parameters(params: dict[str, Any]) -> str:
    shape_items = [
        (key, value)
        for key, value in params.items()
        if key == "shape" or key.endswith("_shape") or key.endswith("_shapes")
    ]
    if len(shape_items) == 1 and shape_items[0][0] == "shape":
        return _display_value(shape_items[0][1])
    if shape_items:
        return "; ".join(f"{key}={_display_value(value)}" for key, value in shape_items)
    remaining = {
        key: value
        for key, value in params.items()
        if key not in {"dtype", "dtypes", "op_params", "case_id", "label"}
        and not key.endswith("_dtype")
    }
    return _display_value(remaining) if remaining else "N/A"


def _dtype(params: dict[str, Any]) -> str:
    items = [
        (key, value)
        for key, value in params.items()
        if key in {"dtype", "dtypes"} or key.endswith("_dtype")
    ]
    if len(items) == 1:
        return _display_value(items[0][1])
    return "; ".join(f"{key}={_display_value(value)}" for key, value in items) or "N/A"


def _case_label(case: dict[str, Any]) -> str:
    label = case.get("label") or (case.get("params") or {}).get("label")
    if label not in (None, ""):
        return str(label)
    return _shape_or_parameters(case.get("params") or {})


def _kernel_name(case: dict[str, Any]) -> str:
    return str((case.get("msprof") or {}).get("kernel_name_resolved") or "N/A")


def _md_cell(value: Any) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")


def _normalize_run_for_render(run: dict[str, Any]) -> dict[str, Any]:
    """Upgrade older run.json records enough for the current renderers."""
    normalized = deepcopy(run)
    correctness = normalized.get("correctness") or {}
    perf_cases = normalized.get("performance", {}).get("cases", [])
    summary = normalized.setdefault("summary", {})
    total = int(summary.get("total_cases", correctness.get("tests", 0)) or 0)
    passed = int(summary.get("passed_cases", correctness.get("passed", 0)) or 0)
    failed = int(
        summary.get(
            "failed_cases",
            int(correctness.get("failed", 0) or 0) + int(correctness.get("errors", 0) or 0),
        )
        or 0
    )
    summary.setdefault("total_cases", total)
    summary.setdefault("passed_cases", passed)
    summary.setdefault("failed_cases", failed)
    summary.setdefault("pass_rate", passed / total if total else None)
    summary.setdefault("case_count", len(perf_cases))

    if not normalized.get("setup"):
        metadata = normalized.get("metadata") or {}
        normalized["setup"] = {
            "metadata": {
                "framework": "TileOPs Reporting",
                "date": metadata.get("generated_at"),
                "evaluation_scope": normalized.get("operator"),
                "profiler": metadata.get("prof_mode_requested"),
                "run_id": metadata.get("run_id"),
                "git_commit": metadata.get("git_commit"),
                "correctness_target": metadata.get("test_file"),
                "benchmark_target": metadata.get("benchmark_file"),
            },
            "environment": {
                "python": metadata.get("python"),
                "os": metadata.get("platform"),
            },
        }

    if not normalized.get("operators"):
        perf_by_operator: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for case in perf_cases:
            perf_by_operator[
                str(case.get("operator") or normalized.get("operator") or "unknown")
            ].append(case)
        names = list(perf_by_operator)
        scope = str(normalized.get("operator") or "unknown")
        if not names and scope != "all":
            names = [scope]
        correctness_by_operator: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for case in correctness.get("cases", []):
            declared = (case.get("properties") or {}).get("op")
            if declared:
                correctness_by_operator[str(declared)].append(case)
                if str(declared) not in names:
                    names.append(str(declared))
            elif len(names) == 1:
                correctness_by_operator[names[0]].append(case)
        operators = []
        for name in names:
            correct_cases = correctness_by_operator.get(name, [])
            op_total = (
                len(correct_cases) if correctness.get("cases") else total if len(names) == 1 else 0
            )
            op_passed = (
                sum(1 for case in correct_cases if case.get("outcome") == "passed")
                if correctness.get("cases")
                else passed
                if len(names) == 1
                else 0
            )
            op_perf = perf_by_operator.get(name, [])
            ratios = [
                float(case["roofline_utilization_percent"])
                for case in op_perf
                if isinstance(case.get("roofline_utilization_percent"), (int, float))
            ]
            operators.append(
                {
                    "operator": name,
                    "cases": op_total,
                    "passed": op_passed,
                    "failed": max(op_total - op_passed, 0),
                    "pass_rate": op_passed / op_total if op_total else None,
                    "avg_max_abs_err": None,
                    "performance_cases": len(op_perf),
                    "ratio_range": {"min": min(ratios), "max": max(ratios)} if ratios else None,
                }
            )
        normalized["operators"] = operators
    correctness_errors: dict[str, list[float]] = defaultdict(list)
    for case in correctness.get("cases", []):
        declared = (case.get("properties") or {}).get("op")
        raw_error = (case.get("properties") or {}).get("max_abs_err")
        if not declared or raw_error is None:
            continue
        try:
            correctness_errors[str(declared)].append(float(raw_error))
        except (TypeError, ValueError):
            continue
    perf_counts: dict[str, int] = defaultdict(int)
    for case in perf_cases:
        perf_counts[str(case.get("operator") or normalized.get("operator") or "unknown")] += 1
    for operator in normalized.get("operators", []):
        name = str(operator.get("operator"))
        errors = correctness_errors.get(name, [])
        operator.setdefault("avg_max_abs_err", sum(errors) / len(errors) if errors else None)
        operator.setdefault("max_abs_err", max(errors) if errors else None)
        operator.setdefault("performance_cases", perf_counts.get(name, 0))
    summary.setdefault("operator_count", len(normalized.get("operators", [])))
    return normalized


def _setup_tables_markdown(run: dict[str, Any]) -> list[str]:
    setup = run.get("setup") or {}
    lines = ["## Experiment Setup / 评测配置", ""]
    for title, values in (
        ("Metadata / 元信息", setup.get("metadata") or {}),
        ("Environment / 运行环境", setup.get("environment") or {}),
    ):
        lines.extend([f"### {title}", "", "| Item | Value |", "|---|---|"])
        lines.extend(
            f"| {_md_cell(key)} | {_md_cell(_display_value(value))} |"
            for key, value in values.items()
            if value not in (None, "")
        )
        lines.append("")
    return lines


def render_markdown(run: dict[str, Any]) -> str:
    """Render the CANN-Bench-style hierarchy in Markdown."""
    run = _normalize_run_for_render(run)
    summary = run.get("summary") or {}
    lines = [f"# TileOPs Evaluation Report: {run.get('operator', 'unknown')}", ""]
    lines.extend(_setup_tables_markdown(run))
    lines.extend(
        [
            "## Results Overview / 结果总览",
            "",
            "| Pass Rate | Operators | Total Cases | Failed Cases |",
            "|---:|---:|---:|---:|",
            f"| {_fmt_rate(summary.get('pass_rate'))} | {summary.get('operator_count', 0)} | "
            f"{summary.get('total_cases', 0)} | {summary.get('failed_cases', 0)} |",
            "",
            "## Operator Analysis / 算子分析",
            "",
            "| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for operator in run.get("operators", []):
        lines.append(
            f"| {_md_cell(operator.get('operator'))} | {_fmt_correctness(operator)} | "
            f"{_fmt_scientific(operator.get('avg_max_abs_err'))} | "
            f"{operator.get('performance_cases', 0)} | "
            f"{_fmt_range(operator.get('ratio_range'), '%', 2)} |"
        )
    lines.extend(
        [
            "",
            "> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。",
            "",
            "## Operator Details / 算子明细",
            "",
        ]
    )
    perf_cases = run.get("performance", {}).get("cases", [])
    for operator in run.get("operators", []):
        name = operator.get("operator")
        cases = [case for case in perf_cases if case.get("operator") == name]
        lines.extend(
            [
                f"### {name}",
                "",
                "| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |",
                "|---|---:|---:|---|---|---|---|---:|",
            ]
        )
        for case in cases:
            params = case.get("params") or {}
            lines.append(
                f"| {_md_cell(_case_label(case))} | {_fmt(case.get('latency_us'))} | "
                f"{_fmt(case.get('roofline_utilization_percent'), 2)} | "
                f"{_md_cell(_shape_or_parameters(params))} | {_md_cell(_dtype(params))} | "
                f"{case.get('prof_mode') or 'N/A'} | {_md_cell(_kernel_name(case))} | "
                f"{_fmt(case.get('bandwidth_tb_s'))} |"
            )
        if not cases:
            lines.append("| - | No performance records | - | - | - | - | - | - |")
        lines.append("")
    warnings = run.get("warnings", [])
    if warnings:
        lines.extend(["## Warnings", "", *(f"- {warning}" for warning in warnings), ""])
    failed_cases = [
        (stage, case)
        for stage, report in (
            ("Correctness", run.get("correctness", {})),
            ("Benchmark", run.get("benchmark_tests", {})),
        )
        for case in report.get("cases", [])
        if case.get("outcome") in {"failed", "error"}
    ]
    if failed_cases:
        lines.extend(
            [
                "## Failed Cases / 失败用例",
                "",
                "| Stage | Test | Outcome | Message |",
                "|---|---|---|---|",
            ]
        )
        lines.extend(
            f"| {stage} | {_md_cell(case.get('nodeid'))} | {case.get('outcome')} | "
            f"{_md_cell(case.get('message') or '')} |"
            for stage, case in failed_cases
        )
        lines.append("")
    return "\n".join(lines)


def _kv_rows(fields: list[tuple[str, Any]], *, code_values: bool = True) -> str:
    def _value_cell(value: Any) -> str:
        rendered = html.escape(_display_value(value))
        return f"<code>{rendered}</code>" if code_values else rendered

    return "".join(
        f'<tr><td class="kv-k">{html.escape(label)}</td><td>{_value_cell(value)}</td></tr>'
        for label, value in fields
        if value not in (None, "")
    )


def _setup_blocks(run: dict[str, Any]) -> str:
    setup = run.get("setup") or {}
    metadata = setup.get("metadata") or run.get("metadata") or {}
    environment = setup.get("environment") or {}
    metadata_labels = {
        "framework": "Framework",
        "date": "Date",
        "evaluation_scope": "Evaluation Scope",
        "benchmark": "Benchmark",
        "profiler": "Profiler",
        "run_id": "Run ID",
        "git_commit": "Git Commit",
        "correctness_target": "Correctness Target",
        "benchmark_target": "Benchmark Target",
    }
    environment_labels = {
        "npu": "NPU",
        "cpu": "CPU",
        "cann": "CANN",
        "driver": "Driver",
        "pytorch": "PyTorch",
        "pytorch_npu": "PyTorch NPU",
        "tilelang": "TileLang",
        "python": "Python",
        "os": "OS",
        "container": "Container",
    }
    meta_fields = [(metadata_labels.get(key, key), value) for key, value in metadata.items()]
    env_fields = [(environment_labels.get(key, key), value) for key, value in environment.items()]
    return (
        '<div class="setup-block"><h4>Metadata / 元信息</h4><table><tbody>'
        f"{_kv_rows(meta_fields, code_values=False)}</tbody></table></div>"
        '<div class="setup-block"><h4>Environment / 运行环境</h4><table><tbody>'
        f"{_kv_rows(env_fields, code_values=False)}</tbody></table></div>"
    )


def _score_class(value: Any, *, high: float, mid: float) -> str:
    if not isinstance(value, (int, float)):
        return ""
    if value >= high:
        return "score-high"
    return "score-mid" if value >= mid else "score-low"


def _operator_analysis_rows(run: dict[str, Any]) -> str:
    rows = []
    for index, operator in enumerate(run.get("operators", []), 1):
        rate = operator.get("pass_rate")
        rows.append(
            "<tr>"
            f"<td>{index}</td>"
            f'<td class="col-name"><a class="case-link" href="#operator-{index}">{html.escape(str(operator.get("operator")))}</a></td>'
            f'<td class="score-cell {_score_class(rate, high=0.8, mid=0.4)}">{html.escape(_fmt_correctness(operator))}</td>'
            f'<td class="number">{_fmt_scientific(operator.get("avg_max_abs_err"))}</td>'
            f"<td>{operator.get('performance_cases', 0)}</td>"
            f"<td>{html.escape(_fmt_range(operator.get('ratio_range'), '%', 2))}</td>"
            "</tr>"
        )
    return "".join(rows) or '<tr><td colspan="6" class="empty">No operator records</td></tr>'


def _performance_rows(cases: list[dict[str, Any]], operator_index: int) -> str:
    rows = []
    for case_index, case in enumerate(cases, 1):
        params = case.get("params") or {}
        ratio = case.get("roofline_utilization_percent")
        rows.append(
            "<tr>"
            f'<td class="col-name"><a class="case-link" href="#case-{operator_index}-{case_index}">{html.escape(_case_label(case))}</a></td>'
            f'<td class="number strong">{_fmt(case.get("latency_us"), 3)}</td>'
            f'<td class="number score-cell {_score_class(ratio, high=70, mid=40)}">{_fmt_with_suffix(ratio, "%", 2)}</td>'
            f"<td><code>{html.escape(_shape_or_parameters(params))}</code></td>"
            f"<td><code>{html.escape(_dtype(params))}</code></td>"
            f"<td><code>{html.escape(_kernel_name(case))}</code>"
            f'<span class="cell-sub">{html.escape(str(case.get("prof_mode") or "N/A"))}</span></td>'
            f'<td class="number">{_fmt(case.get("bandwidth_tb_s"), 3)}</td>'
            "</tr>"
        )
    return "".join(rows) or '<tr><td colspan="7" class="empty">No performance records</td></tr>'


def _case_details(cases: list[dict[str, Any]], operator_index: int) -> str:
    entries = []
    for case_index, case in enumerate(cases, 1):
        params = case.get("params") or {}
        msprof = case.get("msprof") or {}
        fields = [
            ("Internal case ID", case.get("case_id")),
            ("Parameters", json.dumps(params, ensure_ascii=False)),
            ("Kernel config", _display_value(case.get("config"))),
            ("Profiler / kernel", f"{case.get('prof_mode') or 'N/A'} / {_kernel_name(case)}"),
            ("AI (Ops/Byte)", _fmt(case.get("arithmetic_intensity_ops_per_byte"))),
            ("Performance (TOPS)", _fmt(case.get("performance_tops"))),
            ("Computility (TOPS)", _fmt(case.get("computility_tops"))),
            ("Samples", msprof.get("sample_count")),
            (
                "Min / median / max (us)",
                " / ".join(
                    _fmt(msprof.get(key))
                    for key in ("min_latency_us", "latency_us", "max_latency_us")
                ),
            ),
            ("Artifact", case.get("artifact_dir")),
        ]
        entries.append(
            f'<details class="case-detail" id="case-{operator_index}-{case_index}"><summary>'
            f'<span class="case-index">{html.escape(_case_label(case))}</span>'
            f"<strong>{_fmt(case.get('latency_us'), 3)} us</strong>"
            f"<span>{_fmt_with_suffix(case.get('roofline_utilization_percent'), '%', 2)}</span>"
            f"<code>{html.escape(_shape_or_parameters(params))}</code>"
            f"<span>{html.escape(_dtype(params))}</span>"
            "</summary>"
            f'<div class="detail-grid"><table><tbody>{_kv_rows(fields)}</tbody></table></div>'
            "</details>"
        )
    return "".join(entries)


def _operator_details_section(run: dict[str, Any]) -> str:
    perf_cases = run.get("performance", {}).get("cases", [])
    blocks = []
    for operator_index, operator in enumerate(run.get("operators", []), 1):
        name = operator.get("operator")
        cases = [case for case in perf_cases if case.get("operator") == name]
        blocks.append(
            f'<div class="operator-block" id="operator-{operator_index}">'
            f"<h4>4.{operator_index} {html.escape(str(name))}</h4>"
            '<div class="table-wrap"><table>'
            f"<caption>Table 4.{operator_index}. Per-shape performance results</caption>"
            "<thead><tr><th>Label</th><th>Latency (us)</th><th>Ratio</th>"
            "<th>Shape / Parameters</th><th>DType</th><th>Kernel</th>"
            "<th>Bandwidth (TB/s)</th></tr></thead>"
            f"<tbody>{_performance_rows(cases, operator_index)}</tbody></table></div>"
            '<p class="note">Latency 是该 shape 重复测量后的统计值；Ratio 不跨 shape 求平均。</p>'
            f"{_case_details(cases, operator_index)}</div>"
        )
    content = "".join(blocks) or '<p class="empty">No operator details</p>'
    return (
        '<section class="section">'
        '<h3><span class="sec-num">4.</span> Operator Details / 算子明细</h3>'
        f"{content}</section>"
    )


def _alert_section(run: dict[str, Any]) -> str:
    warnings = list(run.get("warnings", []))
    failed = [
        ("Correctness", case)
        for case in run.get("correctness", {}).get("cases", [])
        if case.get("outcome") in {"failed", "error"}
    ]
    failed.extend(
        ("Benchmark", case)
        for case in run.get("benchmark_tests", {}).get("cases", [])
        if case.get("outcome") in {"failed", "error"}
    )
    if not warnings and not failed:
        return ""
    warning_html = ""
    if warnings:
        items = "".join(f"<li>{html.escape(str(item))}</li>" for item in warnings)
        warning_html = (
            f'<div class="callout warning"><strong>Warnings</strong><ul>{items}</ul></div>'
        )
    failure_html = ""
    if failed:
        rows = "".join(
            "<tr>"
            f"<td>{html.escape(stage)}</td>"
            f"<td><code>{html.escape(str(case.get('nodeid') or 'N/A'))}</code></td>"
            f"<td>{html.escape(str(case.get('outcome') or 'N/A'))}</td>"
            f"<td>{html.escape(str(case.get('message') or ''))}</td>"
            "</tr>"
            for stage, case in failed
        )
        failure_html = (
            '<div class="table-wrap"><table><thead><tr><th>Stage</th><th>Test</th><th>Outcome</th>'
            f"<th>Message</th></tr></thead><tbody>{rows}</tbody></table></div>"
        )
    return (
        '<section class="section"><h3><span class="sec-num">!</span> '
        f"Diagnostics / 诊断信息</h3>{warning_html}{failure_html}</section>"
    )


def _abstract(run: dict[str, Any]) -> str:
    summary = run.get("summary") or {}
    return (
        f"本报告覆盖 <strong>{summary.get('operator_count', 0)}</strong> 个算子、"
        f"<strong>{summary.get('total_cases', 0)}</strong> 个精度用例和 "
        f"<strong>{summary.get('case_count', 0)}</strong> 个性能 workload。"
        "精度由原有 pytest golden 对比判定，性能数据来自 BenchmarkReport 与 msprof；"
        "run.json 是机器可读的事实来源。"
    )


def render_html(run: dict[str, Any]) -> str:
    """Fill the standalone report template with normalized run data."""
    run = _normalize_run_for_render(run)
    template = _HTML_TEMPLATE.read_text(encoding="utf-8")
    summary = run.get("summary") or {}
    status = str(run.get("status", "unknown"))
    metadata = run.get("metadata") or {}
    replacements = {
        "{{PAGE_TITLE}}": html.escape(
            f"TileOPs Evaluation Report: {run.get('operator', 'unknown')}"
        ),
        "{{OPERATOR}}": html.escape(str(run.get("operator", "unknown"))),
        "{{STATUS}}": html.escape(status.upper()),
        "{{STATUS_CLASS}}": html.escape(status),
        "{{GENERATED_AT}}": html.escape(str(metadata.get("generated_at") or "N/A")),
        "{{PROFILE_MODE}}": html.escape(str(metadata.get("prof_mode_requested") or "N/A")),
        "{{PERFORMANCE_CASES}}": str(summary.get("case_count", 0)),
        "{{ABSTRACT}}": _abstract(run),
        "{{SETUP_BLOCKS}}": _setup_blocks(run),
        "{{PASS_RATE}}": _fmt_rate(summary.get("pass_rate")),
        "{{PASSED_CASES}}": str(summary.get("passed_cases", 0)),
        "{{OPERATOR_COUNT}}": str(summary.get("operator_count", 0)),
        "{{TOTAL_CASES}}": str(summary.get("total_cases", 0)),
        "{{FAILED_CASES}}": str(summary.get("failed_cases", 0)),
        "{{OPERATOR_ANALYSIS_ROWS}}": _operator_analysis_rows(run),
        "{{OPERATOR_DETAILS_SECTION}}": _operator_details_section(run),
        "{{DIAGNOSTICS_SECTION}}": _alert_section(run),
    }
    for token, value in replacements.items():
        template = template.replace(token, value)
    return template


def write_reports(run: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    """Write canonical JSON plus Markdown and template-backed HTML views."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": root / "run.json",
        "markdown": root / "report.md",
        "html": root / "report.html",
    }
    paths["json"].write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["markdown"].write_text(render_markdown(run), encoding="utf-8")
    paths["html"].write_text(render_html(run), encoding="utf-8")
    return paths
