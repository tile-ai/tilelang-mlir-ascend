"""Gate lint rules for tilelang-op-conductor artifacts (U1 fix, improvement #2).

Mechanically decidable checks only. No domain reasoning: equivalence proofs,
complexity re-computation and semantic consistency stay with LLM reviewers.

Lint groups (dispatched by stage):
  - Stage 0: meta_lint        (.migration_meta.json + TileOPs 7-file scaffold)
  - Stage 1: design_lint      (DESIGN.md section schema / research 4Q / core-split
                               3-elements / placeholder / reference-path existence)
  - Stage 2: review_lint      (REVIEW.md conclusion literal + dimension sections)
  - Stage 3: kernel_lint      ({op}.py AST checks: jit target, golden, test levels)
  - Stage 4: kernel_lint      (perf_opt/{op}.py) + opt_log.md
               + perf_feedback_lint ([DESIGN_LIMIT] artifact, if present — U14 fix #20)
               + perf_records_lint (perf_records.jsonl, if present — A2/B2:
                 per-branch record schema + comparison-table existence +
                 Final Summary winner reconciliation)
  - Stage 5: integration_lint (integration package + wrapper + TileOPs report)
  - always:  state_schema_lint (.stage_state.json schema)

Every failure is a dict {rule_id, file, message}; warnings are separate and
never block. Output is consumed by statectl `gate` / `complete` (exit code +
JSON), which the conductor uses instead of manual Read/Write bookkeeping.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re

REPO_TOP_DIRS = ("docs", "examples", "testing", ".agents")

PLACEHOLDER_PATTERNS = ("待补充", "待填写", "TODO", "FIXME", "{placeholder}")
# Curly-brace groups containing CJK chars, e.g. "{算子名称}" / "{如: ...}".
TEMPLATE_BRACE_RE = re.compile(r"\{[^{}\n]*[\u4e00-\u9fff][^{}\n]*\}")
# Angle-bracket groups containing CJK chars — unfilled <...> template slots in
# perf_feedback.md (whose schema template uses <...> placeholders).
TEMPLATE_ANGLE_RE = re.compile(r"<[^<>\n]*[\u4e00-\u9fff][^<>\n]*>")
# Repo-relative path tokens referenced in prose, e.g. docs/Tilelang.language/xxx
# (full-width punctuation excluded so trailing prose is not swallowed)
REF_PATH_RE = re.compile(
    r"(?:docs|examples|testing|\.agents)/[^\s`'\"<>()\[\]（）【】：，。；、！？]+"
)
HEADING_RE = re.compile(r"^(#{1,6})\s+([0-9]+(?:\.[0-9]+)*)[.\s、)：:]", re.M)


def _read_text(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except (OSError, UnicodeDecodeError):
        return None


def _fail(rule_id: str, file: str, message: str) -> dict:
    return {"rule_id": rule_id, "file": file, "message": message}


def _warn(rule_id: str, file: str, message: str) -> dict:
    return {"rule_id": rule_id, "file": file, "message": message}


def _ref_path_failures(text: str, file_path: str, repo_root: str, rule_id: str):
    """Repo-relative paths cited in prose must exist (negative-assertion
    evidence rule — shared by design_lint / perf_feedback_lint)."""
    failures = []
    seen = set()
    for m in REF_PATH_RE.finditer(text):
        token = m.group(0).rstrip(".,;:!?、。】")
        if token in seen:
            continue
        seen.add(token)
        if "{" in token or "}" in token or "*" in token:
            continue  # template variables / globs are not concrete paths
        if not os.path.exists(os.path.join(repo_root, token)):
            failures.append(
                _fail(
                    rule_id,
                    file_path,
                    f"引用的仓库路径不存在：{token}（负向断言/佐证路径必须真实存在）",
                )
            )
    return failures


def _section_body(text: str, title: str) -> str | None:
    """Body of the first markdown section whose heading starts with `title`
    (heading line excluded), or None if no such heading."""
    m = re.search(r"^#{1,6}\s*" + re.escape(title), text, re.M)
    if not m:
        return None
    rest = text[m.end() :]
    nxt = re.search(r"^#{1,6}\s+", rest, re.M)
    return rest[: nxt.start()] if nxt else rest


# ---------------------------------------------------------------------------
# Markdown section splitting
# ---------------------------------------------------------------------------


def split_sections(text: str) -> tuple[dict[str, str], dict[str, str]]:
    """Split markdown into ({section_number: body}, {section_number: heading}).

    Heading lines are kept separately (keyword checks may need them, e.g.
    ``### 5.5 分核策略``); subsection bodies live under their own number —
    use region()/region_all() to merge a section with its subsections.
    """
    sections: dict[str, list[str]] = {}
    headings: dict[str, list[str]] = {}
    current = None
    for line in text.splitlines():
        m = re.match(r"^#{1,6}\s+([0-9]+(?:\.[0-9]+)*)[.\s、)：:]", line)
        if m:
            current = m.group(1)
            sections.setdefault(current, [])
            headings.setdefault(current, []).append(line)
            continue
        if current is not None:
            sections[current].append(line)
    return (
        {num: "\n".join(body) for num, body in sections.items()},
        {num: "\n".join(lines) for num, lines in headings.items()},
    )


def region(sections: dict[str, str], prefix: str) -> str:
    """Merge section `prefix` bodies with all subsection bodies."""
    parts = [
        body
        for num, body in sections.items()
        if num == prefix or num.startswith(prefix + ".")
    ]
    return "\n".join(parts)


def region_all(sections: dict[str, str], headings: dict[str, str], prefix: str) -> str:
    """Region bodies + heading lines (keywords may live in headings)."""
    return region(sections, prefix) + "\n" + region(headings, prefix)


# ---------------------------------------------------------------------------
# Stage 1: DESIGN.md lint
# ---------------------------------------------------------------------------

# Sections required in every DESIGN.md (conductor gate table, design template).
DESIGN_REQUIRED_SECTIONS = [
    "1",
    "1.1",
    "1.3",
    "1.4",
    "1.6",
    "1.6.0",
    "1.6.1",
    "1.6.2",
    "1.6.3",
    "2",
    "3",
    "3.5",
    "4",
    "4.1",
    "4.2",
    "5",
    "6",
    "7",
    "8",
]
# Migration-only sections (conductor gate table: 0.1/0.3/0.4/0.5/0.6 required;
# 0.2/0.7 recommended -> warning).
DESIGN_MIG_REQUIRED = ["0", "0.1", "0.3", "0.4", "0.5", "0.6"]
DESIGN_MIG_RECOMMENDED = ["0.2", "0.7"]

RESEARCH_4Q = [
    ("R1-等价化简公式", ("等价", "化简")),
    ("R2-在线算法", ("在线",)),
    ("R3-复杂度对比", ("复杂度",)),
    ("R4-硬件亲和", ("亲和",)),
]

CORES_PLAN_KEYWORDS = (
    "无需适配",
    "整数倍",
    "核内串行",
    "串行",
    "persistent",
    "规模判定",
)


def design_lint(design_path: str, migration: bool, repo_root: str):
    """Lint DESIGN.md. Returns (failures, warnings)."""
    failures, warnings = [], []
    text = _read_text(design_path)
    if text is None or not text.strip():
        failures.append(_fail("S1-EXISTS", design_path, "DESIGN.md 不存在或为空"))
        return failures, warnings

    for pat in PLACEHOLDER_PATTERNS:
        if pat in text:
            failures.append(
                _fail(
                    "S1-PLACEHOLDER",
                    design_path,
                    f"含占位符残留「{pat}」，未完成的设计章节禁止通过门禁",
                )
            )
    for m in TEMPLATE_BRACE_RE.finditer(text):
        failures.append(
            _fail(
                "S1-PLACEHOLDER",
                design_path,
                f"疑似模板变量未替换：{m.group(0)}",
            )
        )

    sections, headings = split_sections(text)
    missing = [
        num
        for num in DESIGN_REQUIRED_SECTIONS
        if num not in sections or not region_all(sections, headings, num).strip()
    ]
    if missing:
        failures.append(
            _fail(
                "S1-SECT-MISSING",
                design_path,
                "缺少或为空的必需章节：" + "、".join(f"§{n}" for n in missing),
            )
        )

    if migration:
        mig_missing = [n for n in DESIGN_MIG_REQUIRED if n not in sections]
        if mig_missing:
            failures.append(
                _fail(
                    "S1-MIG-S0",
                    design_path,
                    "迁移任务缺少 §0 必需小节："
                    + "、".join(f"§{n}" for n in mig_missing),
                )
            )
        for n in DESIGN_MIG_RECOMMENDED:
            if n not in sections:
                warnings.append(
                    _warn("S1-MIG-S0", design_path, f"建议补充迁移小节 §{n}")
                )

    # --- §1.6.0 research 4 questions + baseline + conclusion literal ---
    r160 = region_all(sections, headings, "1.6.0")
    if r160.strip():
        missing_q = [
            name for name, kws in RESEARCH_4Q if not all(kw in r160 for kw in kws)
        ]
        if missing_q:
            failures.append(
                _fail(
                    "S1-RESEARCH-4Q",
                    design_path,
                    "§1.6.0 调研四问缺失：" + "、".join(missing_q),
                )
            )
        if "基线" not in r160:
            failures.append(
                _fail(
                    "S1-RESEARCH-BASELINE",
                    design_path,
                    "§1.6.0 候选表缺少基线候选（源算法/输入公式必须作为基线参与对比）",
                )
            )
        if "调研结论" not in r160:
            failures.append(
                _fail(
                    "S1-RESEARCH-CONCL",
                    design_path,
                    "§1.6.0 缺少「调研结论」结论行（若结论为无更优替代须写明调研范围）",
                )
            )

    # --- §1.6.1/1.6.2/1.6.3 conclusion literals ---
    for sub, literal in (
        ("1.6.1", "优化结论"),
        ("1.6.2", "向量化结论"),
        ("1.6.3", "布局决策结论"),
    ):
        if sub in sections and literal not in sections[sub]:
            failures.append(
                _fail("S1-SECT-CONCL", design_path, f"§{sub} 缺少「{literal}」结论行")
            )

    # --- §1.6.1 equivalence machine verification (D-1, S1-EQUIV-EXEC) ---
    # When adopted optimizations exist, the prose equivalence argument must
    # be backed by an executable check: verify_equiv.py next to DESIGN.md and
    # an embedded result table (EQUIV_PASS per adopted item). Designs that
    # explicitly conclude "no optimization space" are exempt.
    r161 = region_all(sections, headings, "1.6.1")
    if r161.strip() and "无优化空间" not in r161:
        op_dir = os.path.dirname(os.path.abspath(design_path))
        equiv_script = os.path.join(op_dir, "verify_equiv.py")
        if not os.path.isfile(equiv_script):
            failures.append(
                _fail(
                    "S1-EQUIV-EXEC",
                    design_path,
                    "§1.6.1 含采纳优化项但缺少等价性机器验证脚本 "
                    f"{os.path.basename(op_dir)}/verify_equiv.py"
                    "（候选式 vs 基线式数值对照，torch CPU + fp64 参照）",
                )
            )
        if "EQUIV_PASS" not in r161:
            failures.append(
                _fail(
                    "S1-EQUIV-EXEC",
                    design_path,
                    "§1.6.1 缺少等价性验证结果表（每采纳项：最大 ulp 差 / "
                    "违反率 / EQUIV_PASS 结论——由 verify_equiv.py 执行产出）",
                )
            )
        if "EQUIV_FAIL" in r161:
            failures.append(
                _fail(
                    "S1-EQUIV-EXEC",
                    design_path,
                    "§1.6.1 存在 EQUIV_FAIL 项——等价性未通过的优化项禁止采纳，"
                    "须放弃该项或修正论证后重跑验证",
                )
            )

    # --- §2 programming mode ---
    sec2 = region_all(sections, headings, "2")
    if sec2.strip() and not re.search(r"[Dd]eveloper|[Ee]xpert|混合", sec2):
        failures.append(
            _fail(
                "S1-MODE", design_path, "§2 编程模式选型缺少 Developer/Expert/混合 结论"
            )
        )

    # --- §5 core-split 3 elements (fall back to whole doc, warn if moved) ---
    sec5 = region_all(sections, headings, "5")
    cores_checks = [
        ("S1-CORES-LOGICAL", "逻辑核数", "① 逻辑核数计算"),
        (
            "S1-CORES-PHYSICAL",
            "get_aicore_num",
            "② 物理核数依据（NPUUtils.get().get_aicore_num() 实查记录）",
        ),
    ]
    for rule_id, kw, desc in cores_checks:
        if kw not in sec5:
            if kw in text:
                warnings.append(
                    _warn(
                        rule_id,
                        design_path,
                        f"分核要素（{desc}）未位于 §5 Tiling 策略章节",
                    )
                )
            else:
                failures.append(_fail(rule_id, design_path, f"§5 缺分核策略要素{desc}"))
    if "分核" not in sec5 or not any(kw in sec5 for kw in CORES_PLAN_KEYWORDS):
        failures.append(
            _fail(
                "S1-CORES-PLAN",
                design_path,
                "§5 缺分核策略要素③ 规模判定与分核方案（无需适配依据 / 对齐物理核整数倍 / 核内串行）",
            )
        )

    # --- §8 verification plan ---
    sec8 = region_all(sections, headings, "8")
    if sec8.strip():
        if "olden" not in sec8:  # matches Golden/golden
            failures.append(
                _fail("S1-GOLDEN-PLAN", design_path, "§8 验证方案缺少 Golden 函数说明")
            )
        if "L0" not in sec8:
            failures.append(
                _fail("S1-L0-PLAN", design_path, "§8 验证方案缺少 L0 门槛测试计划")
            )

    # --- referenced repo paths must exist (negative-assertion evidence rule) ---
    failures.extend(_ref_path_failures(text, design_path, repo_root, "S1-REF-PATHS"))
    return failures, warnings


# ---------------------------------------------------------------------------
# Stage 2: REVIEW.md lint
# ---------------------------------------------------------------------------

CONCLUSION_RE = re.compile(
    r"^\s*(?:\*\*)?结论(?:\*\*)?\s*[:：]\s*(通过|不通过)\s*$", re.M
)
DIM_HEADING_RE = r"^#{1,4}\s*%s[.、\s]"
NEXT_HEADING_RE = re.compile(r"^#{1,4}\s*[0-9]+[.、\s]", re.M)


def _dim_body(text: str, dim: int) -> str | None:
    """Body text of dimension section `dim` (heading excluded), or None."""
    m = re.search(DIM_HEADING_RE % dim, text, re.M)
    if not m:
        return None
    rest = text[m.end() :]
    nxt = NEXT_HEADING_RE.search(rest)
    return rest[: nxt.start()] if nxt else rest


def review_lint(review_path: str, migration: bool):
    """Lint REVIEW.md. Returns (failures, warnings)."""
    failures, warnings = [], []
    text = _read_text(review_path)
    if text is None or not text.strip():
        failures.append(_fail("S2-EXISTS", review_path, "REVIEW.md 不存在或为空"))
        return failures, warnings

    concl = CONCLUSION_RE.findall(text)
    if not concl:
        failures.append(
            _fail(
                "S2-CONCLUSION",
                review_path,
                "缺少结论字面量「结论: 通过」或「结论: 不通过」（不得使用模糊表述）",
            )
        )

    required_dims = list(range(0, 9)) if migration else list(range(1, 9))
    missing_dims = [
        k for k in required_dims if not re.search(DIM_HEADING_RE % k, text, re.M)
    ]
    if missing_dims:
        failures.append(
            _fail(
                "S2-DIMS",
                review_path,
                "缺少检视维度章节："
                + "、".join(f"维度 {k}" for k in missing_dims)
                + "（迁移任务须 9 维度，非迁移 8 维度）",
            )
        )

    m8 = _dim_body(text, 8)
    if m8 is not None:
        has_recheck = ("复核" in m8) or ("复算" in m8)
        has_verify = "核对" in m8
        if not (has_recheck and has_verify):
            failures.append(
                _fail(
                    "S2-DIM8-EVIDENCE",
                    review_path,
                    "维度 8 缺少独立复核证据（需含调研结论「复核/复算」与弃选论证「核对」过程记录）",
                )
            )
    else:
        warnings.append(
            _warn(
                "S2-DIM8-EVIDENCE", review_path, "未找到维度 8 章节，无法核对复核证据"
            )
        )

    if migration:
        dim0_body = _dim_body(text, 0)
        if dim0_body is not None:
            if "n/a" in dim0_body.replace(" ", ""):
                failures.append(
                    _fail(
                        "S2-MIG-DIM0",
                        review_path,
                        "迁移任务维度 0 不得标 n/a（须含源码核对证据）",
                    )
                )
            elif "源码" not in dim0_body:
                failures.append(
                    _fail(
                        "S2-MIG-DIM0",
                        review_path,
                        "迁移任务维度 0 缺少源码核对证据",
                    )
                )

    if concl and concl[0] == "通过" and "检视问题列表" in text:
        failures.append(
            _fail(
                "S2-PASS-NO-ISSUES",
                review_path,
                "结论为通过时不得出现「检视问题列表」章节",
            )
        )
    return failures, warnings


# ---------------------------------------------------------------------------
# Stage 3 / 4: kernel .py AST lint
# ---------------------------------------------------------------------------


def _collect_torch_aliases(tree: ast.AST) -> set[str]:
    aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                root = item.name.split(".")[0]
                if root == "torch":
                    aliases.add(item.asname or "torch")
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.split(".")[0] == "torch"
        ):
            for item in node.names:
                aliases.add(item.asname or item.name)
    return aliases


def _jit_decorators(tree: ast.AST):
    """Yield (call_node, has_target, target_value) for tilelang.jit decorators."""
    tilelang_aliases, jit_names = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                if item.name == "tilelang":
                    tilelang_aliases.add(item.asname or "tilelang")
        elif isinstance(node, ast.ImportFrom) and node.module == "tilelang":
            for item in node.names:
                if item.name == "jit":
                    jit_names.add(item.asname or "jit")

    def is_jit(func) -> ast.expr | None:
        for dec in func.decorator_list:
            target = None
            if isinstance(dec, ast.Call):
                target = dec.func
            elif isinstance(
                dec, (ast.Attribute, ast.Name)
            ):  # bare @tilelang.jit / @jit
                target = dec
            if target is None:
                continue
            if isinstance(target, ast.Attribute) and target.attr == "jit":
                if isinstance(target.value, ast.Name) and (
                    target.value.id in tilelang_aliases
                ):
                    return dec
            elif isinstance(target, ast.Name) and target.id in jit_names:
                return dec
        return None

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            dec = is_jit(node)
            if dec is not None:
                if isinstance(dec, ast.Call):
                    for kw in dec.keywords:
                        if kw.arg == "target":
                            val = getattr(kw.value, "value", None)
                            yield dec, True, val
                            break
                    else:
                        yield dec, False, None
                else:
                    yield dec, False, None


def _uses_namespace(body_nodes, aliases: set[str]) -> bool:
    for node in body_nodes:
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Attribute)
                and isinstance(sub.value, ast.Name)
                and sub.value.id in aliases
            ):
                return True
            if isinstance(sub, ast.Name) and sub.id in aliases:
                return True
    return False


def _is_stub(func: ast.FunctionDef) -> bool:
    meaningful = [
        stmt
        for stmt in func.body
        if not (
            isinstance(stmt, ast.Pass)
            or (
                isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, (str, type(Ellipsis)))
            )
        )
    ]
    return not meaningful


def kernel_lint(py_path: str, op_name: str, rule_prefix: str = "S3"):
    """AST-level lint for {op}.py / perf_opt/{op}.py. Returns (failures, warnings)."""
    failures, warnings = [], []
    src = _read_text(py_path)
    if src is None or not src.strip():
        failures.append(
            _fail(f"{rule_prefix}-EXISTS", py_path, "kernel 文件不存在或为空")
        )
        return failures, warnings
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        failures.append(
            _fail(f"{rule_prefix}-PARSE", py_path, f"Python 语法错误：{exc}")
        )
        return failures, warnings

    for pat in PLACEHOLDER_PATTERNS:
        if pat in src:
            failures.append(
                _fail(
                    f"{rule_prefix}-PLACEHOLDER",
                    py_path,
                    f"含占位符残留「{pat}」",
                )
            )

    # --- tilelang.jit target ---
    jit_found = False
    for _dec, has_target, target_val in _jit_decorators(tree):
        jit_found = True
        if has_target and target_val != "npuir":
            failures.append(
                _fail(
                    f"{rule_prefix}-JIT",
                    py_path,
                    f'@tilelang.jit target="{target_val}"，必须为 target="npuir"',
                )
            )
        elif not has_target:
            warnings.append(
                _warn(
                    f"{rule_prefix}-JIT",
                    py_path,
                    '@tilelang.jit 未显式声明 target="npuir"',
                )
            )
    if not jit_found:
        failures.append(
            _fail(
                f"{rule_prefix}-JIT", py_path, "未找到 @tilelang.jit 装饰的 kernel 函数"
            )
        )

    # --- golden function using torch ---
    torch_aliases = _collect_torch_aliases(tree)
    funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    golden_funcs = [f for f in funcs if f.name.startswith("golden")]
    if not golden_funcs:
        failures.append(
            _fail(
                f"{rule_prefix}-GOLDEN",
                py_path,
                f"缺少 golden 参考实现函数（命名 golden_{op_name} 或 golden*）",
            )
        )
    else:
        if not torch_aliases:
            failures.append(
                _fail(
                    f"{rule_prefix}-GOLDEN-TORCH",
                    py_path,
                    "golden 未引用 torch（必须以 torch.* API 为权威参考实现）",
                )
            )
        elif not any(_uses_namespace(f.body, torch_aliases) for f in golden_funcs):
            failures.append(
                _fail(
                    f"{rule_prefix}-GOLDEN-TORCH",
                    py_path,
                    "golden 函数体未直接调用 torch API（禁止手写归约/激活循环替代 torch 参考）",
                )
            )
        for f in golden_funcs:
            if _is_stub(f):
                failures.append(
                    _fail(
                        f"{rule_prefix}-STUB",
                        py_path,
                        f"golden 函数 {f.name} 为空实现（pass/省略号）",
                    )
                )

    # --- layered test entrypoints ---
    top_names = {f.name for f in funcs}
    for required, label in (
        ("run_L0", "run_L0"),
        ("run_L1", "run_L1"),
        ("run_L2", "run_L2"),
    ):
        if required not in top_names:
            failures.append(
                _fail(
                    f"{rule_prefix}-TEST-LEVELS", py_path, f"缺少分层测试入口 {label}()"
                )
            )
    if not any("boundary" in name.lower() for name in top_names):
        failures.append(
            _fail(
                f"{rule_prefix}-TEST-LEVELS", py_path, "缺少边界测试入口 run_boundary()"
            )
        )
    for f in funcs:
        if (f.name.startswith("run_") or f.name.startswith("golden")) and _is_stub(f):
            failures.append(
                _fail(
                    f"{rule_prefix}-STUB",
                    py_path,
                    f"函数 {f.name} 为空实现（pass/省略号）",
                )
            )

    # --- main with --level ---
    has_level = any(
        isinstance(node, ast.Constant) and node.value == "--level"
        for node in ast.walk(tree)
    )
    if not has_level:
        failures.append(
            _fail(
                f"{rule_prefix}-MAIN-LEVEL",
                py_path,
                "main 入口缺少 --level 参数（L0/all 分层测试入口）",
            )
        )

    # --- T.prim_func usage ---
    has_prim = any(
        (isinstance(node, ast.Attribute) and node.attr == "prim_func")
        or (isinstance(node, ast.Name) and node.id == "prim_func")
        for node in ast.walk(tree)
    )
    if not has_prim:
        warnings.append(
            _warn(
                f"{rule_prefix}-PRIM-FUNC",
                py_path,
                "未检测到 T.prim_func（请确认 kernel 结构）",
            )
        )
    return failures, warnings


# ---------------------------------------------------------------------------
# Stage 4 extras
# ---------------------------------------------------------------------------

# perf_feedback.md fixed schema ([DESIGN_LIMIT] artifact, U14 fix #20).
PERF_FEEDBACK_SECTIONS = (
    "触发判定",
    "参照锚定",
    "假设",
    "实测",
    "影响面",
    "反馈结论",
)
# Quantified structural speedup, e.g. "2.5x" / "3 倍" / "2.8 ×".
SPEEDUP_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:x|×|倍)")
# perf_records.jsonl required fields (schema canonical:
# _shared/standards/signal-registry.md §5).
PERF_RECORD_WORKLOAD_FIELDS = (
    "round",
    "candidate_id",
    "parent_id",
    "kernel_id",
    "workload_id",
    "phase",
    "artifact_path",
    "artifact_sha256",
    "duration_us",
    "l0_pass",
    "msprof_raw_path",
    "timestamp",
)
CURRENT_BEST_RE = re.compile(r"current[- ]best", re.I)
# Reconciliation tolerance: claimed winner duration must match a recorded
# duration_us within 1% (msprof noise floor well below that).
RECON_TOLERANCE = 0.01


def _perf_records_workload_lint(
    records_path: str,
    opt_log_path: str,
    inventory_path: str,
    final_artifact_path: str | None,
):
    """Check benchmark-derived workload coverage and one final version per kernel."""
    failures, warnings = [], []

    def fail(rule: str, path: str, message: str) -> None:
        failures.append(_fail(rule, path, message))

    try:
        with open(inventory_path, encoding="utf-8") as handle:
            inventory = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        fail(
            "S4-WORKLOAD-INVENTORY",
            inventory_path,
            f"无法读取 workload inventory：{exc}",
        )
        return failures, warnings
    if not isinstance(inventory, dict) or not isinstance(
        inventory.get("workloads"), list
    ):
        fail(
            "S4-WORKLOAD-INVENTORY", inventory_path, "inventory 必须包含 workloads 数组"
        )
        return failures, warnings

    expected = {}
    for index, item in enumerate(inventory["workloads"], 1):
        if not isinstance(item, dict):
            fail(
                "S4-WORKLOAD-INVENTORY", inventory_path, f"workloads[{index}] 不是对象"
            )
            continue
        required = (
            "benchmark_source",
            "kernel_id",
            "workload_id",
            "label",
            "marks",
            "shape",
            "dtype",
            "params",
            "kind",
            "reason",
        )
        missing = [name for name in required if name not in item]
        if missing:
            fail(
                "S4-WORKLOAD-INVENTORY",
                inventory_path,
                f"workloads[{index}] 缺少 {', '.join(missing)}",
            )
            continue
        key = (item["kernel_id"], item["workload_id"])
        if (
            any(not isinstance(value, str) or not value for value in key)
            or key in expected
        ):
            fail(
                "S4-WORKLOAD-INVENTORY",
                inventory_path,
                f"workloads[{index}] kernel_id/workload_id 无效或重复",
            )
            continue
        if (
            not isinstance(item["benchmark_source"], str)
            or not item["benchmark_source"]
            or not isinstance(item["label"], str)
            or not isinstance(item["marks"], list)
            or not all(isinstance(mark, str) for mark in item["marks"])
            or not isinstance(item["shape"], (list, dict))
            or not isinstance(item["dtype"], str)
            or not isinstance(item["params"], dict)
            or not isinstance(item["reason"], str)
            or item["kind"] not in ("tune", "smoke", "skipped")
        ):
            fail(
                "S4-WORKLOAD-INVENTORY",
                inventory_path,
                f"workloads[{index}] 字段类型或 kind 无效",
            )
            continue
        marks = {mark.lower() for mark in item["marks"]}
        has_smoke_token = any(
            re.search(r"(?:^|[^A-Za-z0-9])smoke(?:$|[^A-Za-z0-9])", value, re.I)
            for value in (item["workload_id"], item["label"])
        )
        expected_kind = (
            "skipped"
            if "skip" in marks
            else ("smoke" if "smoke" in marks or has_smoke_token else "tune")
        )
        if item["kind"] != expected_kind:
            fail(
                "S4-WORKLOAD-INVENTORY",
                inventory_path,
                f"{key} 分类应为 {expected_kind}，实际为 {item['kind']}",
            )
        if item["kind"] == "tune" and item.get("tuning_status") not in (
            "winner_merged",
            "no_gain",
            "merge_blocked",
        ):
            fail(
                "S4-WORKLOAD-INVENTORY",
                inventory_path,
                f"{key} 缺已完成的 tuning_status",
            )
        if (
            item.get("tuning_status") in ("no_gain", "merge_blocked")
            and not item["reason"]
        ):
            fail(
                "S4-WORKLOAD-INVENTORY",
                inventory_path,
                f"{key} 的无收益/合并受阻结论缺少原因",
            )
        if item["kind"] == "smoke" and item.get("precision_pass") is not True:
            fail(
                "S4-WORKLOAD-INVENTORY",
                inventory_path,
                f"smoke {key} 缺最终精度通过记录",
            )
        if item["kind"] == "smoke" and "full" in marks and not item["reason"]:
            fail(
                "S4-WORKLOAD-INVENTORY",
                inventory_path,
                f"smoke {key} 与 full 标记冲突时须记录原因",
            )
        if item["kind"] == "skipped" and not item["reason"]:
            fail("S4-WORKLOAD-INVENTORY", inventory_path, f"skipped {key} 缺跳过原因")
        expected[key] = item["kind"]

    try:
        with open(records_path, encoding="utf-8") as handle:
            lines = handle.readlines()
    except OSError:
        lines = []
    measurements = {}
    final_versions = {}
    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            fail(
                "S4-PERF-RECORDS-SCHEMA",
                records_path,
                f"第 {line_no} 行非法 JSON：{exc}",
            )
            continue
        if not isinstance(rec, dict):
            fail("S4-PERF-RECORDS-SCHEMA", records_path, f"第 {line_no} 行不是对象")
            continue
        missing = [field for field in PERF_RECORD_WORKLOAD_FIELDS if field not in rec]
        if missing:
            fail(
                "S4-PERF-RECORDS-SCHEMA",
                records_path,
                f"第 {line_no} 行缺少 {', '.join(missing)}",
            )
            continue
        if not isinstance(rec["kernel_id"], str) or not isinstance(
            rec["workload_id"], str
        ):
            fail(
                "S4-PERF-RECORDS-SCHEMA",
                records_path,
                f"第 {line_no} 行 kernel_id/workload_id 须为字符串",
            )
            continue
        key = (rec["kernel_id"], rec["workload_id"])
        if expected.get(key) != "tune":
            fail(
                "S4-PERF-RECORDS-COVERAGE",
                records_path,
                f"第 {line_no} 行不是 inventory 中的 tune workload：{key}",
            )
            continue
        if (
            not isinstance(rec["round"], int)
            or isinstance(rec["round"], bool)
            or not isinstance(rec["candidate_id"], str)
            or not rec["candidate_id"]
            or (rec["parent_id"] is not None and not isinstance(rec["parent_id"], str))
            or rec["phase"] not in ("baseline", "candidate", "merged", "final")
            or not isinstance(rec["duration_us"], (int, float))
            or isinstance(rec["duration_us"], bool)
            or not 0 < rec["duration_us"] < float("inf")
            or rec["l0_pass"] is not True
            or not isinstance(rec["msprof_raw_path"], str)
            or not rec["msprof_raw_path"]
            or not isinstance(rec["artifact_path"], str)
            or not rec["artifact_path"]
            or not isinstance(rec["artifact_sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", rec["artifact_sha256"])
            or not isinstance(rec["timestamp"], str)
            or not rec["timestamp"]
        ):
            fail("S4-PERF-RECORDS-SCHEMA", records_path, f"第 {line_no} 行字段值无效")
            continue
        measurements.setdefault(key, {}).setdefault(rec["phase"], []).append(rec)
        if rec["phase"] == "final":
            final_versions.setdefault(rec["kernel_id"], set()).add(rec["candidate_id"])
            if final_artifact_path:
                recorded_path = rec["artifact_path"]
                if not os.path.isabs(recorded_path):
                    recorded_path = os.path.join(
                        os.path.dirname(records_path), recorded_path
                    )
                if os.path.normcase(os.path.abspath(recorded_path)) != os.path.normcase(
                    os.path.abspath(final_artifact_path)
                ):
                    fail(
                        "S4-PERF-RECORDS-RECON",
                        records_path,
                        f"{key} final 未测最终 perf_opt 文件",
                    )
                elif os.path.isfile(final_artifact_path):
                    with open(final_artifact_path, "rb") as handle:
                        actual_hash = hashlib.sha256(handle.read()).hexdigest()
                    if rec["artifact_sha256"] != actual_hash:
                        fail(
                            "S4-PERF-RECORDS-RECON",
                            records_path,
                            f"{key} final 文件哈希与采集时不一致",
                        )

    for key, kind in expected.items():
        if kind != "tune":
            continue
        phases = measurements.get(key, {})
        if not phases.get("baseline") or not phases.get("final"):
            fail(
                "S4-PERF-RECORDS-COVERAGE",
                records_path,
                f"{key} 缺 baseline 或 final 记录",
            )
        if len(phases.get("final", [])) != 1:
            fail(
                "S4-PERF-RECORDS-RECON",
                records_path,
                f"{key} 必须恰有一条当前最终版本的 final 记录",
            )
    for kernel_id, versions in final_versions.items():
        if len(versions) != 1:
            fail(
                "S4-PERF-RECORDS-RECON",
                records_path,
                f"{kernel_id} 的 final 来自多个候选：{sorted(versions)}",
            )

    log = _read_text(opt_log_path) or ""
    if measurements and ("Task Duration" not in log or not CURRENT_BEST_RE.search(log)):
        fail(
            "S4-OPTLOG-COMPTABLE",
            opt_log_path,
            "缺少候选 vs current best 的 Task Duration 对比表",
        )
    if any(kind == "tune" for kind in expected.values()):
        marker = re.search(r"^#{2,3} Final Performance Test Data\s*$", log, re.M)
        if marker is None:
            fail(
                "S4-PERF-RECORDS-RECON",
                opt_log_path,
                "缺少 Final Performance Test Data 表",
            )
        else:
            claims = {}
            for line in log[marker.end() :].splitlines():
                if line.startswith("## ") or line.startswith("### "):
                    break
                if not line.strip().startswith("|"):
                    continue
                cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
                if (
                    len(cells) != 5
                    or cells[0] in ("kernel_id", "---")
                    or all(set(cell) <= set("-: ") for cell in cells)
                ):
                    continue
                try:
                    claim_key = (cells[0], cells[1])
                    if claim_key in claims:
                        fail(
                            "S4-PERF-RECORDS-RECON",
                            opt_log_path,
                            f"Final 表重复 {claim_key}",
                        )
                    claims[claim_key] = (float(cells[2]), float(cells[3]), cells[4])
                except ValueError:
                    fail(
                        "S4-PERF-RECORDS-RECON",
                        opt_log_path,
                        f"Final 表非法数值：{line}",
                    )
            for key, kind in expected.items():
                if kind != "tune":
                    continue
                claim = claims.get(key)
                phases = measurements.get(key, {})
                if claim is None:
                    fail("S4-PERF-RECORDS-RECON", opt_log_path, f"Final 表缺少 {key}")
                    continue
                for phase, value in (("baseline", claim[0]), ("final", claim[1])):
                    records = phases.get(phase, [])
                    if records and not any(
                        abs(rec["duration_us"] - value)
                        <= max(RECON_TOLERANCE * value, 1e-9)
                        for rec in records
                    ):
                        fail(
                            "S4-PERF-RECORDS-RECON",
                            opt_log_path,
                            f"{key} 的 {phase} 时延与记录不符",
                        )
                finals = phases.get("final", [])
                if len(finals) == 1 and claim[2] != finals[0]["candidate_id"]:
                    fail(
                        "S4-PERF-RECORDS-RECON",
                        opt_log_path,
                        f"{key} 的最终候选与记录不符",
                    )
            for key in claims:
                if expected.get(key) != "tune":
                    fail(
                        "S4-PERF-RECORDS-RECON",
                        opt_log_path,
                        f"Final 表包含非 tune workload：{key}",
                    )
    return failures, warnings


def perf_feedback_lint(path: str, repo_root: str):
    """Lint perf_opt/perf_feedback.md ([DESIGN_LIMIT] artifact).

    Only runs when the file exists (statectl gate 4 wires it via perf_lint).
    Mechanical checks only — whether the attribution/threshold judgment is
    *sound* stays with the conductor and later review.
    """
    failures, warnings = [], []
    text = _read_text(path)
    if text is None or not text.strip():
        failures.append(
            _fail(
                "S4-PERF-FEEDBACK",
                path,
                "perf_feedback.md 不存在或为空（[DESIGN_LIMIT] 信号的必需工件）",
            )
        )
        return failures, warnings

    for pat in PLACEHOLDER_PATTERNS:
        if pat in text:
            failures.append(
                _fail(
                    "S4-PERF-FEEDBACK-PLACEHOLDER",
                    path,
                    f"含占位符残留「{pat}」，未完成的反馈禁止通过门禁",
                )
            )
    for m in TEMPLATE_ANGLE_RE.finditer(text):
        failures.append(
            _fail(
                "S4-PERF-FEEDBACK-PLACEHOLDER",
                path,
                f"疑似模板变量未替换：{m.group(0)}",
            )
        )

    for title in PERF_FEEDBACK_SECTIONS:
        if _section_body(text, title) is None:
            failures.append(
                _fail(
                    "S4-PERF-FEEDBACK-SCHEMA",
                    path,
                    "缺少「{t}」章节（固定 schema：触发判定/参照锚定/假设/实测/"
                    "影响面/反馈结论）".format(t=title),
                )
            )

    # Reference anchoring (T-1): a [DESIGN_LIMIT] ceiling claim must be
    # anchored against the best known same-family implementation, or
    # explicitly downgrade the claim confidence when none is found.
    anchor = _section_body(text, "参照锚定") or ""
    if anchor.strip():
        has_ref = "参照实现" in anchor
        degraded = "未找到参照" in anchor
        argued = "不可移植" in anchor
        if not has_ref:
            failures.append(
                _fail(
                    "S4-PERF-FEEDBACK-ANCHOR",
                    path,
                    "参照锚定缺少「参照实现」来源行（同族已知最优实现路径，"
                    "或显式「未找到参照，天花板结论置信度降级」）",
                )
            )
        elif not degraded and not argued:
            failures.append(
                _fail(
                    "S4-PERF-FEEDBACK-ANCHOR",
                    path,
                    "参照锚定缺少「不可移植论证」（逐条结构差异 + 依据；"
                    "未找到参照时须显式降级标注）",
                )
            )

    trig = _section_body(text, "触发判定") or ""
    if "设计层归因" not in trig:
        failures.append(
            _fail(
                "S4-PERF-FEEDBACK-TRIGGER",
                path,
                "触发判定缺少「设计层归因」（须归因到 DESIGN.md 具体假设/选型）",
            )
        )
    speedups = [float(v) for v in SPEEDUP_RE.findall(trig)]
    if not (any(v > 2 for v in speedups) or "矛盾" in trig):
        failures.append(
            _fail(
                "S4-PERF-FEEDBACK-TRIGGER",
                path,
                "触发判定缺少量化结构性加速估计（>2x，如 2.5x）或「实测与假设矛盾」依据",
            )
        )

    meas = _section_body(text, "实测") or ""
    if "msprof" not in meas:
        failures.append(
            _fail(
                "S4-PERF-FEEDBACK-METRIC",
                path,
                "实测章节缺少 msprof 口径证据（Task Duration，唯一 kernel 时延口径）",
            )
        )

    concl = _section_body(text, "反馈结论") or ""
    if "建议路由" not in concl:
        failures.append(
            _fail(
                "S4-PERF-FEEDBACK-ROUTE",
                path,
                "反馈结论缺少「建议路由」（附录补记 / 设计修订）",
            )
        )

    failures.extend(_ref_path_failures(text, path, repo_root, "S4-PERF-FEEDBACK-REF"))
    return failures, warnings


def perf_records_lint(
    records_path: str,
    opt_log_path: str,
    final_artifact_path: str,
):
    """Check inventory coverage, measured workloads, and the final kernel."""
    inventory_path = os.path.join(
        os.path.dirname(records_path), "workload_inventory.json"
    )
    if not os.path.isfile(inventory_path):
        return [
            _fail(
                "S4-WORKLOAD-INVENTORY",
                inventory_path,
                "Stage 4 缺少 workload_inventory.json",
            )
        ], []
    return _perf_records_workload_lint(
        records_path, opt_log_path, inventory_path, final_artifact_path
    )


def perf_lint(
    perf_py_path: str,
    opt_log_path: str,
    feedback_path: str,
    records_path: str,
    op_name: str,
    repo_root: str,
):
    failures, warnings = [], []
    f, w = kernel_lint(perf_py_path, op_name, rule_prefix="S4")
    failures += f
    warnings += w
    log = _read_text(opt_log_path)
    if log is None or not log.strip():
        failures.append(
            _fail("S4-OPTLOG", opt_log_path, "perf_opt/opt_log.md 不存在或为空")
        )
    elif "Skill Retrospective" not in log:
        warnings.append(
            _warn(
                "S4-OPTLOG",
                opt_log_path,
                "opt_log.md 缺少 Skill Retrospective 复盘章节",
            )
        )
    # [DESIGN_LIMIT] artifact: lint only when present (signal-optional flow).
    if os.path.exists(feedback_path):
        f, w = perf_feedback_lint(feedback_path, repo_root)
        failures += f
        warnings += w
    # Stage 4 requires a workload inventory and measured results for every tune workload.
    f, w = perf_records_lint(records_path, opt_log_path, perf_py_path)
    failures += f
    warnings += w
    return failures, warnings


# ---------------------------------------------------------------------------
# Stage 0: migration meta + TileOPs scaffold lint
# ---------------------------------------------------------------------------


def meta_lint(meta_path: str, repo_root: str):
    """Stage 0 gate: .migration_meta.json schema + TileOPs 7-file scaffold."""
    failures, warnings = [], []
    raw = _read_text(meta_path)
    if raw is None or not raw.strip():
        failures.append(
            _fail("S0-META", meta_path, ".migration_meta.json 不存在或为空")
        )
        return failures, warnings
    try:
        meta = json.loads(raw)
    except json.JSONDecodeError as exc:
        failures.append(
            _fail("S0-META", meta_path, f".migration_meta.json 非法 JSON：{exc}")
        )
        return failures, warnings

    for field in ("op_slug", "family", "extracted_functions"):
        if field not in meta or not meta[field]:
            failures.append(
                _fail(
                    "S0-META-FIELD", meta_path, f".migration_meta.json 缺字段 {field}"
                )
            )
    funcs = meta.get("extracted_functions") or []
    if isinstance(funcs, list) and not funcs:
        failures.append(
            _fail(
                "S0-META-FUNC", meta_path, "extracted_functions 为空（至少 1 个函数）"
            )
        )

    family = meta.get("family")
    op_slug = meta.get("op_slug")
    if family and op_slug:
        checks = [
            ("S0-SCAFFOLD", f"tileops/manifest/{family}.yaml"),
            ("S0-SCAFFOLD", f"tileops/workloads/{family}.py"),
            ("S0-SCAFFOLD", f"tileops/kernels/{family}/{op_slug}/{op_slug}.py"),
        ]
        for test_slug in [meta.get("test_slug")] if meta.get("test_slug") else []:
            checks.append(("S0-SCAFFOLD", f"tests/ops/test_{test_slug}.py"))
        for bench_slug in [meta.get("bench_slug")] if meta.get("bench_slug") else []:
            checks.append(("S0-SCAFFOLD", f"benchmarks/ops/bench_{bench_slug}.py"))
        for rule_id, rel in checks:
            abs_path = os.path.join(repo_root, "examples/TileOPs", rel)
            if not os.path.exists(abs_path):
                failures.append(
                    _fail(
                        rule_id,
                        abs_path,
                        f"TileOPs 脚手架文件缺失：examples/TileOPs/{rel}",
                    )
                )
    return failures, warnings


# ---------------------------------------------------------------------------
# Stage 5: integration package lint
# ---------------------------------------------------------------------------

WRAPPER_PERF_IMPORT_RE = re.compile(r"^\s*#\s*(?:from|import)\s+.*perf_opt", re.M)


def _literal_ascend_mode(source: str) -> str | None:
    """Find the final kernel's explicit mode (legacy setdefault is accepted)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    declared, defaults = [], []
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "ASCEND_MODE"
                for target in node.targets
            )
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            declared.append(node.value.value)
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if (
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "setdefault"
            and isinstance(call.func.value, ast.Attribute)
            and call.func.value.attr == "environ"
            and len(call.args) >= 2
            and isinstance(call.args[0], ast.Constant)
            and call.args[0].value == "TILELANG_ASCEND_MODE"
        ):
            value = call.args[1]
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                defaults.append(value.value)
            elif isinstance(value, ast.Name) and value.id == "ASCEND_MODE" and declared:
                defaults.append(declared[-1])
    modes = {mode.capitalize() for mode in declared + defaults}
    return modes.pop() if len(modes) == 1 and modes <= {"Developer", "Expert"} else None


def integration_lint(migration_state: dict, repo_root: str):
    """Stage 5 gate: integration package, wrapper, and single-op report.

    ``migration_state`` is the parsed .migration_state.json (needs meta_path,
    family, op_slug, functions).
    """
    failures, warnings = [], []
    family = migration_state.get("family")
    op_slug = migration_state.get("op_slug")
    meta_path = migration_state.get("meta_path")
    if not (family and op_slug):
        failures.append(
            _fail("S5-STATE", str(meta_path), ".migration_state.json 缺 family/op_slug")
        )
        return failures, warnings
    pkg_dir = os.path.join(
        repo_root,
        f"examples/TileOPs/tileops/kernels/{family}/{op_slug}/{op_slug}_kernel",
    )
    if not os.path.isdir(pkg_dir):
        failures.append(_fail("S5-PKG", pkg_dir, "集成包目录不存在"))
        return failures, warnings
    for required in ("__init__.py", "integration_log.md", "integration_report.json"):
        if not os.path.exists(os.path.join(pkg_dir, required)):
            failures.append(
                _fail(
                    "S5-PKG", os.path.join(pkg_dir, required), f"集成包缺少 {required}"
                )
            )
    for func in migration_state.get("functions") or {}:
        for suffix in (".py", "_DESIGN.md"):
            path = os.path.join(pkg_dir, f"{func}{suffix}")
            if not os.path.exists(path):
                failures.append(
                    _fail("S5-FUNCS", path, f"集成包缺少函数 {func} 的 {suffix} 文件")
                )
    wrapper = os.path.join(
        repo_root, f"examples/TileOPs/tileops/kernels/{family}/{op_slug}/{op_slug}.py"
    )
    wtext = _read_text(wrapper)
    source_modes = set()
    for func in migration_state.get("functions") or {}:
        kernel_path = os.path.join(pkg_dir, f"{func}.py")
        kernel_text = _read_text(kernel_path)
        if kernel_text is None:
            continue
        mode = _literal_ascend_mode(kernel_text)
        if mode is None:
            failures.append(
                _fail(
                    "S5-ASCEND-MODE",
                    kernel_path,
                    "kernel 缺少明确且一致的 Developer/Expert 模式声明",
                )
            )
        else:
            source_modes.add(mode)
    if len(source_modes) > 1:
        failures.append(
            _fail(
                "S5-ASCEND-MODE",
                pkg_dir,
                "同一 Kernel class 集成了不同编程模式，需逐调用限定模式",
            )
        )
    if wtext is None:
        failures.append(_fail("S5-WRAPPER", wrapper, "wrapper 文件不存在或不可读"))
    else:
        try:
            wrapper_tree = ast.parse(wtext)
            classes = [
                node
                for node in wrapper_tree.body
                if isinstance(node, ast.ClassDef)
                and any(
                    isinstance(base, ast.Name) and base.id == "Kernel"
                    for base in node.bases
                )
            ]
            class_modes = [
                stmt.value.value
                for cls in classes
                for stmt in cls.body
                if isinstance(stmt, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "ascend_mode"
                    for target in stmt.targets
                )
                and isinstance(stmt.value, ast.Constant)
            ]
            if (
                len(classes) != 1
                or len(class_modes) != 1
                or set(class_modes) != source_modes
            ):
                failures.append(
                    _fail(
                        "S5-ASCEND-MODE",
                        wrapper,
                        "wrapper 的 Kernel.ascend_mode 与集成 kernel 模式不一致",
                    )
                )
        except SyntaxError:
            failures.append(
                _fail(
                    "S5-ASCEND-MODE",
                    wrapper,
                    "wrapper Python 语法错误，无法检查编程模式",
                )
            )
        if "perf_opt" not in wtext:
            failures.append(
                _fail(
                    "S5-WRAPPER-SWITCH",
                    wrapper,
                    "wrapper 未生成 baseline/perf_opt 双 import 切换块（未检测到 perf_opt 引用）",
                )
            )
        elif not WRAPPER_PERF_IMPORT_RE.search(wtext):
            warnings.append(
                _warn(
                    "S5-WRAPPER-SWITCH",
                    wrapper,
                    "wrapper 含 perf_opt 引用但未检测到注释态的 perf_opt import（切换块形态请复核）",
                )
            )
    report_ref_path = os.path.join(pkg_dir, "integration_report.json")
    report_ref_text = _read_text(report_ref_path)
    if report_ref_text is None:
        return failures, warnings
    try:
        report_ref = json.loads(report_ref_text)
    except json.JSONDecodeError as exc:
        failures.append(
            _fail("S5-REPORT-REF", report_ref_path, f"报告引用不是合法 JSON：{exc}")
        )
        return failures, warnings
    run_rel = report_ref.get("run_json") if isinstance(report_ref, dict) else None
    if not isinstance(run_rel, str) or not run_rel.endswith("/run.json"):
        failures.append(
            _fail(
                "S5-REPORT-REF",
                report_ref_path,
                "run_json 必须是仓库相对的报告 run.json 路径",
            )
        )
        return failures, warnings
    report_root = os.path.realpath(
        os.path.join(repo_root, "examples", "TileOPs", "reports", "tileops")
    )
    run_path = os.path.realpath(os.path.join(repo_root, run_rel))
    if (
        not run_rel.startswith("examples/TileOPs/reports/tileops/")
        or os.path.commonpath((report_root, run_path)) != report_root
    ):
        failures.append(
            _fail(
                "S5-REPORT-REF", report_ref_path, "run_json 路径不在 TileOPs 报告目录内"
            )
        )
        return failures, warnings
    run_text = _read_text(run_path)
    if run_text is None:
        failures.append(_fail("S5-REPORT", run_path, "报告 run.json 不存在或不可读"))
        return failures, warnings
    try:
        run = json.loads(run_text)
    except json.JSONDecodeError as exc:
        failures.append(
            _fail("S5-REPORT", run_path, f"报告 run.json 不是合法 JSON：{exc}")
        )
        return failures, warnings
    if not isinstance(run, dict):
        failures.append(_fail("S5-REPORT", run_path, "报告 run.json 顶层必须为对象"))
        return failures, warnings

    meta_file = (
        meta_path
        if os.path.isabs(str(meta_path))
        else os.path.join(repo_root, str(meta_path))
    )
    try:
        meta = json.loads(_read_text(meta_file) or "")
    except (json.JSONDecodeError, TypeError):
        meta = None
    if not isinstance(meta, dict) or not meta.get("op_name"):
        failures.append(
            _fail("S5-REPORT-META", meta_file, "迁移元数据缺少 manifest 算子名 op_name")
        )
        return failures, warnings
    if run.get("operator") != meta["op_name"]:
        failures.append(
            _fail("S5-REPORT-OP", run_path, "报告算子与迁移元数据 op_name 不一致")
        )
    metadata = run.get("metadata")
    summary = run.get("summary")
    if not isinstance(metadata, dict) or not isinstance(summary, dict):
        failures.append(_fail("S5-REPORT", run_path, "报告缺少 metadata/summary 对象"))
        return failures, warnings
    expected_test = (
        meta.get("test_path") or f"tests/ops/test_{meta.get('test_slug')}.py"
    )
    expected_bench = (
        meta.get("bench_path") or f"benchmarks/ops/bench_{meta.get('bench_slug')}.py"
    )
    if (
        metadata.get("test_file") != expected_test
        or metadata.get("benchmark_file") != expected_bench
    ):
        failures.append(
            _fail(
                "S5-REPORT-TARGET",
                run_path,
                "报告 test/benchmark 目标与迁移元数据不一致",
            )
        )
    if metadata.get("prof_mode_requested") != "msprof":
        failures.append(_fail("S5-REPORT-PROF", run_path, "报告未请求 msprof 模式"))
    correctness_passed = summary.get("correctness_passed") is True
    benchmark_requested = summary.get("benchmark_requested") is True
    benchmark_passed = summary.get("benchmark_passed") is True
    status = run.get("status")
    correctness_tests = summary.get("correctness_tests")
    if (
        not correctness_passed
        or not isinstance(correctness_tests, int)
        or isinstance(correctness_tests, bool)
        or correctness_tests < 1
    ):
        failures.append(
            _fail("S5-REPORT-TEST", run_path, "报告缺少通过的全量正确性用例")
        )
    if not benchmark_requested:
        failures.append(_fail("S5-REPORT-BENCH", run_path, "报告未实际运行 benchmark"))
    if status not in ("passed", "partial") or (status == "passed") != benchmark_passed:
        failures.append(
            _fail("S5-REPORT-STATUS", run_path, "报告状态与 benchmark 结论不一致")
        )
    if status == "partial" and correctness_passed and benchmark_requested:
        warnings.append(
            _warn(
                "S5-REPORT-BENCH",
                run_path,
                "正确性通过，但 benchmark 未得到有效结果；仅记录，不阻断集成",
            )
        )
    for name in ("report.md", "report.html"):
        path = os.path.join(os.path.dirname(run_path), name)
        if not os.path.isfile(path):
            failures.append(_fail("S5-REPORT-FILE", path, f"缺少 {name}"))
    sources = [wrapper, os.path.join(pkg_dir, "__init__.py")]
    sources.extend(
        os.path.join(pkg_dir, f"{func}.py")
        for func in migration_state.get("functions") or {}
    )
    if all(os.path.isfile(path) for path in sources) and os.path.getmtime(
        run_path
    ) < max(os.path.getmtime(path) for path in sources):
        failures.append(
            _fail(
                "S5-REPORT-STALE",
                run_path,
                "报告早于当前 wrapper/kernel 集成文件，需重新运行",
            )
        )
    return failures, warnings


# ---------------------------------------------------------------------------
# .stage_state.json schema lint (consumed by verify/repair/every mutation)
# ---------------------------------------------------------------------------

SCHEMA = {
    "task_id": str,
    "project_name": str,
    "operator_name": str,
    "scenario": str,
    "migration_mode": (str, type(None)),
    "stage_plan": list,
    "phase": str,
    "user_requirement": (str, type(None)),
    "design_md_path": (str, type(None)),
    "review_md_path": (str, type(None)),
    "kernel_py_path": (str, type(None)),
    "kernel_opt_py_path": (str, type(None)),
    "retry_count": int,
    "max_retry": int,
    "final_artifact": (str, type(None)),
    "stage_status": dict,
    "stage_retry_count": dict,
    "stage3_failure_breakdown": dict,
    "perf_iteration": dict,
    "perf_tuning_requested": (str, type(None), bool),
    "env_check_passed": bool,
    "failure_reason": (str, type(None)),
}
PHASES = {
    "INIT",
    "SCAFFOLD",
    "DESIGN",
    "REVIEW",
    "DEVELOP",
    "TUNING",
    "INTEGRATE",
    "DONE",
    "FAILED",
}
SCENARIOS = {"new_op", "migration", "optimize"}
MIGRATION_MODES = {"harness", "plain"}
STAGE_STATUS_VALUES = {"in_progress", "completed", "failed"}
BLOCKED_CODES = {
    "BLOCKED_DESIGN",
    "BLOCKED_IMPL",
    "BLOCKED_ACCURACY",
    "BLOCKED_SCAFFOLD",
    "BLOCKED_INTEGRATION",
    "BLOCKED_ENVIRONMENT",
    "BLOCKED_SPEC",
}


def state_schema_lint(state: dict, state_path: str):
    failures, warnings = [], []
    for field, expected in SCHEMA.items():
        if field not in state:
            failures.append(
                _fail("SCHEMA-MISSING", state_path, f"状态文件缺字段 {field}")
            )
        elif not isinstance(state[field], expected):
            failures.append(
                _fail(
                    "SCHEMA-TYPE",
                    state_path,
                    f"字段 {field} 类型错误：期望 {expected}，实际 {type(state[field]).__name__}",
                )
            )
    if state.get("scenario") not in SCENARIOS:
        failures.append(
            _fail("SCHEMA-ENUM", state_path, f"scenario 非法：{state.get('scenario')}")
        )
    if state.get("phase") not in PHASES:
        failures.append(
            _fail("SCHEMA-ENUM", state_path, f"phase 非法：{state.get('phase')}")
        )
    mode = state.get("migration_mode")
    if mode is not None and mode not in MIGRATION_MODES:
        failures.append(
            _fail("SCHEMA-ENUM", state_path, f"migration_mode 非法：{mode}")
        )
    reason = state.get("failure_reason")
    if reason is not None and reason not in BLOCKED_CODES:
        failures.append(
            _fail("SCHEMA-ENUM", state_path, f"failure_reason 非法：{reason}")
        )
    for key, val in (state.get("stage_status") or {}).items():
        if val is not None and val not in STAGE_STATUS_VALUES:
            failures.append(
                _fail(
                    "SCHEMA-ENUM",
                    state_path,
                    f"stage_status[{key}] 非法：{val}",
                )
            )
    for k in range(6):
        if str(k) not in (state.get("stage_retry_count") or {}):
            warnings.append(
                _warn(
                    "SCHEMA-MISSING",
                    state_path,
                    f"stage_retry_count 缺键 '{k}'（repair 可补齐）",
                )
            )
    return failures, warnings


DEFAULT_STATE = {
    "task_id": "",
    "project_name": "",
    "operator_name": "",
    "scenario": "new_op",
    "migration_mode": None,
    "stage_plan": [1, 2, 3],
    "phase": "INIT",
    "user_requirement": None,
    "design_md_path": None,
    "review_md_path": None,
    "kernel_py_path": None,
    "kernel_opt_py_path": None,
    "retry_count": 0,
    "max_retry": 3,
    "final_artifact": None,
    "stage_status": {},
    "stage_retry_count": {str(k): 0 for k in range(6)},
    "stage3_failure_breakdown": {"runtime_fail": 0, "precision_fail": 0},
    "perf_iteration": {
        "count": 0,
        "last_improvement": 0.0,
        "consecutive_no_improvement": 0,
    },
    "perf_tuning_requested": None,
    "env_check_passed": False,
    "failure_reason": None,
    # optional fields (backfilled, not required by SCHEMA)
    "perf_feedback": None,
    "budget": {
        "max_wallclock_s": None,
        "max_subagent_dispatches": None,
        "max_stage4_experiments": None,
    },
}


def state_backfill(state: dict) -> dict:
    """Backfill missing schema fields with defaults (conductor rule: schema 补齐)."""
    filled = dict(state)
    for field, default in DEFAULT_STATE.items():
        if field not in filled:
            filled[field] = default
    retry = filled.get("stage_retry_count")
    if not isinstance(retry, dict):
        retry = {}
    for k in range(6):
        retry.setdefault(str(k), 0)
    filled["stage_retry_count"] = retry
    status = filled.get("stage_status")
    if not isinstance(status, dict):
        status = {}
    filled["stage_status"] = status
    if not isinstance(filled.get("stage3_failure_breakdown"), dict):
        filled["stage3_failure_breakdown"] = {"runtime_fail": 0, "precision_fail": 0}
    if not isinstance(filled.get("perf_iteration"), dict):
        filled["perf_iteration"] = {
            "count": 0,
            "last_improvement": 0.0,
            "consecutive_no_improvement": 0,
        }
    if "artifact_hashes" not in filled:
        filled["artifact_hashes"] = {}
    if "last_failure_reason" not in filled:
        filled["last_failure_reason"] = None
    if not isinstance(filled.get("budget"), dict):
        filled["budget"] = {
            "max_wallclock_s": None,
            "max_subagent_dispatches": None,
            "max_stage4_experiments": None,
        }
    return filled
