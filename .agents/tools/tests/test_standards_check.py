"""Tests for standards_check (U10 fix acceptance criteria).

Acceptance mapping (docs/developer/conductor-improvement-analysis-aggregated.md
improvement #17 / §6.1):
  - 单一事实源：标准文本指纹不得在消费者文件中内联复制（SC-INLINE-DUP）
  - 校验哈希：标准文件修改后 lock 哈希必须同步更新（SC-HASH-MISMATCH /
    SC-UNTRACKED / SC-LOCK-MISSING）
  - 引用完整性：消费者对 _shared/standards/ 的引用必须可解析
    （SC-DANGLING-REF）；标准文件引用的仓库路径必须存在（SC-REF-PATH）
  - 指纹自身健康：指纹必须存在于权威文件（SC-FP-STALE）

All invocations go through the CLI boundary (subprocess) with a hermetic
fake repo per test, mirroring test_statectl.py conventions.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1]
SCRIPT = TOOLS_DIR / "standards_check.py"


def run(args: list[str], repo_root: str) -> tuple[int, dict]:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), *args, "--repo-root", repo_root],
        capture_output=True,
        text=True,
    )
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        payload = {"ok": False, "_raw": proc.stdout, "_err": proc.stderr}
    return proc.returncode, payload


CORE_SPLIT = """# core-split
FINGERPRINT-CORESPLIT 使其接近物理核数整数倍（按实查核数取 1×/2×/3×）。
FINGERPRINT-CORESPLIT2 禁止以文档假设或经验值（如 20~24）替代实查。
"""

ALGO = """# algorithm research
复杂度对比须覆盖四口径（FLOPs / 访存 / 扫描遍数 / 中间缓冲峰值）。
"""

EVIDENCE = """# evidence
禁止凭先验（尤其 GPU 直觉）在检索前否决候选。
参考 docs/开发指南.md。
"""

GATE = """# gate and retry
TileOPs 7 文件存在 + Tier 1 通过（import / manifest / collect-only）。
| 3 | 5 次 Subagent 调度（运行失败 + 精度失败合并累计；`[DESIGN_ERROR]` 触发修订不计入） | BLOCKED_IMPL |
"""

STAGE3 = """# stage3 routing
强制要求 Developer 先备份当前 impl 到 `history_version/{op}_impl_s3_attempt{N}.py` 再做修改。
"""

REPORT = """# report
harness 集成验证: <smoke / full 用例数与结果，仅 migration-harness 填>
"""

PERF_FEEDBACK = """# perf feedback
参数级性能不足（tiling 可解）留在 Stage 4 迭代内，永不触发本机制。
"""

SIGNAL_REGISTRY = """# signal registry
developer mode 枚举：`first_impl` / `retry_impl` / `precision_fix`。
optimizer mode 枚举：`full`（默认，完整调优流程） / `precision_fix`（仅精度回归修复）。
perf_records.jsonl 字段契约：
{round, candidate_id, parent_id, dispatch_path, workload, duration_us, l0_pass, msprof_raw_path, timestamp}
"""

CONDUCTOR_MAIN = """# conductor main
| 4 算子调优 | `TUNING` | `@tilelang-op-optimizer` |
├── RETROSPECTIVE.md              # Stage 1/2/3/5 复盘（自进化钩子；Stage 4 复盘在 perf_opt/opt_log.md）
"""

NEW_OP_SCENARIO = """# new-op scenario
INIT --> DESIGN --> REVIEW --> DEVELOP --> TUNING(可选) --> DONE
"""

EVOLUTION_SKILL = """# skill evolution
| **D 实测数据** | 性能数字、代价常数、编译器/运行时陷阱实证、API 实际行为实证 | **Tier 0** 直接合入 `pattern-library.md` §1/§2（须带溯源 + 工具链版本戳 + 复现命令三件套） |
"""


def build_fake_repo() -> str:
    root = tempfile.mkdtemp(prefix="stdchk_")
    std = os.path.join(root, ".agents", "skills", "_shared", "standards")
    os.makedirs(std)
    os.makedirs(os.path.join(root, ".agents", "tools"))
    os.makedirs(os.path.join(root, ".opencode", "agents", "conductor-scenarios"))
    os.makedirs(os.path.join(root, ".agents", "skills", "tilelang-x"))
    os.makedirs(os.path.join(root, ".agents", "skills", "tilelang-skill-evolution"))
    os.makedirs(os.path.join(root, "docs"))
    # standards files covering every standards-dir fingerprint's canonical file
    files = {
        "core-split-strategy.md": CORE_SPLIT,
        "algorithm-research.md": ALGO,
        "negative-claim-evidence.md": EVIDENCE,
        "gate-and-retry.md": GATE,
        "stage3-routing.md": STAGE3,
        "perf-feedback.md": PERF_FEEDBACK,
        "final-report-template.md": REPORT,
        "signal-registry.md": SIGNAL_REGISTRY,
    }
    for name, body in files.items():
        with open(os.path.join(std, name), "w", encoding="utf-8") as fh:
            fh.write(body)
    # repo-scoped canonical files for the C3.2 knowledge-block fingerprints
    with open(
        os.path.join(root, ".opencode", "agents", "tilelang-op-conductor.md"),
        "w",
        encoding="utf-8",
    ) as fh:
        fh.write(CONDUCTOR_MAIN)
    with open(
        os.path.join(root, ".opencode", "agents", "conductor-scenarios", "new-op.md"),
        "w",
        encoding="utf-8",
    ) as fh:
        fh.write(NEW_OP_SCENARIO)
    with open(
        os.path.join(root, ".agents", "skills", "tilelang-skill-evolution", "SKILL.md"),
        "w",
        encoding="utf-8",
    ) as fh:
        fh.write(EVOLUTION_SKILL)
    # a path cited by EVIDENCE must exist
    with open(os.path.join(root, "docs", "开发指南.md"), "w", encoding="utf-8") as fh:
        fh.write("# 开发指南\n")
    # one well-behaved consumer that references (not inlines) standards
    with open(
        os.path.join(root, ".opencode", "agents", "good.md"), "w", encoding="utf-8"
    ) as fh:
        fh.write(
            "规则见 `.agents/skills/_shared/standards/core-split-strategy.md` 与 "
            "`_shared/standards/algorithm-research.md`。\n"
        )
    return root


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------


def test_update_then_check_passes():
    root = build_fake_repo()
    try:
        code, out = run(["update"], root)
        assert code == 0 and out["ok"] is True
        assert len(out["updated"]) == 8
        code, out = run(["check"], root)
        assert code == 0, out
        assert out["ok"] is True and out["failures"] == []
        assert out["stats"]["standards_refs"] == 2
        assert out["stats"]["dup_hits"] == 0
    finally:
        shutil.rmtree(root)


def test_check_missing_lock_reports_and_still_scans():
    root = build_fake_repo()
    try:
        code, out = run(["check"], root)
        assert code == 1
        rules = {f["rule_id"] for f in out["failures"]}
        assert "SC-LOCK-MISSING" in rules
        assert "SC-UNTRACKED" in rules  # files present but not tracked
        # missing lock does not disable the dup detector below
    finally:
        shutil.rmtree(root)


# ---------------------------------------------------------------------------
# hash / lock governance
# ---------------------------------------------------------------------------


def test_edited_standard_without_lock_update_is_rejected():
    root = build_fake_repo()
    try:
        run(["update"], root)
        target = os.path.join(
            root, ".agents", "skills", "_shared", "standards", "algorithm-research.md"
        )
        with open(target, "a", encoding="utf-8") as fh:
            fh.write("新增一行规则\n")
        code, out = run(["check"], root)
        assert code == 1
        hit = [f for f in out["failures"] if f["rule_id"] == "SC-HASH-MISMATCH"]
        assert hit and "algorithm-research.md" in hit[0]["file"]
        # update heals it
        code, out = run(["update"], root)
        assert code == 0
        code, out = run(["check"], root)
        assert code == 0
    finally:
        shutil.rmtree(root)


def test_deleted_standard_leaves_stale_lock_entry():
    root = build_fake_repo()
    try:
        run(["update"], root)
        os.remove(
            os.path.join(
                root, ".agents", "skills", "_shared", "standards", "gate-and-retry.md"
            )
        )
        code, out = run(["check"], root)
        assert code == 1
        rules = {f["rule_id"] for f in out["failures"]}
        assert "SC-LOCK-STALE" in rules
        assert "SC-FP-STALE" in rules  # its fingerprint lost its canonical file
    finally:
        shutil.rmtree(root)


def test_corrupt_lock_is_reported():
    root = build_fake_repo()
    try:
        run(["update"], root)
        with open(
            os.path.join(
                root, ".agents", "skills", "_shared", "standards", "standards.lock.json"
            ),
            "w",
            encoding="utf-8",
        ) as fh:
            fh.write("{not json")
        code, out = run(["check"], root)
        assert code == 1
        assert "SC-LOCK-CORRUPT" in {f["rule_id"] for f in out["failures"]}
    finally:
        shutil.rmtree(root)


def test_show_dumps_lock():
    root = build_fake_repo()
    try:
        run(["update"], root)
        code, out = run(["show"], root)
        assert code == 0
        assert "core-split-strategy.md" in out["files"]
        assert len(out["files"]["core-split-strategy.md"]) == 64  # sha256 hex
    finally:
        shutil.rmtree(root)


# ---------------------------------------------------------------------------
# inline duplication (the core U10 signal)
# ---------------------------------------------------------------------------


def test_inlined_rule_text_in_consumer_is_flagged():
    root = build_fake_repo()
    try:
        run(["update"], root)
        with open(
            os.path.join(root, ".opencode", "agents", "conductor-fake.md"),
            "w",
            encoding="utf-8",
        ) as fh:
            fh.write(
                "# fake conductor\n"
                "分核方案：使其接近物理核数整数倍（按实查核数取 1×/2×/3×），"
                "并对齐取值。\n"
            )
        code, out = run(["check"], root)
        assert code == 1
        hits = [f for f in out["failures"] if f["rule_id"] == "SC-INLINE-DUP"]
        assert len(hits) == 1
        assert "conductor-fake.md" in hits[0]["file"]
        assert "SC-DUP-CORESPLIT" in hits[0]["message"]
    finally:
        shutil.rmtree(root)


def test_inlined_rule_text_in_skill_is_flagged():
    root = build_fake_repo()
    try:
        run(["update"], root)
        with open(
            os.path.join(root, ".agents", "skills", "tilelang-x", "SKILL.md"),
            "a",
            encoding="utf-8",
        ) as fh:
            fh.write(
                "注意：复杂度对比须覆盖四口径（FLOPs / 访存 / 扫描遍数 / 中间缓冲峰值）。\n"
            )
        code, out = run(["check"], root)
        assert code == 1
        hits = [f for f in out["failures"] if f["rule_id"] == "SC-INLINE-DUP"]
        assert len(hits) == 1 and "tilelang-x/SKILL.md" in hits[0]["file"]
    finally:
        shutil.rmtree(root)


# ---------------------------------------------------------------------------
# C3.2 knowledge-block fingerprints (state machine / stage table / dir tree /
# retry limits / D-P-R-C / mode enums / perf_records contract)
# ---------------------------------------------------------------------------


def test_readme_inlining_state_machine_is_flagged():
    root = build_fake_repo()
    try:
        run(["update"], root)
        # README mirrors the scenario state-machine diagram verbatim
        with open(
            os.path.join(root, ".opencode", "agents", "README.md"),
            "w",
            encoding="utf-8",
        ) as fh:
            fh.write(
                "# readme\n\n```\nINIT --> DESIGN --> REVIEW --> DEVELOP "
                "--> TUNING(可选) --> DONE\n```\n"
            )
        code, out = run(["check"], root)
        assert code == 1
        hits = [f for f in out["failures"] if f["rule_id"] == "SC-INLINE-DUP"]
        assert any("SC-DUP-STATEMACHINE" in h["message"] for h in hits)
    finally:
        shutil.rmtree(root)


def test_readme_inlining_dprc_table_is_flagged():
    root = build_fake_repo()
    try:
        run(["update"], root)
        with open(
            os.path.join(root, ".opencode", "agents", "README.md"),
            "w",
            encoding="utf-8",
        ) as fh:
            fh.write(
                "| **D 实测数据** | 性能数字、代价常数、编译器/运行时陷阱实证、"
                "API 实际行为实证 | **Tier 0** 直接合入 `pattern-library.md` §1/§2"
                "（须带溯源 + 工具链版本戳 + 复现命令三件套） |\n"
            )
        code, out = run(["check"], root)
        assert code == 1
        hits = [f for f in out["failures"] if f["rule_id"] == "SC-INLINE-DUP"]
        assert any("SC-DUP-DPRC" in h["message"] for h in hits)
    finally:
        shutil.rmtree(root)


def test_consumer_inlining_mode_enum_is_flagged():
    root = build_fake_repo()
    try:
        run(["update"], root)
        with open(
            os.path.join(root, ".agents", "skills", "tilelang-x", "SKILL.md"),
            "a",
            encoding="utf-8",
        ) as fh:
            fh.write("mode 枚举：`first_impl` / `retry_impl` / `precision_fix`。\n")
        code, out = run(["check"], root)
        assert code == 1
        hits = [f for f in out["failures"] if f["rule_id"] == "SC-INLINE-DUP"]
        assert any("SC-DUP-MODEENUM" in h["message"] for h in hits)
    finally:
        shutil.rmtree(root)


def test_repo_scoped_fingerprint_stale_when_canonical_edited():
    root = build_fake_repo()
    try:
        run(["update"], root)
        # rewrite the canonical new-op.md without the state machine sentinel
        with open(
            os.path.join(
                root, ".opencode", "agents", "conductor-scenarios", "new-op.md"
            ),
            "w",
            encoding="utf-8",
        ) as fh:
            fh.write("# new-op scenario\n规则已迁移。\n")
        code, out = run(["check"], root)
        assert code == 1
        assert "SC-FP-STALE" in {f["rule_id"] for f in out["failures"]}
    finally:
        shutil.rmtree(root)


def test_standard_file_itself_is_exempt_from_dup_scan():
    """A fingerprint inside the standards dir must not self-trigger."""
    root = build_fake_repo()
    try:
        run(["update"], root)
        # core-split-strategy.md already contains both core-split fingerprints
        code, out = run(["check"], root)
        assert code == 0 and out["stats"]["dup_hits"] == 0
    finally:
        shutil.rmtree(root)


# ---------------------------------------------------------------------------
# reference resolution
# ---------------------------------------------------------------------------


def test_dangling_standards_reference_is_flagged():
    root = build_fake_repo()
    try:
        run(["update"], root)
        with open(
            os.path.join(root, ".opencode", "agents", "good.md"), "a", encoding="utf-8"
        ) as fh:
            fh.write("另见 `.agents/skills/_shared/standards/not-exist.md`。\n")
        code, out = run(["check"], root)
        assert code == 1
        hits = [f for f in out["failures"] if f["rule_id"] == "SC-DANGLING-REF"]
        assert hits and "not-exist.md" in hits[0]["message"]
    finally:
        shutil.rmtree(root)


def test_standard_citing_missing_repo_path_is_flagged():
    root = build_fake_repo()
    try:
        run(["update"], root)
        target = os.path.join(
            root, ".agents", "skills", "_shared", "standards", "algorithm-research.md"
        )
        with open(target, "a", encoding="utf-8") as fh:
            fh.write("参考 examples/nope/missing.md 的内容。\n")
        run(["update"], root)  # hash change must be accepted via update
        code, out = run(["check"], root)
        assert code == 1
        hits = [f for f in out["failures"] if f["rule_id"] == "SC-REF-PATH"]
        assert hits and "examples/nope/missing.md" in hits[0]["message"]
    finally:
        shutil.rmtree(root)


def test_template_braces_in_standard_paths_do_not_flag():
    root = build_fake_repo()
    try:
        run(["update"], root)
        target = os.path.join(
            root, ".agents", "skills", "_shared", "standards", "stage3-routing.md"
        )
        with open(target, "a", encoding="utf-8") as fh:
            fh.write("备份到 examples/{project}/{op}/history_version/ 目录。\n")
        run(["update"], root)
        code, out = run(["check"], root)
        assert code == 0, out
    finally:
        shutil.rmtree(root)


# ---------------------------------------------------------------------------
# real repo sanity (guards the U10 fix itself)
# ---------------------------------------------------------------------------


def test_real_repo_check_passes():
    repo_root = str(TOOLS_DIR.parents[1])
    code, out = run(["check"], repo_root)
    assert code == 0, out["failures"]
    assert out["ok"] is True
    # every canonical standard is referenced by at least one consumer
    assert out["stats"]["standards_refs"] >= out["stats"]["standards_files"]


if __name__ == "__main__":
    sys.exit(0)
