"""Tests for the conductor scenario split (improvement #22 / §6.1 item 3).

Locks the structural contract of .opencode/agents/tilelang-op-conductor.md +
.opencode/agents/conductor-scenarios/*.md:

  - main file stays within the line budget (target ≤300 lines, E1.1), keeping
    only scenario routing + state-machine skeleton;
  - all four scenario files exist and are non-trivial;
  - scenario files are agent-discovery-safe: every .md under .opencode/agents/
    is registered as an agent unless it carries `disable: true` frontmatter
    (same pattern as the agents README), so scenario files must start with it;
  - main references every scenario file via the mandatory load table;
  - every scenario file declares its scenario scope and back-references main;
  - scenario-specific rule sentinels must NOT regress back into the main file
    (the split's core invariant: scenario detail lives in scenario files);
  - the scenario dir contains exactly the expected files (no strays).

Inline duplication of _shared/standards text and dangling standards refs in
these same files are separately enforced by standards_check.py (it scans
.opencode/agents recursively), see test_standards_check.py.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MAIN = REPO_ROOT / ".opencode" / "agents" / "tilelang-op-conductor.md"
SCENARIO_DIR = REPO_ROOT / ".opencode" / "agents" / "conductor-scenarios"

SCENARIOS = ["new-op.md", "migration.md", "harness.md", "optimize.md"]

# §6.1 item 3 + E1.1: 主文件保留场景路由 + 状态机骨架（目标 ≤300 行；
# 门禁总表/重试上限/信号枚举引用 _shared/standards，不再内联）
LINE_BUDGET = 300
MIN_LINES = 20

# Scenario-only rule sentinels: (must exist in the scenario file, must NOT
# exist in the main file). Picked as stable verbatim fragments of rules that
# moved out of the main conductor during the split. C3: plain.md merged into
# migration.md §6 — its sentinels now live there.
SENTINELS = {
    "new-op.md": ["编程模式偏好", "调优必要信息收集"],
    "migration.md": ["三问解读", "迁移前后保持不变", "无 Stage 0 / Stage 5"],
    "harness.md": ["extracted_functions", "migration init --op-slug"],
    "optimize.md": ["翻转切换块注释"],
}


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_main_line_budget():
    lines = read(MAIN).splitlines()
    assert len(lines) <= LINE_BUDGET, (
        f"conductor 主文件 {len(lines)} 行超出 §6.1 预算 {LINE_BUDGET} 行——"
        "场景专属规则应下沉 conductor-scenarios/，不得回流主文件"
    )


def test_scenario_files_exist_and_nontrivial():
    for name in SCENARIOS:
        path = SCENARIO_DIR / name
        assert path.is_file(), f"场景文件缺失：{path}"
        n = len(read(path).splitlines())
        assert n >= MIN_LINES, f"场景文件 {name} 过小（{n} 行）——内容可能丢失"


def test_scenario_files_disable_frontmatter():
    # .opencode/agents 下所有 .md 会被 opencode 注册为 agent，场景文件必须
    # disable 以免被当作 agent 加载（与 agents README 同款防护）。
    for name in SCENARIOS:
        text = read(SCENARIO_DIR / name)
        assert text.startswith("---\ndisable: true\n---"), (
            f"{name} 缺少 `disable: true` frontmatter——会被 agent 发现机制注册"
        )


def test_main_references_all_scenario_files():
    text = read(MAIN)
    for name in SCENARIOS:
        assert f"conductor-scenarios/{name}" in text, (
            f"主文件未引用场景文件 {name}（「场景文件加载」表不完整）"
        )
    assert "场景文件加载" in text, "主文件缺少「场景文件加载」章节"
    # mandatory load rule (route -> Read) must stay
    assert "未 Read 场景文件不得开始需求预检或推进任何 Stage" in text


def test_scenario_files_declare_scope_and_backref():
    for name in SCENARIOS:
        text = read(SCENARIO_DIR / name)
        assert "scenario" in text, f"{name} 未声明适用范围（scenario）"
        assert "tilelang-op-conductor.md" in text, f"{name} 未回引用主文件"
        assert "载入" in text, f"{name} 未说明载入时机"


def test_sentinels_not_in_main():
    main = read(MAIN)
    for name, sentinels in SENTINELS.items():
        scenario_text = read(SCENARIO_DIR / name)
        for s in sentinels:
            assert s in scenario_text, f"哨兵 {s!r} 不在 {name}——场景规则疑似丢失"
            assert s not in main, (
                f"哨兵 {s!r} 回流主文件——{name} 的专属规则应在场景文件中单点维护"
            )


def test_no_stray_files_in_scenario_dir():
    if not SCENARIO_DIR.is_dir():
        raise AssertionError(f"场景目录缺失：{SCENARIO_DIR}")
    actual = sorted(p.name for p in SCENARIO_DIR.iterdir() if p.suffix == ".md")
    assert actual == sorted(SCENARIOS), (
        f"conductor-scenarios/ 存在意外文件：{actual}（预期 {sorted(SCENARIOS)}）"
    )


def test_main_load_table_covers_all_scenarios():
    # every loadable scenario combination must name its file(s) explicitly
    text = read(MAIN)
    for combo in ["new_op", "migration-plain", "migration-harness", "optimize"]:
        assert combo in text, f"「场景文件加载」表缺少场景组合 {combo}"
