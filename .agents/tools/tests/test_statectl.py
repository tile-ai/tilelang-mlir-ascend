"""Tests for statectl / gate_lint (U1 fix acceptance criteria).

Acceptance mapping (docs/developer/conductor-improvement-analysis-aggregated.md §1.1):
  - 人工注入 5 类典型错误（跳阶段/漏计数/工件缺失/schema 损坏/重复 complete）全部拦截
  - 同一任务连跑两次状态文件 bit-identical
  - 人为篡改工件 verify 能报警

U14 fix (improvement #20, TUNING→DESIGN 受控逆向反馈):
  - perf_feedback.md schema gate (S4-PERF-FEEDBACK-*)
  - `set --perf-feedback-action` routing record (file precondition + timeline)
  - design-revision path C from Stage 4 (shared retry_count, downstream reset)

Improvement #1.3 (任务时间线事件流):
  - enriched event fields (stage/subagent/mode/verdict/duration_s)
  - duration_s = complete/fail ts − most recent start ts of the same stage
  - `timeline-summary` mechanical aggregation (dispatches / per-stage
    attempts & durations / chronological failure chain) + E-MISSING

All invocations go through the CLI boundary (subprocess), exactly as the
conductor would call it. Fixtures build a hermetic fake repo per test.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1]
STATECTL = TOOLS_DIR / "statectl.py"
FIXED_NOW = "2026-09-05T00:00:00Z"

# ---------------------------------------------------------------------------
# fixtures (minimal-but-gate-passing artifact content)
# ---------------------------------------------------------------------------

DESIGN_MD = """# t 算子设计文档

## 1. 概述

### 1.1 算子名称

t

### 1.2 功能描述

逐元素加一。

### 1.3 数学公式

y = x + 1

### 1.4 算法描述

逐元素映射，与 §1.6.0 选定算法一致。

### 1.5 数据流图

GM -> UB -> GM

### 1.6 算法调研与优化分析

#### 1.6.0 算法调研（Algorithm Research）

R1 等价化简公式：无化简公式（结构依据：单次逐元素映射无公共子表达式）。
R2 在线算法：逐元素映射天然单遍，无在线变体。
R3 复杂度对比：O(N) 单遍扫描，四口径（FLOPs/访存/扫描遍数/中间缓冲）与基线一致。
R4 硬件亲和：Vector 单元亲和。

| 算法候选 | 基线 | 结论 |
|---|---|---|
| 逐元素加一 | 是 | 选定 |

**调研结论**: 沿用基线（调研范围：algorithm-research.md 参考表 + 结构分析）。

#### 1.6.1 数学等价优化（公式级）

**优化结论**: 无优化空间（单次逐元素映射），依据见 1.6.0。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

**向量化结论**: 逐元素计算已全部向量化，保留 0 处标量。

#### 1.6.3 向量化轴与数据布局决策

**布局决策结论**: 选定向量化轴 N，核内布局沿用输入布局。

## 2. 编程模式选型

Developer 模式（自动内存与同步）。

## 3. API 映射设计

### 3.1 公式拆解

见 1.4。

### 3.2 TileLang API 映射

T.vadd（已对照 docs/Tilelang.language/数学操作/ 核对）。

### 3.3 计算伪代码

y[bx, i] = x[bx, i] + 1.0

### 3.4 API 可行性确认

vadd 在 docs/Tilelang.language/数学操作/ 有文档。

## 3.5 技术约束确认

### 3.5.1 本项目已知限制检查

无新增限制。

### 3.5.2 参考实现差异说明

无外部参考。

### 3.5.3 本项目同类实现参考

无。

## 4. 数据规格与内存规划

### 4.1 输入张量

[B, N] float16。

### 4.2 输出张量

同输入。

### 4.3 中间缓冲区

无。

## 5. Tiling 策略

### 5.5 分核策略（物理核数适配）

逻辑核数 = ceil(M / block_M)。
物理核数 = NPUUtils.get().get_aicore_num() 实查值，查询代码已记录。
规模判定：逻辑核数 ≤ 物理核数，无需适配。

## 6. 循环与调度结构

无显式循环（block 级并行）。

## 7. 同步策略

Developer 模式自动同步。

## 8. 验证方案

### 8.1 Golden 函数

golden_t 以 torch.add 为参考实现。

### 8.2 精度标准

L0 atol/rtol 1e-3。
"""

REVIEW_MD = """# t 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息

- 检视维度: 8 项
- 阻塞级问题数: 0

## 检视详情

### 1. API 可行性 — pass

ok

### 2. 内存层级规划 — pass

ok

### 3. Tiling 策略 — pass

ok

### 4. 技术约束检测 — pass

ok

### 5. 循环与同步 — pass

ok

### 6. 验证方案 — pass

ok

### 7. 完整性与一致性 — pass

ok

### 8. 算法优化分析 — pass

调研结论独立复核：复杂度复算一致，四问齐全且含基线候选；
弃选论证已亲自打开 API 文档核对限制条款，未见矛盾。
"""

KERNEL_PY = """import argparse

import tilelang
import tilelang.language as T
import torch


def golden_t(x):
    return torch.add(x, 1.0)


@tilelang.jit(target="npuir")
def t_kernel(B, N, dtype):
    @T.prim_func
    def main(X, Y):
        with T.Kernel(1, threads=1) as bx:
            Y[bx, 0] = X[bx, 0] + 1.0


def run_L0():
    x = torch.randn(4, 8)
    assert golden_t(x).shape == (4, 8)


def run_L1():
    x = torch.randn(8, 16)
    assert golden_t(x).shape == (8, 16)


def run_L2():
    x = torch.randn(3, 7)
    assert golden_t(x).shape == (3, 7)


def run_boundary():
    x = torch.zeros(4, 8)
    assert torch.isfinite(golden_t(x)).all()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "all"])
    args = parser.parse_args()
    if args.level == "all":
        run_L0()
        run_L1()
        run_L2()
        run_boundary()
    else:
        run_L0()
"""

OPT_LOG_MD = """# opt log

## 基线
baseline msprof 记录。

## Skill Retrospective

none
"""

OPT_LOG_RECORDS_MD = """# opt log

## 基线

baseline msprof 记录（round 0，见 perf_records.jsonl）。

## Iteration Log

### Round 1

| 候选 | Task Duration(us) | AICore 利用率 | Memory 指标 | L0 |
|---|---|---|---|---|
| current best (baseline) | 120.5 | 41% | 78% | pass |
| v2_c轴重排 | 100.0 | 67% | 80% | pass |

## Final Summary

best: v2_c轴重排
final_latency: 100.0 us
总提升: 17.0%
中止原因: plateau

## Skill Retrospective

none
"""

PERF_RECORDS_JSONL = (
    '{"round": 0, "candidate_id": "baseline", "parent_id": null, '
    '"dispatch_path": "t_kernel::main", "workload": "w1", "duration_us": 120.5, '
    '"l0_pass": true, "msprof_raw_path": "perf_opt/profiles/baseline/", '
    '"timestamp": "2026-09-06T00:00:00Z"}\n'
    '{"round": 1, "candidate_id": "v2", "parent_id": "baseline", '
    '"dispatch_path": "t_kernel::main", "workload": "w1", "duration_us": 100.0, '
    '"l0_pass": true, "msprof_raw_path": "perf_opt/profiles/round1/", '
    '"timestamp": "2026-09-06T00:01:00Z"}\n'
)

PERF_FEEDBACK_MD = """# 性能反馈（[DESIGN_LIMIT]）

## 触发判定

- 设计层归因: DESIGN.md §1.6.3 选定向量化轴 N，实测标量执行占比 62% 表明天花板在布局设计层，非 tiling 参数可解。
- 结构性加速估计: 3.2x（依据: C 轴融合布局实测外推，round2 对照实验）。

## 假设

DESIGN.md §1.6.3 假设原生布局 + 最内连续轴 N 已最优（先例依据 pattern-library §1）。

## 实测

msprof op Task Duration：主选 412us vs 备选布局变体 129us（同 dispatch、同 workload，raw profile 见 profiles/round2/）。

## 影响面

影响含 2 个以上非 batch 维的逐元素与窗口类算子族；改设计将重做 Stage 1 布局决策与 Stage 3 kernel 布局。

## 反馈结论

当前设计的天花板由布局选型决定，tiling 参数迭代无法收敛到目标。

建议路由: 附录补记
"""


def make_repo(tmp_path: Path, name: str = "repo") -> Path:
    repo = tmp_path / name
    (repo / ".git").mkdir(parents=True)
    (repo / "docs" / "Tilelang.language" / "数学操作").mkdir(parents=True)
    (repo / "examples").mkdir()
    return repo


def op_dir(repo: Path, project: str = "proj", op: str = "t") -> Path:
    d = repo / "examples" / project / op
    d.mkdir(parents=True, exist_ok=True)
    return d


def sc(*argv, now: str | None = FIXED_NOW, cwd: Path | None = None):
    env = dict(os.environ)
    if now:
        env["STATECTL_NOW"] = now
    proc = subprocess.run(
        [sys.executable, str(STATECTL), *argv],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(cwd) if cwd else None,
    )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        payload = {"stdout": proc.stdout, "stderr": proc.stderr}
    return proc.returncode, payload


def write_artifacts(d: Path, design=True, review=True, kernel=True):
    if design:
        (d / "DESIGN.md").write_text(DESIGN_MD, encoding="utf-8")
    if review:
        (d / "REVIEW.md").write_text(REVIEW_MD, encoding="utf-8")
    if kernel:
        (d / "t.py").write_text(KERNEL_PY, encoding="utf-8")


def init_new_op(d: Path, **kw):
    return sc(
        "init",
        "--dir",
        str(d),
        "--project",
        "proj",
        "--op",
        "t",
        "--scenario",
        "new_op",
        **kw,
    )


def happy_path(d: Path, with_stage4: bool = False):
    """Drive a full new_op lifecycle through the CLI. Returns last exit code."""
    write_artifacts(d)
    steps = [
        ("start", "1"),
        ("complete", "1"),
        ("start", "2"),
        ("complete", "2"),
        ("start", "3"),
        ("complete", "3"),
    ]
    rc = 0
    for cmd, stage in steps:
        rc, _ = sc(cmd, stage, "--dir", str(d))
        assert rc == 0, f"{cmd} {stage} failed"
    if with_stage4:
        (d / "perf_opt").mkdir(exist_ok=True)
        (d / "perf_opt" / "t.py").write_text(KERNEL_PY, encoding="utf-8")
        (d / "perf_opt" / "opt_log.md").write_text(OPT_LOG_MD, encoding="utf-8")
        rc, _ = sc("set", "--dir", str(d), "--perf-tuning", "yes")
        assert rc == 0
        rc, _ = sc("start", "4", "--dir", str(d), "--extend")
        assert rc == 0
        rc, _ = sc("complete", "4", "--dir", str(d))
        assert rc == 0
    return rc


def load_state(d: Path) -> dict:
    return json.loads((d / ".stage_state.json").read_text(encoding="utf-8"))


def reach_stage(d: Path, stage: int, op: str = "t", scenario: str = "new_op"):
    """合法推进到 `stage` 并使其处于 in_progress。"""
    write_artifacts(d, kernel=(stage >= 3))
    if scenario == "new_op":
        assert init_new_op(d)[0] == 0
    else:
        rc, _ = sc(
            "init",
            "--dir",
            str(d),
            "--project",
            "proj",
            "--op",
            op,
            "--scenario",
            scenario,
        )
        assert rc == 0
    assert sc("start", "1", "--dir", str(d))[0] == 0
    assert sc("complete", "1", "--dir", str(d))[0] == 0
    if stage >= 2:
        assert sc("start", "2", "--dir", str(d))[0] == 0
        assert sc("complete", "2", "--dir", str(d))[0] == 0
    if stage >= 3:
        assert sc("start", "3", "--dir", str(d))[0] == 0
    if stage >= 4:
        (d / "perf_opt").mkdir(exist_ok=True)
        (d / "perf_opt" / f"{op}.py").write_text(KERNEL_PY, encoding="utf-8")
        (d / "perf_opt" / "opt_log.md").write_text(OPT_LOG_MD, encoding="utf-8")
        assert sc("start", "4", "--dir", str(d), "--extend")[0] == 0


# ---------------------------------------------------------------------------
# init / scenario routing
# ---------------------------------------------------------------------------


def test_init_new_op_plan(tmp_path):
    repo = make_repo(tmp_path)
    d = op_dir(repo)
    rc, out = init_new_op(d)
    assert rc == 0 and out["stage_plan"] == [1, 2, 3]
    assert out["phase"] == "DESIGN"
    state = load_state(d)
    assert state["scenario"] == "new_op"
    assert state["stage_retry_count"] == {str(k): 0 for k in range(6)}


def test_init_migration_modes(tmp_path):
    repo = make_repo(tmp_path)
    d_plain = op_dir(repo, op="tp")
    rc, out = sc(
        "init",
        "--dir",
        str(d_plain),
        "--project",
        "p",
        "--op",
        "tp",
        "--scenario",
        "migration",
        "--migration-mode",
        "plain",
    )
    assert rc == 0 and out["stage_plan"] == [1, 2, 3]

    d_h = repo / "examples" / "slug" / "funcA"
    rc, out = sc(
        "init",
        "--dir",
        str(d_h),
        "--project",
        "slug",
        "--op",
        "funcA",
        "--scenario",
        "migration",
        "--migration-mode",
        "harness",
    )
    assert rc == 0 and out["stage_plan"] == [1, 2, 3]
    assert "harness" in (out["note"] or "")


def test_init_optimize(tmp_path):
    repo = make_repo(tmp_path)
    d = op_dir(repo, op="opt_t")
    rc, out = sc(
        "init",
        "--dir",
        str(d),
        "--project",
        "proj",
        "--op",
        "opt_t",
        "--scenario",
        "optimize",
    )
    assert rc == 0 and out["stage_plan"] == [4] and out["phase"] == "TUNING"


def test_init_invalid_combos(tmp_path):
    repo = make_repo(tmp_path)
    d = op_dir(repo)
    rc, out = sc(
        "init",
        "--dir",
        str(d),
        "--project",
        "proj",
        "--op",
        "t",
        "--scenario",
        "new_op",
        "--migration-mode",
        "harness",
    )
    assert rc == 1 and out["errors"][0]["code"] == "E-SCENARIO"
    rc, out = sc(
        "init",
        "--dir",
        str(d),
        "--project",
        "p",
        "--op",
        "t",
        "--scenario",
        "migration",
    )  # missing mode
    assert rc == 1


def test_init_twice_rejected(tmp_path):
    repo = make_repo(tmp_path)
    d = op_dir(repo)
    assert init_new_op(d)[0] == 0
    rc, out = init_new_op(d)
    assert rc == 1 and out["errors"][0]["code"] == "E-EXISTS"


# ---------------------------------------------------------------------------
# 5 类典型错误拦截：跳阶段 / 重复 complete / 并发 in_progress
# ---------------------------------------------------------------------------


def test_complete_without_start_rejected(tmp_path):
    repo = make_repo(tmp_path)
    d = op_dir(repo)
    assert init_new_op(d)[0] == 0
    rc, out = sc("complete", "1", "--dir", str(d))
    assert rc == 1 and out["errors"][0]["code"] == "E-NOT-STARTED"


def test_skip_stage_rejected(tmp_path):
    repo = make_repo(tmp_path)
    d = op_dir(repo)
    assert init_new_op(d)[0] == 0
    rc, out = sc("start", "3", "--dir", str(d))
    assert rc == 1 and out["errors"][0]["code"] == "E-OUT-OF-ORDER"
    write_artifacts(d)
    assert sc("start", "1", "--dir", str(d))[0] == 0
    rc, out = sc("start", "3", "--dir", str(d))
    # Stage 1 仍 in_progress：并发或乱序拦截均为合法（不能推进到 3）
    assert rc == 1 and out["errors"][0]["code"] in ("E-CONCURRENT", "E-OUT-OF-ORDER")
    assert sc("complete", "1", "--dir", str(d))[0] == 0
    rc, out = sc("start", "3", "--dir", str(d))
    assert rc == 1 and out["errors"][0]["code"] == "E-OUT-OF-ORDER"
    rc, out = sc("complete", "3", "--dir", str(d))
    assert rc == 1  # 未 start，且非 in_progress


def test_duplicate_complete_rejected(tmp_path):
    repo = make_repo(tmp_path)
    d = op_dir(repo)
    write_artifacts(d)
    assert init_new_op(d)[0] == 0
    assert sc("start", "1", "--dir", str(d))[0] == 0
    assert sc("complete", "1", "--dir", str(d))[0] == 0
    rc, out = sc("complete", "1", "--dir", str(d))
    assert rc == 1 and out["errors"][0]["code"] == "E-DUPLICATE-COMPLETE"
    rc, out = sc("start", "1", "--dir", str(d))
    assert rc == 1 and out["errors"][0]["code"] == "E-COMPLETED"


def test_concurrent_in_progress_rejected(tmp_path):
    repo = make_repo(tmp_path)
    d = op_dir(repo)
    write_artifacts(d)
    assert init_new_op(d)[0] == 0
    assert sc("start", "1", "--dir", str(d))[0] == 0
    rc, out = sc("start", "2", "--dir", str(d))
    assert rc == 1 and out["errors"][0]["code"] == "E-CONCURRENT"


def test_terminal_start_rejected(tmp_path):
    repo = make_repo(tmp_path)
    d = op_dir(repo)
    write_artifacts(d)
    assert init_new_op(d)[0] == 0
    assert sc("start", "1", "--dir", str(d))[0] == 0
    rc, _ = sc("fail", "1", "--dir", str(d), "--blocked-code", "BLOCKED_ENVIRONMENT")
    assert rc == 0
    rc, out = sc("start", "1", "--dir", str(d))
    assert rc == 1 and out["errors"][0]["code"] == "E-TERMINAL"
    state = load_state(d)
    assert state["phase"] == "FAILED"
    assert state["failure_reason"] == "BLOCKED_ENVIRONMENT"


def test_duplicate_start_rejected(tmp_path):
    repo = make_repo(tmp_path)
    d = op_dir(repo)
    assert init_new_op(d)[0] == 0
    assert sc("start", "1", "--dir", str(d))[0] == 0
    rc, out = sc("start", "1", "--dir", str(d))
    assert rc == 1 and out["errors"][0]["code"] == "E-ALREADY-STARTED"


# ---------------------------------------------------------------------------
# 漏计数 / 重试上限 / BLOCKED_* 路由
# ---------------------------------------------------------------------------


def test_stage1_retry_limit_auto_blocked(tmp_path):
    repo = make_repo(tmp_path)
    d = op_dir(repo)
    assert init_new_op(d)[0] == 0
    assert sc("start", "1", "--dir", str(d))[0] == 0
    for _ in range(3):
        rc, out = sc("fail", "1", "--dir", str(d))
        assert rc == 0
        assert out["stage_retry_count"] <= 3
    state = load_state(d)
    assert state["phase"] == "FAILED"
    assert state["failure_reason"] == "BLOCKED_DESIGN"
    assert state["stage_retry_count"]["1"] == 3
    rc, out = sc("start", "1", "--dir", str(d))
    assert rc == 1 and out["errors"][0]["code"] == "E-TERMINAL"


def test_stage3_fail_type_routing(tmp_path):
    repo = make_repo(tmp_path)
    d = op_dir(repo, op="rt")
    reach_stage(d, 3, op="rt")

    def drain(fail_type, expect_reason):
        for i in range(5):
            rc, out = sc("fail", "3", "--dir", str(d), "--fail-type", fail_type)
            if out.get("phase") == "FAILED":
                assert out["failure_reason"] == expect_reason, f"iter {i}"
                return
        raise AssertionError("limit never reached")

    drain("precision", "BLOCKED_ACCURACY")

    d2 = op_dir(repo, op="rf")
    reach_stage(d2, 3, op="rf")
    for _ in range(4):
        sc("fail", "3", "--dir", str(d2), "--fail-type", "runtime")
    rc, out = sc("fail", "3", "--dir", str(d2), "--fail-type", "runtime")
    assert out["failure_reason"] == "BLOCKED_IMPL"
    state = load_state(d2)
    assert state["stage3_failure_breakdown"] == {"runtime_fail": 5, "precision_fail": 0}


def test_stage4_limit_ends_done_not_failed(tmp_path):
    repo = make_repo(tmp_path)
    d = op_dir(repo, op="t4")
    sc(
        "init",
        "--dir",
        str(d),
        "--project",
        "proj",
        "--op",
        "t4",
        "--scenario",
        "optimize",
    )
    assert sc("start", "4", "--dir", str(d))[0] == 0
    for _ in range(9):
        rc, out = sc("fail", "4", "--dir", str(d))
        assert out.get("phase") != "FAILED"
    rc, out = sc("fail", "4", "--dir", str(d))
    assert out["phase"] == "DONE" and out["abort"] == "stage4_iteration_limit"


def test_stage2_plain_fail_requires_revision(tmp_path):
    repo = make_repo(tmp_path)
    d = op_dir(repo)
    reach_stage(d, 2)
    rc, out = sc("fail", "2", "--dir", str(d))
    assert rc == 1 and out["errors"][0]["code"] == "E-STAGE2-REVISION"


# ---------------------------------------------------------------------------
# 设计修订循环
# ---------------------------------------------------------------------------


def _reach_stage3(d: Path):
    write_artifacts(d)
    assert init_new_op(d)[0] == 0
    assert sc("start", "1", "--dir", str(d))[0] == 0
    assert sc("complete", "1", "--dir", str(d))[0] == 0
    assert sc("start", "2", "--dir", str(d))[0] == 0


def test_design_revision_flow(tmp_path):
    repo = make_repo(tmp_path)
    d = op_dir(repo)
    _reach_stage3(d)
    # 检视不通过 → 修订路径 A
    (d / "REVIEW.md").write_text(
        REVIEW_MD.replace("结论: 通过", "结论: 不通过"), encoding="utf-8"
    )
    rc, out = sc("fail", "2", "--dir", str(d), "--reason", "design_revision")
    assert rc == 0 and out["retry_count"] == 1
    state = load_state(d)
    # 下游 Stage 状态清空、重试计数清零
    assert "3" not in state["stage_status"]
    assert state["stage_retry_count"]["3"] == 0
    # 已 completed 的 Stage 1 允许经修订通行证重入
    rc, out = sc("start", "1", "--dir", str(d))
    assert rc == 0
    # 通行证一次性：再次 start 1 被拒
    rc, out = sc("start", "1", "--dir", str(d))
    assert rc == 1 and out["errors"][0]["code"] == "E-ALREADY-STARTED"
    assert sc("complete", "1", "--dir", str(d))[0] == 0
    assert sc("start", "2", "--dir", str(d))[0] == 0


def test_design_revision_over_limit(tmp_path):
    repo = make_repo(tmp_path)
    d = op_dir(repo)
    write_artifacts(d)
    assert init_new_op(d)[0] == 0
    assert sc("start", "1", "--dir", str(d))[0] == 0
    assert sc("complete", "1", "--dir", str(d))[0] == 0
    assert sc("start", "2", "--dir", str(d))[0] == 0
    for i in range(3):
        rc, out = sc("fail", "2", "--dir", str(d), "--reason", "design_revision")
        assert rc == 0 and out["retry_count"] == i + 1
        if i < 2:  # 预算未超限：修订后重跑 1→2
            assert sc("start", "1", "--dir", str(d))[0] == 0
            assert sc("complete", "1", "--dir", str(d))[0] == 0
            assert sc("start", "2", "--dir", str(d))[0] == 0
    state = load_state(d)
    assert state["phase"] == "FAILED"
    assert state["failure_reason"] == "BLOCKED_DESIGN"
    assert state["retry_count"] == 3


# ---------------------------------------------------------------------------
# 门禁 lint（工件缺失 / 章节缺失 / 占位符 / 死链 / 结论字面量 / kernel AST）
# ---------------------------------------------------------------------------


def test_gate1_missing_design(tmp_path):
    repo = make_repo(tmp_path)
    d = op_dir(repo)
    assert init_new_op(d)[0] == 0
    rc, out = sc("gate", "1", "--dir", str(d))
    assert rc == 1
    assert out["failures"][0]["rule_id"] == "S1-EXISTS"


def test_gate1_missing_sections_and_4q(tmp_path):
    repo = make_repo(tmp_path)
    d = op_dir(repo)
    (d / "DESIGN.md").write_text("# t\n\n## 1. 概述\n\n只有概述。\n", encoding="utf-8")
    assert init_new_op(d)[0] == 0
    rc, out = sc("gate", "1", "--dir", str(d))
    assert rc == 1
    rules = {f["rule_id"] for f in out["failures"]}
    assert "S1-SECT-MISSING" in rules


def test_gate1_research_4q_incomplete(tmp_path):
    repo = make_repo(tmp_path)
    d = op_dir(repo)
    bad = DESIGN_MD.replace("R2 在线算法：逐元素映射天然单遍，无在线变体。", "")
    bad = bad.replace("R4 硬件亲和：Vector 单元亲和。", "")
    (d / "DESIGN.md").write_text(bad, encoding="utf-8")
    assert init_new_op(d)[0] == 0
    rc, out = sc("gate", "1", "--dir", str(d))
    rules = {f["rule_id"] for f in out["failures"]}
    assert "S1-RESEARCH-4Q" in rules
    msg = next(f for f in out["failures"] if f["rule_id"] == "S1-RESEARCH-4Q")
    assert "R2" in msg["message"] and "R4" in msg["message"]


def test_gate1_placeholder(tmp_path):
    repo = make_repo(tmp_path)
    d = op_dir(repo)
    bad = DESIGN_MD.replace(
        "### 4.3 中间缓冲区\n\n无。", "### 4.3 中间缓冲区\n\n待补充"
    )
    (d / "DESIGN.md").write_text(bad, encoding="utf-8")
    assert init_new_op(d)[0] == 0
    rc, out = sc("gate", "1", "--dir", str(d))
    rules = {f["rule_id"] for f in out["failures"]}
    assert "S1-PLACEHOLDER" in rules


def test_gate1_dead_reference_path(tmp_path):
    repo = make_repo(tmp_path)
    d = op_dir(repo)
    bad = DESIGN_MD.replace(
        "### 3.5.3 本项目同类实现参考\n\n无。",
        "### 3.5.3 本项目同类实现参考\n\n参考 examples/does_not_exist/foo.py。",
    )
    (d / "DESIGN.md").write_text(bad, encoding="utf-8")
    assert init_new_op(d)[0] == 0
    rc, out = sc("gate", "1", "--dir", str(d))
    rules = {f["rule_id"] for f in out["failures"]}
    assert "S1-REF-PATHS" in rules


def test_gate1_cores_elements_missing(tmp_path):
    repo = make_repo(tmp_path)
    d = op_dir(repo)
    bad = DESIGN_MD.replace("逻辑核数 = ceil(M / block_M)。", "按 block 划分。")
    bad = bad.replace(
        "物理核数 = NPUUtils.get().get_aicore_num() 实查值，查询代码已记录。", ""
    )
    bad = bad.replace("规模判定：逻辑核数 ≤ 物理核数，无需适配。", "")
    (d / "DESIGN.md").write_text(bad, encoding="utf-8")
    assert init_new_op(d)[0] == 0
    rc, out = sc("gate", "1", "--dir", str(d))
    rules = {f["rule_id"] for f in out["failures"]}
    assert "S1-CORES-LOGICAL" in rules and "S1-CORES-PHYSICAL" in rules


def test_gate1_valid_design_passes(tmp_path):
    repo = make_repo(tmp_path)
    d = op_dir(repo)
    (d / "DESIGN.md").write_text(DESIGN_MD, encoding="utf-8")
    assert init_new_op(d)[0] == 0
    rc, out = sc("gate", "1", "--dir", str(d))
    assert rc == 0, out["failures"]
    assert out["pass"] is True


def test_gate2_fuzzy_conclusion_rejected(tmp_path):
    repo = make_repo(tmp_path)
    d = op_dir(repo)
    (d / "REVIEW.md").write_text(
        REVIEW_MD.replace("结论: 通过", "结论: 基本通过"), encoding="utf-8"
    )
    assert init_new_op(d)[0] == 0
    rc, out = sc("gate", "2", "--dir", str(d))
    rules = {f["rule_id"] for f in out["failures"]}
    assert "S2-CONCLUSION" in rules


def test_gate2_missing_dimension(tmp_path):
    repo = make_repo(tmp_path)
    d = op_dir(repo)
    bad = REVIEW_MD.replace("### 5. 循环与同步 — pass\n\nok\n", "")
    (d / "REVIEW.md").write_text(bad, encoding="utf-8")
    assert init_new_op(d)[0] == 0
    rc, out = sc("gate", "2", "--dir", str(d))
    rules = {f["rule_id"] for f in out["failures"]}
    assert "S2-DIMS" in rules


def test_gate2_pass_with_issue_list_rejected(tmp_path):
    repo = make_repo(tmp_path)
    d = op_dir(repo)
    bad = REVIEW_MD + "\n## 检视问题列表\n\n### 问题 1: x\n"
    (d / "REVIEW.md").write_text(bad, encoding="utf-8")
    assert init_new_op(d)[0] == 0
    rc, out = sc("gate", "2", "--dir", str(d))
    rules = {f["rule_id"] for f in out["failures"]}
    assert "S2-PASS-NO-ISSUES" in rules


def test_gate2_valid_review_passes(tmp_path):
    repo = make_repo(tmp_path)
    d = op_dir(repo)
    (d / "REVIEW.md").write_text(REVIEW_MD, encoding="utf-8")
    assert init_new_op(d)[0] == 0
    rc, out = sc("gate", "2", "--dir", str(d))
    assert rc == 0, out["failures"]


def test_gate3_valid_kernel_passes(tmp_path):
    repo = make_repo(tmp_path)
    d = op_dir(repo)
    (d / "t.py").write_text(KERNEL_PY, encoding="utf-8")
    assert init_new_op(d)[0] == 0
    rc, out = sc("gate", "3", "--dir", str(d))
    assert rc == 0, out["failures"]


def test_gate3_kernel_ast_failures(tmp_path):
    repo = make_repo(tmp_path)
    d = op_dir(repo)
    assert init_new_op(d)[0] == 0
    # wrong target
    bad = KERNEL_PY.replace('target="npuir"', 'target="cuda"')
    (d / "t.py").write_text(bad, encoding="utf-8")
    rc, out = sc("gate", "3", "--dir", str(d))
    assert rc == 1 and "S3-JIT" in {f["rule_id"] for f in out["failures"]}
    # missing golden + missing run_L1 + no --level
    bad = KERNEL_PY.replace("def golden_t(x):\n    return torch.add(x, 1.0)\n", "")
    bad = bad.replace(
        "def run_L1():\n    x = torch.randn(8, 16)\n    assert golden_t(x).shape == (8, 16)\n",
        "",
    )
    bad = bad.replace(
        '"--level", default="L0", choices=["L0", "all"]', '"--run", default="L0"'
    )
    (d / "t.py").write_text(bad, encoding="utf-8")
    rc, out = sc("gate", "3", "--dir", str(d))
    rules = {f["rule_id"] for f in out["failures"]}
    assert "S3-GOLDEN" in rules and "S3-TEST-LEVELS" in rules
    assert "S3-MAIN-LEVEL" in rules
    # golden not calling torch (hand-written loop)
    bad = KERNEL_PY.replace(
        "def golden_t(x):\n    return torch.add(x, 1.0)",
        "def golden_t(x):\n    out = x.clone()\n    for i in range(x.numel()):\n"
        "        out.view(-1)[i] = out.view(-1)[i] + 1.0\n    return out",
    )
    (d / "t.py").write_text(bad, encoding="utf-8")
    rc, out = sc("gate", "3", "--dir", str(d))
    rules = {f["rule_id"] for f in out["failures"]}
    assert "S3-GOLDEN-TORCH" in rules


def test_gate4_requires_perf_and_optlog(tmp_path):
    repo = make_repo(tmp_path)
    d = op_dir(repo)
    assert init_new_op(d)[0] == 0
    rc, out = sc("gate", "4", "--dir", str(d))
    rules = {f["rule_id"] for f in out["failures"]}
    assert "S4-EXISTS" in rules and "S4-OPTLOG" in rules
    (d / "perf_opt").mkdir()
    (d / "perf_opt" / "t.py").write_text(KERNEL_PY, encoding="utf-8")
    (d / "perf_opt" / "opt_log.md").write_text(OPT_LOG_MD, encoding="utf-8")
    rc, out = sc("gate", "4", "--dir", str(d))
    assert rc == 0, out["failures"]


def test_complete_blocked_by_gate_writes_nothing(tmp_path):
    repo = make_repo(tmp_path)
    d = op_dir(repo)  # 无 DESIGN.md
    assert init_new_op(d)[0] == 0
    assert sc("start", "1", "--dir", str(d))[0] == 0
    before = (d / ".stage_state.json").read_text(encoding="utf-8")
    rc, out = sc("complete", "1", "--dir", str(d))
    assert rc == 1
    assert out["failures"] and out["hint"]
    after = (d / ".stage_state.json").read_text(encoding="utf-8")
    assert before == after  # 门禁失败不写状态


# ---------------------------------------------------------------------------
# 快照 / verify（篡改检测）/ repair（schema 损坏）
# ---------------------------------------------------------------------------


def test_tampered_artifact_detected(tmp_path):
    repo = make_repo(tmp_path)
    d = op_dir(repo)
    write_artifacts(d)
    assert init_new_op(d)[0] == 0
    assert sc("start", "1", "--dir", str(d))[0] == 0
    assert sc("complete", "1", "--dir", str(d))[0] == 0
    rc, out = sc("verify", "--dir", str(d))
    assert rc == 0
    # 篡改 DESIGN.md → verify 报漂移
    (d / "DESIGN.md").write_text(DESIGN_MD + "\n被篡改\n", encoding="utf-8")
    rc, out = sc("verify", "--dir", str(d))
    assert rc == 1
    assert any("漂移" in item["issue"] for item in out["drift"])
    # 删除工件 → verify 报缺失
    (d / "DESIGN.md").unlink()
    rc, out = sc("verify", "--dir", str(d))
    assert rc == 1
    assert any("缺失" in item["issue"] for item in out["drift"])


def test_verify_detects_completed_stage_without_artifact(tmp_path):
    repo = make_repo(tmp_path)
    d = op_dir(repo)
    write_artifacts(d)
    assert init_new_op(d)[0] == 0
    assert sc("start", "1", "--dir", str(d))[0] == 0
    assert sc("complete", "1", "--dir", str(d))[0] == 0
    (d / "DESIGN.md").unlink()
    (d / ".stage_state.json").unlink()
    # 手工构造：completed 但工件不存在（绕过快照）
    state = {
        "task_id": "x",
        "project_name": "proj",
        "operator_name": "t",
        "scenario": "new_op",
        "migration_mode": None,
        "stage_plan": [1, 2, 3],
        "phase": "REVIEW",
        "retry_count": 0,
        "max_retry": 3,
        "stage_status": {"1": "completed"},
        "stage_retry_count": {},
    }
    (d / ".stage_state.json").write_text(
        json.dumps(state, ensure_ascii=False), encoding="utf-8"
    )
    rc, out = sc("verify", "--dir", str(d))
    assert rc == 1
    assert out["schema_failures"]  # 缺字段被报出（backfill 不掩盖）
    assert out["consistency"]


def test_repair_corrupted_json(tmp_path):
    repo = make_repo(tmp_path)
    d = op_dir(repo)
    (d / ".stage_state.json").write_text("{ not json !!!", encoding="utf-8")
    rc, out = sc("repair", "--dir", str(d))
    assert rc == 1 and out["errors"][0]["code"] == "E-PARSE"
    assert "timeline" in out["suggestion"]
    # 其它命令同样拒绝读写损坏文件
    rc, out = sc("start", "1", "--dir", str(d))
    assert rc == 1 and out["errors"][0]["code"] == "E-PARSE"


def test_repair_backfills_missing_fields(tmp_path):
    repo = make_repo(tmp_path)
    d = op_dir(repo)
    state = {
        "task_id": "x",
        "project_name": "proj",
        "operator_name": "t",
        "scenario": "new_op",
        "stage_plan": [1, 2, 3],
        "phase": "DESIGN",
        "stage_status": {},
        "retry_count": 0,
    }
    (d / ".stage_state.json").write_text(
        json.dumps(state, ensure_ascii=False), encoding="utf-8"
    )
    rc, out = sc("repair", "--dir", str(d))
    assert rc == 0 and "max_retry" in out["would_fix"]
    rc, out = sc("repair", "--dir", str(d), "--apply")
    assert rc == 0 and "max_retry" in out["applied"]
    state = load_state(d)
    assert state["max_retry"] == 3
    assert state["stage_retry_count"] == {str(k): 0 for k in range(6)}


# ---------------------------------------------------------------------------
# bit-identical（同一任务连跑两次）
# ---------------------------------------------------------------------------


def test_full_lifecycle_bit_identical(tmp_path):
    outputs = []
    for i in range(2):
        repo = make_repo(tmp_path, name=f"repo{i}")
        d = op_dir(repo)
        assert init_new_op(d)[0] == 0
        assert happy_path(d, with_stage4=True) == 0
        outputs.append(
            (
                (d / ".stage_state.json").read_bytes(),
                (d / ".task_timeline.jsonl").read_bytes(),
            )
        )
    assert outputs[0][0] == outputs[1][0], "state files differ between runs"
    assert outputs[0][1] == outputs[1][1], "timeline differs between runs"
    state = json.loads(outputs[0][0])
    assert state["phase"] == "DONE"
    assert state["final_artifact"].endswith("perf_opt/t.py")
    assert state["stage_status"] == {
        "1": "completed",
        "2": "completed",
        "3": "completed",
        "4": "completed",
    }


# ---------------------------------------------------------------------------
# 可选 Stage 4 / --extend
# ---------------------------------------------------------------------------


def test_extend_stage4_after_done(tmp_path):
    repo = make_repo(tmp_path)
    d = op_dir(repo)
    write_artifacts(d)
    assert init_new_op(d)[0] == 0
    assert happy_path(d) == 0  # plan [1,2,3] → complete 3 后 DONE
    assert load_state(d)["phase"] == "DONE"
    rc, out = sc("start", "4", "--dir", str(d))
    assert rc == 1  # 不在 plan，拒绝
    rc, out = sc("start", "4", "--dir", str(d), "--extend")
    assert rc == 0
    (d / "perf_opt").mkdir()
    (d / "perf_opt" / "t.py").write_text(KERNEL_PY, encoding="utf-8")
    (d / "perf_opt" / "opt_log.md").write_text(OPT_LOG_MD, encoding="utf-8")
    rc, out = sc("complete", "4", "--dir", str(d))
    assert rc == 0
    state = load_state(d)
    assert state["phase"] == "DONE"
    assert state["final_artifact"].endswith("perf_opt/t.py")


# ---------------------------------------------------------------------------
# migration 聚合状态 + gate 0/5
# ---------------------------------------------------------------------------


META = {
    "op_slug": "mop",
    "family": "reduction",
    "extracted_functions": ["funcA", "funcB"],
    "test_slug": "mop",
    "bench_slug": "mop",
}


def _make_harness_repo(tmp_path):
    repo = make_repo(tmp_path)
    slug_dir = repo / "examples" / "mop"
    slug_dir.mkdir(parents=True)
    meta_path = (
        repo
        / "examples"
        / "TileOPs"
        / "tileops"
        / "kernels"
        / "reduction"
        / "mop"
        / ".migration_meta.json"
    )
    meta_path.parent.mkdir(parents=True)
    meta_path.write_text(json.dumps(META), encoding="utf-8")
    return repo, slug_dir, meta_path


def test_migration_aggregate_flow(tmp_path):
    repo, slug_dir, meta_path = _make_harness_repo(tmp_path)
    rc, out = sc(
        "migration",
        "init",
        "--dir",
        str(slug_dir),
        "--op-slug",
        "mop",
        "--family",
        "reduction",
        "--meta-path",
        str(meta_path),
        "--functions",
        "funcA",
        "funcB",
    )
    assert rc == 0 and out["phase"] == "DEV_LOOP"
    # INTEGRATE 前置：全部函数 done
    rc, out = sc("migration", "phase", "--dir", str(slug_dir), "--phase", "INTEGRATE")
    assert rc == 1 and out["errors"][0]["code"] == "E-PHASE-PRECOND"
    for f in ("funcA", "funcB"):
        rc, _ = sc("migration", "func", f, "--dir", str(slug_dir), "--status", "done")
        assert rc == 0
    rc, _ = sc("migration", "phase", "--dir", str(slug_dir), "--phase", "INTEGRATE")
    assert rc == 0
    # DONE 前置：integration pass
    rc, out = sc("migration", "phase", "--dir", str(slug_dir), "--phase", "DONE")
    assert rc == 1 and out["errors"][0]["code"] == "E-PHASE-PRECOND"
    rc, _ = sc("migration", "integrate", "--dir", str(slug_dir), "--status", "pass")
    assert rc == 0
    rc, _ = sc("migration", "phase", "--dir", str(slug_dir), "--phase", "DONE")
    assert rc == 0
    # 非法回退
    rc, out = sc("migration", "phase", "--dir", str(slug_dir), "--phase", "DEV_LOOP")
    assert rc == 1 and out["errors"][0]["code"] == "E-TERMINAL"


def test_migration_integrate_fail_limit(tmp_path):
    repo, slug_dir, meta_path = _make_harness_repo(tmp_path)
    sc(
        "migration",
        "init",
        "--dir",
        str(slug_dir),
        "--op-slug",
        "mop",
        "--family",
        "reduction",
        "--meta-path",
        str(meta_path),
        "--functions",
        "funcA",
    )
    sc("migration", "func", "funcA", "--dir", str(slug_dir), "--status", "done")
    sc("migration", "phase", "--dir", str(slug_dir), "--phase", "INTEGRATE")
    for _ in range(2):
        rc, out = sc(
            "migration", "integrate", "--dir", str(slug_dir), "--status", "fail"
        )
        assert out.get("phase") != "FAILED"
    rc, out = sc("migration", "integrate", "--dir", str(slug_dir), "--status", "fail")
    assert out["phase"] == "FAILED"
    assert out["failure_reason"] == "BLOCKED_INTEGRATION"


def test_gate0_meta_lint(tmp_path):
    repo, slug_dir, meta_path = _make_harness_repo(tmp_path)
    # 脚手架文件缺失 → S0-SCAFFOLD
    rc, out = sc("gate", "0", "--meta", str(meta_path), "--dir", str(slug_dir))
    assert rc == 1
    assert "S0-SCAFFOLD" in {f["rule_id"] for f in out["failures"]}
    # 补齐脚手架
    base = repo / "examples" / "TileOPs"
    for rel in (
        "tileops/manifest/reduction.yaml",
        "tileops/workloads/reduction.py",
        "tileops/kernels/reduction/mop/mop.py",
        "tests/ops/test_mop.py",
        "benchmarks/ops/bench_mop.py",
    ):
        p = base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# scaffold\n", encoding="utf-8")
    rc, out = sc("gate", "0", "--meta", str(meta_path), "--dir", str(slug_dir))
    assert rc == 0, out["failures"]


def test_gate5_integration_lint(tmp_path):
    repo, slug_dir, meta_path = _make_harness_repo(tmp_path)
    sc(
        "migration",
        "init",
        "--dir",
        str(slug_dir),
        "--op-slug",
        "mop",
        "--family",
        "reduction",
        "--meta-path",
        str(meta_path),
        "--functions",
        "funcA",
        "funcB",
    )
    rc, out = sc("gate", "5", "--dir", str(slug_dir), "--migration-dir", str(slug_dir))
    assert rc == 1
    assert "S5-PKG" in {f["rule_id"] for f in out["failures"]}
    pkg = (
        repo
        / "examples"
        / "TileOPs"
        / "tileops"
        / "kernels"
        / "reduction"
        / "mop"
        / "mop_kernel"
    )
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "integration_log.md").write_text("ok\n", encoding="utf-8")
    for f in ("funcA", "funcB"):
        (pkg / f"{f}.py").write_text(KERNEL_PY, encoding="utf-8")
        (pkg / f"{f}_DESIGN.md").write_text(DESIGN_MD, encoding="utf-8")
    wrapper = (
        repo
        / "examples"
        / "TileOPs"
        / "tileops"
        / "kernels"
        / "reduction"
        / "mop"
        / "mop.py"
    )
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text(
        "from .mop_kernel import funcA  # baseline\n"
        "# from .mop_kernel.perf_opt import funcA  # perf_opt\n",
        encoding="utf-8",
    )
    rc, out = sc("gate", "5", "--dir", str(slug_dir), "--migration-dir", str(slug_dir))
    assert rc == 0, out["failures"]


# ---------------------------------------------------------------------------
# show / set / timeline
# ---------------------------------------------------------------------------


def test_show_and_set(tmp_path):
    repo = make_repo(tmp_path)
    d = op_dir(repo)
    assert init_new_op(d)[0] == 0
    rc, out = sc("show", "--dir", str(d))
    assert rc == 0 and out["state"]["phase"] == "DESIGN"
    rc, _ = sc("set", "--dir", str(d), "--perf-tuning", "yes", "--env-check", "true")
    assert rc == 0
    state = load_state(d)
    assert state["perf_tuning_requested"] == "yes"
    assert state["env_check_passed"] is True
    timeline = (d / ".task_timeline.jsonl").read_text(encoding="utf-8").splitlines()
    actions = [json.loads(line)["action"] for line in timeline]
    assert actions == ["init", "set"]


def test_optimize_scenario_gate4_via_complete(tmp_path):
    repo = make_repo(tmp_path)
    d = op_dir(repo, op="t")
    (d / "t.py").write_text(KERNEL_PY, encoding="utf-8")
    (d / "perf_opt").mkdir()
    (d / "perf_opt" / "t.py").write_text(KERNEL_PY, encoding="utf-8")
    (d / "perf_opt" / "opt_log.md").write_text(OPT_LOG_MD, encoding="utf-8")
    rc, _ = sc(
        "init",
        "--dir",
        str(d),
        "--project",
        "proj",
        "--op",
        "t",
        "--scenario",
        "optimize",
    )
    assert rc == 0
    rc, out = sc("start", "4", "--dir", str(d))
    assert rc == 0
    rc, out = sc("complete", "4", "--dir", str(d))
    assert rc == 0, out
    state = load_state(d)
    assert state["phase"] == "DONE"
    assert state["final_artifact"].endswith("perf_opt/t.py")


# ---------------------------------------------------------------------------
# A2：perf_iteration 白名单 flag + budget 字段与 start 预算水位（E1.3）
# ---------------------------------------------------------------------------


def test_set_perf_iteration_flags(tmp_path):
    repo = make_repo(tmp_path)
    d = op_dir(repo)
    assert init_new_op(d)[0] == 0
    rc, _ = sc(
        "set",
        "--dir",
        str(d),
        "--perf-iteration-count",
        "3",
        "--perf-iteration-last-improvement",
        "5.5",
        "--perf-iteration-no-improve",
        "2",
    )
    assert rc == 0
    state = load_state(d)
    assert state["perf_iteration"] == {
        "count": 3,
        "last_improvement": 5.5,
        "consecutive_no_improvement": 2,
    }
    # 负值被拒（E-VALUE），不写状态
    rc, out = sc("set", "--dir", str(d), "--perf-iteration-count", "-1")
    assert rc == 1 and out["errors"][0]["code"] == "E-VALUE"
    assert load_state(d)["perf_iteration"]["count"] == 3


def test_set_budget_json_validation_and_watermark(tmp_path):
    repo = make_repo(tmp_path)
    d = op_dir(repo)
    write_artifacts(d)
    assert init_new_op(d)[0] == 0
    # 非法 JSON / 未知字段 / 非正值都被拒
    assert sc("set", "--dir", str(d), "--budget-json", "not-json")[0] == 1
    assert sc("set", "--dir", str(d), "--budget-json", '{"unknown_dim": 1}')[0] == 1
    assert (
        sc("set", "--dir", str(d), "--budget-json", '{"max_wallclock_s": -5}')[0] == 1
    )
    # 合法写入
    rc, _ = sc(
        "set", "--dir", str(d), "--budget-json", '{"max_subagent_dispatches": 1}'
    )
    assert rc == 0
    assert load_state(d)["budget"]["max_subagent_dispatches"] == 1
    # 第 1 次 dispatch：水位正常（advisory，不拦截）
    rc, out = sc("start", "1", "--dir", str(d))
    assert rc == 0
    assert out["budget"]["dispatches_used"] == 0
    assert out["budget"]["exceeded"] == []
    rc, _ = sc("complete", "1", "--dir", str(d))
    assert rc == 0
    # 第 2 次 dispatch：超出预算 → budget.exceeded 标记（exit 仍 0，conductor 路由）
    rc, out = sc("start", "2", "--dir", str(d))
    assert rc == 0
    assert out["budget"]["dispatches_used"] == 1
    assert out["budget"]["exceeded"] == ["max_subagent_dispatches"]
    # 超限事件进入 timeline（可审计）
    events = [
        json.loads(line)
        for line in (d / ".task_timeline.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert any(
        ev.get("action") == "start" and ev.get("budget_exceeded") for ev in events
    )


def test_budget_absent_keeps_start_payload_unchanged(tmp_path):
    repo = make_repo(tmp_path)
    d = op_dir(repo)
    write_artifacts(d)
    assert init_new_op(d)[0] == 0
    rc, out = sc("start", "1", "--dir", str(d))
    assert rc == 0 and "budget" not in out


# ---------------------------------------------------------------------------
# A2/B2：perf_records.jsonl 记录校验 + winner 对账 + 对比表存在性（gate 4）
# ---------------------------------------------------------------------------


def stage4_artifacts(d: Path, opt_log: str = OPT_LOG_RECORDS_MD):
    (d / "perf_opt").mkdir(exist_ok=True)
    (d / "perf_opt" / "t.py").write_text(KERNEL_PY, encoding="utf-8")
    (d / "perf_opt" / "opt_log.md").write_text(opt_log, encoding="utf-8")


def write_perf_records(d: Path, text: str = PERF_RECORDS_JSONL):
    (d / "perf_opt").mkdir(exist_ok=True)
    (d / "perf_opt" / "perf_records.jsonl").write_text(text, encoding="utf-8")


def test_gate4_perf_records_valid_passes(tmp_path):
    repo = make_repo(tmp_path)
    d = op_dir(repo)
    assert init_new_op(d)[0] == 0
    stage4_artifacts(d)
    write_perf_records(d)
    rc, out = sc("gate", "4", "--dir", str(d))
    assert rc == 0, out["failures"]
    # final_latency 在 1% 容差内匹配记录（100.0 vs 100.5）
    write_perf_records(
        d, PERF_RECORDS_JSONL.replace('"duration_us": 100.0', '"duration_us": 100.9')
    )
    rc, out = sc("gate", "4", "--dir", str(d))
    assert rc == 0, out["failures"]


def test_gate4_perf_records_missing_warns_only(tmp_path):
    repo = make_repo(tmp_path)
    d = op_dir(repo)
    assert init_new_op(d)[0] == 0
    stage4_artifacts(d, OPT_LOG_MD)  # 旧式日志（无对比表/final_latency）
    rc, out = sc("gate", "4", "--dir", str(d))
    assert rc == 0, out["failures"]
    assert "S4-PERF-RECORDS" in {w["rule_id"] for w in out["warnings"]}


def test_gate4_perf_records_recon_mismatch(tmp_path):
    repo = make_repo(tmp_path)
    d = op_dir(repo)
    assert init_new_op(d)[0] == 0
    stage4_artifacts(d)
    write_perf_records(d)
    # Final Summary 声称 50.0us，与记录（100.0 / 120.5）偏差超 1% → 不可对账
    stage4_artifacts(
        d,
        OPT_LOG_RECORDS_MD.replace("final_latency: 100.0 us", "final_latency: 50.0 us"),
    )
    rc, out = sc("gate", "4", "--dir", str(d))
    assert rc == 1
    assert "S4-PERF-RECORDS-RECON" in {f["rule_id"] for f in out["failures"]}
    # records 存在但 Final Summary 无 final_latency 行 → 同样不可对账
    stage4_artifacts(d, OPT_LOG_RECORDS_MD.replace("final_latency: 100.0 us\n", ""))
    rc, out = sc("gate", "4", "--dir", str(d))
    assert rc == 1
    assert "S4-PERF-RECORDS-RECON" in {f["rule_id"] for f in out["failures"]}


def test_gate4_perf_records_schema_and_comptable(tmp_path):
    repo = make_repo(tmp_path)
    d = op_dir(repo)
    assert init_new_op(d)[0] == 0
    stage4_artifacts(d)
    # 非法 JSON 行 + 缺必需字段行
    write_perf_records(d, "not-json\n" + '{"round": 1, "candidate_id": "v2"}\n')
    rc, out = sc("gate", "4", "--dir", str(d))
    assert rc == 1
    rules = {f["rule_id"] for f in out["failures"]}
    assert "S4-PERF-RECORDS-SCHEMA" in rules
    # duration_us 非数字
    write_perf_records(
        d, PERF_RECORDS_JSONL.replace('"duration_us": 100.0', '"duration_us": "fast"')
    )
    rc, out = sc("gate", "4", "--dir", str(d))
    assert rc == 1
    assert "S4-PERF-RECORDS-SCHEMA" in {f["rule_id"] for f in out["failures"]}
    # records 存在但 opt_log 缺「候选 vs current best」对比表（B2）
    write_perf_records(d)
    stage4_artifacts(
        d,
        OPT_LOG_RECORDS_MD.replace(
            "| current best (baseline) | 120.5 | 41% | 78% | pass |",
            "| 基线 | 120.5 | 41% | 78% | pass |",
        ).replace(
            "| 候选 | Task Duration(us) | AICore 利用率 | Memory 指标 | L0 |", ""
        ),
    )
    rc, out = sc("gate", "4", "--dir", str(d))
    assert rc == 1
    assert "S4-OPTLOG-COMPTABLE" in {f["rule_id"] for f in out["failures"]}
    # 空文件（存在但无记录）
    write_perf_records(d, "\n")
    rc, out = sc("gate", "4", "--dir", str(d))
    assert rc == 1
    assert "S4-PERF-RECORDS" in {f["rule_id"] for f in out["failures"]}


# ---------------------------------------------------------------------------
# U14：TUNING→DESIGN 受控逆向反馈（[DESIGN_LIMIT] / perf_feedback.md）
# ---------------------------------------------------------------------------


def write_perf_feedback(d: Path, text: str = PERF_FEEDBACK_MD):
    (d / "perf_opt").mkdir(exist_ok=True)
    (d / "perf_opt" / "perf_feedback.md").write_text(text, encoding="utf-8")


def test_gate4_perf_feedback_valid_passes(tmp_path):
    repo = make_repo(tmp_path)
    d = op_dir(repo)
    assert init_new_op(d)[0] == 0
    (d / "perf_opt").mkdir()
    (d / "perf_opt" / "t.py").write_text(KERNEL_PY, encoding="utf-8")
    (d / "perf_opt" / "opt_log.md").write_text(OPT_LOG_MD, encoding="utf-8")
    write_perf_feedback(d)
    rc, out = sc("gate", "4", "--dir", str(d))
    assert rc == 0, out["failures"]
    # 无 perf_feedback.md 的旧流程不受影响（lint-if-exists）
    (d / "perf_opt" / "perf_feedback.md").unlink()
    rc, out = sc("gate", "4", "--dir", str(d))
    assert rc == 0, out["failures"]


def test_gate4_perf_feedback_schema_failures(tmp_path):
    repo = make_repo(tmp_path)
    d = op_dir(repo)
    assert init_new_op(d)[0] == 0
    (d / "perf_opt").mkdir()
    (d / "perf_opt" / "t.py").write_text(KERNEL_PY, encoding="utf-8")
    (d / "perf_opt" / "opt_log.md").write_text(OPT_LOG_MD, encoding="utf-8")

    def rules():
        rc, out = sc("gate", "4", "--dir", str(d))
        assert rc == 1
        return {f["rule_id"] for f in out["failures"]}

    # 缺「反馈结论」章节（重命名标题）
    write_perf_feedback(d, PERF_FEEDBACK_MD.replace("## 反馈结论", "## 结论"))
    assert "S4-PERF-FEEDBACK-SCHEMA" in rules()
    # 触发判定低于 2x 阈值且无「矛盾」依据
    write_perf_feedback(
        d,
        PERF_FEEDBACK_MD.replace(
            "结构性加速估计: 3.2x（依据: C 轴融合布局实测外推，round2 对照实验）。",
            "结构性加速估计: 1.3x（依据: 微弱外推）。",
        ),
    )
    assert "S4-PERF-FEEDBACK-TRIGGER" in rules()
    # 缺「设计层归因」字样
    write_perf_feedback(d, PERF_FEEDBACK_MD.replace("设计层归因: ", "归因: "))
    assert "S4-PERF-FEEDBACK-TRIGGER" in rules()
    # 实测章节缺 msprof 口径
    write_perf_feedback(
        d, PERF_FEEDBACK_MD.replace("msprof op Task Duration", "event 计时")
    )
    assert "S4-PERF-FEEDBACK-METRIC" in rules()
    # 反馈结论缺「建议路由」
    write_perf_feedback(
        d, PERF_FEEDBACK_MD.replace("建议路由: 附录补记", "路由倾向: 附录补记")
    )
    assert "S4-PERF-FEEDBACK-ROUTE" in rules()
    # 占位符残留 / 未替换模板变量
    write_perf_feedback(
        d, PERF_FEEDBACK_MD.replace("## 假设", "## 假设\n\n影响面待补充。")
    )
    assert "S4-PERF-FEEDBACK-PLACEHOLDER" in rules()
    write_perf_feedback(
        d, PERF_FEEDBACK_MD.replace("## 假设\n", "## 假设\n\n<按模板填写>\n")
    )
    assert "S4-PERF-FEEDBACK-PLACEHOLDER" in rules()
    # 死链引用
    write_perf_feedback(
        d,
        PERF_FEEDBACK_MD.replace(
            "## 影响面\n", "## 影响面\n\n参考 examples/does_not_exist/foo.py。\n"
        ),
    )
    assert "S4-PERF-FEEDBACK-REF" in rules()


def test_complete4_blocks_on_invalid_perf_feedback(tmp_path):
    repo = make_repo(tmp_path)
    d = op_dir(repo)
    write_artifacts(d)
    assert init_new_op(d)[0] == 0
    happy_path(d)  # plan [1,2,3] → DONE
    assert sc("start", "4", "--dir", str(d), "--extend")[0] == 0
    (d / "perf_opt").mkdir()
    (d / "perf_opt" / "t.py").write_text(KERNEL_PY, encoding="utf-8")
    (d / "perf_opt" / "opt_log.md").write_text(OPT_LOG_MD, encoding="utf-8")
    write_perf_feedback(d, PERF_FEEDBACK_MD.replace("## 反馈结论", "## 结论"))
    before = (d / ".stage_state.json").read_text(encoding="utf-8")
    rc, out = sc("complete", "4", "--dir", str(d))
    assert rc == 1 and out["failures"]
    assert (d / ".stage_state.json").read_text(encoding="utf-8") == before


def test_set_perf_feedback_action(tmp_path):
    repo = make_repo(tmp_path)
    d = op_dir(repo)
    assert init_new_op(d)[0] == 0
    # 工件不存在 → 拒绝记录路由
    rc, out = sc("set", "--dir", str(d), "--perf-feedback-action", "archive")
    assert rc == 1 and out["errors"][0]["code"] == "E-MISSING"
    write_perf_feedback(d)
    rc, _ = sc("set", "--dir", str(d), "--perf-feedback-action", "archive")
    assert rc == 0
    state = load_state(d)
    assert state["perf_feedback"]["action"] == "archive"
    assert state["perf_feedback"]["path"].endswith("perf_opt/perf_feedback.md")
    # 非法 action → argparse usage error
    rc, out = sc("set", "--dir", str(d), "--perf-feedback-action", "bogus")
    assert rc == 2
    # timeline 记录路由动作
    timeline = [
        json.loads(line)
        for line in (d / ".task_timeline.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert timeline[-1]["action"] == "set"
    assert timeline[-1]["perf_feedback_action"] == "archive"


def test_design_limit_revision_flow(tmp_path):
    """[DESIGN_LIMIT] 路径 C：Stage 4 in_progress → fail 4 --reason design_revision。"""
    repo = make_repo(tmp_path)
    d = op_dir(repo)
    write_artifacts(d)
    assert init_new_op(d)[0] == 0
    assert happy_path(d) == 0  # plan [1,2,3] → DONE；重开 Stage 4
    assert sc("set", "--dir", str(d), "--perf-tuning", "yes")[0] == 0
    assert sc("start", "4", "--dir", str(d), "--extend")[0] == 0
    (d / "perf_opt").mkdir()
    (d / "perf_opt" / "t.py").write_text(KERNEL_PY, encoding="utf-8")
    (d / "perf_opt" / "opt_log.md").write_text(OPT_LOG_MD, encoding="utf-8")
    write_perf_feedback(d)
    # 附录补记（默认）路径：合法 perf_feedback 过 gate，complete 4 → DONE
    rc, out = sc("gate", "4", "--dir", str(d))
    assert rc == 0, out["failures"]
    assert sc("set", "--dir", str(d), "--perf-feedback-action", "archive")[0] == 0
    rc, _ = sc("complete", "4", "--dir", str(d))
    assert rc == 0
    state = load_state(d)
    assert state["phase"] == "DONE"
    assert state["perf_feedback"]["action"] == "archive"
    # perf_feedback.md 进入 Stage 4 快照，篡改后 verify 报漂移
    assert any(p.endswith("perf_feedback.md") for p in state["artifact_hashes"]), state[
        "artifact_hashes"
    ]
    (d / "perf_opt" / "perf_feedback.md").write_text(
        PERF_FEEDBACK_MD + "\n被篡改\n", encoding="utf-8"
    )
    rc, out = sc("verify", "--dir", str(d))
    assert rc == 1
    assert any("漂移" in item["issue"] for item in out["drift"])


def test_design_limit_revision_from_stage4(tmp_path):
    """设计修订路径 C（Stage 4 触发源）：共享 retry_count、下游清零、修订重入。"""
    repo = make_repo(tmp_path)
    d = op_dir(repo)
    write_artifacts(d)
    assert init_new_op(d)[0] == 0
    assert happy_path(d) == 0
    assert sc("start", "4", "--dir", str(d), "--extend")[0] == 0
    (d / "perf_opt").mkdir()
    (d / "perf_opt" / "t.py").write_text(KERNEL_PY, encoding="utf-8")
    (d / "perf_opt" / "opt_log.md").write_text(OPT_LOG_MD, encoding="utf-8")
    write_perf_feedback(d)
    # 路由记录 + 设计修订（fail 4 --reason design_revision）
    assert sc("set", "--dir", str(d), "--perf-feedback-action", "revise")[0] == 0
    rc, out = sc("fail", "4", "--dir", str(d), "--reason", "design_revision")
    assert rc == 0 and out["retry_count"] == 1
    state = load_state(d)
    # 下游 Stage 2/3/4 状态清空、重试计数清零（全新实现语义）
    assert state["stage_status"] == {"1": "completed"}
    assert state["stage_retry_count"]["4"] == 0
    assert state["perf_feedback"]["action"] == "revise"  # 路由记录保留
    assert state["last_failure_reason"] == "design_revision"
    # 已 completed 的 Stage 1 凭一次性修订通行证重入
    rc, _ = sc("start", "1", "--dir", str(d))
    assert rc == 0
    assert load_state(d)["phase"] == "DESIGN"


# ---------------------------------------------------------------------------
# 1.3：任务时间线事件流（字段丰富 / duration / timeline-summary）
# ---------------------------------------------------------------------------


def read_timeline(d: Path) -> list:
    return [
        json.loads(line)
        for line in (d / ".task_timeline.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]


def init_at(d: Path, now: str, **kw):
    return sc(
        "init",
        "--dir",
        str(d),
        "--project",
        "proj",
        "--op",
        "t",
        "--scenario",
        "new_op",
        "--requirement",
        "r",
        now=now,
        **kw,
    )


def test_timeline_event_fields(tmp_path):
    """start/complete/fail 事件携带 1.3 字段：stage/subagent/mode/verdict/duration_s。"""
    repo = make_repo(tmp_path)
    d = op_dir(repo)
    write_artifacts(d)
    assert init_new_op(d)[0] == 0
    assert read_timeline(d)[0]["mode"] == "new_op"
    assert sc("start", "1", "--dir", str(d))[0] == 0
    assert sc("complete", "1", "--dir", str(d))[0] == 0
    assert sc("start", "2", "--dir", str(d))[0] == 0
    assert sc("fail", "2", "--dir", str(d), "--reason", "design_revision")[0] == 0
    events = read_timeline(d)
    start_ev = [e for e in events if e["action"] == "start" and e["stage"] == 1][-1]
    assert start_ev["subagent"] == "tilelang-op-designer"
    assert start_ev["mode"] == "new_op"
    complete_ev = [e for e in events if e["action"] == "complete"][-1]
    assert complete_ev["verdict"] == "pass"
    assert complete_ev["subagent"] == "tilelang-op-designer"
    assert complete_ev["duration_s"] == 0  # FIXED_NOW → start 到 complete 为 0
    fail_ev = [e for e in events if e["action"] == "fail"][-1]
    assert fail_ev["verdict"] == "design_revision"
    assert fail_ev["subagent"] == "tilelang-design-reviewer"
    assert fail_ev["mode"] == "new_op"
    assert fail_ev["duration_s"] == 0
    # duration_s 从最近一次同名 Stage 的 start 起算（非首个 start）
    t0, t1 = "2026-09-05T01:00:00Z", "2026-09-05T01:05:00Z"
    assert sc("start", "1", "--dir", str(d), now=t0)[0] == 0
    assert sc("complete", "1", "--dir", str(d), now=t1)[0] == 0
    complete_ev = [e for e in read_timeline(d) if e["action"] == "complete"][-1]
    assert complete_ev["duration_s"] == 300.0


def test_timeline_mode_labels(tmp_path):
    """mode = scenario(-migration_mode) 组合标签，与最终报告 vocabulary 一致。"""
    repo = make_repo(tmp_path)
    d_h = op_dir(repo, op="th")
    sc(
        "init",
        "--dir",
        str(d_h),
        "--project",
        "p",
        "--op",
        "th",
        "--scenario",
        "migration",
        "--migration-mode",
        "harness",
    )
    assert read_timeline(d_h)[-1]["mode"] == "migration-harness"
    d_p = op_dir(repo, op="tp")
    sc(
        "init",
        "--dir",
        str(d_p),
        "--project",
        "p",
        "--op",
        "tp",
        "--scenario",
        "migration",
        "--migration-mode",
        "plain",
    )
    assert read_timeline(d_p)[-1]["mode"] == "migration-plain"
    d_o = op_dir(repo, op="to")
    sc(
        "init",
        "--dir",
        str(d_o),
        "--project",
        "p",
        "--op",
        "to",
        "--scenario",
        "optimize",
    )
    assert read_timeline(d_o)[-1]["mode"] == "optimize"


def test_timeline_duration_omitted_without_start(tmp_path):
    """BLOCKED_* 直接终态且该 Stage 从未 start → 无 duration_s 字段。"""
    repo = make_repo(tmp_path)
    d = op_dir(repo)
    assert init_new_op(d)[0] == 0
    rc, _ = sc("fail", "1", "--dir", str(d), "--blocked-code", "BLOCKED_SPEC")
    assert rc == 0
    fail_ev = [e for e in read_timeline(d) if e["action"] == "fail"][-1]
    assert fail_ev["verdict"] == "BLOCKED_SPEC"
    assert "duration_s" not in fail_ev


def test_timeline_summary(tmp_path):
    """timeline-summary 机械汇总：dispatches / 各 Stage attempts 与耗时 / 失败链。"""
    repo = make_repo(tmp_path)
    d = op_dir(repo)
    write_artifacts(d)
    t0, t1 = "2026-09-05T00:00:00Z", "2026-09-05T00:01:40Z"
    assert init_at(d, t0)[0] == 0
    assert sc("start", "1", "--dir", str(d), now=t0)[0] == 0
    assert sc("fail", "1", "--dir", str(d), now=t0)[0] == 0  # attempt 1 失败（0s）
    assert sc("start", "1", "--dir", str(d), now=t0)[0] == 0
    assert sc("complete", "1", "--dir", str(d), now=t1)[0] == 0  # attempt 2 通过
    rc, out = sc("timeline-summary", "--dir", str(d), now=t1)
    assert rc == 0
    assert out["mode"] == "new_op"
    assert out["dispatches"] == 2
    assert out["events_total"] == 5
    assert out["first_ts"] == t0 and out["last_ts"] == t1
    assert out["elapsed_s"] == 100.0
    assert out["action_counts"] == {"init": 1, "start": 2, "fail": 1, "complete": 1}
    (s1,) = out["stages"]
    assert s1["stage"] == 1 and s1["subagent"] == "tilelang-op-designer"
    assert s1["attempts"] == 2 and s1["completed"] == 1 and s1["failed"] == 1
    assert s1["durations_s"] == [0.0, 100.0]
    assert s1["duration_s_total"] == 100.0
    (failure,) = out["failures"]
    assert failure["stage"] == 1
    assert failure["subagent"] == "tilelang-op-designer"
    assert failure["verdict"] == "runtime"  # 默认 fail_type
    assert failure["ts"] == t0 and failure["duration_s"] == 0.0


def test_timeline_summary_missing_file(tmp_path):
    repo = make_repo(tmp_path)
    d = op_dir(repo)
    rc, out = sc("timeline-summary", "--dir", str(d))
    assert rc == 1 and out["errors"][0]["code"] == "E-MISSING"


def test_timeline_summary_migration_aggregate(tmp_path):
    """migration 聚合时间线（无 stage 字段）：dispatches=0、action_counts 汇总。"""
    repo, slug_dir, meta_path = _make_harness_repo(tmp_path)
    assert (
        sc(
            "migration",
            "init",
            "--dir",
            str(slug_dir),
            "--op-slug",
            "mop",
            "--family",
            "reduction",
            "--meta-path",
            str(meta_path),
            "--functions",
            "funcA",
        )[0]
        == 0
    )
    assert (
        sc("migration", "func", "funcA", "--dir", str(slug_dir), "--status", "done")[0]
        == 0
    )
    rc, out = sc("timeline-summary", "--dir", str(slug_dir))
    assert rc == 0
    assert out["dispatches"] == 0
    assert out["stages"] == [] and out["failures"] == []
    assert out["action_counts"] == {"migration-init": 1, "migration-func": 1}
