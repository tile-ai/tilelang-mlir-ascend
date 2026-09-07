# 信号与契约注册表（signal registry）⭐

> **单一事实源**：本文件是 conductor 体系全部「跨文件契约字面量」的唯一出处——完成/失败/通过信号 token、Subagent 调度 mode 枚举、状态字段枚举、`perf_iteration` 与 `perf_records.jsonl` 字段契约、关键术语的执行定义。conductor / 各 Stage Subagent / 场景文件 / README 引用本文件（一句话 + 路径引用），禁止在别处内联整表；**修改任何枚举值必须先改本文件**，再同步 `statectl`（`.agents/tools/statectl.py`）的 argparse choices 与 `gate_lint.py` 机械规则（两者同步演进，见 [README.md](README.md) 维护规则 5）。
>
> 边界：`statectl` 的 `E-*` 拦截码与命令行参数以代码为权威实现；本表只登记 LLM 可见的契约字面量。

## 1. 阶段完成 / 失败 / 通过信号

| Stage | 完成信号 | 失败 / 判定信号 |
|-------|---------|----------------|
| 0 迁移脚手架 | `SCAFFOLD_COMPLETED` | `[SCAFFOLD_FAIL]` |
| 1 算子设计 | `DESIGN_COMPLETED` | fail + 缺失章节列表 |
| 2 设计检视 | `REVIEW_COMPLETED` | `结论: 不通过`（REVIEW.md 字面量） |
| 3 算子开发 | `DEVELOP_COMPLETED` | 四出口：`[PRECISION_PASS]` / `[PRECISION_FAIL]` / `[DESIGN_ERROR]` / `RUNTIME_FAIL` |
| 4 算子调优 | `TUNING_COMPLETED` | `TUNING_FAILED`；非阻塞逆向信号 `[DESIGN_LIMIT]`（区别于正确性阻塞信号 `[DESIGN_ERROR]`） |
| 5 迁移集成 | `INTEGRATE_COMPLETED` | `[INTEGRATE_FAIL]` / `[DESIGN_ERROR]` |
| 终态蒸馏（非 Stage） | `EVOLVE_COMPLETED` | `[EVOLVE_SKIP]` / `[EVOLVE_FAIL]` |

`BLOCKED_*` 终态码（`failure_reason` 合法值）：`BLOCKED_DESIGN / BLOCKED_IMPL / BLOCKED_ACCURACY / BLOCKED_SCAFFOLD / BLOCKED_INTEGRATION / BLOCKED_ENVIRONMENT / BLOCKED_SPEC`（映射表见 [gate-and-retry.md](gate-and-retry.md) §3）。

## 2. Subagent 调度 mode 枚举

| Subagent | mode 枚举 | 语义 |
|----------|----------|------|
| designer | `first_design` / `revision` | 首次设计 / 设计修订重做（携 `last_design_path` + `design_error_summary` + `previous_revisions`） |
| reviewer | （无 mode） | 每次调用执行完整维度检视（迁移 0–8 / 非迁移 1–8） |
| developer | `first_impl` / `retry_impl` / `precision_fix` | 首次实现 / 运行失败重试 / 精度失败修复 |
| optimizer | `full`（默认，完整调优流程） / `precision_fix`（仅精度回归修复） | `precision_fix`：只跑 L0/L1 回归修复，不重走 Phase 1 采数与已完成轮次，从当前最优版本继续（optimize 场景回归失败重调度专用，计入 `stage_retry_count[4]`） |
| evolver | `distill` / `apply` | 任务终态蒸馏 / 执行用户已批准的 Tier 2 提案 |

## 3. 状态字段枚举（与 statectl schema 同步）

| 字段 | 合法值 | 写入方式 |
|------|--------|---------|
| `scenario` | `new_op` / `migration` / `optimize` | `statectl init --scenario` |
| `migration_mode` | `harness` / `plain` | `statectl init --migration-mode` |
| `programming_mode`（`op_requirements` 字段） | `developer` / `expert` / `hybrid` | 预检归一化后传入 designer（用户可答 Developer/Expert/混合，写入前一律归一为小写枚举） |
| `perf_tuning_requested` | `yes` / `no` | `statectl set --perf-tuning yes\|no` |
| `perf_feedback.action` | `archive` / `revise` / `none` | `statectl set --perf-feedback-action`（前置：`perf_opt/perf_feedback.md` 存在） |
| `phase` | `SCAFFOLD / DESIGN / REVIEW / DEVELOP / TUNING / INTEGRATE / DONE / FAILED` | statectl 迁移 |
| `budget` | `{max_wallclock_s, max_subagent_dispatches, max_stage4_experiments}`，每项为正数或 `null`（=不限；默认全 `null`，建议值见 [gate-and-retry.md](gate-and-retry.md) §2） | `statectl set --budget-json '{"max_subagent_dispatches": 40}'` |

## 4. `perf_iteration` 字段（Stage 4 迭代计数）

| 字段 | 类型 | 语义 | 写入方式 |
|------|------|------|---------|
| `count` | int ≥ 0 | 已完成调优轮数 | `statectl set --perf-iteration-count N` |
| `last_improvement` | float | 最近一次超噪声阈值的提升幅度（%） | `statectl set --perf-iteration-last-improvement F` |
| `consecutive_no_improvement` | int ≥ 0 | 连续无有效提升轮数（plateau 中止条件的机械跟踪） | `statectl set --perf-iteration-no-improve N` |

## 5. `perf_opt/perf_records.jsonl` 字段契约（append-only）

每轮每个实验分支验证后追加一行 JSON（与 `.task_timeline.jsonl` 同纪律，只追加不改写不删行）：

`{round, candidate_id, parent_id, dispatch_path, workload, duration_us, l0_pass, msprof_raw_path, timestamp}`

| 字段 | 类型 | 语义 |
|------|------|------|
| `round` | int | 调优轮次（baseline 记 0） |
| `candidate_id` | str | 候选标识（如 `baseline` / `v2_c轴重排`） |
| `parent_id` | str \| null | 派生来源候选（baseline 为 null） |
| `dispatch_path` | str | 目标 kernel 的 dispatch 路径 |
| `workload` | str | 代表 workload 标识 |
| `duration_us` | number | `msprof op Task Duration(us)`（唯一 kernel 时延口径） |
| `l0_pass` | bool | 该分支 L0 精度回归是否通过 |
| `msprof_raw_path` | str | raw profile 路径（如 `perf_opt/profiles/round2/`） |
| `timestamp` | str | ISO 8601 UTC |

机械校验（gate 4 规则 `S4-PERF-RECORDS-*` / `S4-OPTLOG-COMPTABLE`，文件存在时执行）：① 每行必含全部必需字段且类型正确（`S4-PERF-RECORDS-SCHEMA`）；② `opt_log.md` 每轮须含「候选 vs current best」对比表——Task Duration(us)、AICore 利用率、memory 指标、L0 结果，数据取自本文件（`S4-OPTLOG-COMPTABLE`，B2 结构化回流）；③ Final Summary 须含 `final_latency: {N} us` 行且与记录中某 `duration_us` 偏差 ≤ 1%（`S4-PERF-RECORDS-RECON`，winner 对账——把最终加速比从自述变为可对账）。文件不存在 → 仅告警不阻塞（旧流程兼容）。

## 6. 关键术语的执行定义

- **二次校验精度**：conductor 在 Stage 3 返回 `[PRECISION_PASS]` 后**重新跑全量** `python {op}.py --level all` 确认结果真实性（不只复跑 L0）。
- **四出口判定（Stage 3）**：`[PRECISION_PASS]` / `[PRECISION_FAIL]` / `[DESIGN_ERROR]`（三者均为带标记返回）/ `RUNTIME_FAIL`（无标记且 exit code ≠ 0）。路由表见 [stage3-routing.md](stage3-routing.md) §2。
