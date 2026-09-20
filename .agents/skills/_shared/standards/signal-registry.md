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

从迁移后 benchmark 展开全部案例，先写 `perf_opt/workload_inventory.json`，再逐测量向 `perf_opt/perf_records.jsonl` 追加一行（只追加不改写不删行）。

`workload_inventory.json`：`{"workloads": [...]}`。每项至少含 `benchmark_source`（非 TileOPs 独立算子的显式性能目标记 `explicit_target`）、`kernel_id`（可唯一定位实现文件与函数）、`workload_id`、`label`、`marks`（pytest 标记名数组）、`shape`（单输入可为列表，多输入可为按输入名组织的对象）、`dtype`、`params`、`kind`（`tune` / `smoke` / `skipped`）和 `reason`；同一 `(kernel_id, workload_id)` 不重复。`tune` 项在调优后须写 `tuning_status`：`winner_merged` / `no_gain` / `merge_blocked`，分别表示 winner 已核内合并、完整迭代无有效增益、或找到局部 winner 但未能安全合并；后两种状态须在 `reason` 说明原因，未处理或预算中断的 workload 不得伪装成已完成。`smoke` 项在最终及每次合并后只做精度回归，最终设置 `precision_pass: true/false`；`skipped` 项保留 benchmark skip 原因。若显式 `full` 标记与独立 smoke ID/label 冲突，按 smoke 排除性能调优并在 `reason` 留痕。不得仅凭 shape 或参数化顺序判定 smoke。

`perf_records.jsonl` 每行字段：

`{round, candidate_id, parent_id, kernel_id, workload_id, phase, artifact_path, artifact_sha256, duration_us, l0_pass, msprof_raw_path, timestamp}`

`phase` 为 `baseline` / `candidate` / `merged` / `final`。`artifact_path` 指向实际测量的候选 kernel 文件，`artifact_sha256` 是采集时该文件的 SHA256；final 记录的路径和哈希须与最终 `perf_opt/{op}.py` 一致。只有 inventory 中 `kind=tune` 的项可以有性能记录；每项必须有有效 baseline 和 final 记录。对同一个 `kernel_id`，全部 final 记录须使用同一个 `candidate_id`，且在最终文件上重新测得；合并时与上一已合并版本逐 workload 比较，最终再与 baseline 比较。Final Summary 须有 `Final Performance Test Data` 表，逐项列 `kernel_id | workload_id | baseline_us | final_us | final_candidate_id`，不得拼接各实验分支的最好成绩。`msprof op Task Duration(us)` 是唯一时延口径。无 tune 项的 kernel 可记录 `no_tunable_workload`，不生成虚假性能数据。

机械校验（gate 4）：依据 inventory 核对分类与覆盖，逐 tune workload 校验 baseline/final、同一 kernel 的 final candidate 一致、Final Performance Test Data 逐项时延与 JSONL 偏差 ≤1%；smoke 仅核对 `precision_pass`，skip 核对 reason，二者不得有性能记录；迭代日志仍须含「候选 vs current best」对比表。

## 6. 关键术语的执行定义

- **二次校验精度**：conductor 在 Stage 3 返回 `[PRECISION_PASS]` 后**重新跑全量** `python {op}.py --level all` 确认结果真实性（不只复跑 L0）。
- **四出口判定（Stage 3）**：`[PRECISION_PASS]` / `[PRECISION_FAIL]` / `[DESIGN_ERROR]`（三者均为带标记返回）/ `RUNTIME_FAIL`（无标记且 exit code ≠ 0）。路由表见 [stage3-routing.md](stage3-routing.md) §2。
