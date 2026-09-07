# TUNING→DESIGN 受控逆向反馈标准（`[DESIGN_LIMIT]`）⭐

> **单一事实源**：本文件是「`[DESIGN_LIMIT]` 信号契约 / 触发条件 / `perf_feedback.md` 固定 schema / 路由与防抖动规则」的权威版本。conductor（路由）、optimizer（产出）、`tilelang-op-optimize` skill（执行）、README（摘要）只引用本文件，不复制本文文本。机械子集（schema 存在性 / 量化 / 口径 / 占位符 / 引用路径）由 `statectl gate 4`（`gate_lint.py` 规则 `S4-PERF-FEEDBACK-*`）执行，两者同步演进。
>
> **背景**（U14 修复，改进项 #20）：调优阶段发现的算法/设计层性能天花板（如「存在更优算法族但设计已冻结」）此前被「调优不逆向反馈」硬禁止，opt_log 里的算法级发现无法回流 Stage 1。本机制以**受控旁路**打开回流通道——调优仍自完成最优版本，反馈经**信号 + 工件 + 用户路由**三重受控（依据：Reflexion/ExpeL「失败反馈回流才产生学习」；AlphaEvolve 程序数据库本质是「测量结果反哺生成」）。

## 1. 触发条件（optimizer 判定，两项须同时满足）

| # | 条件 | 判定要点 |
|---|------|---------|
| ① | 性能天花板由**算法/设计层**决定 | 非 tiling/参数可解；须归因到 DESIGN.md 具体假设或选型（§1.6 调研选定 / §1.6.1 优化后公式 / §1.6.3 布局决策 / §5 分核方案 / 实验裁决主选） |
| ② | 结构性加速估计 > **2x** 或实测与设计假设**直接矛盾** | 估计依据须为实测数据外推 / 文献 / pattern-library 先例；矛盾须指认被推翻的具体假设与实测证据 |

条件不满足时**禁止产出** `perf_feedback.md`——参数级性能不足（tiling 可解）留在 Stage 4 迭代内，永不触发本机制。

`[DESIGN_LIMIT]` 是**非阻塞**信号：伴随 `TUNING_COMPLETED` 返回（调优闭环照常收束最优版本），**不是** `[DESIGN_ERROR]`（正确性阻塞信号，Stage 3 中断语义）。实验裁决 A/B 的「备选结构性胜出 / 实测推翻判定常数」是该信号的一等来源（[core-split-strategy.md](core-split-strategy.md) §2.4）。

## 2. `perf_feedback.md` 固定 schema（`perf_opt/perf_feedback.md`）

```markdown
# 性能反馈（[DESIGN_LIMIT]）

## 触发判定

- 设计层归因: <DESIGN.md 具体章节/假设>
- 结构性加速估计: <N>x（依据: <实测外推/文献/pattern-library 先例>）或 实测与假设矛盾: <描述>

## 假设

<DESIGN.md 中的原假设/选型，含章节引用>

## 实测

<msprof op 口径证据：dispatch、Task Duration、对比数据、raw profile 引用>

## 影响面

<该设计假设影响的算子族/shape 范围；改设计预期影响的 Stage 与工作量>

## 反馈结论

<一句话结论>

建议路由: 附录补记 / 设计修订
```

机械校验（`S4-PERF-FEEDBACK-*`，文件存在时由 gate 4 强制执行）：文件存在且非空、无占位符与未替换模板变量、五个章节（触发判定/假设/实测/影响面/反馈结论）齐全、触发判定含「设计层归因」与量化加速（>2x，如 2.5x）或「矛盾」依据、实测章节含 `msprof` 口径、反馈结论含「建议路由」、引用的仓库路径真实存在。归因是否成立、估计是否可信、影响面是否准确等**语义判定仍归 conductor 与后续检视**。

## 3. 路由（conductor 亲自执行，Primary 上下文 AskUserQuestion）

前置：`statectl gate 4` 通过（含 perf_feedback.md schema 校验）；信号声称但文件缺失/不合规 → 按「门禁失败处理流程」`fail_stage(4)` 重调度 optimizer。`optimize` 场景先过精度回归 gate 再路由。

| 路由 | 适用条件 | 动作 | 预算 |
|------|---------|------|------|
| **附录补记**（默认推荐） | 任意场景 | 向 DESIGN.md **追加**「性能反馈附录」小节（一段摘要 + `perf_feedback.md` 路径引用，只追加不删改既有章节）→ `statectl snapshot --dir ...` 重记哈希 → `statectl set --perf-feedback-action archive` → `complete_stage(4)` 正常收束 | 不耗 `retry_count`、不重开 Stage |
| **设计修订**（路径 C） | `1 ∈ stage_plan`（new_op / migration-plain）且 `retry_count < max_retry`；optimize 场景无 Stage 1、harness 场景无 Stage 4，均不适用 | 备份 `DESIGN.md` 与 `perf_feedback.md` 到 `history_version/` → `statectl set --perf-feedback-action revise` → `fail_stage(4, reason=design_revision)`（共享预算、清空下游）→ `start_stage(1)` → 重调度 designer（`mode=revision`，`design_error_summary` = 触发判定 + 反馈结论 + 建议路由） | **共享 `retry_count`**（与路径 A/B 合并累计，`max_retry` 封顶） |
| **不处理** | 任意场景 | `statectl set --perf-feedback-action none` → `complete_stage(4)`；`perf_feedback.md` 留档（终态蒸馏仍读取） | 无 |

## 4. 防抖动（对应聚合报告风险 7.5）

1. 触发门槛双条件（设计层归因 + 结构性加速 > 2x / 假设矛盾）**同时满足**——参数级不足永不触发；
2. 设计修订共享 `retry_count`（`max_retry` 自然封顶）；附录补记与不处理不耗任何预算、不重开 Stage；
3. 同一 `[DESIGN_LIMIT]` 信号只路由一次；修订后 Stage 4 重跑再次给出信号时按新信号路由（仍受预算约束）；
4. 附录补记只追加不删改，且追加后必须 `statectl snapshot` 重记哈希——否则 `statectl verify` 会把附录误报为 DESIGN.md 工件漂移。

## 5. 与自进化的衔接

`perf_feedback.md` 列入 evolver 终态蒸馏工件清单（**D 类实测数据优先蒸馏**，带证据三件套溯源与工具链版本戳）；附录补记使同算子后续 optimize / 再设计任务在 DESIGN.md 中直接可见该发现；同族算子后续任务经 pattern-library 案例索引自动引用（与改进项 #14 跨算子复用衔接）。
