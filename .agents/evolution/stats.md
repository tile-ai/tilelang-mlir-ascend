# 进化统计（Evolution Stats）

> 由 `tilelang-skill-evolver` 在每次蒸馏（distill 模式）后更新。用于度量自进化机制的有效性：条目是否被读到、同类错误是否复发。

## 1. 任务蒸馏记录

> 每次蒸馏任务追加一行。`vp_count` 按 D/P/R/C 分列计数（候选数，非合入数）。

| date | task_id | scenario | final_phase | vp_count (D/P/R/C) | merged | enqueued | verdict |
|------|---------|----------|-------------|--------------------|--------|----------|---------|
| 2026-09-07 | lerp_tensor-_make_lerp_tensor_kernel-20260907T010433Z | migration-harness | DONE | 3/3/5/1 | pattern-library §1.5×1（host IntImm 折叠）、§2×2（threads= 无效果、torch golden fp16 opmath 分歧）、§4×1（lerp 迁移精度案例） | VP-2026-0002/0003/0006/0007（Tier 1）、VP-2026-0008/0009/0012（Tier 2） | EVOLVE_COMPLETED |
| 2026-09-07 | lerp_tensor-_make_lerp_tensor_kernel-20260907T025419Z | optimize | DONE | 3/3/2/1 | bottleneck-patterns BP_run_state_bimodality（VP-2026-0001，mish+lerp 2/2） | VP-2026-0004/0005（Tier 1）、VP-2026-0010/0011（Tier 2）；查重命中 4（§1.6 copy-floor/MTE2 曲线、§2 multi-buffer、§4 perf_opt 案例行——optimizer 已任务内回写） | EVOLVE_COMPLETED |

## 2. 条目命中统计

> 统计 pattern-library / bottleneck-patterns 条目被后续任务引用的情况。
>
> **计数方式**（由 evolver 在蒸馏时增量更新，不要求全量重扫）：
> - **主动命中**：任务工件（DESIGN.md / opt_log.md / REVIEW.md / RETROSPECTIVE.md）中显式引用了 `pattern-library §x.x` / `BP-xxx` 条目——每个任务每条目计 1 次。
> - **被动命中**：条目位于某 skill 的强制读取路径（design 强制步骤 0.5 / optimize Phase 0 / develop Phase 1 / conductor 带记忆重试注入）且该任务执行过对应阶段——不计入（只统计主动引用）。
>
> **治理规则**：某条目累计主动命中为 0 且超过 2 个蒸馏周期 → consolidate 候选；同类错误在已有对应条目的情况下复发（复盘章节可识别）→ 检索注入点缺失，优先补「失败触发读取」而非新增条目。

| entry | hits | last_hit_task | note |
|-------|------|---------------|------|
| pattern-library §4 mish perf_opt 案例行 | 1 | lerp_tensor-…T025419Z | lerp optimizer 经 §4 路由引用 mish 先例（run 双态协议 / Expert 流水 / multi-buffer 教训） |
| pattern-library §2 event 口径弃用行 | 1 | lerp_tensor-…T025419Z | opt_log Skill Flow Issue 引用「§2 已弃用 event 口径」 |
| pattern-library §2 auto-multi-buffer UB 膨胀行 | 1 | lerp_tensor-…T025419Z | Iteration 1 blocked 分析复现 mish BP_multi_buffer_ub_budget（fp32 bs12288 同报错） |
| bottleneck-patterns BP_run_state_bimodality（2026-09-07 新增） | 1 | lerp_tensor-…T025419Z | 合入前经 mish 档案被 lerp 任务应用（Iteration 4 A/B/A/B 协议）；本轮 VP-2026-0001 达 2/2 合入 |
| pattern-library §1.6 copy-floor/MTE2 曲线（2026-09-07 optimizer 任务内回写） | 0 | — | 新条目初始化（非检索命中） |
| pattern-library §2 threads= 无效果行（2026-09-07 新增） | 0 | — | 新条目初始化 |
| pattern-library §2 torch golden fp16 opmath 行（2026-09-07 新增） | 0 | — | 新条目初始化 |
| pattern-library §1.5 host IntImm 折叠行（2026-09-07 新增） | 0 | — | 新条目初始化 |
| pattern-library §4 lerp 迁移精度案例行（2026-09-07 新增） | 0 | — | 新条目初始化 |

## 3. 系统指标

> 北极星指标。原始数据来自各任务 `.stage_state.json` 与 RETROSPECTIVE.md；由 evolver 记录原始数据，趋势分析可人工或后续工具化。

| 指标 | 定义 | 当前值 |
|------|------|--------|
| first_pass_rate | Stage 3 `attempt=1` 即 `[PRECISION_PASS]` 的任务占比（长期应随模式库增长而上升） | 0/1（n=1；lerp_tensor 迁移 attempt-1 precision fail → attempt-2 pass。根因：golden opmath 域未核实——VP-2026-0008/VP-2026-0002 旨在消除该失败类） |
| 同类错误复发率 | 已有 D/P 条目对应的失败模式在后续任务中复发占比（复发=检索注入失效） | 暂无数据（lerp_tensor 的 fp16 opmath 分歧为新模式，无既有条目可复发） |
| design_revision_avg | 任务平均设计修订次数（`retry_count` 均值） | 0（n=1；lerp_tensor 迁移 retry_count=0，precision 重试计入 stage_retry_count[3]） |

## 4. 变更日志

| date | by | change |
|------|-----|--------|
| 2026-09-01 | 初始化（conductor-self-evolution-design 落地） | 创建骨架 |
| 2026-09-07 | tilelang-skill-evolver | 双任务蒸馏（lerp_tensor 迁移 + optimize）：Tier 0 合入 pattern-library §1.5×1/§2×2/§4×1；Tier 1 达 2/2 合入 bottleneck-patterns BP_run_state_bimodality（mish+lerp 双证据，VP-2026-0001）；入队 6×Tier 1（VP-2026-0002..0007）+ 5×Tier 2（VP-2026-0008..0012）。**快照披露**：pattern-library.md 蒸馏前已有未提交改动（本任务族 optimizer 回写 §1.6/§2 multi-buffer/§4 lerp 行 + mish 档案行 + D2 溯源头注），与进化增量同任务 lineage，按单一进化快照一并提交（未采用 stash 拆分——拆分将使 commit 缺失回写查重基线且 stash pop 必然冲突）；如需回滚整体 revert 本次 commit。 |
