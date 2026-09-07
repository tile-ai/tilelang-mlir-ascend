# 最终输出报告模板 ⭐

> **单一事实源**：本文件是任务终态结构化摘要的权威模板。conductor 在流程结束时按本模板输出（Read 本文件后填充）；README 只链接摘要。占位符 `{...}` 由状态文件与任务工件实际值填充。

```markdown
## 开发结果
- scenario: {new_op / migration-harness / migration-plain / optimize}    project: {project}    算子: {op}    phase: DONE / FAILED    failure_reason: <FAILED 时填>    design_revisions: {retry_count}
- design: examples/{project}/{op}/DESIGN.md（无则标 N/A）
- review: examples/{project}/{op}/REVIEW.md（无则标 N/A）
- kernel: examples/{project}/{op}/{op}.py（含 kernel + golden + 分层测试套件 L0/L1/L2/Boundary）
- final_artifact: {final_artifact 路径，若有调优则指向 perf_opt/{op}.py；harness 迁移指向集成包}
- migration: <仅 harness：函数列表与各自状态、meta 路径、集成结果、bench 报告摘要>

## 精度结果
- status: PASS / FAIL / UNKNOWN    accuracy_fix_count: {stage3 precision_fix 次数}
- harness 集成验证: <smoke / full 用例数与结果，仅 migration-harness 填>

## 性能结果（若进入 Stage 4）
- iterations: {perf_iteration.count}    last_improvement: {perf_iteration.last_improvement}
- final_artifact: {kernel_opt_py_path}
- 回归: <仅 optimize：perf_opt 回归 L0/L1 结果>

## 时间线（可观测性，statectl 机械生成）
- timeline_summary: <粘贴 `statectl timeline-summary --dir examples/{project}/{op}` 的 JSON 输出（dispatches / 各 Stage attempts 与耗时 / 失败事件链）；migration-harness 逐函数算子目录 + op 级聚合目录各跑一次；`.task_timeline.jsonl` 缺失时标 N/A>

## 成本（预算水位，E1.3）
- dispatches: {timeline_summary.dispatches}    wallclock_s: {timeline_summary.elapsed_s}    budget: {各维度 used/max 摘要；未设置预算时标 未设置}
- 各 Stage 耗时: <从 timeline_summary.stages 拼接「stage: attempts N 次 / duration_s_total X」；纯拼接，无新数据>

## 进化结果（自进化）
- verdict: {EVOLVE_COMPLETED / [EVOLVE_SKIP] / [EVOLVE_FAIL] / skipped(无信号)}
- merged: <Tier 0 合入摘要（pattern-library 条目名）>
- enqueued: <入队 proposal_id 摘要>
- pending_tier2: {待人工审批 R 类提案数与摘要；处理方式见 conductor「自进化机制」第 4 条}
```
