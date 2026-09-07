---
disable: true
---

# conductor 场景文件：new_op（新算子生成，默认场景）

> **适用范围**：`scenario=new_op`（场景路由的默认分支）。**载入时机**：场景路由确定为 new_op 后、需求预检开始前，按主文件 `.opencode/agents/tilelang-op-conductor.md`「场景文件加载」指引 Read 载入；续跑 / 恢复 / 设计修订重入时按 `.stage_state.json` 的 `scenario` 字段重新载入。本文件与主文件冲突时以本文件为准；与 `.agents/skills/_shared/standards/` 权威标准冲突时以标准文件为准。

## 1. 状态机

```
INIT --> DESIGN --> REVIEW --> DEVELOP --> TUNING(可选) --> DONE
  ^                 |
  |___ 修订循环 ____|  (retry_count < max_retry)
  |___ 超限 _______> FAILED
```

## 2. 项目名称与算子名称解析（预检最先执行）

从用户提示词中解析**项目名称（project）**和**算子名称（op）**，二者决定全流程的目录与文件路径：

| 名称 | 解析来源 | 解析不出时 |
|------|----------|-----------|
| 算子名称（op） | 用户消息中的明确算子名（如 `softmax`、`layer_norm`） | 必须通过 AskUserQuestion 向用户追问，不得跳过 |
| 项目名称（project） | 用户消息中提及的项目分组（如"norm 项目下的 layer_norm"、"gemm 项目的 matmul"） | **`project = op`**（用算子名称作为项目名称） |

- `project_name` 决定项目目录 `examples/{project}/`（可含多个算子）；`op_name` 决定算子目录 `examples/{project}/{op}/` 及其中文件名（完整算子目录结构见主文件「标准工件契约」）。
- 解析完成后将 `project_name` 与 `operator_name` 写入 `.stage_state.json`（经 `statectl init` 参数）。后续所有 Subagent 调度 prompt 中必须同时传入 `project_name` 和 `op_name`，Subagent 据此确定工件落盘路径。

## 3. 5 个必需字段清单

进入 Stage 1 之前必须确保以下字段**全部齐全**（来源可以是用户消息中已说明，或你通过 AskUserQuestion 问到的）：

| 字段 | 判定齐全的标准 | 缺失时的提问内容 |
|------|-------------|-----------------|
| 算子名称 | 用户消息中含明确算子名（如 softmax、layer_norm）；或可从功能描述无歧义推断 | "请告诉我算子名称（用作算子文件名和函数名，如 `softmax`）" |
| 数学公式 / 计算语义 | 用户给出公式 / 标准 API 名（如"参考 PyTorch 的 F.softmax"）；标准算子可由你查知识库 | "请给出算子的数学公式或参考实现（如 `softmax(x)=exp(x)/sum(exp(x))`，或 `参考 torch.nn.functional.softmax`）" |
| 输入张量规格 | **shape + dtype 都明确**（shape 可含动态维度 `B`、`N` 等符号，但需明确哪些动态）。该 shape 作为 L0 代表性规则 shape；更全面的不规则/异常/边界覆盖由 Stage 1 的 L0 计划与 Stage 3 的扩展自动产生 | "请告诉我输入张量的 shape 和 dtype（如 `[B, N] float16`，其中 B 是动态、N 是静态）" |
| 输出张量规格 | shape + dtype 都明确；若与输入一致可允许"同输入"作为回答 | "请告诉我输出张量的 shape 和 dtype（与输入相同时回答`同输入`即可）" |
| **编程模式偏好** ⭐ | 用户明确写 `Developer` / `Expert` / `混合` 三者之一（写入 `op_requirements` 前归一化为小写枚举 `developer` / `expert` / `hybrid`——枚举唯一出处见 `_shared/standards/signal-registry.md` §3） | "请选择编程模式：Developer（自动化）/ Expert（手动控制 L1/UB/L0）/ 混合（关键路径用 Expert）。**这条不能默认填，必须由你选择**" |

### 预检执行规则

1. **逐字段扫描** 按上表顺序扫描用户消息（含初始描述 + 后续回答），标记每个字段为 `provided` 或 `missing`。
2. **每次只问一个 missing 字段**（不批量问），按表格顺序问，已 `provided` 的跳过。
3. **编程模式必须显式问**——只要用户没说就必须问，不能跳过、不能用默认值。
4. **可选字段**（精度容忍度 atol/rtol、性能目标、动态轴范围等）有合理默认值，由 op-design skill 内部处理，不在本预检范围。

### 失败处理

| 情况 | 处理 |
|------|------|
| 用户拒绝回答某字段 | 重新询问 1 次，仍拒绝则置 `phase=FAILED`、`failure_reason=BLOCKED_SPEC` 并报告"用户未提供 X 字段，无法启动开发"。**不允许用默认值绕过**（特别是编程模式） |
| 用户回答模糊（如"差不多"、"随便"） | 用 AskUserQuestion 用 multipleChoice 列出具体选项让用户选 |
| 用户中途要求改字段 | 接受，更新结构化对象，**重新触发**预检确认是否仍齐全 |

预检是 Stage 1 启动的硬前置，**不能委托给 Subagent**。

## 4. 传给 designer 的字段格式

5 个字段齐全后汇总成结构化对象，作为调度 `@tilelang-op-designer`（`mode=first_design`）的 prompt 输入；同时写入临时区便于失败重试时不重复问用户；designer 调用 `tilelang-op-design` skill 时带上这些字段，skill 看到字段齐全后跳过提问环节，直接走技术约束检测和 design 生成。

```yaml
op_requirements:
  project_name: <项目名，解析不出时与 op_name 相同>
  op_name: <算子名>
  math_formula: <公式或参考 API 名；迁移任务可由 designer 从源码解读得出后回填>
  input_spec:
    shape: <如 [B, N]>
    dtype: <如 float16>
    dynamic_axes: <如 [B]>  # 可选，shape 含符号时必填
  output_spec:
    shape: <如 [B, N] 或 same_as_input>
    dtype: <如 float16 或 same_as_input>
  programming_mode: developer | expert | hybrid   # 用户回答 Developer/Expert/混合 后归一化（signal-registry.md §3）
  # ⬇ 迁移任务必填（scenario=migration，字段语义见 conductor-scenarios/migration.md）
  source_op_path: <源算子文件路径；plain=用户给出，harness=.migration_meta.json 中该函数的 GPU 源码路径>
  source_output_shape: <源算子输出 shape，如 (M, N)；无法从源码推断时由 designer 解读后回填并标注依据>
```

## 5. Stage 4 进入前的用户确认（new_op 与 migration-plain 共用）

> migration-harness 跳过 Stage 4（见 harness.md）；optimize 场景调优即任务本身，进入时已收集调优信息，不再询问。

Stage 3 返回 `[PRECISION_PASS]` 且二次校验通过后，你**必须**先向用户说明当前状态（算子已精度通过，给出 kernel 路径），**主动询问**："是否需要进行性能调优？"

| 用户回答 | 行为 |
|---------|------------------|
| 不需要 / 否 / no / 跳过 | 经 `statectl set --perf-tuning no` 写入 `perf_tuning_requested=no`、置 `phase=DONE`，输出最终报告，流程结束 |
| 需要 / 是 / yes | 继续询问调优必要信息（下表），收集完成后经 `statectl set --perf-tuning yes` 写入 `perf_tuning_requested=yes` 并 `start_stage(4)` |
| 未明确回答 | 重新询问一次；二次仍不明确视为"不需要"，置 `phase=DONE` |

### 调优必要信息收集

| 字段 | 必填 | 默认值 | 说明 |
|------|---------|--------|------|
| 性能目标类型 | ✅ | — | `latency` / `throughput` / `baseline_compare`（与 PyTorch/同类对比）/ `best_effort` |
| 目标数值 | ⭕ (type=latency/throughput 时必填) | — | 如 `< 100us` 或 `> 10 GFLOPS` |
| Baseline 路径 | ⭕ (type=baseline_compare 时必填) | — | 对比基线代码路径或 PyTorch API |
| 测试 shape | ⭕ | DESIGN.md 已有 shape | 性能基准对应的输入规格 |
| 噪声阈值 | ⭕ | 3% | 覆盖 optimizer 默认采纳门槛 |
| 最大迭代数 | ⭕ | 10 | 覆盖默认迭代上限 |

信息收集后**追加**写回 `examples/{project}/{op}/DESIGN.md` 的"性能目标"章节（不覆盖既有内容），再执行 `statectl snapshot --dir ...` 重记 DESIGN.md 哈希（防止后续 `verify` 把追加误报为工件漂移），然后 `start_stage(4)`。

### Stage 4 中止条件

满足任一即结束：① 迭代次数达到用户指定上限（默认 10）；② 连续三次无性能提升；③ 达到用户指定的性能目标（type=latency/throughput/baseline_compare 时）。中止后 `phase=DONE`，`final_artifact` 指向 `perf_opt/{op}.py`（或最优版本）。
