---
name: tilelang-op-optimize
description: "对精度已通过的 TileLang-NPUIR 算子做 Stage 4 性能调优，产出 perf_opt/{op}.py、msprof op 性能数据和调优日志。触发：性能调优、optimize、性能优化、perf tuning、Stage 4 调优。"
---

# TileLang-NPUIR 算子性能调优

## 目标

对 Stage 3 精度通过的 `{op}.py` 做真实性能调优，产出：

- `examples/{project}/{op}/perf_opt/{op}.py`
- `examples/{project}/{op}/perf_opt/opt_log.md`（每轮含候选 vs current best 对比表）
- `examples/{project}/{op}/perf_opt/perf_records.jsonl`（**结构化性能记录，append-only**：每轮每实验分支一行，字段契约唯一出处 `_shared/standards/signal-registry.md` §5——gate 4 据此做 winner 对账）
- `perf_opt/profiles/` 下的 raw `msprof op` 数据
- `perf_opt/logs/` 下的实验 stdout/stderr 过程日志
- `examples/{project}/{op}/perf_opt/perf_feedback.md`（可选，仅 `[DESIGN_LIMIT]` 设计层天花板时产出，见 Phase 3）

若项目流程要求交付摘要，可额外生成 `examples/{project}/{op}/Optimize.md`，但它只能从 `perf_opt/opt_log.md` 摘要，不重复记录全过程。

**性能口径唯一**：`msprof op Task Duration`（kernel-only 时延）是本 skill 唯一的 kernel 时延测量方式与优化主指标。不关注算子端到端时延，不采集 NPU event、runner 端到端时间等其他口径；一切优化点的讨论、实验对比、winner 判定和性能目标达成判断，均以降低 `msprof op` 测得的 `Task Duration(us)` 为准。候选差异落入噪声阈值或多次复测打平时，用更大 `launch-count` 复测并以 `Task Duration`、负载均衡和资源占用决胜，不得引入其他测量口径。

## 资源索引

按阶段读取，不要在启动时一次性读取所有资源：

- Phase 0 加载上下文时读取：[hardware-context.md](references/hardware-context.md)、[pattern-library.md](references/pattern-library.md)（**必读**：已验证性能模式/代价表/编译器陷阱版本戳/证伪协议）
- Phase 1 性能采集时读取：[profile-collection.md](references/profile-collection.md)
- Phase 2 每轮现象分析时读取：[iteration-diagnosis.md](references/iteration-diagnosis.md)
- Phase 2 生成候选优化点时按需读取：[bottleneck-patterns.md](references/bottleneck-patterns.md)
- Phase 2 候选优化点包含 autotune 时读取：[autotune.md](references/autotune.md)
- Phase 3 判定设计层天花板时读取：[perf-feedback.md](../_shared/standards/perf-feedback.md)（`[DESIGN_LIMIT]` 触发条件与 `perf_feedback.md` 固定 schema——单一事实源）
- Phase 4 调优复盘时读取：[skill-retrospective.md](references/skill-retrospective.md)（产出按 vp_type D/P/R/C 标注 + 证据三件套，供任务终态 `tilelang-skill-evolver` 蒸馏）

按需参考同类 skill：

- 只有当前算子类型已判断为 cube 时，参考：[tilelang-cube-skill](../tilelang-cube-skill/)
- 只有当前算子类型已判断为 vector 时，参考：[tilelang-vector-skill](../tilelang-vector-skill/)
- 只有当前算子类型已判断为 mix 时，参考：[tilelang-mixcv-skill](../tilelang-mixcv-skill/)

## 主流程

### Phase 0：加载上下文

1. 读取 `{op}.py`、`DESIGN.md` 和 [hardware-context.md](references/hardware-context.md)、[pattern-library.md](references/pattern-library.md)。
2. 判断算子类型：`cube / vector / mix`。
3. **首轮必查项（来自 pattern-library，不得跳过）**：
   - **向量化轴与布局重估**：核对当前实现的向量化轴与核内布局——即使上游设计已选定，其结论可能基于旧工具链或未枚举重排布局变体（I/O layout 是契约、核内布局是设计变量；含 ≥2 个非 batch 维的逐元素/窗口/规约类算子须对照 pattern-library §1 评估「原生布局+最内连续轴」vs「核内重排布局+高整除性轴（如 C 轴融合转置链）」两条路线）；
   - **编译器陷阱版本戳与 origin_task 核对**：pattern-library §2 的陷阱结论绑定工具链版本与来源任务（origin_task）——若 tilelang 源码被修改/重编译过，相关结论自动视为待重验，不得直接引用；引用命中条目时在 opt_log 中标注其 origin_task，据此判别可信度；
   - 若 `DESIGN.md` 含 §1.6.3（向量化轴与数据布局决策），本轮调优须对照该决策与实测现象（标量占比、带宽利用率）——现象与决策矛盾时优先重验布局路线；
   - **实验裁决执行（DESIGN 含实验裁决三件套时必做）**：若 `DESIGN.md` §1.6 含「主选 + 备选 + 实验裁决计划」（判定裕度依赖未实证常数的备选方案），A/B 实测是本轮调优的必做项，不是可选项——按裁决计划执行：① 在 `perf_opt/` 下实现备选变体（基准 `{op}.py` 与 wrapper 不动）；② 按计划的代表 shape 与主选同口径对比（msprof op）；③ 实测/反解裁决所依赖的未知常数（如转置吞吐、跨步代价）；④ 按计划判定阈值裁决——备选胜出（全局或按 shape 分片）则采纳备选变体为候选 best，主选胜出则以实测数字固化设计判定；⑤ 实测数据与裁决结论写入 `opt_log.md`；若裁决构成设计层天花板（备选结构性胜出满足 [perf-feedback.md](../_shared/standards/perf-feedback.md) §1 触发条件，或实测推翻设计判定假设），按其 §2 产出 `perf_feedback.md` 并在返回中附 `[DESIGN_LIMIT]` 信号交 conductor 受控路由（附录补记回写 DESIGN.md / 设计修订路径 C）；新实测常数追加回 pattern-library.md，见 Phase 4。
4. 搜索同类算子或历史优化实现，尤其关注：
   - `T.serial`
   - `T.Pipelined`
   - multi-buffer
   - 片上 buffer 生命周期复用
   - dtype-aware 参数
5. 记录可参考的结构性策略，但不要照搬；必须用当前算子的 profile 验证。

### Phase 1：采集初始 baseline

按 [profile-collection.md](references/profile-collection.md) 执行轻量多 dispatch 采集：

```text
找出真实 dispatch path
-> 每个 dispatch path 选择一个代表 workload
-> 串行运行 msprof op
-> 校验 profile 有效性
-> 记录 Performance Test Data
-> 追加 perf_records.jsonl（round 0，candidate_id=baseline，parent_id=null）
```

要求：

- 普通 shape、tile、axis 数值差异不算 dispatch path，除非它触发真实代码分支。
- 每个 dispatch path 默认只采一个代表 workload。
- 无效 profile 不能进入诊断。
- **baseline 采集完成后即向 `perf_opt/perf_records.jsonl` 追加首行**（append-only，字段契约见 `_shared/standards/signal-registry.md` §5）——后续每轮的对比表与 gate 4 对账都以它为基准。

### Phase 2：优化闭环

Phase 2 是多轮闭环。优化点分析不做成一次性前置步骤；每轮都基于当前 best 的最新 profile 重新分析当前现象，并生成多个候选优化点。

每轮执行：

1. 固定本轮 base：当前 best 版本和它的最新 profile。
2. 读取 [iteration-diagnosis.md](references/iteration-diagnosis.md) 与 [bottleneck-patterns.md](references/bottleneck-patterns.md)（**每轮必读**——模式库是候选优化点的直接来源，且模式库只有在被读的位置上才不会遗忘），基于 base profile 整理当前现象。
3. **先看表再分析**（B2 结构化回流）：从 `perf_opt/perf_records.jsonl` 汇总「候选 vs current best」对比表（Task Duration(us)、AICore 利用率、memory 指标（msprof 可得）、L0 结果）作为本轮分析上下文——把「读散文日志复盘」变成「看表决策」，也直接提升 `[DESIGN_LIMIT]` 设计层归因的证据质量。
4. 从同一个 base 派生多个实验分支：`perf_opt/{op}_opt_v{iter}_{opt_id}.py`。
5. 每个实验分支只改一个主要优化点。
6. 每个实验分支跑 L0 精度回归。
7. 对精度通过的分支，用 `msprof op` 采集目标 kernel 性能；**验证完成后向 `perf_records.jsonl` 追加一行**（含 `parent_id` = 派生来源候选；append-only，禁止改写/删除既有行）。
8. 若结构性分支正确性通过且方向有效，先围绕该结构暴露的关键参数做一轮 coarse autotune 或等价手动粗搜；再检查 autotune top-k 与 winner 邻域，必要时做手动/脚本精搜，最后把精搜 winner 作为该结构分支的候选版本复测。
9. 在同一 `(dispatch_path, workload_id)` 内比较 valid 分支；按本轮主指标（`msprof op Task Duration(us)`）选择候选 winner；Task Duration 打平时用负载均衡、资源占用和代码复杂度决胜。
10. 候选 winner 更新为全局 current best 前，必须确认必测 dispatch 没有超过噪声阈值的性能回退；若只在部分 dispatch 提升但其它必测 dispatch 明显回退，不更新全局 current best，并记录 rollback/defer 原因。
11. 若所有分支无提升、无效或阻塞，current best 保持不变。
12. 记录本轮现象、候选优化点、实验分支、性能、精度、必测 dispatch 非回退检查和 winner/rollback 结论；**本轮记录必须含第 3 步的结构化对比表**（数据来自 perf_records.jsonl 与 raw profile——gate 4 `S4-OPTLOG-COMPTABLE` 机械校验其存在性）。
13. 未满足终止条件则进入下一轮，重新分析当前现象。

终止条件：

- success：达到用户指定性能目标。
- budget_exhausted：达到 `max_rounds` 或 `max_experiments`，默认 `max_rounds=10`，`max_experiments=30`。
- plateau：连续 3 轮没有任何 valid 实验分支带来超过噪声阈值（默认 3%）的提升，且主要结构候选已经得到充分验证；单个配置的 `config_no_gain` 或单个 blocked 分支不能单独证明整类方向无效。
- blocked：剩余候选优化点均因精度失败、编译失败、profile invalid 或实现约束无法继续。
- user_stop：用户要求停止。

### Phase 3：产物收束

1. 选 current best 作为 `perf_opt/{op}.py`。
2. 确认 `perf_opt/opt_log.md` 已完整记录过程和最终结论；**Final Summary 必须含 `final_latency: {N} us` 行**——N 取自 perf_records.jsonl 中 winner 的 `duration_us` 记录（gate 4 `S4-PERF-RECORDS-RECON` 按 ≤1% 偏差对账，把最终加速比从「自述」变为「可对账」）。
3. TileOPs 集成算子（算子目录为 `tileops/kernels/{family}/{op_slug}/{op_slug}_kernel/`）：`perf_opt/` 建在该目录下；wrapper 的 baseline/perf_opt 双 import 切换块由 conductor 在回归通过后翻转采纳（perf_opt 默认激活），本 skill 不修改 wrapper。若 tuned kernel 与基准 kernel 的默认参数不同（如 block_size），须在 `perf_opt/{op}.py` 中以模块级常量暴露 tuned 默认值，供 wrapper 切换块成对引用。
4. **设计层天花板判定（[DESIGN_LIMIT]，可选产出）**：读取 [perf-feedback.md](../_shared/standards/perf-feedback.md)，逐条核对触发条件——① 性能天花板由算法/设计层决定（非 tiling/参数可解，归因到 DESIGN.md 具体假设）；② 结构性加速估计 > 2x 或实测与设计假设直接矛盾。**两项同时满足** → 按其 §2 固定 schema 产出 `perf_opt/perf_feedback.md`（实测章节必须为 msprof op 口径，反馈结论含建议路由），返回时附 `[DESIGN_LIMIT]` 信号；任一不满足 → **禁止产出**该文件（参数级不足留在迭代内，不触发逆向反馈）。

### Phase 4：调优复盘与最终交付

1. 读取 [skill-retrospective.md](references/skill-retrospective.md)。
2. 回看 `perf_opt/opt_log.md` 中的 baseline、多 dispatch 数据、候选优化点、实验分支、winner、rollback、blocked、`config_no_gain` 和 `family_no_gain`。
3. 判断本次调优是否暴露出 skill 流程问题。
4. 判断是否需要提出新的 `BP_xxx`，或更新已有 `BP_xxx`。
5. 把复盘写入 `perf_opt/opt_log.md` 的 `Skill Retrospective` 章节。复盘产出按 **vp_type（D/P/R/C）** 标注并写全**证据三件套**（evidence + repro + toolchain_stamp），供任务终态 `tilelang-skill-evolver`（`tilelang-skill-evolution` skill）蒸馏与分级合入——D/C 类自动合入（Tier 0），P 类入队等 2 次独立证据（Tier 1），R 类入队等人工审批（Tier 2）。
6. **回写 pattern-library.md（例外授权）**：本次调优产出的新模式（含代码形态与实测代价）、新代价数据、编译器陷阱新实证/证伪更正，**必须**追加到 `references/pattern-library.md` 对应章节（含任务溯源与工具链版本戳）——该文件是数据积累文件而非流程文档，本条是"不自动修改 skill 文档"禁令的唯一例外。流程/结构层面的改进仍只提 proposal，不自动改。**本条任务内回写与终态 evolver 蒸馏互补防双写：已回写的条目在复盘表 evidence 中注明回写位置。**
7. 如果项目流程要求 `Optimize.md`，从最终 `opt_log.md` 摘要最终结果、关键有效优化点、中止原因、遗留问题和复盘摘要。
8. 不自动修改 skill 流程文档（SKILL.md 及 references 除 pattern-library.md 外）；只在最终报告里列出建议和 BP proposal。
9. 返回 `TUNING_COMPLETED`。

## 核心防呆

### 每轮都重新分析现象

优化会改变瓶颈。不要只沿用初始 baseline 的现象判断或优化点。

典型变化：

```text
block_size / DMA 效率问题
-> launch/scheduling overhead
-> UB buffer 压力
-> HBM 带宽极限
```

### 每个实验分支只改一个主要优化点

一轮可以尝试多个候选优化点，但每个实验分支只能验证一个主要优化点，且必须从同一个 current best 派生。若确实需要组合策略，先拆成可验证的小步，确认单点有效后再组合。

### 先确认测量分辨力

当候选的 `Task Duration(us)` 差异小于噪声阈值或多次复测打平时，先用更大 `launch-count` 复测并比较多次独立运行的 `msprof op` 结果；不要用单次 msprof 结果对平区内的 `num_cores / block_size` 候选排序。

### Autotune 只用于参数选择

autotune 只负责在给定搜索空间中选参数，不是最终裁判。若主要瓶颈是结构问题，先改结构；结构性分支通过正确性并显示方向有效后，再按 [autotune.md](references/autotune.md) 对该结构做 coarse search，随后检查 top-k 和 winner 邻域，最后用 `msprof op` 复测最终 winner。

### 经验结论不要过度泛化

具体策略细节以 [bottleneck-patterns.md](references/bottleneck-patterns.md) 为准。某个优化点在当前环境失败，只能记录为当前 workload / TileLang-NPUIR / CANN / Developer 模式下的实测结论。

### 证伪协议（否定一个 API/模式前必须遵守）⭐

1. **必须用文档合法形态测试**：否定任何 API/模式前，先查 `docs/Tilelang.language/` 确认其合法参数与形式，穷举代表性写法后再下结论。禁止以单一非法形态（如 3-cycle permutation、非文档规定的循环形式）的编译失败否定整个 API 类别——实测教训：曾因此误判"T.transpose 链/C 轴累加不可用"，掩盖了 2–13x 收益（详见 pattern-library §2 证伪协议条目）。
2. **编译器约束结论必须绑定工具链版本戳**：记录 tilelang build/commit 信息；工具链变更后旧结论自动视为待重验。
3. **证伪更正须留痕**：推翻旧结论时在 opt_log.md 写明"误判根因 + 合法形态 + 新数据"，并同步更新 pattern-library.md §2 的状态列。
4. **诊断信号强制触发换轴分析**：msprof 显示热点段标量执行占比 > 50% 时，强制评估向量化轴/布局重排候选（pattern-library §1），不得只在原轴上微调参数。

## 日志最小要求

`perf_opt/opt_log.md` 至少记录：

- Performance Test Data：每个 dispatch 的 workload、target kernel、Task Duration、raw profile。
- Iteration Log：每轮现象、候选优化点、实验分支、latency、精度、`config_no_gain/family_no_gain` 范围、必测 dispatch 非回退检查、winner/rollback；**每轮含「候选 vs current best」对比表**（Task Duration(us)、AICore 利用率、memory 指标、L0 结果——B2 结构化回流，gate 4 `S4-OPTLOG-COMPTABLE` 校验存在性）。
- Autotune Log：若使用 autotune，则记录 search space、best config、正确性、winner 的 `msprof op` 复测结果。
- Final Summary：best 版本、**`final_latency: {N} us` 行**（与 perf_records.jsonl 可对账，`S4-PERF-RECORDS-RECON`）、总提升、中止原因。
- Skill Retrospective：skill 流程问题、建议修改、`BP_xxx` proposal。

`perf_opt/perf_records.jsonl`：每轮每分支一行（baseline 为 round 0 首行），append-only，字段契约唯一出处 `_shared/standards/signal-registry.md` §5；gate 4 `S4-PERF-RECORDS-*` 机械校验。

产物布局要求：

- `perf_opt/` 顶层只放最终 `{op}.py`、实验分支 `{op}_opt_v*.py`、`opt_log.md`、`perf_records.jsonl`、`perf_feedback.md`（仅 `[DESIGN_LIMIT]` 时）、`profiles/`、`logs/` 和必要 runner/helper 脚本。
- 实验 stdout/stderr 写入 `perf_opt/logs/{stage_or_round}/`。
- 不在 `perf_opt/` 顶层生成 `op*.log`、`probe*.log`、`final*.log` 或 `debug_log.md`。

`Optimize.md` 不是必需过程日志。只有项目流程要求交付摘要时才生成，且内容必须来自 `perf_opt/opt_log.md`。

## 交付报告

```markdown
## Stage Result

- stage: 4
- project: {project}
- operator: {op}
- output: examples/{project}/{op}/perf_opt/{op}.py
- log: examples/{project}/{op}/perf_opt/opt_log.md
- summary_doc: examples/{project}/{op}/Optimize.md 或 none
- verdict: TUNING_COMPLETED
- design_limit_signal: none 或 [DESIGN_LIMIT]
- perf_feedback: none 或 examples/{project}/{op}/perf_opt/perf_feedback.md
- iterations: {N}
- primary_metric: msprof_task_duration
- baseline_latency: {v} us
- final_latency: {v} us
- improvement: {x}%
- stop_reason: {reason}
- skill_retrospective: {none_or_summary}
- value_point_proposals: {none_or_list}   # 含 BP_xxx 与 D/P/R/C 各类，逐条带 vp_type
- skills_consulted: {paths}
- summary: {one_sentence}
- issues: {none_or_notes}
```
