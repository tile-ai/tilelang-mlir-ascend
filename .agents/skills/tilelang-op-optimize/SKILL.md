---
name: tilelang-op-optimize
description: "对精度已通过的 TileLang-NPUIR 算子做 Stage 4 性能调优，产出 perf_opt/{op}.py、msprof op / NPU event 性能数据和调优日志。触发：性能调优、optimize、性能优化、perf tuning、Stage 4 调优。"
---

# TileLang-NPUIR 算子性能调优

## 目标

对 Stage 3 精度通过的 `{op}.py` 做真实性能调优，产出：

- `examples/{project}/{op}/perf_opt/{op}.py`
- `examples/{project}/{op}/perf_opt/opt_log.md`
- `perf_opt/profiles/` 下的 raw `msprof op` 数据，以及必要时的 NPU event 数据
- `perf_opt/logs/` 下的实验 stdout/stderr 过程日志

若项目流程要求交付摘要，可额外生成 `examples/{project}/{op}/Optimize.md`，但它只能从 `perf_opt/opt_log.md` 摘要，不重复记录全过程。

性能口径分两层：`msprof op Task Duration` 用于判断目标 kernel 内部执行时间；NPU event median 用于判断设备侧调度、派发和真实完成时间。若用户或 conductor 指定唯一主指标，按指定主指标交付；若未指定，默认用 `msprof op` 做 kernel 优化主口径。当 `Block Dim` 远超 AI Core Count、单 block 工作量很小、或 event 与 `msprof` 明显背离时，必须同时记录并分析 NPU event。event 用于调度结构粗筛和回退门禁；若多个候选 event 差异落入噪声阈值或形成平区，标记 `flat_response`，不能用单 pass event 排序，可用 `msprof Task Duration`、负载均衡和资源占用决胜。

## 资源索引

按阶段读取，不要在启动时一次性读取所有资源：

- Phase 0 加载上下文时读取：[hardware-context.md](references/hardware-context.md)、[pattern-library.md](references/pattern-library.md)（**必读**：已验证性能模式/代价表/编译器陷阱版本戳/证伪协议）
- Phase 1 性能采集时读取：[profile-collection.md](references/profile-collection.md)
- Phase 2 每轮现象分析时读取：[iteration-diagnosis.md](references/iteration-diagnosis.md)
- Phase 2 生成候选优化点时按需读取：[bottleneck-patterns.md](references/bottleneck-patterns.md)
- Phase 2 候选优化点包含 autotune 时读取：[autotune.md](references/autotune.md)
- Phase 4 调优复盘时读取：[skill-retrospective.md](references/skill-retrospective.md)

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
   - **编译器陷阱版本戳核对**：pattern-library §2 的陷阱结论绑定工具链版本——若 tilelang 源码被修改/重编译过，相关结论自动视为待重验，不得直接引用；
   - 若 `DESIGN.md` 含 §1.6.3（向量化轴与数据布局决策），本轮调优须对照该决策与实测现象（标量占比、带宽利用率）——现象与决策矛盾时优先重验布局路线；
   - **实验裁决执行（DESIGN 含实验裁决三件套时必做）**：若 `DESIGN.md` §1.6 含「主选 + 备选 + 实验裁决计划」（判定裕度依赖未实证常数的备选方案），A/B 实测是本轮调优的必做项，不是可选项——按裁决计划执行：① 在 `perf_opt/` 下实现备选变体（基准 `{op}.py` 与 wrapper 不动）；② 按计划的代表 shape 与主选同口径对比（msprof op + 必要时 event）；③ 实测/反解裁决所依赖的未知常数（如转置吞吐、跨步代价）；④ 按计划判定阈值裁决——备选胜出（全局或按 shape 分片）则采纳备选变体为候选 best，主选胜出则以实测数字固化设计判定；⑤ 实测数据与裁决结论写入 `opt_log.md`，并报告 conductor 以触发设计修订回写（DESIGN.md 的判定依据从"先例/下界估算"升级为"实测数字"；新实测常数追加回 pattern-library.md，见 Phase 4）。
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
-> 必要时采集 NPU event median
-> 校验 profile 有效性
-> 记录 Performance Test Data
```

要求：

- 普通 shape、tile、axis 数值差异不算 dispatch path，除非它触发真实代码分支。
- 每个 dispatch path 默认只采一个代表 workload。
- 无效 profile 不能进入诊断。
- `msprof op` 与 NPU event 是不同口径，不能直接横比；调度结构类优化必须先看 event 是否明显回退，若 event 清晰改善则优先保留，若 event 打平则按 `msprof` 和结构证据决胜。

### Phase 2：优化闭环

Phase 2 是多轮闭环。优化点分析不做成一次性前置步骤；每轮都基于当前 best 的最新 profile 重新分析当前现象，并生成多个候选优化点。

每轮执行：

1. 固定本轮 base：当前 best 版本和它的最新 profile。
2. 读取 [iteration-diagnosis.md](references/iteration-diagnosis.md)，基于 base profile 整理当前现象；生成候选优化点时按需查阅 [bottleneck-patterns.md](references/bottleneck-patterns.md)。
3. 从同一个 base 派生多个实验分支：`perf_opt/{op}_opt_v{iter}_{opt_id}.py`。
4. 每个实验分支只改一个主要优化点。
5. 每个实验分支跑 L0 精度回归。
6. 对精度通过的分支，用 `msprof op` 采集目标 kernel 性能；若分支涉及 `Block Dim / num_cores / T.serial / T.Pipelined / multi-buffer` 等调度结构变化，同步采集 NPU event median，并按 [profile-collection.md](references/profile-collection.md) 检查 event 测量质量。
7. 若结构性分支正确性通过且方向有效，先围绕该结构暴露的关键参数做一轮 coarse autotune 或等价手动粗搜；再检查 autotune top-k 与 winner 邻域，必要时做手动/脚本精搜，最后把精搜 winner 作为该结构分支的候选版本复测。
8. 在同一 `(dispatch_path, workload_id)` 内比较 valid 分支；按本轮主指标选择候选 winner。kernel 内部优化默认看 `Task Duration(us)`；调度结构优化要求 event 不明显回退，若 event 清晰改善则优先保留，若 event 打平则用 `msprof Task Duration`、负载均衡和资源占用决胜。
9. 候选 winner 更新为全局 current best 前，必须确认必测 dispatch 没有超过噪声阈值的性能回退；若只在部分 dispatch 提升但其它必测 dispatch 明显回退，不更新全局 current best，并记录 rollback/defer 原因。
10. 若所有分支无提升、无效或阻塞，current best 保持不变。
11. 记录本轮现象、候选优化点、实验分支、性能、精度、必测 dispatch 非回退检查和 winner/rollback 结论。
12. 未满足终止条件则进入下一轮，重新分析当前现象。

终止条件：

- success：达到用户指定性能目标。
- budget_exhausted：达到 `max_rounds` 或 `max_experiments`，默认 `max_rounds=10`，`max_experiments=30`。
- plateau：连续 3 轮没有任何 valid 实验分支带来超过噪声阈值（默认 3%）的提升，且主要结构候选已经得到充分验证；单个配置的 `config_no_gain` 或单个 blocked 分支不能单独证明整类方向无效。
- blocked：剩余候选优化点均因精度失败、编译失败、profile invalid 或实现约束无法继续。
- user_stop：用户要求停止。

### Phase 3：产物收束

1. 选 current best 作为 `perf_opt/{op}.py`。
2. 确认 `perf_opt/opt_log.md` 已完整记录过程和最终结论。

### Phase 4：调优复盘与最终交付

1. 读取 [skill-retrospective.md](references/skill-retrospective.md)。
2. 回看 `perf_opt/opt_log.md` 中的 baseline、多 dispatch 数据、候选优化点、实验分支、winner、rollback、blocked、`config_no_gain` 和 `family_no_gain`。
3. 判断本次调优是否暴露出 skill 流程问题。
4. 判断是否需要提出新的 `BP_xxx`，或更新已有 `BP_xxx`。
5. 把复盘写入 `perf_opt/opt_log.md` 的 `Skill Retrospective` 章节。
6. **回写 pattern-library.md（例外授权）**：本次调优产出的新模式（含代码形态与实测代价）、新代价数据、编译器陷阱新实证/证伪更正，**必须**追加到 `references/pattern-library.md` 对应章节（含任务溯源与工具链版本戳）——该文件是数据积累文件而非流程文档，本条是"不自动修改 skill 文档"禁令的唯一例外。流程/结构层面的改进仍只提 BP proposal，不自动改。
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

当候选性能差异小于噪声阈值、event 曲线呈平区，或同一配置跨 session 漂移明显时，先执行 event 复测和 anchor 漂移检查。不要用单 pass event 对平区内的 `num_cores / block_size` 候选排序。

### Autotune 只用于参数选择

autotune 只负责在给定搜索空间中选参数，不是最终裁判。若主要瓶颈是结构问题，先改结构；结构性分支通过正确性并显示方向有效后，再按 [autotune.md](references/autotune.md) 对该结构做 coarse search，随后检查 top-k 和 winner 邻域，最后用 `msprof op` 和必要的 NPU event 复测最终 winner。

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
- Event Test Data：触发调度结构分析时，记录 NPU event median、重复次数、runner/command、独立 pass 数、anchor 漂移状态、`flat_response/noisy_invalid` 判断和适用 workload。
- Iteration Log：每轮现象、候选优化点、实验分支、latency、event、精度、`config_no_gain/family_no_gain` 范围、必测 dispatch 非回退检查、winner/rollback。
- Autotune Log：若使用 autotune，则记录 search space、best config、正确性、winner 的 `msprof op` 和必要的 `event_quality`。
- Final Summary：best 版本、最终 latency、总提升、中止原因。
- Skill Retrospective：skill 流程问题、建议修改、`BP_xxx` proposal。

产物布局要求：

- `perf_opt/` 顶层只放最终 `{op}.py`、实验分支 `{op}_opt_v*.py`、`opt_log.md`、`profiles/`、`logs/` 和必要 runner/helper 脚本。
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
- iterations: {N}
- primary_metric: {msprof_task_duration_or_event_median}
- baseline_latency: {v} us
- baseline_event_median: {v_or_na} us
- final_latency: {v} us
- final_event_median: {v_or_na} us
- final_event_quality: {valid/flat_response/noisy_invalid/na}
- improvement: {x}%
- stop_reason: {reason}
- skill_retrospective: {none_or_summary}
- bp_proposals: {none_or_list}
- skills_consulted: {paths}
- summary: {one_sentence}
- issues: {none_or_notes}
```
