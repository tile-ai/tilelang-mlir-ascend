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

- Phase 0 加载上下文时读取：[hardware-context.md](references/hardware-context.md)、[pattern-library/INDEX.md](references/pattern-library/INDEX.md)（**必读**：条目索引 + 证伪协议；命中条目精读对应主题文件，或经 `python3 .agents/tools/kb_search.py "<算子族/症状/API/dtype>"` 检索——已验证性能模式/代价表/编译器陷阱版本戳；工具链变更后先跑 `python3 .agents/tools/kb_stale_check.py` 列出待重验条目）
- Phase 1 性能采集时读取：[profile-collection.md](references/profile-collection.md)
- Phase 2 每轮现象分析时读取：[iteration-diagnosis.md](references/iteration-diagnosis.md)
- Phase 2 生成候选优化点时按需读取：[bottleneck-patterns.md](references/bottleneck-patterns.md)
- Phase 2 候选优化点包含 autotune 时读取：[autotune.md](references/autotune.md)（含**实验批处理 runner 模式**：`run_experiments.py` 模板——T-4，见其末节）
- Phase 3 判定设计层天花板时读取：[perf-feedback.md](../_shared/standards/perf-feedback.md)（`[DESIGN_LIMIT]` 触发条件（含参照锚定门槛）与 `perf_feedback.md` 固定 schema——单一事实源）
- Phase 4 调优复盘时读取：[skill-retrospective.md](references/skill-retrospective.md)（产出按 vp_type D/P/R/C 标注 + 证据三件套，供任务终态 `tilelang-skill-evolver` 蒸馏）与 pattern-library [repro/README.md](references/pattern-library/repro/README.md)（ED-B repro 规范——回写新条目时同步沉淀最小可复现代码）

按需参考同类 skill：

- 只有当前算子类型已判断为 cube 时，参考：[tilelang-cube-skill](../tilelang-cube-skill/)
- 只有当前算子类型已判断为 vector 时，参考：[tilelang-vector-skill](../tilelang-vector-skill/)
- 只有当前算子类型已判断为 mix 时，参考：[tilelang-mixcv-skill](../tilelang-mixcv-skill/)

## 主流程

### Phase 0：加载上下文

1. 读取 `{op}.py`、`DESIGN.md` 和 [hardware-context.md](references/hardware-context.md)、[pattern-library/INDEX.md](references/pattern-library/INDEX.md)（命中条目精读主题文件；conductor 调度 prompt 附带的 kb_search 预注入条目同样在本阶段消费）。
2. **版本失效核对（E-5）**：工具链（tilelang 重编译 / CANN 升级）相对条目版本戳有变更时，先跑 `python3 .agents/tools/kb_stale_check.py` 列出待重验条目清单——本任务引用的条目若在清单内，结论标注"待重验"并优先安排一次同口径复测（可用 `python3 .agents/tools/repro_runner.py --filter <条目ID>` 重验有 repro 的条目）。
3. 判断算子类型：`cube / vector / mix`。
4. **首轮必查项（来自 pattern-library，不得跳过）**：
   - **向量化轴与布局重估**：核对当前实现的向量化轴与核内布局——即使上游设计已选定，其结论可能基于旧工具链或未枚举重排布局变体（I/O layout 是契约、核内布局是设计变量；含 ≥2 个非 batch 维的逐元素/窗口/规约类算子须对照 pattern-library layout.md 评估「原生布局+最内连续轴」vs「核内重排布局+高整除性轴（如 C 轴融合转置链）」两条路线）；
   - **编译器陷阱版本戳与 origin_task 核对**：pattern-library traps-*.md 的陷阱结论绑定工具链版本与来源任务（origin_task）——若 tilelang 源码被修改/重编译过，相关结论自动视为待重验，不得直接引用；引用命中条目时在 opt_log 中标注其 origin_task，据此判别可信度；
   - 若 `DESIGN.md` 含 §1.6.3（向量化轴与数据布局决策），本轮调优须对照该决策与实测现象（标量占比、带宽利用率）——现象与决策矛盾时优先重验布局路线；
   - **实验裁决执行（DESIGN 含实验裁决三件套时必做）**：若 `DESIGN.md` §1.6 含「主选 + 备选 + 实验裁决计划」（判定裕度依赖未实证常数的备选方案），A/B 实测是本轮调优的必做项，不是可选项——按裁决计划执行：① 在 `perf_opt/` 下实现备选变体（基准 `{op}.py` 与 wrapper 不动）；② 按计划的代表 shape 与主选同口径对比（msprof op）；③ 实测/反解裁决所依赖的未知常数（如转置吞吐、跨步代价）；④ 按计划判定阈值裁决——备选胜出（全局或按 shape 分片）则采纳备选变体为候选 best，主选胜出则以实测数字固化设计判定；⑤ 实测数据与裁决结论写入 `opt_log.md`；若裁决构成设计层天花板（备选结构性胜出满足 [perf-feedback.md](../_shared/standards/perf-feedback.md) §1 触发条件，或实测推翻设计判定假设），按其 §2 产出 `perf_feedback.md` 并在返回中附 `[DESIGN_LIMIT]` 信号交 conductor 受控路由（附录补记回写 DESIGN.md / 设计修订路径 C）；新实测常数追加回 pattern-library/constants.md（含 repro，ED-C），见 Phase 4。
5. 搜索同类算子或历史优化实现，尤其关注：
   - `T.serial`
   - `T.Pipelined`
   - multi-buffer
   - 片上 buffer 生命周期复用
   - dtype-aware 参数
6. 记录可参考的结构性策略，但不要照搬；必须用当前算子的 profile 验证。

### Phase 1：采集初始 baseline

按 [profile-collection.md](references/profile-collection.md) 建立 benchmark workload 清单并逐 kernel 采集：

```text
展开迁移后 benchmark 的全部参数化案例并分类 tune / smoke / skipped
-> 确认每个案例实际调用的 Stage 3 kernel
-> 对每个 kernel 的全部 tune workload 串行运行 msprof op baseline
-> 校验 profile 有效性
-> 基于实测瓶颈排序并记录理由
-> 逐 workload 记录 Performance Test Data 与 baseline 记录
```

要求：

- TileOPs 迁移/集成场景的性能目标只取 benchmark 中可运行的非 smoke workload；smoke 不做性能采集或调优，但每次合并及最终用于精度回归；skip 记录原因。非 TileOPs 且确无迁移 benchmark 的独立算子使用用户明确指定的性能 workload，并逐 kernel/逐 workload 建立 inventory。
- 不额外枚举 manifest 或测试案例，不提前运行正式 benchmark、接入 wrapper 或新增候选注入接口。TileOPs 原有 op 层分派保持不变，仅用来确定 workload 实际进入哪个 kernel。
- 一个 optimizer subagent 逐 kernel、逐 workload 串行执行；`msprof op` 全局串行。
- 无效 profile 不能进入诊断。
- **每个 tune workload 的 baseline 采集完成后即向 `perf_opt/perf_records.jsonl` 追加一行**（append-only，字段契约见 `_shared/standards/signal-registry.md` §5）；完整清单写入 `perf_opt/workload_inventory.json`，供 gate 4 核对覆盖范围。
- **设计估算 vs 实测偏差行（D-2 回填）**：`DESIGN.md` §1.6.0 含「设计期估算」行时，baseline 采集后在 `opt_log.md` Baseline 段追加**「设计估算 vs 实测」偏差行**（估算下界 vs baseline Task Duration；偏差 >2x 时注明失准项——流量/发射/容量）；该偏差行是 `[DESIGN_LIMIT]` 归因的量化证据（hardware-cost-model.md §3）。

### Phase 2：优化闭环

Phase 2 对每个 kernel 的 tune workload 按 Phase 1 排序逐个执行多轮闭环。排序默认优先实测性能负担大、结构收益明显的大 shape，再处理小 shape、其他 dtype、尾块与资源临界案例；排序只影响先后，不跳过任何 tune workload。每轮基于当前已合并 kernel 的最新 profile 分析现象并生成候选优化点。

**实验批处理 runner（T-4）**：首轮生成 `perf_opt/run_experiments.py`（从模板 [references/run_experiments_template.py](references/run_experiments_template.py) 拷贝并适配实验清单）后，每轮的「实验分支执行」（L0 回归 → msprof 采集 → perf_records.jsonl 追加）交由该脚本批量完成，你每轮只需读它输出的汇总表决策——把「agent 往返驱动实验」变成「脚本执行、agent 读表」（预期单轮交互轮次下降 60%+，并根治大工件会话超限类失败，VP-2026-0021）。脚本模板与适配说明见 [autotune.md](references/autotune.md)「实验批处理 runner」节。

每轮执行：

1. 固定本轮 base：当前 best 版本和它的最新 profile。
2. 读取 [iteration-diagnosis.md](references/iteration-diagnosis.md) 与 [bottleneck-patterns.md](references/bottleneck-patterns.md)（**每轮必读**——模式库是候选优化点的直接来源，且模式库只有在被读的位置上才不会遗忘），基于 base profile 整理当前现象。
3. **先看表再分析**（B2 结构化回流）：从 `perf_opt/perf_records.jsonl` 汇总「候选 vs current best」对比表（Task Duration(us)、AICore 利用率、memory 指标（msprof 可得）、L0 结果）作为本轮分析上下文——把「读散文日志复盘」变成「看表决策」，也直接提升 `[DESIGN_LIMIT]` 设计层归因的证据质量。
4. 从同一个 base 派生多个实验分支：`perf_opt/{op}_round{iter}_{opt_id}.py`。
5. 每个实验分支只改一个主要优化点。
6. 每个实验分支跑 L0 精度回归。
7. 对精度通过的分支，用 `msprof op` 采集目标 kernel 性能；**验证完成后向 `perf_records.jsonl` 追加一行**（含 `parent_id` = 派生来源候选；append-only，禁止改写/删除既有行）。
8. 若结构性分支正确性通过且方向有效，先围绕该结构暴露的关键参数做一轮 coarse autotune 或等价手动粗搜；再检查 autotune top-k 与 winner 邻域，必要时做手动/脚本精搜，最后把精搜 winner 作为该结构分支的候选版本复测。
9. 在同一 `kernel_id + workload_id` 内比较 valid 分支；按 `msprof op Task Duration(us)` 选择该 workload 的 winner；打平时用负载均衡、资源占用和代码复杂度决胜。**<5% 差异与平区决胜（T-2）**：小 kernel（Task Duration <20µs）或逼近带宽/搬运地板的 workload，候选差异 <5% 量级时须先经交错 A/B/A/B 多 run 协议裁决——调用 `python3 .agents/tools/ab_test.py --a-cmd "<baseline bench cmd>" --b-cmd "<candidate bench cmd>" --workdir perf_opt/ab_<round>`，按其 JSON 输出裁决采纳/tie/inconclusive；未过协议的 <5% 增益不得记为 winner。
10. **每个 workload 完成调优后合并一次**：先将 winner 的改动直接合入同一个 `@T.prim_func`；若本 kernel 其他 tune workload 出现可信回退，则尝试在该 kernel 内按可见 shape/dtype/参数设置条件路径。分支的编译期折叠、buffer 作用域、精度与资源限制须用当前工具链实测；不把新分派放到 Python factory、默认配置、op 层或 wrapper。合并后的实际 kernel 须复测本 kernel 的全部 tune workload（性能与精度）和 smoke workload（仅精度），通过后才更新已合并 current best；失败则保留此前版本并记录未能安全合并。后续 workload 从此已合并版出发。
11. 若所有分支无提升、无效或阻塞，current best 保持不变。
12. 记录本轮现象、候选优化点、实验分支、性能、精度、当前 kernel 全部 tune workload 的合并非回退检查、smoke 精度检查和 winner/rollback 结论；**本轮记录必须含第 3 步的结构化对比表**（数据来自 perf_records.jsonl 与 raw profile——gate 4 `S4-OPTLOG-COMPTABLE` 机械校验其存在性）；经 ab_test 裁决的决策附其 JSON 摘要。
13. 未满足当前 workload 的终止条件则进入下一轮；完成时在 inventory 写 `tuning_status=winner_merged/no_gain/merge_blocked`，再处理排序中的下一个 tune workload。只有所有 tune workload 均有结论且最终全量复测完成，才能报告流程已完整执行；若存在 `merge_blocked`、目标未达标或预算中断，须如实报告，不能以其它 workload 的成功代替，也不能把流程完成等同于性能目标达成。

终止条件：

- success：当前 workload 达到用户指定性能目标；若未设逐 workload 数值目标，则以通过核内合并与全量非回退检查的实测最优结果作为该 workload 的结论。完整 Stage 4 还要求所有 tune workload 有调优结论、最终全量记录以及 smoke 精度通过，`merge_blocked` 不得报告为性能目标已达成。
- budget_exhausted：达到 `max_rounds` 或 `max_experiments`，默认 `max_rounds=10`，`max_experiments=30`。
- plateau：连续 3 轮没有任何 valid 实验分支带来超过噪声阈值（默认 3%）的提升，且主要结构候选已经得到充分验证；单个配置的 `config_no_gain` 或单个 blocked 分支不能单独证明整类方向无效。**plateau ≠ 天花板（T-1）**：进入 plateau 判定前，强制做一次**参照结构 diff 检查**（morph ladder 轻量版）——检索同族算子已知最优实现（pattern-library cases.md 参考实现集 + `examples/` 同类 + kb_search），找到参照时对照其结构形态（grid 组织 / 数据流分相位 / 片上驻留策略 / dtype 路径）与本轮 current best 的差异，差异点即下一轮的结构变异候选（不直接照搬，走正常实验分支验证）；未发现结构差异才允许进入终止或 `[DESIGN_LIMIT]` 流程。
- blocked：剩余候选优化点均因精度失败、编译失败、profile invalid 或实现约束无法继续。
- user_stop：用户要求停止。

### Phase 3：产物收束

1. 选各 kernel 已合并 current best 作为 `perf_opt/{op}.py`，对所有 tune workload 在最终文件上重新采集 `phase=final` 记录，并对 smoke 做最终精度回归。
2. `perf_opt/opt_log.md` 的 Final Summary 逐 `kernel_id + workload_id` 列出 baseline、同一最终候选的时延、提升/回退及 profile 证据，列出 smoke/skip 排除与精度结果；最终表的全部时延必须来自同一最终候选。gate 4 依据 `workload_inventory.json` 与 `perf_records.jsonl` 逐项对账。
3. TileOPs 集成算子（算子目录为 `tileops/kernels/{family}/{op_slug}/{op_slug}_kernel/`）：`perf_opt/` 建在该目录下；wrapper 的 baseline/perf_opt 双 import 切换块由 conductor 在回归通过后翻转采纳（perf_opt 默认激活），本 skill 不修改 wrapper。若 tuned kernel 与基准 kernel 的默认参数不同（如 block_size），须在 `perf_opt/{op}.py` 中以模块级常量暴露 tuned 默认值，供 wrapper 切换块成对引用。
4. **设计层天花板判定（[DESIGN_LIMIT]，可选产出）**：读取 [perf-feedback.md](../_shared/standards/perf-feedback.md)，逐条核对触发条件——① 性能天花板由算法/设计层决定（非 tiling/参数可解，归因到 DESIGN.md 具体假设）；② 结构性加速估计 > 2x 或实测与设计假设直接矛盾。**两项同时满足** → 按其 §2 固定 schema（含**参照锚定**章节）产出 `perf_opt/perf_feedback.md`（实测章节必须为 msprof op 口径，反馈结论含建议路由），返回时附 `[DESIGN_LIMIT]` 信号；任一不满足 → **禁止产出**该文件（参数级不足留在迭代内，不触发逆向反馈）。**「当前工具链/硬件上限内不可达」类断言的证据门槛**（2026-09-09 attention expert 三轮闭环实证：第二轮 [DESIGN_LIMIT] 断言 fa4096 结构性地板 ≈125–131µs > 100µs 目标，第三轮两相位重构同 API 达标 98.05µs 推翻）：① 断言前强制检索仓库内同族先例（`examples/`/`testing/` 是否已有同 API 可达结构），找到先例时同口径实测 + morph 式最小增量对照（iteration-diagnosis.md「参考实现对照」）排除「冻结设计族局部地板」误判——实测外推的地板只对实测过的结构族有效（前序实证证据边界，见 negative-claim-evidence.md 第 6 条 / queue VP-2026-0025）；② 否决备选结构的理由须为实测或文档条款，纯推理否决（如「S 需全任务驻留 → flag 预算超限」）标注为未实证假设；③ 已产出的 [DESIGN_LIMIT] 被后续实测推翻时不删改原文（append-only），以「修正附录」回填 perf_feedback.md（结论限定于冻结设计族 + 修正证据表 + 存活硬上限清单 + 根因记录——本轮已验证形态）。

### Phase 4：调优复盘与最终交付

1. 读取 [skill-retrospective.md](references/skill-retrospective.md)。
2. 回看 `perf_opt/opt_log.md` 中逐 kernel、逐非 smoke workload 的 baseline、候选优化点、实验分支、winner、合并/rollback、blocked、`config_no_gain` 和 `family_no_gain`。
3. 判断本次调优是否暴露出 skill 流程问题。
4. 判断是否需要提出新的 `BP_xxx`，或更新已有 `BP_xxx`。
5. 把复盘写入 `perf_opt/opt_log.md` 的 `Skill Retrospective` 章节。复盘产出按 **vp_type（D/P/R/C）** 标注并写全**证据三件套**（evidence + repro + toolchain_stamp），供任务终态 `tilelang-skill-evolver`（`tilelang-skill-evolution` skill）蒸馏与分级合入——D/C 类自动合入（Tier 0），P 类入队等 2 次独立证据（Tier 1），R 类入队等人工审批（Tier 2）。
6. **回写 pattern-library（例外授权 + repro 沉淀，ED-C）**：本次调优产出的新模式（含代码形态与实测代价）、新代价数据、编译器陷阱新实证/证伪更正，**必须**追加到 `references/pattern-library/` 对应主题文件（含任务溯源与工具链版本戳；条目按 INDEX.md §5 front-matter schema 登记）——该目录是数据积累文件而非流程文档，本条是"不自动修改 skill 文档"禁令的唯一例外。**同步 repro 责任（经验与过程文件解耦——经验需要代码时用最少的代码表达）**：新 D 类条目（陷阱实证/代价常数）与**优化模式条目（P 类，含正向/反向）**在知识域 `references/pattern-library/repro/` 写对应最小代码：陷阱/常数类按 ED-B 分级断言（自包含 + 断言）；**优化点 = 慢→快的关键代码更改**（从实验分支/探针/参考 kernel 裁出最小 diff：完整 before/after 对照或 delta 骨架）**+ 优化见解摘要**（机制归因到硬件常数/编译器行为，自包含于条目正文与 repro 头部）——经验不随过程文件或最终调优 kernel 的存亡而失效；**实际跑一遍验证**（本 skill 有 NPU 环境，数据真实性由来源任务负责），条目 front-matter 登记 `repro: repro/<file>.py`，无法当场最小化的结论标 `repro-missing` 并在复盘表注明待补。新硬件常数追加到 `pattern-library/constants.md`（Tier 0，同步 DESIGN 期 roofline 口径，hardware-cost-model.md §3）。流程/结构层面的改进仍只提 proposal，不自动改。**本条任务内回写与终态 evolver 蒸馏互补防双写：已回写的条目在复盘表 evidence 中注明回写位置。**
7. 如果项目流程要求 `Optimize.md`，从最终 `opt_log.md` 摘要最终结果、关键有效优化点、中止原因、遗留问题和复盘摘要。
8. 不自动修改 skill 流程文档（SKILL.md 及 references 除 pattern-library/ 目录外）；只在最终报告里列出建议和 BP proposal。
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

当候选的 `Task Duration(us)` 差异小于噪声阈值或多次复测打平时，先用更大 `launch-count` 复测并比较多次独立运行的 `msprof op` 结果；不要用单次 msprof 结果对平区内的 `num_cores / block_size` 候选排序。**<5% 差异走 `ab_test.py` 交错多 run 协议**（Phase 2 第 10 步，BP_run_state_bimodality 对策——单 run 双态膨胀曾两次误判采纳决策）。

### Autotune 只用于参数选择

autotune 只负责在给定搜索空间中选参数，不是最终裁判。若主要瓶颈是结构问题，先改结构；结构性分支通过正确性并显示方向有效后，再按 [autotune.md](references/autotune.md) 对该结构做 coarse search，随后检查 top-k 和 winner 邻域，最后用 `msprof op` 复测最终 winner。

### 经验结论不要过度泛化

具体策略细节以 [bottleneck-patterns.md](references/bottleneck-patterns.md) 为准。某个优化点在当前环境失败，只能记录为当前 workload / TileLang-NPUIR / CANN / Developer 模式下的实测结论。

### 证伪协议（否定一个 API/模式前必须遵守）⭐

1. **必须用文档合法形态测试**：否定任何 API/模式前，先查 `docs/Tilelang.language/` 确认其合法参数与形式，穷举代表性写法后再下结论。禁止以单一非法形态（如 3-cycle permutation、非文档规定的循环形式）的编译失败否定整个 API 类别——实测教训：曾因此误判"T.transpose 链/C 轴累加不可用"，掩盖了 2–13x 收益（详见 pattern-library/INDEX.md §2 证伪协议）。
2. **编译器约束结论必须绑定工具链版本戳**：记录 tilelang build/commit 信息；工具链变更后旧结论自动视为待重验（`kb_stale_check.py` 机械检测）。
3. **证伪更正须留痕**：推翻旧结论时在 opt_log.md 写明"误判根因 + 合法形态 + 新数据"，并同步更新 pattern-library/traps-*.md 对应条目的 `status` front-matter（overturn 时保留原文 + deprecate 标注）。
4. **诊断信号强制触发换轴分析**：msprof 显示热点段标量执行占比 > 50% 时，强制评估向量化轴/布局重排候选（pattern-library/layout.md），不得只在原轴上微调参数。

## 日志最小要求

`perf_opt/opt_log.md` 至少记录：

- Performance Test Data：每个 tune workload 的目标 kernel、Task Duration、raw profile；另列 smoke/skip 排除原因。
- Iteration Log：每轮现象、候选优化点、实验分支、latency、精度、`config_no_gain/family_no_gain` 范围、该 kernel 全部 tune workload 的合并非回退检查、smoke 精度及 winner/rollback；**每轮含「候选 vs current best」对比表**（Task Duration(us)、AICore 利用率、memory 指标、L0 结果——B2 结构化回流，gate 4 `S4-OPTLOG-COMPTABLE` 校验存在性）。
- Autotune Log：若使用 autotune，则记录 search space、best config、正确性、winner 的 `msprof op` 复测结果。
- Final Summary：每个 kernel 的最终候选版本、每个 tune workload 的 baseline/final 对账表、smoke 最终精度、总体结论与中止原因。
- Skill Retrospective：skill 流程问题、建议修改、`BP_xxx` proposal。

`perf_opt/perf_records.jsonl`：每次 kernel–workload 测量追加一行（baseline 记 round 0），append-only，字段契约唯一出处 `_shared/standards/signal-registry.md` §5；gate 4 `S4-PERF-RECORDS-*` 机械校验。

产物布局要求：

- `perf_opt/` 顶层只放最终 `{op}.py`、实验分支 `{op}_round*.py`、`opt_log.md`、`perf_records.jsonl`、`workload_inventory.json`、`perf_feedback.md`（仅 `[DESIGN_LIMIT]` 时）、`profiles/`、`logs/` 和必要 runner/helper 脚本。
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
- baseline_latency_by_workload: {kernel_id + workload_id -> us}
- final_latency_by_workload: {kernel_id + workload_id -> us}
- improvement_by_workload: {kernel_id + workload_id -> %}
- stop_reason: {reason}
- skill_retrospective: {none_or_summary}
- value_point_proposals: {none_or_list}   # 含 BP_xxx 与 D/P/R/C 各类，逐条带 vp_type
- skills_consulted: {paths}
- summary: {one_sentence}
- issues: {none_or_notes}
```
