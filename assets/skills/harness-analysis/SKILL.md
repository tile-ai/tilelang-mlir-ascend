---
name: harness-analysis
description: 针对已有 TileLang kernel，结合标杆、移植差异、测试反馈、仿真和流水证据定位缺陷，交接可验证的修改假设。无候选时先移植标杆，用于 Operator Harness 的 analysis 节点。
---

# Harness Analysis

按 [项目管理契约](../harness-new-project/references/project-management.md) 选择项目并核对写入归属；从 STATE 和当前 TASK 接续。

## 节点执行边界

按 [节点契约](../harness-resume/references/node-contract.md) 执行收到的 TASK；直接交互先用 resume 补齐任务。节点写所属产物，协调者接收并更新共享记录。

默认流程是 `establish-baseline → port-baseline → analysis → develop → analysis …`；已有适用候选时跳过移植。目标已达成或用户要求停止时收尾，不为循环新增分析。读取 `.workspace/<project>/STATE.md`、TASK 固定的候选、标杆和最近测试结果，明确当前实现的待解释问题。

## 输入必须有实际候选

绑定实际 TileLang 候选、case/输出契约、确切 BASELINE 身份及最新测试结果。新生成的初版同时读取 PORTING，了解参考阶段与候选的对应、能力替代（尤其 AICPU）及性能差距。已接入的现有候选没有移植记录是正常情况，直接核对实际实现与证据，不补造历史。

没有候选时推荐 [标杆移植](../harness-port-baseline/SKILL.md)，不把只研究参考算子写成本节点已完成当前 kernel 诊断。标杆用于对照解释候选，不再单独承担初版方案生成；仍可在本节点读取参考源码、比较布局/划分/同步/融合策略。当前相对性能判断需要的标杆缺失/失效时交接 establish-baseline，已授权的离线候选分析可以继续并注明限制。

## 定位当前 kernel 的问题

按构建日志、逐 case 性能差距和已有采集确定问题。初版慢于标杆直接进入诊断，不要求先证明思路失败；已有历史失败候选或用户明确请求诊断时可分析反例，但不能据此接收当前 port-baseline/develop 内的精度失败。已有分析按输入身份复用，标杆或契约改变时只分析受影响部分。

优先读取 port-baseline/develop 交付的 ANALYSIS_DATA 及其原始上板流水/指标，核对真实 kernel、case、候选/构建、采集配置和覆盖范围。按 [上板数据分析方法](references/analysis-method.md#上板流水与指标) 形成优化判断；已有数据足够时不重复采集。

按问题需要执行仿真和流水分析，定位源码/IR/生成代码中的问题，提出下一次 develop 的具体修改假设、预期效果与验证办法；本节点不修改正式候选。明确反例已足以定位时可做简短分析，不强制采集。无动态证据时静态结论标为假设，不能虚构瓶颈或收益。

发现相关实现/版本、先前调查遗漏、工具链变化或连续优化无进展时，按 [标杆来源复查规则](../harness-establish-baseline/references/source-investigation.md) 给出有范围的 establish-baseline 任务建议；不自行更换分母。既有标杆是否失效按实际身份与条件判断，不因远端出现新提交就要求整条优化链重跑。

## 按实际分析方式绑定环境

执行新仿真时使用 TASK 固定的 **simulation 专属约定快照**（旧 trace 分析使用其历史身份）；实时 ENVIRONMENT 索引只核对变化与撤权，不替换本次绑定，按 [环境复用规则](../harness-prepare-environment/references/environment-contract.md) 核对必需能力。仿真有自己的机器/容器、工具链、运行限制、授权和预算；上板环境可用或允许测试，不代表允许仿真。只读源码或已有 trace 不要求新仿真环境 ready。新增实卡流水采集由 develop 的 collect-only 模式 按 device 约定完成后返回 analysis；不得借 analysis 扩大设备权限。

沿用该约定，不重复环境问询。仿真未启用、未验证或禁止运行时记录具体状态，必要时在 RESULT 建议主 agent 派发环境准备补齐该节点；不能把任务自动转到上板机器仿真。分析已有产物可继续，但必须核对来源，不能当成本轮新采集。

## 明确本次问题与采集范围

先说明要验证的假设、相关 case、需要的输出或 trace 指标。使用已记录的 simulator 或项目工具及对应版本文档确定命令，只采集能回答问题的有限 case。源码/二进制、ABI、launch、输入、模型/架构、采集配置均按 [证据记录](../harness-resume/references/execution-records.md) 绑定。

若仿真构建链与实卡不同，分别保存产物及共同的 DSL 来源；不能将不同编译产物当成完全相同的流水。需要创建小复现时记录与完整算子的差别，不能用子问题结果替代整算子结果。

## 提交与检查

使用实际可用的 simulator CLI 或 MCP。MCP 尚未接入时直接使用已验证入口，不假造工具存在。长任务保存作业 ID、日志和回收方式；按 simulation 约定限制 CPU、并发、时长和输出大小。超时或断线后先查询原作业，保留 partial 产物，不盲目重放。

依次检查进程退出状态、模型错误、输出是否完整及数值比较、trace 是否覆盖所声明范围、解析工具是否成功。程序退出 0 或生成了目录均不足以判仿真成功。数值规则沿用任务契约，不能无条件拒绝契约允许的 NaN/Inf。

## 从证据到假设

使用 [分析方法](references/analysis-method.md) 将观测、模型解释和优化建议分开，引用具体 trace 区间、指令/事件或统计文件。单位与统计范围要明确，模型频率未知时不换算 cycle 为设备时间。

输出错误或 trace 残缺时，可做局部诊断并注明可见范围，但不做完整 kernel 性能结论。模型不支持或复现持续超时，记录限制并停止预算外重试；需要实卡佐证时在 RESULT 建议主 agent 派发 develop collect-only 任务，使用上板自己的约定，不自动扩大权限。

在本 run 保存原始产物、解析命令、RESULT 与建议下一实验，由主 agent 接收并更新共享记录。仿真是可选时，失败不抹去已有实卡结果；任务明确要求仿真验收时保留未达成项。任何优化收益最终用同版本候选的实卡精度与性能证明。

## 交付与下一节点

RESULT 记录具体候选与问题、确切输入身份、源码/测试/trace 证据、观察与推断、修改假设、未决问题、验证 case 和完成/停止条件；纯源码分析将仿真与 trace 判定写为 not_run 或 not_applicable，不能伪造采集。候选继续遵守不引入 AICPU 的项目能力边界，不将新增 AICPU 支持交给 develop。

正常完成后的主推荐是 [harness-develop](../harness-develop/SKILL.md)，交接上述方案/假设；确有必需采集或能力缺口时，推荐具体 develop collect-only 采集或 prepare-environment 补齐动作，并说明返回本 analysis 的条件。用户要求本节点后停止时只给推荐。已接收且输入仍适用的分析不重复做，下一步按 [统一路由](../harness-resume/references/next-action.md) 选择。

启用搜索树时，沿用 TASK 的 `search_node_id`，在 RESULT（以及 develop 生成的候选清单）保留同一归属；阶段执行者不自行切换策略节点。将可复用发现与证据交给协调者归入关联节点的 hot，区分观测、假设和结论。见 [搜索与知识](../harness-resume/references/search-knowledge.md)。

## Develop 的退化交接

默认接收 develop 已通过全部必选精度、实现符合原方案且有效性能证据指向思路问题的交接。读取退化现象、逐 case 父候选/标杆对比、原始样本、实际实现核对和排除实现缺陷的依据，再形成下一优化方向。精度失败、实现不正确或原因未明不构成该交接条件，应由原 develop 在节点内解决；不将这些未验收结果当成正常进入下一轮的依据。

## 多方案交接

有多个独立优化点时，分别给出 plan_id、输入候选、证据/假设、预期收益、改动范围、风险和逐 case 验证计划；说明方案是互斥比较、可独立尝试还是依赖前一方案结果。允许多个 analysis RESULT 分别形成方案，不为凑并发重复分析同一问题。

协调者接收后，独立方案可按 [并行开发契约](../harness-resume/references/parallel-develop.md) 分配给不同 worktree 的 develop，分别实现并验证。analysis 不自行派发/改写兄弟候选；方案依赖或输入不可比时先顺序推进。

较大实验前使用 [执行控制中的机制查重](../harness-resume/references/execution-control.md) 对照同机制成功/失败/关闭节点，保存 search-plan 的新增依据、可证伪条件、保留/淘汰标准及简化计划。已有叠加优化按预算选有价值的移除对照，收益相当时优先简单实现，不强制为凑表采集。
