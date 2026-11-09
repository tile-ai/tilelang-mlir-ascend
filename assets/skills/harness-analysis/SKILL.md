---
name: harness-analysis
description: 首次分析标杆的实现思路并形成 TileLang 开发方案；后续结合测试反馈、仿真和流水证据定位当前 kernel 问题，交接可验证的修改假设。用于 Operator Harness 的 analysis 节点。
---

# Harness Analysis

先按 [项目管理契约](../harness-new-project/references/project-management.md) 解析 workspace_root、project_id 和 project_dir。用户/TASK 的显式选择优先；多项目不猜测，不使用全局当前项目指针。写入前核对项目 active 与会话 owner，其他会话须先交接；freezing 只允许安全收尾，frozen/archived 交给 harness-restore-project。所有源码修改限项目独立 worktree，环境/源码位置从绑定读取。

## 节点执行边界

当前会话或独立 agent 均可执行本节点。已收到 TASK 的子 agent 直接按 [节点契约](../harness-resume/references/node-contract.md) 读取输入，不递归执行主 agent 的派发流程；直接交互时先按 [harness-resume](../harness-resume/SKILL.md) 恢复或补齐 TASK。记录任务理解与结果，确保不继承聊天的新 agent 也能接手。委派子 agent 只写所分配产物，由主 agent 中继问询并汇总共享记录；直接交互时当前 agent 完成同样的收尾检查后可继续后续节点，不要求新会话。

默认流程是 `establish-baseline → analysis → develop → analysis …`；develop 内完成实现与测试。目标已达成或用户要求停止时收尾，不为循环新增分析。读取 `.workspace/<project>/STATE.md`、TASK 固定的标杆/候选和最近测试结果，明确本次是首次方案分析还是后续问题诊断；不能只凭 run 编号判断首次。

## 首次：分析标杆与开发思路

首次进入本任务的 analysis 时，读取确切 BASELINE ID/修订/摘要、来源覆盖与候选选择依据、选中实现源码和适配、case/输出契约及测量边界；源码与实际测量包无法对应的范围须保留为限制。分析算法阶段与完整调用链、数据组织/布局、并行划分、同步和中间存储，解释逐 shape 分派及其依据；将源码事实、测量观察和待验证推断分开，不从耗时单独猜实现。

结合指定 TileLang/后端能力提出可执行初版方案：哪些思路可复用、哪些需要重新映射、关键正确性风险，以及交给 develop 的一个实现方案和后续验证 case。尚无目标候选是正常情况，不要求先创建 kernel 或先做目标 kernel 仿真。仅离线任务可以分析已有参考并注明未测边界，不强制先建立无关性能标杆；当前任务依赖的标杆缺失/失效时明确交接 establish-baseline。

## 后续：定位当前 kernel 的问题

绑定当前候选与最新测试结果，按精度反例、构建日志、逐 case 性能差距和已有采集，确定本次要解释的问题。首次进入但已有失败候选时，既补齐标杆思路，也分析该候选。已有有效首次分析可直接引用；标杆或契约改变时只重新分析受影响部分。

优先读取 develop 交付的 ANALYSIS_DATA 及其原始上板流水/指标，核对真实 kernel、case、候选/构建、采集配置和覆盖范围。按 [上板数据分析方法](references/analysis-method.md#上板流水与指标) 形成优化判断；已有数据足够时不重复采集。

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

RESULT 记录分析类型、确切输入身份、源码/测试/trace 证据、观察与推断、具体方案或修改假设、未决问题、验证 case 和完成/停止条件；纯源码分析将仿真与 trace 判定写为 not_run 或 not_applicable，不能伪造采集。

正常完成后的主推荐是 [harness-develop](../harness-develop/SKILL.md)，交接上述方案/假设；确有必需采集或能力缺口时，推荐具体 develop collect-only 采集或 prepare-environment 补齐动作，并说明返回本 analysis 的条件。用户要求本节点后停止时只给推荐。已接收且输入仍适用的分析不重复做，下一步按 [统一路由](../harness-resume/references/next-action.md) 选择。

启用搜索树时，沿用 TASK 的 `search_node_id`，在 RESULT（以及 develop 生成的候选清单）保留同一归属；阶段执行者不自行切换策略节点。将可复用发现与证据交给协调者归入关联节点的 hot，区分观测、假设和结论。见 [搜索与知识](../harness-resume/references/search-knowledge.md)。

## Develop 的退化交接

默认接收 develop 已通过全部必选精度、实现符合原方案且有效性能证据指向思路问题的交接。读取退化现象、逐 case 父候选/标杆对比、原始样本、实际实现核对和排除实现缺陷的依据，再形成下一优化方向。精度失败、实现不正确或原因未明不构成该交接条件，应由原 develop 在节点内解决；不将这些未验收结果当成正常进入下一轮的依据。

## 多方案交接

有多个独立优化点时，分别给出 plan_id、输入候选、证据/假设、预期收益、改动范围、风险和逐 case 验证计划；说明方案是互斥比较、可独立尝试还是依赖前一方案结果。允许多个 analysis RESULT 分别形成方案，不为凑并发重复分析同一问题。

协调者接收后，独立方案可按 [并行开发契约](../harness-resume/references/parallel-develop.md) 分配给不同 worktree 的 develop，分别实现并验证。analysis 不自行派发/改写兄弟候选；方案依赖或输入不可比时先顺序推进。
