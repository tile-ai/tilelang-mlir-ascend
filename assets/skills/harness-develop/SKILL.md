---
name: harness-develop
description: 根据已明确的算子契约编写、修复或优化昇腾 TileLang 候选，维护独立参考、case 和可复现编译产物，并交接实卡验证。用于 Operator Harness 的算子编写节点。
---

# Harness Develop

先按 [项目管理契约](../harness-new-project/references/project-management.md) 解析 workspace_root、project_id 和 project_dir。用户/TASK 的显式选择优先；多项目不猜测，不使用全局当前项目指针。写入前核对项目 active 与会话 owner，其他会话须先交接；freezing 只允许安全收尾，frozen/archived 交给 harness-restore-project。所有源码修改限项目独立 worktree，环境/源码位置从绑定读取。

## 节点执行边界

当前会话或独立 agent 均可执行本节点。已收到 TASK 的子 agent 直接按 [节点契约](../harness-resume/references/node-contract.md) 读取输入，不递归执行主 agent 的派发流程；直接交互时先按 [harness-resume](../harness-resume/SKILL.md) 恢复或补齐 TASK。记录任务理解与结果，确保不继承聊天的新 agent 也能接手。委派子 agent 只写所分配产物，由主 agent 中继问询并汇总共享记录；直接交互时当前 agent 完成同样的收尾检查后可继续后续节点，不要求新会话。

按 TASK 输入清单读取当前任务、已接收的 analysis 方案/诊断、父候选和相关实验，使用 TASK 固定的 development 约定快照；实时 ENVIRONMENT 索引只检查漂移/撤权，不重新替换绑定。默认由 analysis 交接首次标杆思路或后续修改假设；缺少适用分析时按统一路由先进入 [harness-analysis](../harness-analysis/SKILL.md)。用户明确只要求编写时遵循该范围并记录所据方案与未验证项。缺少环境信息时只补当前动作所需能力，允许继续授权内的离线编写。

需要性能比较时，读取 TASK 固定的 BASELINE 记录，复用已验收的参考、case 和协议；缺少/失效时建议 [标杆建立](../harness-establish-baseline/SKILL.md)。仅编写或精度任务仍可在明确范围内创建独立参考，不以未测性能标杆阻塞授权内编写；不得把候选的自测时间直接发布为标杆。

## 建立可检验的起点

确认功能语义、shape/动态性、dtype/累加、layout/stride、尾部、原地/别名、非法输入、误差与性能目标。任务中未确定且会改变实现的事项先问清，不从示例的固定尺寸推断通用支持。

检查独立参考与 case：小型可手算样例、边界输入和数学不变量；随机输入保留 seed，必要时保存实际输入。正确性参考和性能基线可以不同。修复参考、容限或 case 需要向主 agent 提交变更提案，接收并更新任务后再执行受影响验证，不能为了候选通过而同时放宽验收。

## 按目标版本编写

先读取 TASK 固定的 [项目契约](../harness-new-project/references/project-contract.md)，遵守本次目标仓库的 API、注册、lint、测试/打包与依赖功能约束。规则与当前源码冲突时保留依据并交协调者修订，不能等到发布前才调查基本集成要求。

使用 [TileLang 开发与定位](references/tilelang-workflow.md)，以环境中绑定的 TileLang 仓库/revision、Ascend dialect、目标架构及后端为依据查 API 和示例。首批语言来源是 tile-ai/tilelang；不要默用旧 NPUIR 分支、OpenTile 方言或其他后端同名 API。

同时读取 [算子生成补充](../tilelang-ascend-generation/SKILL.md)，将适用的阶段拆分、VF 选择、存储预算和流水方法用于候选；在现有 RESULT 中对应到源码/生成代码，说明相关方法未采用的具体原因。

先实现覆盖契约的最小版本。资源布局、尾部 mask、同步、Cube/Vector 分工和 launch 参数都必须与所选目标一致。一次优化围绕一个可验证假设，记录父候选、改变和预期影响。重试受任务预算限制；只有新证据或条件变化时才重复失败路线。

编译/lowering 必须按本次开发约定执行。目标产物若需要在上板或仿真环境编译，交给相应节点，并携带其环境绑定；开发环境的成功不能替代另一工具链编译。无需每次重新询问已确定的环境。

## 交付候选与验证任务

首次候选交给 device-test 前完成项目检查计划中适用的低成本项，按 [PRECHECK 模板](assets/templates/precheck.md) 保存输入身份、实际命令、判定与日志；后续只重跑受改动影响的检查。失败/未运行/工具缺失不写通过，自动修复改动后重新固定候选。CPU 可发现的集成错误先修复；必须先设备诊断的例外在 TASK 说明范围，不将诊断当发布验收。高成本检查仍受环境和预算约束。

在 `.workspace/<project>/runs/<实验编号>/` 保存 [证据记录](../harness-resume/references/execution-records.md)：候选源码快照或 commit 加补丁、依赖/配置摘要、参考/case/协议身份、命令和日志、编译产物及入口/ABI/launch 信息。多 kernel 算子记录整条调用链；JIT 首次运行会触发的编译也计入预算与记录。

使用 [候选模板](../harness-resume/assets/templates/candidate.md) 交付 CANDIDATE，包含可重建源码及下一节点的具体入口。正常交付后的主推荐是 [harness-device-test](../harness-device-test/SKILL.md)，验证新候选的完整输出和任务要求的性能；编译通过只标编译通过。遇到阻塞实现的构建错误或分析假设不成立时，保留日志并推荐 analysis 定位，不在本节点无依据反复调参。后续修改引用具体分析、反例与候选，不依赖之前聊天。

在 RESULT 说明当前候选、尚未验证部分及下一步，由主 agent 接收后更新共享记录与最佳候选。用户只要求编写时完成候选与交接说明，不自行扩大到未授权环境操作。

启用搜索树时，沿用 TASK 的 `search_node_id`，在 RESULT（以及 develop 生成的候选清单）保留同一归属；阶段执行者不自行切换策略节点。将可复用发现与证据交给协调者归入关联节点的 hot，区分观测、假设和结论。见 [搜索与知识](../harness-resume/references/search-knowledge.md)。
