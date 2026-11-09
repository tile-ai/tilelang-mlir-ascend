---
name: harness-resume
description: 从项目记录接续 Operator Harness 算子任务，支持同一会话连续推进或交给独立 agent，维护可独立恢复的节点输入、结果和共享状态。
---

# Harness Resume

先按 [项目管理契约](../harness-new-project/references/project-management.md) 解析 workspace_root、project_id 和 project_dir。用户/TASK 的显式选择优先；多项目不猜测，不使用全局当前项目指针。写入前核对项目 active 与会话 owner，其他会话须先交接；freezing 只允许安全收尾，frozen/archived 交给 harness-restore-project。所有源码修改限项目独立 worktree，环境/源码位置从绑定读取。

目标是让当前 agent 与全新 agent 都能从记录理解任务、执行方式和约束。允许用户在同一会话连续完成多个节点；不强制新 agent 或新会话。先读 [节点交接契约](references/node-contract.md)，按需使用六份模板。[automatic](../harness-automatic/SKILL.md) 已提供当前宿主内的连续协调模式，不把其编排策略施加到普通交互工作。

已收到 TASK 的子 agent 直接执行指定节点 skill，不递归执行本技能的协调/派发步骤。委派时主 agent 负责协调，子 agent 只执行分配节点。直接交互时当前 agent 顺序完成执行、收尾检查和共享记录更新，无须额外角色审批。

## 入口选择

仅查看进展用 [status](../harness-status/SKILL.md)，只读且推荐下一步；直接推进一次用 [next](../harness-next/SKILL.md)；连续推进当前目标用 [automatic](../harness-automatic/SKILL.md)。三者共用 [下一步选择](references/next-action.md)，不复制选题规则。resume 继续作为恢复、执行和接收的共用流程，节点子 agent 不调用协调入口递归编排。

## 搜索树与知识

算子探索使用 [搜索树与 hot/cooled 知识](references/search-knowledge.md)：策略节点包含多轮 analysis/develop，阶段 node 与 search_node_id 分开。协调者用随包 searchctl.py 维护树和证据引用；新任务初始化，已有任务在无在途工作时接入，不伪造历史。交付时整理 hot，只有真实人工批准的准确修订可以晋升 cooled；发布独立授权。

## 恢复与选题

从 `.workspace/<project>/STATE.md` 开始，按链接读取需求、相关实验、决定和在途任务。用 [工作记录与交接](references/working-context.md) 区分当前候选、最佳已验证候选和未验证项，不从源码推断历史结果。

读取 [项目契约](../harness-new-project/references/project-contract.md) 的确切修订，检查本次仓库规则、早期检查、交付证据与授权范围；旧 PROJECT 只补当前动作必需的缺口，不重新询问已有决定。

先确认已有执行者和外部 job 是否仍活跃，未确认结束不得重复执行或提交。根据目标和预算选择下一项具体工作；重复失败路线需要新的依据或条件变化。用户指令改变范围时先写入项目决定和相关需求，不能让后续执行者依赖聊天。

## 准备并执行节点

按 [TASK 模板](assets/templates/node-task.md) 为有明确目标的节点工作写记录，绑定输入快照、工具入口/步骤、环境/权限、本次预算、可写范围、交付和停止条件。短任务可以简短填写，阶段内连续操作不逐次建模板；只补齐当前工作必需的缺口。

- 环境问询、能力缺口或环境变更：[harness-prepare-environment](../harness-prepare-environment/SKILL.md)。
- 建立、选择或更新正确性参考与性能标杆：[harness-establish-baseline](../harness-establish-baseline/SKILL.md)。
- 首次标杆实现思路分析，或后续 kernel 问题定位、仿真和流水分析：[harness-analysis](../harness-analysis/SKILL.md)。
- 根据已接收分析编写、修复或优化候选，并在同一任务内完成精度和性能验证：[harness-develop](../harness-develop/SKILL.md)。
- 单独精度/性能、诊断采集或作业恢复：[harness-develop](../harness-develop/SKILL.md) 的 verify-only/collect-only 模式。

标杆建立后默认进入 `analysis → develop → analysis …`，达成目标即结束。分析是否完成以其绑定的输入和已接收结果判断，旧的“下一步 develop”提示不能跳过新规则要求的首次分析；详见统一路由。

解析本节点环境为确定的 ID/修订/快照；上板与仿真授权和验证状态独立。首次准备可以携带未知项，不捏造已验证环境。按资源账本分配本次额度，不能以旧环境余额覆盖实际占用。

当前 agent 可以继续执行，也可按需要委派。委派给全新 agent 时仅传工作目录根、TASK 路径和读取指令，实质约束必须先落盘，记录实际 agent ID。上下文过长时可建议用户新开会话；用户继续当前会话或没有委派工具时，仍沿相同记录推进，不作为阻塞条件。

## 问询与中断

用户问题、真实回答及作用域落到 RESULT/DECISIONS。委派的子 agent 经主 agent 中继，直接交互由当前 agent 完成；不因已有答案重新问整套环境。普通补充与暂停恢复可以继续原任务；改变冻结目标/范围/预算时建立新 TASK，执行者不必更换。详见 [问询与恢复](references/node-contract.md#6-问询检查点和中断)。

恢复不是重跑。先读检查点与外部 job 身份，查询、回收或确认取消；unknown 作业保持预算占用。更换执行者时确认原执行者已停止写入，再在 STATE 记录接管。

模式切换先落盘再等待，暂停须传给活跃节点；接管协调者与替换节点是两件事，按 [共用切换规则](references/next-action.md#6-模式切换与接管) 处理。检查点恢复与接收中断分别按节点契约恢复执行或补齐对账，不能重复派发、结算。

## 收尾与下一步

项目交付时按 [DELIVERY 模板](assets/templates/delivery.md) 保存 `.workspace/<project>/deliveries/<delivery-id>/DELIVERY.md`，逐仓库对照实测输入与最终发布输入，列出可复用、待补测、未验证的范围及理由。静态一致不冒充发布版实卡重测；目标 base 更新要检查集成影响。提前约定的 required/diagnostic 分类继续有效，不能无依据追加附件门槛。先完成授权内可审阅的源码、检查和 PR/MR 正文，发布动作分别核对已有授权；缺许可只阻塞对应动作，不重复问已有许可。

按 TASK 完成条件检查 RESULT、产物与身份。委派时由主 agent 接收，直接交互由当前 agent 完成同样检查；completed 不等于精度或性能通过；develop 只有全部必选精度通过且性能退化排除实现问题后才可接收。精度失败/实现问题在原节点修复，预算阻塞留检查点；其他缺失或矛盾时保留原结果，建立具体补充任务，不用换 agent 来代替修复信息缺口。

协调角色统一更新 STATE、EXPERIMENTS、DECISIONS、环境索引、发布约定和资源账本；委派子 agent 不直接改共享文件。接收环境提案保留旧修订，接收候选检查可重建清单，更新最佳候选须有任务要求的实际证据。

完成记录后可在同一会话继续用户已授权的下一节点，或交给其他 agent。最终回复与 RESULT 都说明结果、差距和一个主推荐：具体节点/skill、原因与证据、依赖/缺口、完成或停止条件。即使用户要求本节点后停止，也要给出建议而不执行；没有适用节点时明确指出流程缺口。记录检查不替代尚未实现的程序 gate。

多个已接收的独立分析方案按 [并行开发契约](references/parallel-develop.md) 建立隔离 worktree 与有界 develop 批次。共享记录仍由单一协调者更新；子任务分别实现并测试，不共用源码或可写缓存。
