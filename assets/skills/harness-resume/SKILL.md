---
name: harness-resume
description: 从项目记录接续 Operator Harness 算子任务，支持同一会话连续推进或交给独立 agent，维护可独立恢复的节点输入、结果和共享状态。
---

# Harness Resume

目标是让当前 agent 与全新 agent 都能从记录理解任务、执行方式和约束。允许用户在同一会话连续完成多个节点；不强制新 agent 或新会话。先读 [节点交接契约 v1](references/node-contract.md)，按需使用五份模板。[automatic](../harness-automatic/SKILL.md) 已提供当前宿主内的连续协调模式，不把其编排策略施加到普通交互工作。

已收到 TASK 的子 agent 直接执行指定节点 skill，不递归执行本技能的协调/派发步骤。委派时主 agent 负责协调，子 agent 只执行分配节点。直接交互时当前 agent 顺序完成执行、收尾检查和共享记录更新，无须额外角色审批。

## 入口选择

仅查看进展用 [status](../harness-status/SKILL.md)，只读且推荐下一步；直接推进一次用 [next](../harness-next/SKILL.md)；连续推进当前目标用 [automatic](../harness-automatic/SKILL.md)。三者共用 [下一步选择](references/next-action.md)，不复制选题规则。resume 继续作为恢复、执行和接收的共用流程，节点子 agent 不调用协调入口递归编排。

## 恢复与选题

从 `.workspace/STATE.md` 开始，按链接读取需求、相关实验、决定和在途任务。用 [工作记录与交接](references/working-context.md) 区分当前候选、最佳已验证候选和未验证项，不从源码推断历史结果。

先确认已有执行者和外部 job 是否仍活跃，未确认结束不得重复执行或提交。根据目标和预算选择下一项具体工作；重复失败路线需要新的依据或条件变化。用户指令改变范围时先写入项目决定和相关需求，不能让后续执行者依赖聊天。

## 准备并执行节点

按 [TASK 模板](assets/templates/node-task.md) 为有明确目标的节点工作写记录，绑定输入快照、工具入口/步骤、环境/权限、本次预算、可写范围、交付和停止条件。短任务可以简短填写，阶段内连续操作不逐次建模板；只补齐当前工作必需的缺口。

- 环境问询、能力缺口或环境变更：[harness-prepare-environment](../harness-prepare-environment/SKILL.md)。
- 编写、修复或优化候选：[harness-develop](../harness-develop/SKILL.md)。
- 实卡精度、性能或作业恢复：[harness-device-test](../harness-device-test/SKILL.md)。
- 仿真、已有 trace 分析或作业恢复：[harness-simulate](../harness-simulate/SKILL.md)。

解析本节点环境为确定的 ID/修订/快照；上板与仿真授权和验证状态独立。首次准备可以携带未知项，不捏造已验证环境。按资源账本分配本次额度，不能以旧环境余额覆盖实际占用。

当前 agent 可以继续执行，也可按需要委派。委派给全新 agent 时仅传项目根、TASK 路径和读取指令，实质约束必须先落盘，记录实际 agent ID。上下文过长时可建议用户新开会话；用户继续当前会话或没有委派工具时，仍沿相同记录推进，不作为阻塞条件。

## 问询与中断

用户问题、真实回答及作用域落到 RESULT/DECISIONS。委派的子 agent 经主 agent 中继，直接交互由当前 agent 完成；不因已有答案重新问整套环境。普通补充与暂停恢复可以继续原任务；改变冻结目标/范围/预算时建立新 TASK，执行者不必更换。详见 [问询与恢复](references/node-contract.md#6-问询检查点和中断)。

恢复不是重跑。先读检查点与外部 job 身份，查询、回收或确认取消；unknown 作业保持预算占用。更换执行者时确认原执行者已停止写入，再在 STATE 记录接管。

模式切换先落盘再等待，暂停须传给活跃节点；接管协调者与替换节点是两件事，按 [共用切换规则](references/next-action.md#6-模式切换与接管) 处理。检查点恢复与接收中断分别按节点契约恢复执行或补齐对账，不能重复派发、结算。

## 收尾与下一步

按 TASK 完成条件检查 RESULT、产物与身份。委派时由主 agent 接收，直接交互由当前 agent 完成同样检查；completed 不等于精度或性能通过。缺失或矛盾时保留原结果，建立具体补充任务，不用换 agent 来代替修复信息缺口。

协调角色统一更新 STATE、EXPERIMENTS、DECISIONS、环境索引、发布约定和资源账本；委派子 agent 不直接改共享文件。接收环境提案保留旧修订，接收候选检查可重建清单，更新最佳候选须有任务要求的实际证据。

完成记录后可在同一会话继续用户已授权的下一节点，或交给其他 agent。向用户说明结果、差距与下一步；记录检查不替代尚未实现的程序 gate。
