---
name: harness-next
description: 根据 Operator Harness 项目记录直接执行当前任务的一个下一步，完成必要接收与状态更新后返回。用于 next、推进下一步；不自动连续跑完所有阶段。
---

# Harness Next

按 [项目管理契约](../harness-new-project/references/project-management.md) 选择项目并核对写入归属；从 STATE 和当前 TASK 接续。

这是执行入口。按 [统一下一步规则](../harness-resume/references/next-action.md) 读取记录、核对当前目标和身份，选择一个可完成的动作并直接推进，不以复述计划结束。

普通的“执行下一步”不再询问是否开始；只问实际缺失的语义、环境选择或操作范围。环境、授权与预算已确定则复用。用户只要求上板精度时不能擅自加性能或仿真。

使用 [harness-resume](../harness-resume/SKILL.md) 的 TASK/RESULT、节点路由、额度预留和接收流程。已有在途任务先恢复，已有待接收结果先接收，不盲目创建重复 run。当前 agent 或委派 agent 均可执行；同会话连续工作不受限制。

已有 automatic 或用户中途切换模式时，按 [模式切换与接管](../harness-resume/references/next-action.md#6-模式切换与接管) 先保存单步意图及其绑定的动作。另一协调者仍活跃时交给它处理，不能另开执行者；单步请求不增加设备额度，暂停后的问题回答也不恢复连续模式。

一次推进到所选动作的完成或明确等待/阻塞边界，完成本步共享记录更新后返回。若需要用户回答，实际提出对应问题并保存 needs_input；目标已完成就汇总，不自动选择新任务。

最终说明本步动作、产物、实际结果/未测项、阻塞（若有）和新的推荐下一步。给建议不等于继续第二步；只有用户另有连续执行要求时才切换到 [automatic](../harness-automatic/SKILL.md)。
