---
name: harness-status
description: 只读汇报 Operator Harness 当前任务的进展、候选与证据、阻塞和预算，并推荐一个下一步。用于 status、查看进展或询问接下来做什么，不自动执行。
---

# Harness Status

先按 [项目管理契约](../harness-new-project/references/project-management.md) 解析 workspace_root、project_id 和 project_dir。用户/TASK 的显式选择优先；多项目不猜测，不使用全局当前项目指针。本入口只读，可列出多个项目摘要；冻结/归档项目仅在显式选择或查看历史时展开，不取得 owner。

使用 [统一下一步规则](../harness-resume/references/next-action.md) 读取当前项目记录，判断当前目标、实际证据与最值得推进的动作。用户明确选择的任务优先于 STATE 的旧指针。

默认不修改任何项目记录，不创建 TASK，不启动节点、环境探针或作业，不改变 automatic 状态。可以读取已有结果核实摘要，但把“尚未接收的交付”与已接收结果分开；用户额外要求实时状态时才按已有约定做有限只读查询。

活跃 automatic 中的普通查询报告后继续原模式；独立查询者只报告，不接管。若用户还明确要求“先停下/别继续”，由协调流程先处理该模式指令，再做只读报告，见 [模式切换](../harness-resume/references/next-action.md#6-模式切换与接管)。

简洁回报：

1. 当前任务与目标，已完成什么、目前在哪一步。
2. 当前候选与最佳已验证候选，精度/性能/仿真各自证据或未测项。
3. 在途任务、阻塞与预算；automatic 的状态和已用步数（若存在）。并行时逐成员列出方案、worktree、候选、run/agent/job、测试状态与设备等待，不只显示最后一个 run。
4. 推荐下一步做什么、为什么、有哪些前置缺口；提示可用 `harness-next` 执行该步或 `harness-automatic` 连续推进。

附能核查的记录链接，说明状态来自记录还是实时查询。缺少 STATE/任务选择时报告缺口并给出最小建议；不自动初始化，不假定记录缺失等于没有在途任务。不要给无依据的完成百分比。
