---
name: harness-archive-project
description: 冻结或归档已选定的 Harness 项目，保存源码、依赖和任务恢复现场，检查执行者与在途作业；保留目录和未提交改动，不自动清理或提交。
---

# Harness Archive Project

按 [项目管理契约](../harness-new-project/references/project-management.md) 确定工作目录、project_id 与 owner，先读取 project.json、STATE、AUTOMATIC、资源账本及在途 TASK。用户说暂停项目时使用 frozen；结束/归档项目时使用 archived。含义不明确且会影响范围时只补问该项。

使用同契约的 projectctl `begin-close --target frozen|archived` 保存 freezing 状态，再通知原执行者停止新增动作。当前不可中断调用先按约定收尾；不能只凭会话结束推断设备作业退出。仅按既有授权查询/回收，不杀其他用户进程，不扩大设备许可。

保存真实生命周期检查点、源码/依赖身份、未提交改动、未决问题与恢复入口。未终态但安全暂停的 run 保留原 TASK/检查点并列入 paused_runs，不伪造 failed/completed。存在未知作业、未结算资源或仍写入的执行者时保持 freezing，报告具体阻塞。

条件满足后调用 `finish-close --checkpoint <项目内相对路径>`。工具保留仓库快照并更新状态；不会删除源码、构建或依赖，不移动目录、不自动提交、推送或合并。它只核对记录及本地 Git 状态，外部作业检查仍由实际执行工具提供证据。

最后报告 frozen/archived 是否已完成、未决项、原位保存位置和 harness-restore-project 恢复入口。归档不改变精度/性能结论。
