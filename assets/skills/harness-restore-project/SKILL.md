---
name: harness-restore-project
description: 恢复被冻结或归档的 Harness 项目，核对 worktree、依赖、检查点和作业状态，使项目重新可开发；不自动启动优化或设备测试。
---

# Harness Restore Project

按 [项目管理契约](../harness-new-project/references/project-management.md) 用显式项目名定位 frozen/archived 项目，读取关闭事件与恢复入口。不要按默认 active 列表推断归档项目不存在。

同一 owner 可继续；新会话先核实旧执行者已停止写入，按 claim 的显式交接流程取得归属，不能仅因时间过去就接管。projectctl `inspect` 只读检查已登记仓库；对比关闭快照中的 HEAD、分支和未提交改动，核对实际依赖、环境绑定、证据和暂停 run。

源码/分支缺失或未决作业无法核实，保持原状态并给出具体恢复动作，不 reset/clean/stash 或重投作业。源码、工具链或环境变化时记录影响并使相关旧证据失效；恢复项目不等于这些能力重新通过验证。

完成核实后保存真实恢复检查点，并调用 projectctl `restore --checkpoint <项目内相对路径>`。只恢复 active 状态，原 automatic 保持暂停，不自动开启实验。用户同时明确要求继续开发时，再交给 harness-resume 按当前身份与授权推进。

最后报告恢复结果、变化与证据缺口、下一步。需要迁移目录或清理依赖时另行处理，本入口不承担自动历史迁移。
