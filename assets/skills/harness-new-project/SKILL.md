---
name: harness-new-project
description: 通过连续追问帮助用户明确新的昇腾 TileLang 算子项目目标、范围和验收条件，并在 .workspace 中建立项目与路线图。用于“初始化算子项目”或 harness-new-project。
---

# Harness New Project

将用户的想法整理成可继续工作的项目。首批语言为 TileLang，复用会话和文档中已有的决定。

## 了解起点

按 [项目管理契约](references/project-management.md) 确定 workspace_root；使用用户指定的工作目录，未指定且当前上下文明确时使用该目录，不自动提升到 Git 根。先只读列举 project.json，区分创建新项目和继续已有项目。一个项目可包含多个算子，当前不创建父子项目。

- 只有用户选择继续的 active 项目才沿用已有计划；多个项目时不按最近修改时间猜测，frozen/archived 交给恢复入口。
- 已有代码且尚未梳理时，按同级 [harness-onboard](../harness-onboard/SKILL.md) 了解现状，再继续规划。
- 新项目从用户的想法开始。文件不齐时按需要补充，不重新初始化整套记录。

## 连续追问

读取并使用 [追问指南](references/questioning.md)。沿上一轮回答中的痛点、含糊目标和取舍继续深入，用实际场景和例子检验理解。用户想自由讲述时停止选项式提问，先听完再继续。

理解充分后复述目标与范围，让用户选择进入规划或继续探索。已有“直接生成 / 就这样继续”的指令时沿用。研究只围绕当前需要查清的问题展开，材料按需放入 `.workspace/<project>/research/`。需要落盘讨论草稿前先确定项目名；名字未定时保持对话，不把不同项目草稿混入共享根目录。

## 建立项目记录

与用户敲定稳定 project_id 和展示名称，核对重名与保留名。用项目管理契约的 projectctl create 建立元数据和本会话 owner；按需使用 add-repo 为每个可写源码仓库创建项目独立分支/worktree，依赖与构建归项目所有。已有源仓库的未提交改动不会自动带入，先明确需要导入的范围。

按 [项目记录](references/project-records.md) 在 `.workspace/<project>/` 编写 Markdown：项目说明、需求、路线图和当前状态。根据实际内容组织文档，不要求用户凑齐固定字段。未决事项写明影响及下一步，能够独立推进的工作继续安排。

写完后读一遍，检查是否忠实表达用户目标、需求是否进入路线图、下一步是否具体。向用户简要说明记录位置、主要决定及待决问题。

本入口完成项目规划；用户有明确后续指令时，按同级 [harness-resume](../harness-resume/SKILL.md) 接续执行和记录实验。流程来源和许可见 [来源说明](references/provenance.md)。
