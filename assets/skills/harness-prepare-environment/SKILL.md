---
name: harness-prepare-environment
description: 通过多轮交互确定 Python、TileLang、torch/torch_npu 与 CANN 的选择，准备并验证可共享的上板环境。用于首次环境准备、补齐缺口或变更环境；simulator 接入另行设计。
---

# Harness Prepare Environment

先按 [项目管理契约](../harness-new-project/references/project-management.md) 选择项目，核对 active 与 owner。读取 STATE、PROJECT、已有 ENVIRONMENT 和 DECISIONS，复用已确定的选择；同名环境先核实身份，不覆盖其他环境。源码修改限项目独立 worktree，实际安装位置由环境约定确定。

## 节点边界

目标是交付可直接使用的 TileLang + CANN 上板环境，包括匹配的 Python 配套、激活/执行入口和实际验证证据。共享环境按 onboard/simulator 分类，本轮实现 onboard 的指示与模板；simulator 的完整接入后续设计。开发、lowering、编译和上板是执行阶段，可使用同一环境，不要求第三类独立开发环境。

当前会话或独立 agent 均可执行；按 [节点契约](../harness-resume/references/node-contract.md) 写 TASK/RESULT。子 agent 只写获准产物，由协调者中继问询和发布；直接交互由当前 agent 完成相同工作。允许只完成访谈，不以环境未装好判定访谈交付失败。

## 多轮确定选择

使用 [上板环境问询](references/onboard-questioning.md)，每轮围绕一个主题，根据真实回答决定下一问，不一次抛出整张配置表，也不预先替用户选完再要求批准。

1. 明确本次采用哪个环境、local/remote 与 host/container，以及当前需要的能力。优先读取已有环境使用文档。
2. **先敲定 Python 环境**：conda、venv 或用户指定的其他方式；复用哪个环境还是新建，Python 版本与位置。未确定前不安装 Python 包或修改现有环境。
3. **敲定 TileLang 及 Python 配套**：TileLang 来源/版本与安装方式，torch 和 torch_npu 各自的版本与来源。读取目标项目约束，与候选 CANN 一起核对兼容性；发现冲突回到相关选择，不静默换版本。
4. **敲定 CANN**：是否复用已有安装、使用哪个版本和激活入口。没有适用安装时，与用户确定安装版本、来源/方式、位置与允许改动范围，再执行安装。
5. 补齐本次准备必需的资源和使用约定，简要复述已选方案、具体操作与未决项。已有明确授权覆盖时直接执行，不重复问许可；关键选择仍缺失时等待回答，只继续无依赖工作。

已有明确回答不重复问；事实可自行读取，用户没有偏好且明确委托选型时，依据实际兼容信息选择并记录委托来源。推荐选项、沉默和超时不算回答。每轮决定落到项目 DECISIONS，待答问题留在 RESULT 检查点，恢复从该处继续。未知项只阻塞依赖它的动作。

## 准备与最小验证

按已选方案复用、构建或安装，记录实际命令与结果。包操作显式使用已选解释器；激活 CANN 后核对实际加载的版本/位置，不只检查磁盘上是否存在安装包。已有共享安装不因一个项目需要不同版本就原地替换，先按约定使用独立位置。

从一条小型自校验路径确认 Python/包加载、TileLang 编译、指定设备运行和输出比较；优先采用所选源码版本已有的样例。需要性能验收时再验证对应计时入口，诊断 profile 按需求补充。可用一个样例覆盖多项能力，不强制拆成很多探针。探针只证明环境能力，不证明目标算子达标。

沿用已有适配和队列约定；普通环境直接执行。特殊初始化、提权、锁和修复步骤只写入适用环境文档，不作为通用要求。失败时先定位当前环节，在既有范围内有限修复；新版本/安装方式等选择发生变化时再问对应问题，不重开整套访谈。长任务状态未知先查原任务，不重复提交。

## 保存与交接

按 [环境契约](references/environment-contract.md)，使用 [共享上板环境模板](assets/templates/environment.md) 和 [项目绑定模板](assets/templates/environment-index.md)。本 run 的 `environment-proposals/` 保存提案；协调者检查后发布到 `.workspace/environments/onboard/<environment-id>/`，项目 ENVIRONMENT 绑定确切修订及本项目的使用约定。共享事实与项目授权分开，不因另一项目验证通过而继承其许可。

RESULT 分别报告用户选定配置、实际安装/加载结果、已验证能力和缺口。环境描述不能用 planned 填成 verified；仅复用证据时保留原身份与适用范围。已发布旧修订及历史记录保留，新模板不自动迁移旧环境或配置真实机器。

交付具体激活方式、执行 cwd/入口和下一节点所需材料。环境满足当前任务需要而标杆缺失时推荐 [harness-establish-baseline](../harness-establish-baseline/SKILL.md)；已有标杆按 [下一步规则](../harness-resume/references/next-action.md) 推进。明确指出剩余缺口，只问阻塞下一动作的问题。
