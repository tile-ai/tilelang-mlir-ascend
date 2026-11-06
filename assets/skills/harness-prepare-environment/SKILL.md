---
name: harness-prepare-environment
description: 通过问询分别确定算子开发、上板测试和仿真分析的环境与使用约定，执行有限探针并保存证据。用于首次环境准备、补齐环境缺口或显式变更环境。
---

# Harness Prepare Environment

先按 [项目管理契约](../harness-new-project/references/project-management.md) 解析 workspace_root、project_id 和 project_dir。用户/TASK 的显式选择优先；多项目不猜测，不使用全局当前项目指针。写入前核对项目 active 与会话 owner，其他会话须先交接；freezing 只允许安全收尾，frozen/archived 交给 harness-restore-project。所有源码修改限项目独立 worktree，环境/源码位置从绑定读取。

## 节点执行边界

当前会话或独立 agent 均可执行本节点。已收到 TASK 的子 agent 直接按 [节点契约](../harness-resume/references/node-contract.md) 读取输入，不递归执行主 agent 的派发流程；直接交互时先按 [harness-resume](../harness-resume/SKILL.md) 恢复或补齐 TASK。记录任务理解与结果，确保不继承聊天的新 agent 也能接手。委派子 agent 只写所分配产物，由主 agent 中继问询并汇总共享记录；直接交互时当前 agent 完成同样的收尾检查后可继续后续节点，不要求新会话。

让后续节点能够直接使用已约定且验证过的环境。先读 `.workspace/<project>/STATE.md`、目标需求与已有环境索引，使用 [环境约定](references/environment-contract.md) 保存选择和证据。

## 按实际任务问清环境

复用 TASK 与 DECISIONS 中已经记录的用户选择，从配置和文档查清事实；不能从 SSH 别名、安装包存在或旧测试报告推定环境可用。按 [追问指南](../harness-new-project/references/questioning.md) 连续问询，重点是后续实验所需的环境、资源和权限，而不是重新访谈算子语义。

分别讨论开发、上板、仿真的执行链。上板与仿真使用**各自的约定**，即使同一台机器或容器，也分别记录工具链、资源额度、允许操作与禁止事项；一方的授权不传给另一方。先明确本轮及后续计划启用哪些能力，暂不使用的仿真可记录为 deferred，不声称已经准备好。

从用户已知信息开始，每轮聚焦一个真实缺口：

- 代码编辑、TileLang lowering、设备编译、实卡运行、仿真分别在哪里执行；每一步是本地/远程、host/docker，使用哪个项目、解释器、工具链与目标芯片。
- 远程连接与认证引用、容器镜像身份、工作目录和挂载、数据传输/回收路径；不要索要或记录私钥与令牌正文。
- 每个环境允许哪些安装、编译、JIT、测试、profiling 或仿真操作；设备范围、队列/锁、并发、CPU/内存/磁盘、超时、总预算和清理边界是什么。小型探针同样计入相应环境的额度。
- 离线/代理、依赖来源和版本固定要求、宿主修改限制、采集与日志保留要求。已确定的约束直接沿用，不再次征求相同许可。

事实可自行检测；会改变用户选择或执行范围的缺口才问。未知项写明阻塞哪个能力，不替用户选择另一台机器、容器或无限预算。环境限制迫使修改 shape、精度或性能目标时，回到需求记录明确变化。

## 验证每条执行链

先依据约定和实际工具文档制定小型探针，确认它们在各自允许范围内，再执行：连接与路径读写/回收、版本与架构识别、依赖导入、最小 lowering/编译。上板另外验证指定设备上的 launch、同步、输出比较与所需计时能力；仿真另外验证模拟执行、错误日志、输出比较以及所需 trace 的可解析性。检查容器内的实际链路，宿主探针不能替代容器验证。

优先用项目已有的小型自校验用例，绑定来源版本和命令；没有用例时构造一个有独立预期值的最小用例。具体探针由实际 Onboard、simulator 或项目工具提供，不虚构 harness CLI。工具缺失时记录 blocked，不能用工具安装或 `--help` 成功代替运行验证。

探测与修复分开：仅在已约定的可变更范围内修复依赖；超出范围时说明具体新增需求。发生资源忙、超时或断线，保留结果并确认原作业状态，不重复提交。没有执行权限的能力保持未验证，允许继续离线设计与编写。

## 保存并交接

使用 [环境索引模板](assets/templates/environment-index.md) 和 [单环境模板](assets/templates/environment.md)，在本 run 的 `environment-proposals/` 输出索引与各角色约定提案，附问询决定与探针证据。未知和未验证如实记录；上板与仿真不得合并成一份约定。

在 RESULT 列出各环境已验证能力、缺口、提案和拟失效范围。协调者（直接交互时即当前 agent）核对后发布到 `.workspace/<project>/ENVIRONMENT.md` 与工作目录级 `.workspace/environments/`，更新共享状态并继续或委派下一节点；委派子 agent 不直接发布，不增加“是否保存”审批。仅有访谈范围时完成访谈交付，不能自行开始未授权探针。

收尾时在 RESULT 和最终回复给出一个明确的下一节点建议及原因、依赖和完成条件。环境已满足当前任务需要但正确性参考或性能标杆尚未建立/失效时，推荐 [harness-establish-baseline](../harness-establish-baseline/SKILL.md)；已有可用标杆则按 [统一下一步规则](../harness-resume/references/next-action.md) 选择。用户要求到此停止时只建议，不继续执行。
