# 项目记录

项目记录直接使用 Markdown，用户和 agent 都可以编辑。项目流程产物放在工作目录的 `.workspace/<project>/`，与 harness 仓库自身的开发文档 `docs/` 分开。

| 文件 | 内容 |
| --- | --- |
| `project.json` | projectctl 管理的身份、生命周期、owner 与仓库清单；不手工改状态绕过检查 |
| `PROJECT.md` | 版本化项目契约：目标/范围、仓库集成规范、依赖能力、早期检查、交付证据与授权引用；使用 [项目模板](../assets/templates/project.md) |
| `REQUIREMENTS.md` | 具体需求及可观察的完成条件；用简短 ID 便于引用 |
| `ROADMAP.md` | 按依赖组织阶段，关联需求并说明阶段完成条件 |
| `STATE.md` | 当前进展、当前候选与最佳已验证候选、性能概况、待决问题和下一步 |
| `EXPERIMENTS.md` | 开始分析或调优后，按实验记录尝试、结果、取舍和产物链接 |
| `BASELINE.md` | 协调者接收后发布的标杆索引，指向 run 内确切 BASELINE 修订、参考/case/协议与测量证据 |
| `ENVIRONMENT.md` | 环境准备后建立的执行阶段绑定与项目使用约定；开发/上板可共用环境描述 |
| 工作目录 `.workspace/environments/onboard/<id>/` | 共享上板环境的描述修订与证据；项目只引用事实，另保留本项目授权和资源约定 |
| `DECISIONS.md` | 实际用户回答、问题 ID、作用域、日期与替代关系；主 agent 维护 |
| `AUTOMATIC.md` | 自动模式启用时的批次、控制步数、协调者与停止原因，status 只读，普通项目无需空建 |
| `RESOURCE_USAGE.md` | 资源额度预留、消耗与未知作业占用；与环境约定修订分离 |
| `codebase/MAP.md` | 已有项目的代码现状与有用的文件入口，由 onboard 维护 |
| `runs/<run-id>/PRECHECK.md` | develop 的实际低成本检查、输入身份与日志；计划仍在 PROJECT |
| `deliveries/<delivery-id>/DELIVERY.md` | 最终发布版与实测版的差异、证据覆盖、补测/未验证项及发布动作 |

按对话进展逐步建立这些记录，不为凑齐文件写空模板。定位、隔离与归档遵循 [项目管理契约](project-management.md)。同一项信息放在最适合的位置，其他文档引用即可。研究材料和临时笔记只在需要时写入 `research/`、`drafts/`。

[项目契约](project-contract.md) 规定 PROJECT 的必需信息、修订和早期检查；它不与 project.json 生命周期、REQUIREMENTS 验收细则或 STATE 进展重复记账。模板是 agent 编写辅助，不要求用户自行填写。

更新前读取相关文档，保留仍然有效的内容并直接修改变化部分。人工编辑是正常输入。未决问题说明影响什么、下一步怎么处理；不妨碍当前工作的细节可以留到后续阶段。

写需求时按实际任务考虑：输入输出与语义、shape 范围及动态行为、dtype、layout/stride、误差要求、边界 case、独立精度参考、性能基线、目标芯片与工具链、执行环境和交付条件。这些维度用于查漏，不是每次都要填写的表格。

规划结束时，检查重要需求是否进入路线图，验收是否具体，下一步是否可执行。编译、精度与性能的实际结果在后续执行中记录。

初次建立 `STATE.md` 时，没有候选或测量就写“尚无 / 未测”。后续分析和调优使用 [工作记录与交接](../../harness-resume/references/working-context.md)，将详细材料放到 `runs/<实验编号>/`；只有发生实际工作才添加实验条目。

进入实验前用 [环境准备](../../harness-prepare-environment/SKILL.md) 补齐实际环境选择并验证。初始化阶段已确定的答案直接沿用；这里不提前生成未经问询或验证的环境。

环境已具备当前任务所需能力后，需要建立参考或相对性能标杆时显式读取 [标杆建立](../../harness-establish-baseline/SKILL.md)，不要在初始化中用猜测性能补齐基线。

同一会话可以连续执行五节点，也可以由全新 agent 接续。进入节点前按 [节点契约与模板](../../harness-resume/references/node-contract.md) 写完整 TASK，把初始化问询中会影响后续执行的决定与范围落盘；不能只依赖当前聊天。需求文件无固定标题要求，但 TASK 引用的算子语义、验收规则和未知项必须可明确定位。
