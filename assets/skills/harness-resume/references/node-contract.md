# 节点交接契约

项目定位和写入归属先遵循 [项目管理契约](../../harness-new-project/references/project-management.md)。

本契约让五个节点既能在同一会话连续工作，也能被全新 agent 独立接续。项目事实来自项目记录，执行方法来自所分配 skill；不要求节点读取历史聊天或依赖派发者记忆。它是当前可用的 Markdown 交接契约，不是已实现的自动执行器或程序 gate。

## 1. 执行边界

当前交互模式允许同一个 agent 在同一会话完成多个节点或任务。每次有明确目标的节点工作留下 TASK 与 RESULT，节点切换时完成记录再继续；不要求每个工具调用都新建记录，不强制更换 agent 或开启新会话。信息必须可从项目记录恢复，当前聊天只是便利，不能是唯一事实源。

可以按任务复杂度委派子 agent，或在上下文过长时建议用户开启新会话。用户选择继续就在当前会话推进；没有子 agent/隔离工具不构成阻塞。使用全新 agent 时只传工作目录根与 TASK 路径，不继承聊天也应能完成任务。它是接续能力的验收方式，不是所有工作的前置条件。

[automatic 模式](../../harness-automatic/SKILL.md) 使用当前宿主的主/子 agent 能力协调节点，默认独立上下文并通过记录交接，无子 agent 工具时明确采用 inline；不要求用户手动开新会话。这是 skill 驱动的有界循环，不是独立后台调度服务。以下“主 agent”表示协调/接收角色，“节点 agent”表示执行角色；同一个 agent 可以顺序承担两者，不需要额外审批或新的聊天。

委派时子 agent 只执行被分配范围，由主 agent 决定后续；直接交互时当前 agent 完成本节点记录与检查后，可以继续用户已授权的下一节点。并行仅用于输入和可写范围不冲突的工作，硬件锁/队列仍遵守各环境约定。

## 2. 六份模板与位置

| 模板 | 实际记录位置 | 写入者 |
| --- | --- | --- |
| [节点任务](../assets/templates/node-task.md) | `.workspace/<project>/runs/<run-id>/TASK.md` | 协调者执行/派发前填写，开始后冻结 |
| [节点结果](../assets/templates/node-result.md) | 同目录 `RESULT.md`；等待/中断时也是检查点 | 节点 agent 写，主 agent 只读取 |
| [标杆记录](../../harness-establish-baseline/assets/templates/baseline.md) | 本 run 的 `BASELINE.md`；共享 `.workspace/<project>/BASELINE.md` 仅索引确切记录 | establish-baseline 生成，主 agent 接收后发布索引 |
| [候选清单](../assets/templates/candidate.md) | 产生候选的运行目录 `CANDIDATE.md` | develop；其他节点产生的构建物另存清单，不改候选 |
| [环境索引](../../harness-prepare-environment/assets/templates/environment-index.md) | `.workspace/<project>/ENVIRONMENT.md` | 主 agent 接收环境结果后更新 |
| [单环境约定](../../harness-prepare-environment/assets/templates/environment.md) | `.workspace/environments/<id>.md` | prepare 在自己的输出目录生成提案，主 agent 接收后发布 |

automatic 另按其 [控制模板](../../harness-automatic/assets/templates/automatic.md) 保存批次、步数与停止原因，不改变六份节点记录的职责。status 不创建这些文件，next 仅在执行需要时建立记录。

按需实例化，不预填假数据，不一次创建空项目骨架。模板中的 `{{...}}` 必须替换，未知用 `unknown：原因/影响`，不适用用 `not_applicable：原因`，不能以空字符串假装满足门槛。模板按各自声明的契约版本定义字段；扩展小节可以增加，必填语义不能删除。

新 TASK 使用 node-task@2，固定 workspace_root、project_id、project_dir 与 owner。workspace_root 相对 TASK 所在目录解析（标准 `.workspace/<project>/runs/<id>/TASK.md` 使用 `../../../..`）；project_dir 必须是该工作目录下的 `.workspace/<project_id>`，身份须与 project.json 一致。输入和产物路径字段相对 workspace_root，Markdown 链接仍按文件自身位置解析。源码 cwd 位于 project_dir/repos 中，不等于记录根。旧 TASK v1 的 project_root 仍按旧语义解析，不自动迁移。命令另写执行机器、容器、工作目录及环境变量。远端路径标明所属环境，不能拿本地相对路径代替。运行目录和输入快照在结束后保留，复制到另一 worktree 时保持相对布局；确有外部依赖则明确解析入口，缺失即报告缺口。

## 3. TASK 的公共必填内容

- 契约版本、唯一 run ID、节点和本次操作范围、项目根定位方式、父运行或首次任务说明。
- 本次目标、完成条件、明确不包含的动作；只有“继续优化”不够，须有具体问题与停止条件。
- 有顺序的输入清单：STATE 导航、任务需求、用户决定、候选/参考/case/协议、节点环境、相关前序结果；每项写作用、路径、读取策略与身份。候选/需求/参考/case/协议/环境约定使用 snapshot，固定 revision/摘要或不可变副本；STATE 导航、DECISIONS 增量与 RESOURCE_USAGE 使用 live，记录派发时水位和实际读取版本。可变候选指针须在派发前解析成具体输入。
- PROJECT 按 [项目契约](../../harness-new-project/references/project-contract.md) 固定适用修订/摘要或不可变副本，提供本次集成规则、检查计划与交付条件；已有 PRECHECK 需绑定实际候选，缺失则列出本次适用检查。
- 操作步骤与已知工具入口：启动位置、实际工具/脚本、命令或参数来源、依赖、预期产物与成功检查、状态查询/回收/取消入口；不知道接口时把“查清接口”明确为任务，不能派发假定可直接上板的任务。
- 授权来源与作用域、预算计数和本次分配额度、排他资源/锁、可改文件、只读文件、输出目录、在途作业。
- 必交付文件与证据、未决问题、受阻时可以独立做的工作，以及下一节点所需信息。

派发提示仅给工作目录根、TASK 路径和“按 TASK 指定 skill 执行”；所有会影响执行的事实、范围和权限先写入记录。若提示中出现新的事实，先落盘再派发。不能用“按上次讨论”“同前环境”或仅给一个目录来替代输入与操作说明。

live 记录允许正常追加，不以整个文件摘要变化判定候选失效；新决定是否影响任务要检查作用域。snapshot 输入不一致则停止受影响动作并回报，不继续使用被改写的内容。STATE 用于导航和检查是否有新决定，不可在 TASK 执行中自动改换目标。已冻结输入与最新用户限制冲突时停止受影响动作，回报差异；尤其不能凭旧快照继续已撤销的许可。

## 4. 节点专属输入与输出

| 节点 | 除公共字段外必须提供 | 必交付 |
| --- | --- | --- |
| prepare-environment | 本任务计划启用的能力、已知环境事实/选择、各侧已有权限来源、尚未解决的问题；允许只做访谈 | RESULT；各角色约定提案和索引提案；实际探针证据或逐能力未测原因 |
| establish-baseline | 语义与必选 case、正确性参考或允许创建范围、目标/实现限制、调查与选择规则、已有标杆与变更原因；实测绑定 development/device 约定和测量协议（允许在 run 内先生成并冻结） | RESULT；BASELINE；可重建参考/适配、case/协议快照、候选调查/排除理由、全部必需正确性与原始性能证据或逐项缺口；复测入口 |
| analysis | 首次：确切标杆/参考、实现源码与适配、case/契约、测量边界、目标TileLang能力；后续：当前候选、最新测试结果、具体问题及已有分析；新仿真另绑定simulation约定/模型/配置，旧trace绑定历史身份 | RESULT；首次实现思路与开发方案，或当前kernel问题诊断；源码/测试/trace证据与限制、观察/解释/可验证假设、交给develop的修改及验证计划；无采集时如实注明 |
| develop | 可执行语义与输入输出、边界/动态约束、case/参考/容限、性能目标或本次不测、已接收analysis方案/诊断、父候选与前序失败、开发约定；显式仅编写任务注明所据方案；初版无参考时明确创建范围与独立检查 | RESULT；可重建源码；CANDIDATE；已做编译检查及未验证项；device-test入口 |
| device-test | 确切候选、全部必选 case、输入与参考/容限、device 约定、精度/性能/profile 范围；测性能时提供基线、协议及匹配的精度证据或明确先执行精度 | RESULT；每 case 输出比较与原始样本（若采样）、构建/作业清单、日志及回收状态；失败反例 |

生成候选与执行候选分开：develop 的源码交付可以 completed，但设备精度仍 not_run。缺少必要环境时可以完成 TASK 已授权的离线编写；不能隐式扩展成另一个节点。已有 trace 分析不要求重新准备环境，但不能掩盖原产物的身份或正确性缺口。

默认顺序为标杆建立后 `analysis → develop → device-test → analysis …`，具体选择与目标达成后的停止遵循 [统一路由](next-action.md)。首次analysis没有目标候选不构成缺口；其输入是标杆/参考的实现和契约。后续分析必须引用对应候选及测试身份，不能仅凭“曾分析过”跳过新问题。

历史兼容：旧TASK的 `node: simulate` 和 `harness-simulate` 入口按原任务范围映射到 analysis 的仿真/诊断职责；用户称 `harness-simulator` 时也定位到此入口。保留历史TASK/RESULT，不将旧仿真结果改写成已完成首次标杆分析。新任务统一写 `node: analysis` / `harness-analysis`，不新增重复安装的别名skill。

## 5. 冷启动自检与结果接收

节点开始先读 TASK、指定 skill 和按序输入，检查路径可读、版本对应、操作可执行、权限与预算覆盖。执行前在 RESULT 写简短“任务理解”：本次做什么、从哪个候选和环境开始、允许哪些动作、怎样验证、何时停止。发现阻塞项只回报具体缺口，不向用户重新问已经落盘的环境。

协调者执行/派发前检查 TASK 完整；本节点收尾或收到 RESULT 后逐项检查交付文件存在、身份一致、未运行项如实记录、在途作业和预算已交代，再决定接收。接收是内部记录检查，不是新增用户审批。`completed` 只是本次节点任务完成，不意味着算子已达标；必选测试失败也可以得到完整的节点结果，测试判定仍 failed。

接收/发布/结算可能被中断，恢复须先对账：用 run ID 与终态 RESULT 摘要标识同一次交付，检查 EXPERIMENTS 接收记录、STATE 指针、已发布环境修订及各 reservation 已结算量，仅补未完成部分。不得重复扣减同一消耗、重复发布环境修订或以重跑节点修复摘要未同步；账本冲突先保留依据并核实，不猜余额。接收记录写明产物摘要及尚未完成的更新，最后才将控制步骤标为 accepted。

共享记录由协调角色统一写入，决定是否更新当前/最佳候选。委派的子 agent 只能写自己的运行目录和 TASK 授予的源码范围；单 agent 工作时，当前 agent 在节点收尾检查后直接汇总共享记录。标杆建立只生成本 run 的 BASELINE；协调者验收后发布 `.workspace/<project>/BASELINE.md` 索引，下游解析并冻结具体版本，只有证据齐全才能用作相对性能验收。候选基线、case、语义、环境或协议改变时保留旧版，不能默换分母。环境准备先输出 `environment-proposals/`，主 agent 核对后发布；协议/参考/容限变更也是提案，不能由修复候选的节点静默改验收。

结果不完整或引用失效，主 agent 保留原目录，在新 run 中记录具体补充任务，由当前或另一个 agent 继续，不擦写旧结果。下游收到具体产物与证据入口，不需要重新检索全部实验历史。接口未知、硬件不可用与代码失败分别记录，不因“agent 已返回”标成功。

### 资源余额单独记账

环境约定只保存预算政策与上限，实时余额不属于环境身份。主 agent 在 `.workspace/<project>/RESOURCE_USAGE.md` 维护最小账本：`reservation_id、环境/共享资源ID、run_id、额度单位与预留量、已结算量、未决占用、状态、job/RESULT依据`。STATE 链接账本，TASK 固定本次 reservation 和分配额度。节点只能在已分配额度内执行，并在 RESULT 报告增量消耗，不能把旧环境快照中的“剩余”当作可用余额。

首次预留前创建账本，协调者执行/派发前串行预留，按实际结果结算；unknown 作业保留未决占用，新的恢复任务引用同一作业与预留，不重新释放后再分配。共享物理资源另受已有锁/队列约束，本账本不替代跨进程设备锁。仅余额变化不递增环境修订、不使探针失效；预算政策/授权范围变化才更新约定，并检查受影响的在途任务。

## 6. 问询、检查点和中断

五节点在开始下一次探针、编译、设备/仿真提交、重试或新的文件修改前，以及工具返回后，检查 live DECISIONS 中作用于本任务的暂停、撤权或模式决定。协调者收到停止请求先落盘并通知节点；节点在当前不可中断调用返回或可安全保存处写检查点，停止新增操作，不能因“一节点算一步”继续跑完所有实验。已提交 job 仍按原约定记录、查询/回收，未经确认不称已取消、不释放未决占用。当前调用不能立刻中断时如实报告。

转为 next 允许完成其绑定的当前有界动作；显式停止则在安全检查点停下。节点不得自行选择下一节点。详见 [模式切换](next-action.md#6-模式切换与接管)；直接执行与委派执行适用同样边界。

需要问用户时先记录问题 ID、具体问题、影响范围与现有依据；需要等待时在 RESULT 写 needs_input。委派的子 agent 通知主 agent 中继，直接交互则由当前 agent 提问。实际回答、日期、问题 ID、作用域和被替代决定写入 `.workspace/<project>/DECISIONS.md` 后供当前或其他 agent 继续使用。没有真实回答就保持未知；推荐选项或等待超时不是授权。

普通回答或恢复可在当前 agent/会话中继续，不因问询或中断自动新建 run。若答案改变了冻结 TASK 的目标、可写范围或预算，保留旧任务并建立新 TASK/run，执行者可以不变。更换 agent 接续同一未结束任务时沿用原 run 与检查点，协调者先确认原执行者不再写入，并在 STATE 记录接管；已结束任务的补充/新尝试另建 run。以上均不要求继承聊天。

RESULT 作为当前检查点可以更新；覆盖前将已交接或用于问询的版本保留到 `checkpoints/<递增编号>.md`。终态 RESULT 不再修改。在 STATE 中记录 run、在途 job，以及实际发生委派时的 agent ID/派发状态，避免重复执行仍活跃的工作。没有 agent ID 就注明当前会话执行或交接位置，不伪造查询能力。

在外部提交前保存提交意图与 request ID；提交后立即记录 backend job ID、环境身份、命令、最后已知状态、产物位置、查询/回收/取消方式和预算占用，不能等最终总结才写。这些可放 RESULT 的作业小节或其链接的作业日志。丢失响应时状态 unknown；不能只因新 agent 启动就重投。节点停止不代表外部作业停止。

## 7. 状态和版本

- 节点记录状态：`running / needs_input / blocked / completed / failed / interrupted`。blocked 是本次动作条件不满足，附解除条件；failed 是节点本身不能完成交付，不与测试 failed 混同。
- `completed / failed` 是不可改写的终态；`needs_input / blocked / interrupted` 是可接续检查点，恢复遵守同一 TASK 与历史版本保存规则。控制批次 paused 不表示节点 completed；需补新尝试时另建 run。
- 证据判定：编译、精度、性能目标、采集完整性分别用 `passed / failed / not_run / invalid / not_applicable`，每项带证据或理由。
- 后端作业状态独立，沿用 queued/running/unknown 等实际语义；completed 节点也可能只完成对 unknown 作业的诊断，必须留下一步恢复入口。
- 委派时协调者记录派发状态：`prepared / dispatched / accepted / rejected / blocked`；直接交互则在 EXPERIMENTS 写本节点接收结论，不需虚构派发状态。没有通过接收检查的结果不能作为下游的“已验证”输入。

TASK、RESULT、candidate 新模板分别使用 `operator-harness/node-task@2`、`node-result@2`、`candidate@2`，必填 project_id；TASK 同时固定工作目录、记录目录和 owner。其余记录保留各自版本。v1 旧记录只读兼容；新工具不将旧项目根解释为新记录目录。增加可选字段不改变既有语义，必填字段或含义改变则升主版本；不理解版本时停止并报告。已有记录迁移只补缺失事实与引用，保留旧证据；没有历史信息就标 unknown，不补造身份或许可。当前按此契约人工/agent 检查；未来程序校验首先验证这些稳定字段与引用，不另造第二套事实源。

## 搜索树兼容扩展

TASK、RESULT、candidate 的 search_node_id 在未启用树时可省略；未启用树的旧记录仍可读取。启用树后的新探索记录必须填写准确策略节点 ID。原 `node` 始终是工作流阶段，run_id 是阶段执行身份，candidate_id 是代码候选身份；三者都不代替搜索节点。参见 [搜索与知识](search-knowledge.md)。

## 项目交付记录

项目级 [PROJECT](../../harness-new-project/assets/templates/project.md)、开发 run 内 [PRECHECK](../../harness-develop/assets/templates/precheck.md) 和收尾时 [DELIVERY](../assets/templates/delivery.md) 补充上述节点记录，不新增节点。PROJECT 定义规则，PRECHECK 保存实际检查，DELIVERY 对照最终发布版与实测版并说明证据适用范围；详细规则见 [项目契约与交付检查](../../harness-new-project/references/project-contract.md)。
