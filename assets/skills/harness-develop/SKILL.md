---
name: harness-develop
description: 根据已明确的算子契约编写、修复或优化昇腾 TileLang 候选，在同一任务内完成编写、精度验证与性能测量；也支持只验证或诊断采集。用于 Operator Harness 的开发与验证节点。
---

# Harness Develop

按 [项目管理契约](../harness-new-project/references/project-management.md) 选择项目并核对写入归属；从 STATE 和当前 TASK 接续。

## 节点执行边界

按 [节点契约](../harness-resume/references/node-contract.md) 执行收到的 TASK；直接交互先用 resume 补齐任务。节点写所属产物，协调者接收并更新共享记录。

按 TASK 输入清单读取当前任务、已接收的 analysis 方案/诊断、父候选和相关实验，按本次操作使用 TASK 固定的 development/device 约定快照；实时 ENVIRONMENT 索引只检查漂移/撤权，不重新替换绑定。implement-and-validate 落实针对已有候选的修改假设；缺少适用分析时先进入 [harness-analysis](../harness-analysis/SKILL.md)，没有候选且有适用参考源码则先进入 [标杆移植](../harness-port-baseline/SKILL.md)；没有适用性能源码时按本文 initial-and-validate 路径建立初版。用户明确只要求编写时遵循该范围并记录所据方案与未验证项。缺少环境信息时只补当前动作所需能力，允许继续授权内的离线编写。

延续初版的能力约束：TileLang 候选不引入 AICPU，不将其支持开发列为优化前置；保持已验证替代方案的语义和计时边界，后续优化也不能借 host/AICPU 回退绕过约束。

需要性能比较时，读取 TASK 固定的 BASELINE 记录，复用已验收的参考、case 和协议；缺少/失效时建议 [标杆建立](../harness-establish-baseline/SKILL.md)。仅编写或精度任务仍可在明确范围内创建独立参考，不以未测性能标杆阻塞授权内编写；不得把候选的自测时间直接发布为标杆。

## 任务模式与隔离

默认 `operation: implement-and-validate`：落实一个已接收 analysis 方案，完成低成本检查、固定候选、全部必选精度及任务要求的性能测量，再交付同一 run 的 RESULT。不能写完代码就结束节点或另派 test。

用户只要求测试时使用 `verify-only`；analysis 请求实卡诊断时使用 `collect-only`，均复用确切候选，不强制先改代码，也不额外要求首次方案分析。用户明确只要求代码时使用 `code-only`，记录未验证项。默认任务缺少必要环境/预算时可以完成允许的离线部分，但保留 blocked/needs_input 检查点，不能静默改成 code-only 并宣称完成。

并行任务按 [并行开发契约](../harness-resume/references/parallel-develop.md) 执行。只修改 TASK 分配的独立 worktree 与运行目录，不能修改兄弟 worktree、共享安装、全局候选指针或 search/AUTOMATIC/资源账本。不回退别人的改动。每个任务使用独立构建、JIT/cache、远端源码和输出目录；依赖源码需修改时也必须有独立 worktree。

## 建立可检验的起点

按任务加载内部能力：implement-and-validate / initial-and-validate 使用 [实现与性能方案落实](references/tilelang-workflow.md)，涉及索引、规约、缓冲或同步时使用 [正确性检视](references/correctness-review.md)；验证适配、精度故障及 verify-only 使用 [测试可信度](references/test-credibility.md)；collect-only 沿用 [上板诊断](references/onboard-diagnostics.md)；code-only 仅使用已授权的实现与低成本检查。方法吸收自 cannbot 的开发、设计、性能实践、review 和测试设计，安装后的 Harness 可独立使用，不要求另装外部 skill。外部资料仅作匹配版本的参考，不引入其编排、设备命令或报告格式。

确认功能语义、shape/动态性、dtype/累加、layout/stride、尾部、原地/别名、非法输入、误差与性能目标。任务中未确定且会改变实现的事项先问清，不从示例的固定尺寸推断通用支持。

检查独立参考与 case：小型可手算样例、边界输入和数学不变量；随机输入保留 seed，必要时保存实际输入。正确性参考和性能基线可以不同。修复参考、容限或 case 需要向主 agent 提交变更提案，接收并更新任务后再执行受影响验证，不能为了候选通过而同时放宽验收。

## 按目标版本编写

先读取 TASK 固定的 [项目契约](../harness-new-project/references/project-contract.md)，遵守本次目标仓库的 API、注册、lint、测试/打包与依赖功能约束。规则与当前源码冲突时保留依据并交协调者修订，不能等到发布前才调查基本集成要求。

使用 [TileLang 开发与定位](references/tilelang-workflow.md)，以环境中绑定的 TileLang 仓库/revision、Ascend dialect、目标架构及后端为依据查 API 和示例。首批语言来源是 tile-ai/tilelang；不要默用旧 NPUIR 分支、OpenTile 方言或其他后端同名 API。

先实现覆盖契约的最小版本。资源布局、尾部 mask、同步、Cube/Vector 分工和 launch 参数都必须与所选目标一致。一次优化围绕一个可验证假设，记录父候选、改变和预期影响。重试受任务预算限制；只有新证据或条件变化时才重复失败路线。

编译/lowering 必须按本次开发约定执行。目标产物若需要在上板环境编译，在本任务验证阶段按 device 绑定执行；分析请求的新仿真仍交给 analysis；开发环境的成功不能替代另一工具链编译。无需每次重新询问已确定的环境。

## 固定候选并立即验证

首次候选进入设备验证前完成项目检查计划中适用的低成本项，按 [PRECHECK 模板](assets/templates/precheck.md) 保存输入身份、实际命令、判定与日志；后续只重跑受改动影响的检查。失败/未运行/工具缺失不写通过，自动修复改动后重新固定候选。CPU 可发现的集成错误先修复；必须先设备诊断的例外在 TASK 说明范围，不将诊断当发布验收。高成本检查仍受环境和预算约束。

在 `.workspace/<project>/runs/<实验编号>/` 保存 [证据记录](../harness-resume/references/execution-records.md)：候选源码快照或 commit 加补丁、依赖/配置摘要、参考/case/协议身份、命令和日志、编译产物及入口/ABI/launch 信息。多 kernel 算子记录整条调用链；JIT 首次运行会触发的编译也计入预算与记录。

使用 [候选模板](../harness-resume/assets/templates/candidate.md) 固定 CANDIDATE，包含可重建源码和具体执行入口。随后在同一 run 执行 [设备验证](references/device-validation.md) 与 [验证协议](references/validation-protocol.md)：核对 device 绑定与设备权限，按需编译/传输，先全部必选精度，再按固定协议测量完整调用的性能。采集适配、原始样本、退出码与回执核对均由本节点负责，不因缺 benchmark 接线将工作退回环境准备。设备测量按共享资源约定排队，不能与兄弟任务在同一资源上并发计时。

每次尝试有独立 candidate ID 与 `attempts/<id>/CANDIDATE.md`，保存源码/补丁、PRECHECK、命令、精度反例和性能样本。修改后先固定新候选再重验，不覆盖旧证据。TASK 明确修复次数、耗时/实验预算与停止条件；没有修复额度时只执行本次实现及一次计划验证。精度失败和性能实现问题必须在本节点定位、修复并重验；原因未知时继续有界诊断，不能直接交给下一 analysis。预算耗尽或环境受阻时保存 blocked/needs_input/interrupted 检查点，节点不接收、不晋级，也不自行加额。

## 生成下一轮分析数据

性能优化任务默认在本节点生成 [上板流水与指标](references/onboard-diagnostics.md)，交付 [ANALYSIS_DATA](assets/templates/analysis-data.md)。TASK 先固定 diagnostics_required、代表性/退化 case、真实 kernel、指标、必需/可选范围与预算；精度后按需复用或采集，不能等下一 analysis 才补基本证据。仅代码/精度任务可明确不采，但仍记录原因和影响。

完整多指标诊断优先用 msprof op / msopprof，一次传入所需指标组合，由工具负责 PMU 分组、重放和流水插桩；不在 Python 中重复实现。按真实 kernel/启动序号选择，torch_npu MSTX 绑定验证后可用脚本选区。原生 torch_npu profiler 只按需补充调用映射和基础指标，不是固定前置步骤或完整多指标替代品；二者分开运行，不能把 CLI 多指标字符串传给 AiCMetrics。

交付原始数据、实际 kernel/构建/输入/环境身份、计数及时间线、重放分组/工具内存恢复范围、命令/日志、覆盖限制和待解释问题。不把指标数当作重放次数，也不把多轮合成报告当成单次同步观测。详细诊断采集独立于正式性能验收，不替换标杆或加速比。

必需采集失败时在本节点修复适配或保存阻塞检查点，不把“有日志/文件”当作完成。采集不支持或预算不足须给具体证据；已有通过的精度/性能仍保留，不能擅自把必需数据降为可选。下一 analysis 优先消费已交付数据，不无依据重复上板。

## 接收与下一步

RESULT 同时交付实现与测试结果，分别标注编译、精度、性能目标、采集完整性，保留失败、慢 case、未运行项与在途作业。编译成功不能替代设备验证；节点 completed 也不代表性能达标。verify-only/collect-only 不重复生成代码候选，只引用原候选与新增证据。

**精度是节点接收的硬条件。** implement-and-validate、initial-and-validate 与 verify-only 的全部必选精度必须通过；精度失败不能用 completed/failed 交付绕过，不能以“交给 analysis 定位”结束本节点。保留反例，在本节点修复并重新验证全部必选 case。verify-only/collect-only 若只获准只读操作，则保留未验收检查点，请协调者补齐修复范围；不能自行扩大修改权限，也不能接受失败结果。

性能退化先核对计时有效性，再对照 analysis 方案、实际源码/生成物与流水证据判断原因（即使仍满足目标，也需解释相对父候选的退化）：

- **优化实现不正确**：没有落实预期的数据复用、并行或流水，意外 scalar 化、冗余搬运等。在本节点修复，重新固定候选，重跑精度及性能；不能将实现缺陷当作思路无收益。
- **优化思路不正确**：实现已符合方案且精度通过，有效测量仍显示该策略无收益/退化。记录受影响 case、父候选/标杆对比、原始样本、退化现象、归因依据和已排除的实现问题，将这些结果直接交给下一 [harness-analysis](../harness-analysis/SKILL.md)。此时节点可接收，但候选不因此成为最佳。
- **原因未明或测量无效**：留在本节点诊断/修复测试链路，不猜成思路问题。达到预算或权限边界则保留未验收检查点。

目标达成且没有未解决的接收条件时交协调者汇总/交付；有效但未达性能目标的结果也须先完成上述归因，再交给 analysis。明确 code-only 或 collect-only 沿用 TASK 的有限范围，不能将失败任务事后改成这些模式规避验收。

启用搜索树时沿用 TASK 的 `search_node_id`。search@2 在 RESULT 保留 queue_item_id 和适用的 analysis_run，冻结 source/build/launch 后交付；旧批次保留 batch_id、analysis_run。执行者只交付本 run，协调者登记候选、接收结果、确认停止并释放空间，最佳候选另行晋升。候选谱系引用真实输入候选，不能把兄弟的通过标记继承给合并结果。

## 初版构建与程序验收

没有候选且无适用高性能源码时，允许 `initial-and-validate`：先有独立正确性参考、有限来源调查结果和明确的绝对/模型目标，固定 initial_design，再构建最小初版；无需虚构已有候选分析或性能源码移植。遵守同样的 TileLang/AICPU、预检、完整精度、测量与诊断要求。初版实现符合方案且精度/测量有效时，目标未达可带差距进入 analysis；不能将实现或测量错误归入该例外。已有候选仍使用常规 implement-and-validate。

新 TASK/RESULT 按 [结构化验收](../harness-resume/references/execution-control.md) 使用 @4，注册前固定 evaluation_plan。使用 [评测入口](references/evaluation-runner.md) 执行冻结适配器，自动生成逐 case 证据和 RESULT 草稿。正式接收前核对程序检查、科学内容、job/资源及未测边界。范围变化需要新任务时先安全 close-run，不能改冻结任务或把失败称为通过。

已有本地前台命令可使用 [本地 runner](references/local-runner.md) 保留意图、日志和退出回执；真实设备执行仍使用已核实适配。进程回执不等于设备或性能验收。
