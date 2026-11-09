---
name: harness-port-baseline
description: 将已选定标杆的算法与实现策略移植为 TileLang 初版，在同一节点完成精度、性能测量和分析数据交付；自行替代 AICPU 等已知能力缺口。已有适用候选时跳过，用于 Operator Harness 的标杆移植节点。
---

# Harness Port Baseline

先按 [项目管理契约](../harness-new-project/references/project-management.md) 确定 workspace_root、project_id、project_dir，核对项目 active 与 owner。源码和依赖改动限项目独立 worktree；读取 STATE、需求/决定、PROJECT、TASK 和固定的 BASELINE、case/容限/计时协议及 development/device 约定。接续、暂停、作业和预算遵循 [节点契约](../harness-resume/references/node-contract.md)，不以聊天代替输入身份。

## 进入条件与职责

正常路径是 `establish-baseline → port-baseline → analysis → develop → analysis …`。本节点在缺少适用 TileLang 候选时，将参考策略落实为可分析的初版；不要求先有 analysis。已有适用候选时保留并进入验证或 analysis，不因第一次进入工作流、标杆更新或节点新增而重写已有实现。标杆变化先检查语义、比较范围与现有候选的适用性。

默认且可接收的操作为 `port-and-validate`：理解参考、制定移植方案、实现并在本节点完成精度与性能测量，交付 TASK 要求的诊断数据。只做离线部分时保留未验收检查点；用户明确仅要代码可使用 develop 的 code-only 范围，不声称已完成本节点。无可读取/对应的参考实现时记录缺口；只有 Torch 参考或推断性能目标的通用初版构建路径仍待设计，不虚构已经移植了高性能源码。

## 从参考到可执行方案

1. 绑定实测参考的源码、分派分支、适配包装与完整调用边界，核对全部输出、排序/稳定性、容量、padding、别名、状态与同步语义。源码不能对应测量包时明确限制。
2. 分析算法阶段、数据布局、并行划分、融合、同步和中间存储，映射到项目所选 TileLang/后端的实际能力。可复用既有参考研究，不为走流程重新调查全部实现。
3. 在 [PORTING.md](assets/templates/porting.md) 记录参考步骤到候选源码的对应关系、替代方案、正确性风险与验证 case。移植实现策略，不机械翻译 API；不把尚无实测的推断写为候选瓶颈。
4. 完成初版及必要修复，不在本节点持续探索多种性能优化方向。每次尝试固定独立 candidate 身份与证据；达到节点交付条件后交给 analysis。

## AICPU 是已知能力缺口

本工作流中的 TileLang 候选不具备、也不引入 AICPU 能力。发现参考依赖 AICPU 时，由本节点在已授权实现范围内自行寻找替代方案，不重复询问是否实现 AICPU，不以扩展编译器/后端支持 AICPU 为前置，也不将该已知缺口推回 prepare-environment。

先识别该阶段的必要语义，再评估 TileLang 的 AIC/AIV 实现：例如 prefix-sum/count、查找、归约、scatter/gather、必要的排序或阶段融合。若参考排序仅用于后续索引构造，可以研究直接构造；若输出有排序或稳定性要求，则必须保留其语义。后端不支持所需跨核同步时，可以评估分阶段 kernel；若项目要求融合，则替代方案也必须满足融合要求，不能静默放宽。

只依赖静态 shape/配置的元数据，可在既定 host 包装范围内预计算；依赖输入 tensor 的运行时计算不能悄悄搬到 host、借 .cpu()/同步绕开约束，或调用实际落到 AICPU 的 Torch/ACLNN 算子。核对候选完整调用链及实际采集，记录使用了什么执行单元、传输和同步；参考本身可以继续使用 AICPU，它与候选限制分别记录。

替代必须保持全部输出、case、容限、融合等项目要求，保留原计时协议。不能排除候选新增的必需计算/搬运以制造加速，也不能因为标杆允许忽略某段 AICPU 时间，就把候选的替代计算从计时范围中删除。初版因此更慢可作为性能差距交付；精度失败不可以。实现无法满足契约时，保留已尝试方案、失败证据、具体未满足条件及解除方式，节点保持 blocked/needs_input；只有确需改变用户要求时才沟通。

## 本节点内验证

复用 develop 的 [PRECHECK](../harness-develop/assets/templates/precheck.md)、[设备验证方法](../harness-develop/references/device-validation.md)、[验证协议](../harness-develop/references/validation-protocol.md) 和 [上板诊断](../harness-develop/references/onboard-diagnostics.md)，不另派 test/develop 来完成初版验收。复用方法和脚本，不重复实现采集工具；性能退化归因采用本节的初版规则。

先完成适用低成本检查，再通过全部必选精度，随后按冻结协议测量性能。默认交付代表性/慢 case 的 ANALYSIS_DATA，范围和预算在 TASK 固定；明确不要求某项诊断时记录依据，不能事后降级。详细采集与正式性能测量分开，设备共享资源遵循排队约定。

精度、编译、移植实现和测量问题在本节点有界修复。预算耗尽或能力受阻则保存检查点，不以 failed/completed 绕过验收。只要实际实现符合记录的移植方案、精度通过、性能测量有效、必需诊断齐全，性能目标未达也允许接收；记录差距即可，不要求先证明“优化思路失败”。这与 develop 优化一个已有候选后的退化接收规则不同。

## 交付与下一步

交付 RESULT、可重建 CANDIDATE、PRECHECK、全部必选精度、逐 case 正式性能样本及比较、ANALYSIS_DATA 和 PORTING。移植说明中的 `aicpu_status` 仅可为 `not_used`（参考相关路径不依赖 AICPU）或 `replaced`（已替代且验证）；仍有依赖、替代未完成或调用链未知时不能声明 complete。

RESULT 的接收字段见 [节点结果模板](../harness-resume/assets/templates/node-result.md)。协调者检查源码/调用链和证据内容后接收；账本字段与摘要检查不证明 AICPU 已实际消除。执行者只写所属产物，协调者更新共享候选和搜索记录；本节点使用独占源码范围，不加入多 analysis 的 develop-batch。

初版未达最终目标时推荐 analysis，交接当前 kernel、移植差异、性能差距和诊断数据。目标已达且交付条件齐全时直接收尾，不为流程增加分析。遵循用户停止指令；没有完成验收时只交付检查点和恢复入口。
