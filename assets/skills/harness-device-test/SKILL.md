---
name: harness-device-test
description: 使用已绑定的上板环境约定，执行昇腾算子的实卡精度检查和稳态性能测量，保存逐 case 结果与失败证据。用于 Operator Harness 上板测试，不是项目接入。
---

# Harness Device Test

## 节点执行边界

当前会话或独立 agent 均可执行本节点。已收到 TASK 的子 agent 直接按 [节点契约](../harness-resume/references/node-contract.md) 读取输入，不递归执行主 agent 的派发流程；直接交互时先按 [harness-resume](../harness-resume/SKILL.md) 恢复或补齐 TASK。记录任务理解与结果，确保不继承聊天的新 agent 也能接手。委派子 agent 只写所分配产物，由主 agent 中继问询并汇总共享记录；直接交互时当前 agent 完成同样的收尾检查后可继续后续节点，不要求新会话。

读取 `.workspace/STATE.md`、任务契约、候选与 [证据记录](../harness-resume/references/execution-records.md)，使用 TASK 固定的 **device 专属约定快照**；实时 ENVIRONMENT 索引只核对漂移/撤权，不替换本次绑定。按 [环境复用规则](../harness-prepare-environment/references/environment-contract.md) 检查必需能力、身份、预算与设备队列/锁，沿用已确定的连接、容器和限制，不重新问整套环境。

仿真或开发环境的许可不能授权本节点。缺少上板约定或必要能力时，在 RESULT 提出对应缺口，由主 agent 派发环境准备；不能静默改用另一环境。默认按约定串行测量共享设备，发现忙时执行约定的有限等待或报告 blocked。

相对性能验收前读取 TASK 固定的 BASELINE ID/修订/摘要，检查其 ready 状态、case、来源/产物、环境和协议是否适用；无可用标杆时建议 [harness-establish-baseline](../harness-establish-baseline/SKILL.md)。本节点可按固定协议复测同一标杆，不自行重新选择或改写它。仅精度或用户约定的绝对性能目标不强制建立相对性能标杆。

## 绑定待测产物

核对完整候选快照、任务/参考/case/协议版本、lowering 与编译链、目标架构、入口和 launch。必要编译在上板约定指定的位置进行；复用已有二进制前检查身份和兼容性。检查本次运行的代码确实来自该候选，不能仅信缓存文件名或 Git HEAD。

优先适配项目已验证的 Onboard/DeepJIT 或现有测试命令；从其实际接口查参数，不臆造统一 CLI。远程模式记录本地 lowering 与远端构建/执行两段信息。检查产物传输摘要、退出状态与输出回收；SSH 请求成功不等于设备执行成功。

## 性能工具与节点职责

采用 `torch_npu.profiler` 测量时，直接执行 [采集方法](references/torch-npu-measurement.md)：在实际执行 NPU kernel 的进程中包围一次完整调用，保存原始 trace，并按绑定协议统计。配套 [采样脚本](scripts/torch_npu_measure.py) 和 [设备区间解析器](scripts/profile_parser.py) 提供确定性的采集与汇总；先核对脚本支持的协议与本次 TASK 一致。脚本不启动远端、不分配设备许可，也不替代完整调用的覆盖核查。

本节点负责在私有运行目录编写算子调用适配、连接已有 profiler、配置协议参数和汇总数据。缺 benchmark 脚本、多阶段入口或结果解析接线属于本节点工具工作；在既有权限和预算内先完成适配及小 case 采集检查，不能仅因“尚未实现/验证”返回 prepare-environment。只有实际 SDK、依赖、设备权限或采集接口故障且超出本节点修复范围时，才携证据建议环境准备。不能把某次外部进程冲突推断为环境没有采集能力。

上传入口同时核对内容摘要和执行权限。候选编译/运行/精度失败与测试准备错误分别处理：尚未提交设备的可修复路径、权限、脚本调用错误，在原范围内修正后继续，保留每次失败文件，不自动追加设备重试。按后端容器/guard/job 证据标 submitted/queued；正常排队到约定上限，不因暂时无输出提前报 blocked。断线先核对真实进程与原作业，不能仅凭工具 session 消失判定设备已停止。

## 先精度，再常规性能

依据 [实卡验证协议](references/validation-protocol.md) 对全部必选 case 比较全部输出。未执行、缺输出、超时、运行时错误分别保留，不能只凭退出码标 pass。失败时停止该候选的常规性能验收，保存反例与最大误差位置，在 RESULT 推荐 [harness-analysis](../harness-analysis/SKILL.md) 定位当前 kernel 的问题，不在本节点修改候选。

仅当本次任务包含性能验收、全部必选精度通过且身份未变化，预算、许可和计时能力均满足时，按固定协议执行基线与候选的预热、同步、重复采样和统计，保存原始样本与逐 case 对比。用户只要求精度时到此收尾，性能记 not_run。性能基线本身也须满足相同语义和精度要求。记录设备/环境负载和测量范围；无法取得可比样本就报告性能未验证，不输出伪加速比。

若绑定协议使用 profiler 的设备记录作为性能指标，profiling 就是本次性能测量方式，不另设一个尚不存在的“常规计时”前置条件。不要把采集进程墙钟当设备耗时。额外的流水指标/诊断 profiling 是另一范围，需相应任务和预算；采集失败不抹掉已有精度证据。

## 结果与恢复

保存任务状态、输出比较、原始样本、环境快照、命令和日志；分别判定精度、性能目标、采集完整性。保留慢的 case、精度失败与未运行项。超时或断线先查原作业，确认结束/取消和清理后才考虑预算内重试；不能盲目再提交。

独立保存 controller、SSH/worker、guard 的退出码及本作业私有 receipt；入口必须传播非零退出或显式返回失败，不能用 controller=0 覆盖 guard 冲突。guard 非零、receipt 缺失、冲突或观察丢失时性能 invalid，保留原始采样；按任务约定决定是否有额度再测，不改规则忽略冲突。退出码的采集与最终 receipt 验收由外层执行入口完成，不能在 guard 尚未结束时宣称性能有效。

按 RESULT 模板写本次结果，由主 agent 接收并更新实验记录。所有必选 case 达标才宣称目标达成；预算耗尽交付当前最好证据与差距。候选修改后旧通过标记不适用于新候选。

测试完成且需继续改进时，主推荐 analysis，交接确切候选、失败输入/日志、逐 case 性能差距及已有流水证据，形成 `analysis → develop → device-test → analysis` 循环。若本次仅为 analysis 请求的实卡采集，交回原分析问题，不自动进入 develop。目标已达成时推荐结束/交付；用户要求停止或预算不足时记录边界，不为循环新增任务。

启用搜索树时，沿用 TASK 的 `search_node_id`，在 RESULT（以及 develop 生成的候选清单）保留同一归属；阶段执行者不自行切换策略节点。将可复用发现与证据交给协调者归入关联节点的 hot，区分观测、假设和结论。见 [搜索与知识](../harness-resume/references/search-knowledge.md)。
