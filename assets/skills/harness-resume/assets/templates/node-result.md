# 节点结果 / 检查点

- contract: operator-harness/node-result@3
- project_id: {{与 TASK/project.json 一致}}
- search_node_id: {{启用搜索树时填写策略节点 ID；旧任务可省略本行}}
- run_id: {{与 TASK 一致}}
- node: {{与 TASK 一致}}
- task: {{冻结 TASK 的路径与摘要}}
- record_revision: {{递增编号}}
- status: {{running / needs_input / blocked / completed / failed / interrupted}}
- updated_at: {{日期时间与时区}}

## 任务理解

{{基于项目记录：本次目标、候选、节点环境、操作范围、验证与停止条件。缺少信息指出位置}}

## 输入身份与变更

{{实际读取的任务/候选/参考/case/协议/环境 snapshot 身份及 live 记录水位；与 TASK 的差异及处理；关联用户决定 ID}}

## 结果与证据

| 项目 | 判定 | 证据路径、身份与范围；未运行/不适用的原因 |
| --- | --- | --- |
| 编译 | {{判定}} | {{证据或理由}} |
| 精度 | {{判定}} | {{全部必选 case 的结果入口或理由}} |
| 性能目标 | {{判定}} | {{原始样本、统计与逐 case 目标或理由}} |
| 采集完整性 | {{判定}} | {{覆盖范围、缺失、解析器或理由}} |

判定使用 passed/failed/not_run/invalid/not_applicable，不以节点 completed 替代。

## 交付清单

| 产物/变更 | 路径与摘要 | 用途与下游使用方式 |
| --- | --- | --- |
| {{产物}} | {{路径/身份}} | {{作用}} |

{{环境/需求/协议变更只提交提案，说明是否改变权限或旧证据适用性}}

## 作业、资源与预算

{{每个 request/job 的所属环境、最后状态、记录时间、查询/回收/取消方式、未回收产物、锁占用与清理；无作业写 none}}
{{本次消耗、剩余、未知作业占用、是否仍有后台动作}}

## 问题与阻塞

| 问题 ID | 精确问题或阻塞原因 | 阻塞动作 | 已有决定/缺口 | 可继续工作 |
| --- | --- | --- | --- | --- |
| {{ID 或 none}} | {{内容}} | {{范围}} | {{来源}} | {{独立部分}} |

{{回答由协调角色保存到 DECISIONS（直接交互时由当前 agent 完成）；执行者引用决定，不将建议当真实答案}}

## 结论与下一节点建议

{{已完成/未完成、观察与假设的区别、应保留/放弃的候选、一个主推荐的节点/skill、原因与证据、依赖/缺口及完成/停止条件、精确恢复点；委派子 agent 只建议；直接交互在收尾记录后可继续授权内下一节点；最终回复同样呈现该建议，不能仅给记录链接}}

## 并行成员交付（非成员删除）

- batch_id: {{与 TASK 一致}}
- analysis_run: {{与 TASK 一致}}

{{逐次候选与真实父候选、最终选定候选、对共同基线逐 case 比较、环境/协议差异、仍在运行的作业、源码是否仍有写入者。主 agent 核对后接收；成员不得自行更新全局最佳或合并兄弟实现。}}

## Develop 接收判定（其他节点删除）

- analysis_data: {{本 run 的 ANALYSIS_DATA.md 路径，相对 workspace_root；不采也交付范围/原因}}
- accuracy_status: {{passed / failed / not_run / invalid / not_applicable}}
- accuracy_evidence: {{精度 passed 时必填证据文件路径，相对 workspace_root；覆盖全部必选 case 并绑定候选}}
- performance_status: {{passed / failed / not_run / invalid / not_applicable；指目标判定}}
- performance_evidence: {{性能 passed/failed 时必填逐 case 原始样本及比较证据文件路径}}
- regression_cause: {{none / idea / implementation / unknown / measurement；未达性能目标或有任何退化时必须归因}}
- regression_evidence: {{idea 时必填独立归因文件路径；其余可注明不适用}}

上述字段与结果表一致。归因文件包含相对父候选和标杆的逐 case 变化、有效测量依据、预期优化与实际实现的核对、已排除的实现问题、退化现象及交给下一 analysis 的结论。不修改验收阈值或隐去慢 case。

精度失败、实现问题、未知归因不能接收；在本节点修复后重验。预算/环境/权限阻塞保存 blocked/needs_input/interrupted，不以 failed 终态绕过。实现符合方案且精度通过的思路退化，可在保留性能 failed 的同时 completed 并交回 analysis；接收不等于最佳候选晋升。

ANALYSIS_DATA 使用 [分析数据模板](../../../harness-develop/assets/templates/analysis-data.md)，描述原始文件、摘要、实际 kernel/case/构建/环境、指标覆盖、诊断命令与观察。TASK diagnostics_required=true 时须核对到 ready 才能接收；false 时也保留未采/不适用依据，不将 pending_review 的采集脚本输出直接认作完整交付。
