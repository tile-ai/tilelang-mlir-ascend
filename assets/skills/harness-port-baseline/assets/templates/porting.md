# 标杆移植对应说明

- contract: operator-harness/porting@1
- project_id: {{项目身份}}
- run_id: {{port-baseline run}}
- candidate_id: {{本次交付的确切候选}}
- status: {{complete / incomplete}}
- aicpu_status: {{not_used / replaced / unresolved；not_used 指参考相关路径无依赖，replaced 指候选已替代}}

## 输入与方案

{{固定 BASELINE 身份/摘要、实测源码/分派/包装、case/全部输出/容限、计时边界、PROJECT 融合等要求、所选 TileLang/后端能力。来源不可对应处注明限制。}}

| 参考步骤与源码位置 | 所需语义及依赖 | TileLang 候选源码/阶段/执行单元 | 差异、替代原因与验证证据 |
| --- | --- | --- | --- |
| {{步骤}} | {{含排序/稳定性/容量/同步等要求}} | {{实际实现，不只写计划}} | {{精度 case、调用链及诊断证据}} |

## AICPU 替代

{{逐项说明参考的 AICPU 依赖、替代候选与选择依据、实际采用的 AIC/AIV 算法；为什么可以消除某个参考步骤，或如何保留其语义。无依赖时记录核对范围和证据。}}

{{核对候选的 Torch/ACLNN/host 包装与实际调用链，证明未绕回 AICPU 或将依赖输入的计算搬到 host。列出新增阶段、传输、同步及其计时归属；只允许既定包装范围内的静态元数据预计算。}}

## 实现核对与交付

{{候选已落实的方案、可重建 CANDIDATE、PRECHECK、全部必选精度、逐 case 正式性能及 ANALYSIS_DATA 路径/摘要。未满足的契约与实现缺口不能藏在“待优化”中。}}

## 给 analysis 的差距与问题

{{相对标杆的慢 case、移植差异及可观察证据；哪些原因尚待分析。性能慢可以交付，不要求证明优化思路失败；精度/实现/测量问题必须先解决。}}
