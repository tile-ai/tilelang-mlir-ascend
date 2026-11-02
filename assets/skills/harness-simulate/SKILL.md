---
name: harness-simulate
description: 使用独立的仿真环境约定执行昇腾算子仿真、检查输出与采集完整性，并从 trace 或模型结果形成可验证的优化假设。用于 Operator Harness 仿真分析节点。
---

# Harness Simulate

## 节点执行边界

当前会话或独立 agent 均可执行本节点。已收到 TASK 的子 agent 直接按 [节点契约](../harness-resume/references/node-contract.md) 读取输入，不递归执行主 agent 的派发流程；直接交互时先按 [harness-resume](../harness-resume/SKILL.md) 恢复或补齐 TASK。记录任务理解与结果，确保不继承聊天的新 agent 也能接手。委派子 agent 只写所分配产物，由主 agent 中继问询并汇总共享记录；直接交互时当前 agent 完成同样的收尾检查后可继续后续节点，不要求新会话。

读取 `.workspace/STATE.md`、任务/候选和本次分析问题，使用 TASK 固定的 **simulation 专属约定快照**（旧 trace 分析使用其历史身份）；实时 ENVIRONMENT 索引只核对变化与撤权，不替换本次绑定，按 [环境复用规则](../harness-prepare-environment/references/environment-contract.md) 核对必需能力。仿真有自己的机器/容器、工具链、运行限制、授权和预算；上板环境可用或允许测试，不代表允许仿真。

沿用该约定，不重复环境问询。仿真未启用、未验证或禁止运行时记录具体状态，必要时在 RESULT 建议主 agent 派发环境准备补齐该节点；不能把任务自动转到上板机器仿真。分析已有产物可继续，但必须核对来源，不能当成本轮新采集。

## 明确本次问题与采集范围

先说明要验证的假设、相关 case、需要的输出或 trace 指标。使用已记录的 simulator 或项目工具及对应版本文档确定命令，只采集能回答问题的有限 case。源码/二进制、ABI、launch、输入、模型/架构、采集配置均按 [证据记录](../harness-resume/references/execution-records.md) 绑定。

若仿真构建链与实卡不同，分别保存产物及共同的 DSL 来源；不能将不同编译产物当成完全相同的流水。需要创建小复现时记录与完整算子的差别，不能用子问题结果替代整算子结果。

## 提交与检查

使用实际可用的 simulator CLI 或 MCP。MCP 尚未接入时直接使用已验证入口，不假造工具存在。长任务保存作业 ID、日志和回收方式；按 simulation 约定限制 CPU、并发、时长和输出大小。超时或断线后先查询原作业，保留 partial 产物，不盲目重放。

依次检查进程退出状态、模型错误、输出是否完整及数值比较、trace 是否覆盖所声明范围、解析工具是否成功。程序退出 0 或生成了目录均不足以判仿真成功。数值规则沿用任务契约，不能无条件拒绝契约允许的 NaN/Inf。

## 从证据到假设

使用 [分析方法](references/analysis-method.md) 将观测、模型解释和优化建议分开，引用具体 trace 区间、指令/事件或统计文件。单位与统计范围要明确，模型频率未知时不换算 cycle 为设备时间。

输出错误或 trace 残缺时，可做局部诊断并注明可见范围，但不做完整 kernel 性能结论。模型不支持或复现持续超时，记录限制并停止预算外重试；需要实卡佐证时在 RESULT 建议主 agent 派发新的上板节点，使用上板自己的约定，不自动扩大权限。

在本 run 保存原始产物、解析命令、RESULT 与建议下一实验，由主 agent 接收并更新共享记录。仿真是可选时，失败不抹去已有实卡结果；任务明确要求仿真验收时保留未达成项。任何优化收益最终用同版本候选的实卡精度与性能证明。
