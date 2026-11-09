# 候选清单

- contract: operator-harness/candidate@2
- project_id: {{与 TASK/project.json 一致}}
- search_node_id: {{启用搜索树时填写策略节点 ID；旧任务可省略本行}}
- candidate_id: {{项目内唯一候选编号}}
- producer_run: {{产生此候选的 run/TASK}}
- parent_candidate: {{父候选清单；初版写 not_applicable}}
- task_contract: {{需求路径、修订、摘要}}

## 可重建来源

{{源码快照路径，或可取回 commit + 完整补丁 + 未跟踪输入；文件清单与 SHA-256。不得只给当前工作树或 HEAD}}
{{语言仓库/revision、dialect、后端、依赖锁定/实际加载位置与构建配置摘要}}

## 语义与测试输入

{{输入输出/多输出顺序、shape/动态或特化键、dtype/累加、layout/stride、别名/workspace；引用确切任务条目}}
{{case 清单、输入/seed/生成器版本、独立参考与容限身份、性能基线/协议；缺项的影响}}

## 构建与调用

| 阶段 | 适用节点环境/目标 | 工具入口、cwd、命令/参数来源 | 产物或预期产物 |
| --- | --- | --- | --- |
| {{lowering/编译/调用}} | {{绑定}} | {{具体入口}} | {{路径、摘要或未构建原因}} |

{{函数入口、参数 ABI/shape、buffer 所有权、launch/grid/stream/同步；多 kernel 调用链。对 JIT API 记录调用签名和参数，不臆造原生 ABI}}

## 已验证范围与交接

{{PROJECT 契约修订/摘要、适用仓库规则；PRECHECK 路径/摘要及输入范围，不适用或未完成说明原因}}
{{编译/精度/性能各自对应 RESULT；没有就明确 not_run。下游产生的新二进制清单归该 run，不改此候选身份}}
{{本次修改和假设、已知限制、本 develop 内验证及后续复测的确切入口；不能从仿真或编译推导实卡通过}}

## 结构化验证身份

- source_sha256: {{实际源码快照或源码文件清单的 SHA256}}
- build_sha256: {{构建产物/编译配置清单的 SHA256}}
- launch_sha256: {{实际入口/ABI/launch 清单的 SHA256}}

新 @4 的 validation.artifacts 引用上述三个确切文件。清单中所有运行依赖还应列入 validation 的原始依赖引用；摘要证明文件身份，不证明编译器实际消费了这些字段。
