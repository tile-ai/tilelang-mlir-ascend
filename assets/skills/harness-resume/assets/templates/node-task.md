# 节点任务

- contract: operator-harness/node-task@3
- search_node_id: {{启用搜索树时填写策略节点 ID；旧任务可省略本行}}
- run_id: {{项目内唯一编号}}
- node: {{prepare-environment / establish-baseline / analysis / develop，仅一个}}
- operation: {{本次具体操作，develop 为 implement-and-validate / verify-only / collect-only / code-only；其他节点写具体操作}}
- workspace_root: {{相对 TASK 所在目录的工作目录根路径；标准 .workspace/<project>/runs/<id>/ 下为 ../../../..}}
- project_id: {{与 project.json 一致}}
- project_dir: .workspace/{{project_id}}
- owner: {{项目写入协调者的宿主会话标识；子 agent 沿用并受 TASK 白名单约束}}
- skill: {{工作目录根下的 skill 路径}}
- parent_run: {{前序 RESULT 路径；首次写 not_applicable 和原因}}
- created_at: {{日期时间与时区}}

## 目标与完成条件

{{本次要解决的问题、可观察交付、停止条件；与算子最终达标的区别}}
不包含：{{本次不执行的相邻阶段或操作}}

## 输入与读取顺序

以下路径字段相对 workspace_root；每项写确切文件，不只写目录。按节点契约补齐适用的候选、需求、用户决定、环境、case/参考/协议和前序证据。

PROJECT.md 的确切修订与摘要或不可变副本作为 snapshot，绑定本次适用的仓库规则、检查计划和交付要求；未知项注明其阻塞范围。develop 同时携带已有 PRECHECK 身份或本次补齐计划，不把计划当实际通过。

| 顺序 | 用途 | 路径 | snapshot/live | 快照身份或 live 起始水位 | 必需性与未知项影响 |
| --- | --- | --- | --- | --- | --- |
| {{顺序}} | {{用途}} | {{路径}} | {{读取策略}} | {{身份/水位}} | {{说明}} |

## 执行方式

| 步骤 | 机器/容器、cwd、环境激活 | 工具/脚本与参数来源 | 预期输出及检查 | 超时/失败时动作 |
| --- | --- | --- | --- | --- |
| {{步骤}} | {{位置}} | {{入口}} | {{验证}} | {{恢复}} |

{{若接口尚不明确，明确本次仅调查接口及输出要求，不伪造可运行命令}}

## 权限、预算与归属

- 本节点环境绑定及快照：{{角色、ID、修订、路径；仅离线任务可说明不需要执行环境}}
- 授权/禁止事项来源：{{DECISIONS 或独立约定中的具体决定；不使用“已在聊天授权”}}
- 资源与预算：{{计数单位、单次/总上限、本次 reservation_id/分配额度、账本入口、探针/JIT/失败/失联扣减}}
- 排他资源与获取/释放：{{设备、锁/队列入口、等待上限；不适用说明原因}}
- 可修改文件：{{确切文件或有边界的目录}}
- 只读/不得修改：{{需求、参考、容限、其他候选和共享摘要等}}
- 输出目录：{{本 run 专属目录}}
- 在途作业与恢复入口：{{job/request ID、状态、查询/回收/取消命令与占用；确无作业写 none}}

## 必交付与接收检查

{{RESULT.md 及本节点所需产物；每项路径/内容/身份要求。analysis 包含首次标杆方案或当前kernel诊断、证据/假设和验证计划；develop 包含适用的 CANDIDATE、PRECHECK、精度/性能/采集判定、ANALYSIS_DATA 与原始证据，prepare 包含独立环境提案，establish-baseline 包含 BASELINE、参考/case/协议及调查/测量证据}}

## 未决问题与交接

{{已知缺口、会阻塞哪些动作、可独立推进的部分；没有则写 none}}
{{给下一节点的建议及其所需输入；仅建议，由主 agent 决定并新建任务}}

## Develop 执行约定（仅 develop 填写）

- performance_required: {{true / false；按任务固定；code-only/collect-only 为 false}}
- diagnostics_required: {{true / false；性能优化默认 true；仅代码/精度等有限范围明确 false 并说明依据}}
- 分析数据计划：{{代表性/退化 case、真实 kernel 来源；主采集 msprof op/msopprof 的指标组合、重放模式与状态恢复适用性；按需补充的 torch_npu 调用映射/基础指标；必需/可选项、含重放/插桩的预算、复用证据或不采理由；ANALYSIS_DATA 交付位置}}
- 修复预算与停止条件：{{最多几次修复/实验、总额度；精度失败或实现问题留在节点内修复；额度耗尽则未验收，不自行加额}}
- 工作树与构建隔离：{{注册 repo_id、branch、起始源码/补丁身份；本地/远端 build、JIT/cache、源码和输出独立目录}}
- 验证范围：{{必选 case、全部输出、精度后性能、诊断例外、用户明确 code-only 的依据}}

并行成员另在文件顶层填写以下精确字段；非成员删除：

- batch_id: {{develop-batch 的 id}}
- analysis_run: {{已接收 analysis run ID}}

并绑定批次 inputs 的同一快照与摘要；源码白名单列出批次 repo_ids，所有可变目录独占。详见 [并行开发契约](../../references/parallel-develop.md)。
