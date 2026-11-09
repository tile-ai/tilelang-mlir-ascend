# 结构化验收与执行控制

新 TASK/RESULT 使用 @4。`searchctl` 将契约、评测输入和工具身份写入现有 search/tree.json；不建立另一份独立控制数据库。历史 @1–@3 或无版本记录保持原文，补登记时必须在 run/develop-batch 操作中显式给 `legacy: true`，它们不获得新版候选晋升资格。显式未知版本始终拒绝，不能用 legacy 绕过。

## 评测输入与接收

在修改候选前创建 `evaluation-plan@1` JSON，TASK 顶层 `evaluation_plan` 指向该文件。以 [模板](../assets/templates/evaluation-plan.json) 为起点，所有占位内容必须来自实际任务。引用为 `{"path": "工作目录相对路径", "sha256": "实际摘要"}`，可以用以下只读命令生成：

```bash
python3 .agents/skills/harness-resume/scripts/searchctl.py \
  --workspace . --project example reference .workspace/example/inputs/protocol.json
```

plan 固定 requirements/reference/comparator/cases/protocol/environment 六项依赖、全部必选 case 与输出、统计方式、最少样本及逐 case 的最大耗时。协议若使用加速比，预先从冻结标杆及目标换算成逐 case 最大耗时，并将换算依据留在 protocol；工具不接受事后挑选分母。当前程序支持 mean/median 和逐 case latency 上限，不支持任意汇总权重或统计模型；不适用时保留契约缺口，不能修改指标含义来套格式。

`performance_kind` 为 measured/absolute/model/none。模型必须在 protocol 保存推导、硬件来源、假设及用户目标依据；none 只用于明确不要求性能的任务。与 TASK 的 performance_required 一致。注册 run 时冻结 plan 及其直接输入摘要；漂移后不能接收原任务的结果。

新 implement-and-validate、initial-and-validate、verify-only、port-baseline 的 RESULT 增加：

- `candidate_manifest`：实际输出候选；CANDIDATE @2 顶层包含 source_sha256/build_sha256/launch_sha256，分别绑定源码快照、构建清单和 launch 清单。
- `validation_evidence`：validation@1 JSON，字段见 [模板](../assets/templates/validation.json)。绑定 plan、候选、source/build/launch 文件、逐 case 原始报告、独立执行回执和原始诊断清单。
- 保留 accuracy/performance/regression/analysis_data 字段及人工解释，程序声明必须与结构化结果一致。@4 的 accuracy_evidence 和适用的 performance_evidence 都指向同一完整 validation 文件，逐 case 原始文件由该清单引用，避免另一个摘要混入旧结果。

每份原始 case 报告包含 schema、project_id/run_id/candidate_id/case_id/request_id、evaluation_plan 引用、与 validation 相同的 artifacts、非空 raw_files 摘要清单。accuracy@1 的 outputs 是全部要求输出到布尔比较结果的映射；performance@1 的 samples_ms 是完整有效样本序列。程序核对身份、输出/case 覆盖、样本数量与有限正值，重算固定统计量和目标判定。多文件依赖使用 artifact-index@1 JSON（非空 files 引用列表），程序递归冻结该显式索引；不递归猜测任意 Markdown 链接。源码/比较器依赖必须完整列入索引，未声明的文件无法自动发现。原始输入/输出、比较器日志和 profiler 文件仍要保存；JSON 中的比较结果来自被冻结的 comparator，工具不自动重新执行任意 tensor 比较，也不能认证是谁写的报告。

execution-receipt@1 包含项目/run/candidate/request 身份、与 validation 相同的 artifacts/evaluation_plan、status=exited、各执行层 exit_codes 和 valid=true。valid 由实际执行适配器核验资源约定与回收结果后给出，不是执行者手填成功标记；普通本地进程回执不能冒充设备回执。当前没有通用 NPU 回执适配器，使用任务已有且经过核实的适配，明确覆盖范围。

code-only/collect-only 不要求完整评测 plan，也不能晋升最佳候选。二者仍交付 candidate_manifest 和 ANALYSIS_DATA，accuracy_status/performance_status 使用 not_run/not_applicable；历史精度以证据链接复用，不能在有限模式声明本轮新正式验收。collect-only 另交 diagnostic-evidence@1 JSON，包含 project_id/run_id、candidate 引用、原始 artifacts 引用列表；报告对应本次采集，既有精度结果以链接复用，不将诊断采集称为新正式性能验收。

新 develop-batch 成员必须使用完整 plan；requirements/cases/protocol/environment 与批次输入一致，baseline 必须在显式评测依赖中，各成员比较输入、case/目标与统计规则一致。不一致时整批拒绝登记。

接收保存原始文件依赖摘要，后续 check 会检出漂移。新 run 同时固定验收脚本摘要和随包 skill 资产摘要；执行期间升级会阻止旧 run 用新规则 accept，应使用记录对应的安装包或安全关闭后建立新任务。历史已接收结果仍按其 policy 读取；未来评测语义变化须新增 policy 实现，不能原地替换旧语义。

`promote-candidate` 与节点 accept 分开：

```json
{"event_id":"choose-1","op":"promote-candidate","run_id":"d1","reason":"共同契约下全部必选项通过；具体选择依据"}
```

只允许已登记的确切候选、结构化精度通过及适用性能通过进入 best。慢但有效的思路失败可以 accept，不能 promote。已有 best 的比较输入必须一致；新比较契约的跨版本 best 迁移仍需另行设计，不能手改 tree 绕过。STATE 只引用该事实；排名、科学归因及最终交付仍由协调者核验。

## 安全关闭失败或被替代的任务

范围内修复继续原 run。已无法继续、需修改冻结范围/预算或被明确淘汰时，先停止执行者、回收原 job 并结算，再提交 `close-run`：

```json
{"event_id":"close-r1","op":"close-run","id":"r1","disposition":"superseded","successor":"r2","reason":"只读验证发现缺陷，新的修复任务获准改源码","checkpoint":".workspace/example/runs/r1/RESULT.md","evidence":".workspace/example/runs/r1/closure.json"}
```

closure.json 使用 run-closure@1，包含 project_id/run_id、summary、writers_stopped=true、jobs=[]、unsettled_resources=[]、非空 evidence 引用列表。必须来自实际核验；工具检查文件和声明，不能自行探测外部进程。checkpoint 可以是 blocked/interrupted/needs_input/failed/completed，但不能仍 running。终态 RESULT 和冻结 TASK 均不改写。

disposition 为 rejected/abandoned/superseded；只有 superseded 填新的 successor ID。closed 不等于 accepted，不能被用作已通过 analysis 或候选证据。所有成员 accepted 或安全 closed 后，才可切策略/开新批次；兄弟结果保留。重复 event_id 幂等，内容冲突拒绝。已有控制账本中的 unknown job 或未结算预留会阻止关闭，即使 closure 文件声称没有作业。

## 控制预算与恢复

新 automatic 批次在没有在途 run 的安全边界初始化控制账本；已有人工 RESOURCE_USAGE 先核对真实余额，再显式导入剩余额度与来源，不自动迁移历史。启用后 tree.control 是该批次步数、资源和 job 的事实源；RESOURCE_USAGE/AUTOMATIC/STATE 保存索引或摘要，不重复手工扣额。

authority 引用独立、不可变的决定快照，DECISIONS 只追加其索引；不要对持续追加的整个 DECISIONS 文件取摘要，否则正常新增决定会使旧引用失效。

所有操作通过 `searchctl apply operation.json`、使用真实 owner 和唯一 event_id：

| op | 主要字段 | 行为 |
| --- | --- | --- |
| control-init | step_limit, budgets, authority | 初始化一次；budgets 为 pool → {unit, limit}；authority 是真实决定的不可变快照路径 |
| control-pause / control-resume | authority | 保存模式决定；恢复不清零步数、不加额 |
| control-extend | step_limit, limits, authority | 有真实新额度决定才扩大原上限；limits 包含全部既有 pool，保持已用量 |
| control-reserve | id, run_id, pool, amount | 原子预留，余额不足拒绝；每项预留只有一种单位 |
| control-job | id, status, evidence；首次另有 reservation_id | id 是提交前固定的 request_id；首先记录 intent，之后更新实际观察 |
| control-settle | id, used, evidence | id 是 reservation_id；未决 job 禁止释放；真实超额照实登记并阻止后续超额预留 |

五节点 run 登记各占一步，develop-batch 按成员数占步，失败和安全关闭均不返还步数。接收/恢复既有工作不再占一步；宿主额外的问询/恢复控制动作仍须按 AUTOMATIC 约定记录，本工具不自动计量 agent 时间。暂停阻止新 run、预留与新 job 意图，允许记录已有 job 的新观察、结算和收尾。

job 状态：intent → submitted/running/unknown/exited/not_submitted；submitted/running 可进入 unknown/exited/cancelled，unknown 只能凭新观察恢复或确认终结。终态不重开。每次 evidence 是独立的 job-observation@1 JSON，包含项目/run/request/status、带时区 observed_at 和实际查询来源 source；不得重写已引用的观测文件。新观察时间不能早于旧观察。status=unknown 保留预留，不把会话退出、超时或空日志当作 job 已退出。

```bash
python3 .agents/skills/harness-resume/scripts/searchctl.py \
  --workspace . --project example reconcile
```

reconcile 只读输出事件水位、best、未结 run、关闭记录、旧规则接收、冻结输入漂移、预算余额和在途 job。它不解析任意 Markdown、不查询远端、不自动修复 STATE，也没有共享环境发布锁。协调者据此更新摘要，再按实际后端补充观察；查询工具不能认证用户授权或外部状态真实性。

## 搜索纪律

较大实验前用 `searchctl mechanisms` 查已有机制。add-node 的可选 plan 指向 search-plan@1 JSON，包含 project_id、mechanism_id、expected、falsifier、keep、discard、simplification、prior_nodes、new_evidence 引用列表及 changed_conditions。对同一 mechanism_id 必须列出全部已有节点，并提供新证据或明确变化条件；无依据的重复登记会拒绝。

原有节点不强制补造历史。机制查重只按稳定 ID，不能自动识别换名后的语义重复；协调者负责归一化。同机制的旧失败和关闭记录要参与判断。叠加优化写明有价值的移除对照或不适用原因，收益相当时优先简单方案；不为满足表格强制多跑设备。
