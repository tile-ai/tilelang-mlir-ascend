# 结构化验收与执行控制

TASK/RESULT @4 使用单个头部 `<!-- harness-metadata -->` 区块保存机器字段，正文自由记录分析。候选 parent_candidate 使用 ID；历史模板的指定字段章节和父候选路径保持兼容。显式未知/重复版本拒绝，旧 @1–@3/无版本记录须以 legacy=true 补登记。

## 从生成到接收

1. 注册 run 前固定 evaluation-plan：需求、参考、比较器、case、协议、环境，以及必填 execution 中的适配器、backend 和超时。依赖引用统一相对 workspace_root，`searchctl reference PATH` 生成摘要；多文件用 artifact-index@1 的 files 列表。
2. 按 [评测入口](../../harness-develop/references/evaluation-runner.md) 执行。工具生成 accuracy/performance、执行回执、原始文件索引及 RESULT。RESULT 只需要一个 validation_evidence；原来的 accuracy_evidence/performance_evidence 仅用于 legacy 接收。
3. 接收前核对实现说明、适用预检和诊断覆盖。searchctl 检查当前任务、候选、全部必选输出、样本、回执和控制 job，并计算逐 case 目标。精度/实现错误留在节点修复；性能思路问题保留归因证据。
4. accept 与 promote-candidate 分开。晋升重新检查该 run 的证据，比较契约必须与已有 best 一致，包含 backend 和测量指标。code-only/collect-only 不获得最佳候选资格。best 是协调者选择的合格候选，工具不自动排序；协调者按预先固定的逐 case/业务权重规则记录选择理由。

旧 best 使用旧比较摘要时，只能从仍可核验的原冻结 plan 重建比较域：原 plan 必须具有 execution，旧摘要及已接收身份须完整匹配，backend/metric 与新候选相同后才允许新验证候选替换。旧历史与证据保持原样；缺 execution 时无法推断指标。其他跨比较契约迁移仍需另行设计。

plan 支持 performance_kind=measured/absolute/model/none，以及 mean/median 和逐 case 最大耗时。相对目标先由冻结标杆换算上限，换算依据及标杆文件列入 protocol 依赖。模型注明推导和假设；用户仅要求精度时，port-baseline/develop 都可使用 none。

implement-and-validate 必须产生新候选且父候选是本次输入。verify-only/collect-only 登记时必须已有候选，执行后保持同一清单。声明比较的成员须有相同输入、case/目标和统计方式，baseline 列入评测依赖；search@2 的独立队列项不因同时执行就被视为可比较。

采集脚本接受 evidence_context 后生成标准证据。CPU execution-receipt 关联实际子进程退出与冻结依赖；成功的 torch_npu 执行先保留 pending_guard_receipt，取得实际 device-receipt 后沿原 request 完成核验和结果整理。受控请求在此之前保留设备预留；最终回执的 resource_usage 绑定原 pool/unit：executions 的 used 为整数 1；seconds 为整个 guard 生命周期的实际正有限用量，且不小于原 process-receipt.elapsed_seconds。核验后幂等结算，纯设备 busy 秒数不能代替完整占用。精度失败保留反例且不能接收。process-receipt 只描述本地进程；设备最终回执、退出层及冲突观察另行绑定。完整性检查覆盖显式原始依赖，科学归因与诊断覆盖由协调者审阅。

## 读取、漂移与恢复

load/show 只检查账本结构；历史文件损坏不阻断读取、资源回收或安全关闭。

```bash
python3 .agents/skills/harness-resume/scripts/searchctl.py --workspace . --project example reconcile
python3 .agents/skills/harness-resume/scripts/searchctl.py --workspace . --project example check
```

reconcile 汇报事件水位、best、未结任务、预算/job、stale_inputs 和 corrupted_artifacts；check 显式扫描历史摘要，有损坏返回非零。当前任务接收和已接收候选的再次晋升仍严格校验其依赖；修复历史材料或新建验证任务后再复用。

run 绑定评测 Python 代码身份，技能说明和模板的文字改动不会阻断接收。harness_bundle 仅记录来源；更改评测代码需要沿用原代码完成该 run，或安全关闭后重新登记。旧接收记录保留原快照，新规则需要的补验通过新 verify-only 任务取得。

需要改范围、放弃或替代任务时，停止写入、核对原 job 并结算，再 close-run：

```json
{"event_id":"close-r1","op":"close-run","id":"r1","disposition":"superseded","successor":"r2","reason":"修复需要新范围","checkpoint":".workspace/example/runs/r1/RESULT.md","evidence":".workspace/example/runs/r1/closure.json"}
```

closure 使用 run-closure@1，包含 project_id/run_id、summary、writers_stopped=true、jobs=[]、unsettled_resources=[] 和非空 evidence 引用；引用实际查询/回收记录。checkpoint 可为 blocked/interrupted/needs_input/failed/completed。disposition 可选 rejected/abandoned/superseded，只有 superseded 关联新 successor。search@1 全批收尾才启动新工作；search@2 按 [队列依赖](task-queue.md) 选择后继，关闭后仍要显式 release-workspace。未知作业和未结预留继续阻止关闭。

## 控制账本

新 automatic 批次在无在途 run 的安全边界启用 tree.control。旧人工资源账本先核对余额再导入；STATE/AUTOMATIC 只保存摘要和水位。authority 引用真实决定的不可变快照，已有授权可直接复用。

所有操作通过 `searchctl apply operation.json`，带 owner 和唯一 event_id：

| op | 主要字段 | 行为 |
| --- | --- | --- |
| control-init | step_limit, budgets, authority | 初始化；budgets 是 pool → {unit, limit} |
| control-pause / control-resume | authority | 暂停新动作 / 按剩余额度恢复 |
| control-extend | step_limit, limits, authority | 根据新决定加额，保留已用量 |
| control-reserve | id, run_id, pool, amount | 原子预留 |
| control-job | id, status, evidence；首次另有 reservation_id | 提交前 intent，之后写实际观察 |
| control-settle | id, used, evidence | job 全部终结后按实际用量结算 |

runctl/evaluate 自动预留、观察与结算，并生成独立 job-observation；外部后端也须提供带时区 observed_at、source 及对应身份的观察。request_id 在项目内唯一，启动前核对所属 run、完整请求摘要及 pool/amount；冲突在启动前返回 not_submitted。已启动后的记账异常保留实际进程状态，unknown 保留预留并查询原请求。暂停允许原作业观察、结算和收尾。

程序 `control.step_limit` 计准入 run：队列项/批次成员各一步，失败不退步。AUTOMATIC 的 `max_steps/steps_used` 计宿主控制动作，另包含独立问询、接收或恢复；原节点的重连与收尾沿用原步骤。协调者分别保存宿主计数与程序计数水位，启动动作同时检查两类上限。程序目前不自动观测所有宿主动作，不能把 run 数称为完整控制步数，也不能用修改 Markdown 续增程序额度。

## 操作对象与变更权限

| 类型 | 对象 | 变更规则 |
| --- | --- | --- |
| 冻结输入 | TASK、plan、参考/比较器/case/协议、评测代码身份、候选快照 | 准入前由协调者固定；执行节点只读，变更保留旧任务并按安全关闭/后继流程处理 |
| 可编辑内容 | TASK 白名单内 worktree、独占 build/cache、非终态 RESULT 与运行产物 | 所属执行者在原范围与预算内更新，冻结后的候选和终态证据使用新身份 |
| 保留历史 | 账本事件、候选谱系、失败/关闭记录、已接收结果、hot/cooled 修订 | 工具追加事件或后继版本；保留原结果、证据摘要和审批来源 |
| 用户决定 | 范围/验收规则变更、加额、知识冷却、推送/合并/发布及破坏性操作 | 引用真实且适用的决定，既有授权直接复用；工具校验记录不等于认证授权来源 |

project.json 管理项目归属；search/tree.json 管理 run/candidate/接收/best 及启用后的 scheduling/control。STATE、EXPERIMENTS、AUTOMATIC、RESOURCE_USAGE 保存导航和带水位的投影；原始证据与回执支持判定。文件锁、摘要和范围登记面向合作工具，任意同账号代码隔离由独立后端提供。

## 搜索纪律

较大实验前使用 `searchctl mechanisms`。add-node 的 plan 可指向 search-plan@1，记录 mechanism_id、expected、falsifier、keep、discard、simplification、prior_nodes、new_evidence 及 changed_conditions。相同机制必须覆盖已有历史并给出新依据。旧历史无需补造字段；语义归一与有价值的消融由分析者判断。
