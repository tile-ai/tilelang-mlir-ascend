# 依赖队列与工作空间生命周期

`search@2` 在同一 `search/tree.json` 保存队列、run 与工作空间占用。编排 agent 选择何时执行哪个任务，工具检查依赖、输入、资源和预算。没有全局 FIFO，也没有默认“全部 develop 完成才 analysis”的屏障。

探索树组织机制和反例历史，队列组织工作，run 是一次执行。一个 analysis 可提出多个 plan，分别产生 develop 队列项；一个 analysis 也可读取多个已收尾结果，但仍有一个主候选和 search_node_id。current_node 只是导航，不能覆盖 TASK 的归属。队列本身不生成结论、不自动晋升候选或冷却知识。

## 启用与读取

所有 CLI 使用同一 workspace 根和协调 owner；下文 `$SEARCHCTL` 指已安装的 harness-resume/scripts/searchctl.py，变量由当前任务设置。

```bash
python "$SEARCHCTL" --workspace . --project <project> --owner <owner> apply queue-enable.json
python "$SEARCHCTL" --workspace . --project <project> ready
python "$SEARCHCTL" --workspace . --project <project> reconcile
```

写操作均为 JSON，带项目内唯一 `event_id`，通过 `apply` 提交。例如启用操作：

```json
{"event_id":"enable-queue","op":"queue-enable","max_parallel":2,"authority":".workspace/<project>/decisions/queue-authority.md"}
```

在无未结 run/job、资源已结算的边界显式升级。已有 search@1 项目保持 [旧批次契约](parallel-develop.md)，运行中的任务先按原工具束收尾，不自动迁移。新建项目在进入 analysis/develop 前启用队列；max_parallel 默认 2 或使用已有宿主配置，analysis 同样占槽。旧工具拒绝 search@2；启用后保留账本，不能删字段降级。

`ready` 是只读投影，显示 ready/blocked/in_flight/accepted/closed/cancelled、首个阻塞原因、优先级、延后历史和未释放占用。它依据提案的依赖与资源判断，完整 TASK/评测计划/比较契约在准入时再校验。只按 priority 排序，选择权属于编排 agent。`reconcile` 同时报告漂移、未结作业、控制余额和队列。STATE/AUTOMATIC 仅引用账本与事件水位，不再维护第二份可编辑队列。

## 冻结输入

```bash
python "$SEARCHCTL" --workspace . --project <project> --owner <owner> freeze <workspace-relative-file>
```

返回 `{path, sha256}`，副本位于 `.workspace/<project>/snapshots/<sha256>/<filename>`。普通文件按字节复制；artifact-index@1、evaluation-plan@1、diagnostic-evidence@1 中显式的引用递归复制并更新摘要。每次复用仍校验摘要，内容寻址不等于操作系统只读权限。

多文件源码、补丁和未跟踪文件先构成 `{"schema":"operator-harness/artifact-index@1","files":[{"path":"...","sha256":"..."}]}`，枚举完整重建所需内容再 freeze。冻结索引保留 `restore_paths`（原 workspace 相对路径到快照路径），恢复时映射到本 run 被分配的 worktree，不能照原路径覆盖兄弟目录。未列出的依赖不会自动收集。普通 JSON 内的自由文本路径不自动解析；适配器应使用冻结 context 的 artifact 引用。

候选 source/build/launch 必须指向上述快照；只有分支名、commit、活动 worktree 路径或可漂移的压缩包旁边文件不足以准入。已有候选要重新固定完整制品、生成新候选 ID，不能改旧 manifest。队列 run 的 evaluate.py candidate 自动冻结所提供的 source/build/launch（含显式索引），随后生成候选清单；执行适配器也必须读取清单中的确切快照。

评测计划先固定所有输入，再 `freeze plan.json`，TASK 指向返回的计划；initial_design 单独 freeze。队列 inputs 中的需求、环境、方案及其他必需材料也用快照引用。前序 RESULT/关闭记录及候选 manifest 已有终态身份，由准入自动绑定摘要；写入者不得覆盖终态证据。引用原始上板数据时列全文件，冻结索引后再交 analysis。

## 入队、修订与依赖

```json
{
  "event_id":"enqueue-Q1","op":"enqueue","id":"Q1",
  "spec":{
    "stage":"develop","operation":"implement-and-validate","search_node_id":"N1",
    "candidate_id":"C0","analysis_run":"A0","plan_id":"P1","mechanism_id":"reuse",
    "requires":[{"run_id":"A0","condition":"accepted"}],
    "inputs":{},"repo_ids":["variant1"],
    "workspaces":[{"domain":"local","path":".workspace/<project>/builds/Q1"}],
    "priority":10,"reason":"落实已接收的数据复用方案","budget_requirements":{}
  }
}
```

spec 支持 stage（analysis/develop）、operation、search_node_id、candidate_id、analysis_run、plan_id/mechanism_id、requires、inputs、repo_ids、workspaces、priority、reason、budget_requirements、supersedes。analysis 的 operation 为 analyze 或 diagnose-failure；develop 沿用五种模式。首次构建/明确 code-only 可无候选；尚未绑定输出的 analysis 可先入队 candidate_id=null，准入前修订为确切候选。

依赖用 `run_id` 或 `queue_item_id` 二选一，目标须已登记，工具拒绝环：

- `accepted`：前序结果已接收且 status=completed；不会以失败/关闭代替。
- `settled`：前序已接收或安全关闭；尚未准入的队列项显式 cancelled 也满足收尾条件，记为未执行，不补造验证结果。

只分析 D1 就依赖 D1；等待 D1、D2 汇总就列两个 settled 条件。比较分组不会隐式增加依赖。常规候选还要求其 producer 已接收，并且是接收的那个输出；安全关闭的失败候选只能用于显式 diagnose-failure，不能据此启动普通优化。

`queue-revise` 使用 `id、revision、spec` 整体替换尚未准入提案，revision 自动加一；`queue-cancel` / `queue-defer` 使用 `id、revision、reason`。延后记录原因但仍可就绪；真正等待用依赖表示。准入后输入不可修改，需要新任务则安全关闭原 run 并建后继。精确重复的待办会拒绝，语义重复由编排 agent 结合 mechanism 历史判断。

## 准入、派发和工作空间

先准备独立 TASK@4（标准 runs/<run_id>/TASK.md），其 metadata 包含 queue_item_id、operation 和适用的 analysis_run、plan_id、mechanism_id。TASK 输入表与 queue spec/候选清单一致，写清获准目录、build/JIT/cache、远端路径和环境。implementation 必须引用同一主候选的已接收 analysis，多个方向可以共用该 analysis。

```json
{"event_id":"admit-Q1","op":"queue-admit","id":"Q1","revision":1,
 "run_id":"D1","request_id":"dispatch-D1","task":".workspace/<project>/runs/D1/TASK.md",
 "reason":"依赖齐全，工作空间和槽位可用"}
```

同一事务登记 run、占用代次、queue_inputs 和 control 步数；失败不留下半份账本。队列项和 request_id 只能绑定一次，重复同一 event 内容返回原结果。文件准备成功不代表准入或派发成功，准入后还需宿主保存/查询派发意图并启动执行者，记录实际 agent ID。响应丢失先查原 request，不能重派。

每个可写 develop 使用 projectctl 注册的独立 repo_ids，实际真实路径/分支必须一致；只读 analysis 不要求 worktree。附加本地工作空间只允许项目 builds 下的有界目录，run 输出自动占用。所有可写构建、JIT/cache、依赖源码都在 TASK 白名单及占用记录中，禁止共享可变安装或默认全局缓存。worktree 的共享 Git 元数据也不可并发修改。

远端资源使用 `domain: remote:<canonical-machine/container/mount>` 与规范绝对 path，并附 identity_evidence 引用。证据为 workspace-identity@1 JSON，包含相同 domain/path、canonical=true 和实际查询 source；宿主根据环境身份与 realpath 结果填写，不能只用 SSH 别名区分同一目录。工具检查声明与路径重叠，无法代替远端身份核验或任意 shell 沙箱。

可用槽位按未释放占用计数，接收后仍未停止的 writer 不视为空闲。`budget_requirements` 检查当前可用额度但不预留设备，实际执行前仍 control-reserve/control-job。共享设备预算池可声明 `exclusive_resource` 为同一个已核实物理资源 ID，项目内所有同 ID 池在预留未结算时互斥；跨项目仍使用环境/宿主共享锁。unknown 保留占用，不因预算池改名绕过互斥。

## 接收、释放与比较

接收沿用 accept 的精度/性能/采集 gate，晋升仍单独执行。D1 被接收且快照完整后，可在 D2 运行时准入 analysis A1；若槽位耗尽，先核实并释放已停止 D1 的占用。A1 的输入不会因 D2 后来完成而变化，纳入新证据需新分析任务。

释放操作带 `id: <run_id>、generation、evidence: <path>`。证据 JSON：

```json
{"schema":"operator-harness/workspace-release@1","project_id":"<project>","run_id":"D1",
 "generation":1,"writers_stopped":true,"jobs":[],"unsettled_resources":[],
 "evidence":[{"path":"<writer-and-job-observation>","sha256":"<digest>"}]}
```

run 必须已接收或安全关闭，控制作业和预留已结算，输出候选仍可独立读取；确认 agent 与远端 writer 已停止后才能填写真实观察。工具拒绝旧代次释放。释放不删除/重置 worktree，下次使用前保存未知改动并按新父候选恢复。路径登记和 generation 是协作约束，无法阻止越权 shell，宿主仍须落实执行范围。

`comparison-group` 操作包含 id、固定 members（queue ID）与 inputs，inputs 必须有 requirements/baseline/cases/protocol/environment 五个快照引用；在成员准入前创建，各成员必须是同父候选的 develop。准入校验评测计划的相同契约；group 不触发派发，也不代表全部通过。需要等全组时显式列全部成员依赖。不同条件结果只能定性分析，组合方案必须新候选重验。

## 恢复与生命周期

暂停先写 control-pause，再使所有节点停在安全检查点；已准入的接收/结算/释放可以继续，不能新准入或重投 unknown。next 只选择一个明确动作，不把整个队列算一步；automatic 可在预算和资源内选择多个 ready 项，并记录未选项的延后原因。owner 接管先确认原协调者/执行者状态，不重置步数。

prepare-environment、establish-baseline、port-baseline 保持维护屏障：全部 run 收尾、队列占用释放后才能执行，排队意图保留但须复核旧输入。停放/关闭搜索节点需要其关联 run 与 writer 收尾，无关策略不构成屏障。

冻结/归档沿用项目检查点，对仍保留的占用增加 `stopped_workspace_leases: {"<run_id>": <generation>}`，精确覆盖全部未释放占用。未结 run 还需原 paused_runs 检查点。确认 writer 已停后冻结项目，保留占用用于恢复；归档本身不释放或复用空间。恢复先核对原 owner、检查点与派发身份，再恢复原 run 或完成释放。
