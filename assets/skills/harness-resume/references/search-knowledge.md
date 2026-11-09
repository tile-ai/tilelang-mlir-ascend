# 项目搜索树与分层知识

搜索节点 `search_node_id` 表示一个优化方向/核心假设。现有 TASK/RESULT 的 `node` 仍表示 analysis、develop 等工作流阶段。一个搜索节点包含多轮阶段执行和多个候选；报错修复、参数调整和同一假设的验证留在原节点，需要独立比较的新策略建立子节点。树的父子关系不是候选的 `parent_candidate` 谱系。

## 协调与恢复

- 每个项目有独立 `.workspace/<project>/search/tree.json`，绑定项目级目标契约。项目可包含多个算子及融合策略，operator 字段用于描述该探索对象或算子组；节点 scope 与 TASK 明确具体范围。不同任务的结果不可仅凭同项目归属混作一个基线。根目标改变时保留旧项目并建立新项目，不改写旧树身份。
- 新的算子探索初始化 N0；已有任务在下一个安全边界接入。历史归属不明确时不臆造节点、循环或结论，原记录仍可读取。已有在途任务先按旧契约回收。
- 先恢复在途 run/job、接收终态结果，再选择策略节点和阶段。`select` 不执行实验；切换前必须接收树内所有未结 run，协调者还须检查外部作业和资源账本。工具不判断 job 是否已退出。
- 同一假设继续原节点；新策略用 `add-node`，再 `select` 并记原因；回到旧分支同样 select。closed/parked 节点先用 node-status 重新 open。失败和无收益的分支保留。
- TASK、RESULT、候选清单记录准确的 `search_node_id`；重测旧候选只新增 run，不复制候选或节点。一个 run 对应一次阶段调用，不强制固定数量的 run 构成完整 loop。
- 树存引用和 SHA256，不复制候选代码。Hot 登记时将所引用的证据字节保存到 `.workspace/<project_id>/knowledge/evidence/<sha256>.snapshot`，并记录源路径供定位；校验以不可变快照为准，源 RESULT 的后续更新不影响历史 hot。STATE 链接树和当前节点，仍管理当前/最佳已验证候选；EXPERIMENTS 仍记录执行结果。切分支不会自动更换最佳候选、重置预算或提升验证状态。
- 仅当前项目 owner 协调者写账本。helper 核对 project.json、project_id 和生命周期，再对修改加 POSIX 文件锁并原子替换 JSON；每次操作使用稳定 event_id，相同内容重试不重复登记，不同内容复用 ID 报错。锁串行化写入，不授予协调者身份。
- TASK、候选清单、已接收的终态 RESULT 和知识快照冻结后不能原地编辑；变化应建立新 run/候选/hot 修订。运行中的 RESULT 可按原节点契约更新，hot 自动保存其当时的字节，但阶段状态和证据是否充分仍需人工/agent 判断。旧版本中直接绑定源文件的 hot 不自动迁移：摘要已失配时，须从备份或 checkpoints 恢复匹配的原始字节，再登记新修订；不得改写旧摘要掩盖历史变化。中断可能留下未索引的 hot 内容文件，恢复重试原操作即可，不重做实验。跨机器恢复须带上完整 `.workspace/` 和引用产物。

## Hot → cooled

hot 是本次任务知识，关联一个或多个树节点；每次修订保留原版本。明确 observation / hypothesis / conclusion，附适用条件、限制/反例和原始证据。假设可以被保留，但不能包装成已经验证的结论。

交付时运行 review，整理、去重并选择值得共享的条目；在 hot 中先完成文字与适用范围的修订，再请人审批准确修订。未选择、证据不足或未批准的知识继续保持 hot，不阻塞算子交付。冷却不删除 hot。

promote 要求真实人工决定的来源记录、reviewer、日期、交付记录、hot 修订及摘要。Agent 不得自己编造批准，也不得把自动模式或完成任务当成批准。**本地 CLI 只校验记录，不能认证人类身份或证明审批记录真实**；宿主/协作流程负责取得真实批准。

cooled 默认写到工作目录的 `knowledge/cooled/<project>/`，也可用 `--output-dir` 指定工作目录内、`.workspace/` 外的版本管理目录。共享包包含适用条件、限制、可独立理解的 evidence_summary、源 hot/证据摘要及审批溯源，不携带本地证据路径或完整审批对话。人工检查摘要足以复核，必要时附可发布的复现材料。工具不自动脱敏自由文本、不上传远端；发布由用户授权的独立 Git 流程处理。

外部 cooled 先复制/检出到项目内，按节点需要用 use-cooled 绑定准确版本并解释适用性，不把全库塞进上下文。一个目录代表完整知识集合，分发时保留全部版本；不得单独拷贝已过期的旧 active 文件冒充最新知识。invalidate-cooled 生成 needs_revalidation 或 superseded 后继版本，不修改旧文件；存在后继的旧版本不能新增引用。已记录的历史引用仍可审计，继续探索前检查后继/状态和工具链条件。重新验证后创建新的 hot 并经人工批准产生新的 active 版本。没有全局跨仓库撤销服务。

## CLI（Python 3.9+，Linux/macOS，无第三方依赖）

安装器会随 harness-resume 分发脚本。项目先由 projectctl 创建；以下命令从工作目录根执行，SESSION 使用实际协调者标识；仓库开发时把 `.agents` 换为 `assets` 也可。

```bash
python3 .agents/skills/harness-resume/scripts/searchctl.py --workspace . --project example --owner SESSION init \
  --operator example-op --task-contract requirements-v1 \
  --hypothesis '建立初始实现' --scope '需求规定的全部 case'
python3 .agents/skills/harness-resume/scripts/searchctl.py --workspace . --project example --owner SESSION apply operation.json
python3 .agents/skills/harness-resume/scripts/searchctl.py --workspace . --project example show
python3 .agents/skills/harness-resume/scripts/searchctl.py --workspace . --project example check
python3 .agents/skills/harness-resume/scripts/searchctl.py --workspace . --project example review
python3 .agents/skills/harness-resume/scripts/searchctl.py --workspace . --project example --owner SESSION promote decision.json
```

`show/check/review` 只读，不创建目录、锁或修复记录；输出 JSON；show/check 的 cooled_warnings 标明历史引用的知识已失效、出现后继版本，或所在集合有不可读/非对象 JSON。use-cooled 和 invalidate-cooled 将无效邻居报告到 stderr 后继续检查有效对象；promote 遇到这种集合会停止，以免无法确认审批 ID 唯一性。`check` 验证树结构、候选谱系、归属和引用摘要，不证明精度、性能、审批真实性或资源额度。坏记录停止写入，保留现场，不自动修复。操作 JSON/decision JSON 是输入文件，路径相对当前 shell；内部 artifact 路径一律相对 `--workspace`（旧参数名 --project-dir 仅为此参数别名）。禁止逃出工作目录或引用其他项目私有记录；共享环境与工作目录内已发布知识可按范围读取。旧布局只用 `--legacy show/check/review` 读取，不自动迁移。

### apply 操作

每条操作带唯一 `event_id`。节点 ID、候选 ID、run ID、hot ID 独立。下例可各自保存为 operation.json；顺序依赖须先满足。

```json
{"event_id":"e1","op":"add-node","id":"N1","parent":"N0","hypothesis":"减少重复访存","scope":"大 shape"}
```

| op | 除 event_id、op 外的字段 | 约束 |
| --- | --- | --- |
| add-node | id, parent, hypothesis, scope | 父节点已存在；不改父关系 |
| select | id, reason | 无未接收 run，节点 open |
| node-status | id, status, reason | open / parked / closed；有未结 run 不可关闭 |
| candidate | id, search_node_id, parent_candidate, manifest | parent_candidate 是已登记 ID 或 null；manifest 中对应字段为父 manifest 路径或 not_applicable（可附 `: 原因` / `：原因`） |
| develop-batch | id, candidate_id, inputs, members | 至少两个独立 develop；具体字段与前置检查见 [并行开发契约](parallel-develop.md) 和 JSON 模板；跨 open 策略原子登记，不切换 current_node |
| run | id, search_node_id, stage, candidate_id, task | 当前 open 节点；stage 为 port-baseline / analysis / develop，port-baseline 和普通 develop 独占执行；candidate_id 为输入候选或 null（尚无候选） |
| accept | id, result；旧 develop 可补 scope | develop 要求 completed 并通过精度/性能归因检查；port-baseline 要求 completed、精度通过、有效性能和完整移植证据，允许性能未达标；其他阶段为 completed/failed；checkpoint 不接收。scope 仅用于新规则前登记任务的独立范围说明，含 project_id/run_id/operation/performance_required |
| hot | id, base_revision, kind, statement, applicability, limitations, nodes, evidence | base_revision 初次 0，后续须匹配最新；nodes 为节点 ID 列表，evidence 为项目内证据路径列表 |
| use-cooled | search_node_id, artifact, reason | artifact 指向当前 active cooled JSON；reason 解释条件匹配 |

TASK/RESULT/candidate 必须有与所选项目一致的 `- project_id: ...`。TASK/RESULT 用模板的 `- run_id: ...`、`- node: ...`、`- search_node_id: ...` 精确字段；candidate 使用 `- candidate_id: ...` 和同样的 search_node_id。新测试任务使用 develop 的 verify-only/collect-only，历史 device-test 记录只读/接收兼容。TASK 在执行前登记，RESULT 接收前由协调者按节点契约检查任务摘要、交付、环境、作业和资源；helper 的 accept 对 develop 校验精度/性能与退化归因，对 port-baseline 校验精度/有效性能、PORTING 完成与 AICPU 替代声明及候选身份，冻结证据摘要；不解析全部样本或证明归因正确，不能代替完整结果验收。候选在生产后登记，develop 的输入候选可以来自父节点或其他分支。

```json
{
  "event_id": "e-hot-1", "op": "hot", "id": "H1", "base_revision": 0,
  "kind": "observation", "statement": "布局调整在本次 case 上减少了访存耗时",
  "applicability": "列出 shape、dtype、设备和工具链版本、测量协议",
  "limitations": "小 shape 未验证；不能推断端到端收益",
  "nodes": ["N0", "N1"], "evidence": [".workspace/example/runs/r1/RESULT.md"]
}
```

### 人工决定与失效

review 的 `latest_hot` 数组输出每条最新知识的 `hot`（含 revision）和 `artifact`（含 sha256），与树中的代码 `candidates` 区分。审批绑定两者；hot 更新后旧审批不能触发新的晋升；已完成晋升的相同决定仍可幂等重试。同一输出知识目录内，决定 ID 不能绑定另一条 hot 或另一份审批；失效后继可以保留原审批来源。相同 hot 修订和决定 ID 的重复 promote 返回同一不可变文件，内容冲突报错。每个决定 ID 应唯一标识一次决定，不得复用于不同审批。

```json
{
  "schema": "operator-harness/cooling-decision@1",
  "id": "review-1", "approved": true,
  "reviewer": "actual-reviewer", "reviewed_at": "2026-01-01T00:00:00Z",
  "delivery": "delivery-1", "approval_source": ".workspace/example/decisions/review-1.md",
  "hot_id": "H1", "revision": 1, "hot_sha256": "COPY_FROM_REVIEW",
  "evidence_summary": "可独立复核的条件、样本/比较方法、实测结论和边界；不能只写私人路径"
}
```

```bash
python3 .agents/skills/harness-resume/scripts/searchctl.py --workspace . --project example --owner SESSION invalidate-cooled \
  knowledge/cooled/example/H1--review-1.json --status needs_revalidation \
  --version revalidation-1 --reason '工具链变化，需要重新验证'
```

此版本不提供 MCTS、自动策略评分、向量检索或自动发布。由 agent 基于已有证据选择节点；搜索树组织决策，现有阶段工作流执行实验。

search@1 新增可选 batches 索引，成员 run 带 batch_id、analysis_run、repo_ids；旧树不要求这些字段。普通 run 仍受 current_node 约束；只有 develop-batch 可跨策略登记。同批成员乱序 accept，未接收完不切换策略或启动新工作；登记幂等、全部成功或全部不落盘。helper 不核实科学结论，不代替主 agent 的任务输入/权限/资源与结果接收检查。

develop 新登记 run 带 acceptance_policy；接收保存 acceptance 的证据摘要，后续读取校验漂移。历史已接收记录不伪造新版接收结论；旧未接收任务补 scope 时保留原 TASK，协调者核对补充范围与原约定一致，不能通过补充降低标准。新任务禁止替换冻结 scope。
