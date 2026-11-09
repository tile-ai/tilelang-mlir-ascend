# 诊断如何进入探索树

analysis 消费树中的机制历史，产出可接收的证据和实验提案；树与阶段的职责沿用 [搜索与知识](../../harness-resume/references/search-knowledge.md) 和 [执行控制](../../harness-resume/references/execution-control.md)。不创建独立的评分器、预算账本或第二棵树。

## 读取历史与确定归属

从 TASK 的 `search_node_id`、输入 candidate、既有 hot/cooled 引用和 `searchctl mechanisms` 开始，查看同机制成功、失败、parked/closed 节点以及适用条件。`mechanism_id` 按作用机制归一（例如减少某阶段重复 GM 读），不要因 tile 数值、技能名称或方案标题不同就当作新机制。

| 本次发现 | 提案 |
| --- | --- |
| 同一假设的参数调整、编译/实现修复或补证 | 留在原策略节点；新的 develop/collect-only run 仍沿用同一归属 |
| 有独立反证标准的新算法/布局/分片策略 | 建议创建子节点，说明父节点、输入候选和与其他方案关系 |
| 重试失败或关闭的机制 | 列全相关 prior_nodes，给出新证据或已变化条件；纯改名不构成重试理由 |
| 原观测无效、精度反例、所用知识条件改变 | 记录失效范围，建议重验/修复；不抹去失败节点或历史结论 |

analysis 的 RESULT 始终保留原 TASK 归属。子节点创建/选择、新 run 派发、hot 写入由 owner 协调者在接收后完成；切换前回收或安全关闭未结 run/job，遵守现有 control 余额。一个方案不是一次新实验，不因生成更多建议扣 run 步数或重置额度。

## 每个方案的最小交接

在 [诊断报告](../assets/templates/diagnosis.md) 中记录 `plan_id`、`mechanism_id`、输入候选、目标 case/阶段、证据定位和源码位置。给出以下内容：

- **解释与优先级**：目前瓶颈、证据强度和竞争解释；按目标差距、影响范围、验证成本和风险排序，不使用固定利用率阈值评分。
- **改动与准入**：最小修改、后端/API/lowering 依据、资源约束；implementable / experiment / design-only。有证据的静态假设可进入 experiment，但仍需满足可执行与契约条件。
- **验证与去留**：预期计数/流水变化、可证伪条件、全部必选精度与正式性能协议、重点退化 case、保留/淘汰条件及简化计划。指标改善但正式性能无收益时不能判成成功。
- **关系与成本**：独立、互斥或依赖；同机制历史、新证据或条件变化；原节点继续还是建议子节点、设备/仿真补采需要和有界预算请求。

较大实验的方案同时保存为现有 `operator-harness/search-plan@1`，供协调者 add-node 的 `plan` 引用；文件内至少包含 `project_id, mechanism_id, expected, falsifier, keep, discard, simplification, prior_nodes, new_evidence`，重试可增加 `changed_conditions`。`new_evidence` 是 `searchctl reference PATH` 生成的 `{path, sha256}` 引用列表，路径相对 workspace_root；不发明新的 JSON 字段代替契约。原节点内的小调整可在 RESULT 留同等内容，已有 search-plan 作为历史引用，不原地修改冻结提案。

例如，在同一布局机制失败后，新的输入分布证据可以支持重新比较；只有“再试一次”不能支持新分支。若当前只有 util 比例，无法区分访存延迟和同步等待，主推荐应是有界 collect-only，写明目标 case、真实 kernel/launch、所需 pipe/PMU、成功判据和返回 analysis 的条件。补采仍属于验证当前假设，不自动成为新策略。

## 接收后的知识

将可复用内容分成 observation、hypothesis、conclusion，关联原始证据、case/架构/版本/构建条件及反例，建议写入关联节点 hot。当前未验证的优化建议保持 hypothesis；设备实验验证后才能据实际范围形成 conclusion。cooled 的人工批准、失效版本处理及远端发布权限沿用现有契约。

交付一个明确主推荐和剩余候选方向；只在收益可独立归因且输入可比较时建议并行 develop。analysis 不自行派发其他 agent、改写兄弟候选或切换策略。

协调者按 [依赖队列](../../harness-resume/references/task-queue.md) 将 plan 转为工作项；一个 analysis 可指导多个 develop。后续 analysis 固定一个主候选和问题，按需读取多份冻结结果；可先分析已接收成员，也可显式等一组 settled 结果。它不读取仍在写入的候选目录，结果后来到达不能改当前输入。全组等待是编排决策，比较分组本身不形成屏障。
