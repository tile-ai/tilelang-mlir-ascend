# 工作记录与交接

记录服务于不继承聊天的新 agent。[节点交接契约](node-contract.md) 规定六份模板、必填内容和所有权；有明确目标的节点工作留下 TASK 和 RESULT，短任务可简写，阶段内连续操作无需逐个建模板。信息完整是接续要求，不强制切换 agent 或会话。

## 主 agent 的读取入口

先按 [项目管理契约](../../harness-new-project/references/project-management.md) 选择项目，以 `.workspace/<project>/` 为记录根：

1. `STATE.md`：当前任务与待派发/在途 run、实际 agent ID、下一步。
2. `REQUIREMENTS.md`：语义、case、精度与性能验收；执行前读取 PROJECT 的适用契约修订，按需展开 ROADMAP。
3. `DECISIONS.md`：已回答的问题、作用域、来源及替代关系；没有新决定时不创建空文件。
4. `ENVIRONMENT.md`：执行阶段到确切环境修订的绑定、本项目使用约定；development/device 可共用 onboard 描述，仿真按其独立约定。
5. `RESOURCE_USAGE.md`：首次预留资源前建立的预留、消耗、未结算作业账本。
6. `EXPERIMENTS.md`：只展开与当前任务有关的实验和证据。
7. `BASELINE.md`（相关时）：标杆索引，解析到确切 run 的参考/测量记录与摘要。
8. `AUTOMATIC.md`（如有）：当前自动批次、协调者、已用步数、在途步骤与停止原因，避免重复协调。

多算子项目由 STATE 指向本任务的需求与记录，不混用候选。记录与产物通常被 Git 忽略，迁移 worktree/机器必须携带相关输入和证据。

PROJECT 按 [项目契约](../../harness-new-project/references/project-contract.md) 保存集成规则、检查计划与交付要求，TASK 固定适用修订。实际早期检查位于 run 内 PRECHECK.md，最终版本对照位于 deliveries/<delivery-id>/DELIVERY.md；STATE 只链接这些事实，不复制检查结果。

## 节点的读取入口

节点从派发的 `runs/<run-id>/TASK.md` 开始，读取指定 skill、STATE 中的新限制及 TASK 的有序输入清单。TASK 指向具体需求、候选、环境、前序结果与执行方法；不遍历整个仓库猜主 agent 想做什么，也不由 STATE 的下一步自行换任务。

新执行者先在 RESULT 记录恢复出的任务理解与输入自检；原会话连续推进也保持同样的事实记录。缺口、用户问题、在途 job 立即写入记录；执行期间只改 TASK 允许的文件，完成后输出结果路径。主 agent 的派发消息不能成为唯一事实来源。

## 当前状态与实验索引

STATE 保留以下简短事实及链接：任务目标、当前标杆 ID/修订/可用性、当前分析及其绑定的标杆/候选/测试结果、当前候选、最佳已验证候选、逐 case 性能概况与测量边界、当前 run/agent/job、未决问题、下一步。修改候选不自动更新最佳；尚无证据就写未测。

EXPERIMENTS 按运行记录目标/假设、父候选、改变、精度/性能或未测项、接收结论和 RESULT 链接，失败和无收益也保留。详细日志、源码快照、结果表和约定快照放各自 run 目录，索引不复制它们。

委派时主 agent 接收结果后更新共享文件；单 agent 工作时当前 agent 完成节点收尾检查后直接汇总。委派的子 agent 只提交环境/协议/参考变更提案，不能直接改全局记录。已有用户决定仍有效时复用；新冲突只补问相关部分。

## 派发示例

派发前已创建完整 TASK。消息只需：

```text
工作目录根：{{workspace_root}}
任务文件：.workspace/<project>/runs/E004/TASK.md
读取任务及其指定 skill，执行本节点并将结果写回指定目录。
```

这是使用全新上下文时的交接方式，不要求每次都创建 agent。当前会话可以连续推进多个节点，复杂工作可委派；上下文过长时只建议用户新开会话，没有隔离工具仍可继续。automatic 由主 agent 在当前宿主编排子 agent，无委派工具时注明 inline，不要求用户手动开会话。任何执行方式下共享设备都按环境约定和资源账本安排。

## 可复现证据

[执行记录](execution-records.md) 规定实际命令、候选、环境、输入与原始输出之间的关联。源码快照或 commit 加完整补丁必须包含实际构建输入；多 case 比较保留全部结果。仿真 cycle、设备时间与端到端延迟分开。

中断时节点 RESULT 留下最后完成步骤、实际 job、产物位置、剩余额度和恢复命令。主 agent 先确认原 agent/job 状态，再由当前或新的 agent 接续；未结束任务可以沿用原 run，接管前确认原执行者不再写入。不能把启动新上下文当作重跑许可。

## 搜索与知识索引

启用搜索树后，STATE 链接 `.workspace/<project>/search/tree.json` 并摘要 current_node；详细策略、run 和候选归属以树为准，当前/最佳候选仍按已有验收规则更新。EXPERIMENTS 继续记录实验，不承担树结构。节点内读取相关 hot 和准确版本的 cooled；恢复时校验引用摘要，后继版本/工具链变化触发适用性复核。详见 [搜索与知识](search-knowledge.md)。
