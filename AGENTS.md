# Harness development

本仓库的开发限于 `harness` 分支及其派生分支。开发文档放在 `docs/`，中间产物放在 `.workspace/`。对外技能维护在 `assets/skills/`，`.agents/skills/` 通过软链接使用同一份内容。

产品资源不得包含具体工作记录、私人机器路径或写死的外部仓库 commit。外部版本由用户项目选择；任务的实际版本与测量记录保存在私有运行产物中，通用模板和版本一致性检查仍需保留。

<!-- operator-harness -->
## Operator Harness

按 `.agents/skills/harness-new-project/references/project-management.md` 选择工作目录和项目；多项目需明确选择。继续任务先读 `.workspace/<project>/STATE.md`，再读当前 TASK 与关联证据。写入前核对 project.json 的状态和 owner。

新项目使用 harness-new-project，已有代码首次接入使用 harness-onboard。项目记录与独立源码 worktree 放在 `.workspace/<project>/`；共享环境描述放在 `.workspace/environments/`，项目绑定确切修订。PROJECT 固定集成规则、低成本预检及交付条件，规则见项目契约。

工作流：
- prepare-environment 确定并验证 Python、TileLang、torch/torch_npu 和 CANN；复用已有选择。
- establish-baseline 固定独立正确性参考、case、协议及适用性能目标。
- 无候选且有参考源码时，port-baseline 移植并验证初版；无性能源码时用 develop 的 initial-and-validate。
- 已有候选按 analysis → develop → analysis 迭代。develop 内完成实现、精度及任务要求的性能验证；单独测试使用 verify-only，诊断采集使用 collect-only。
- 精度或实现问题在本节点修复；有效但未达性能目标的初版/优化思路交回 analysis。TileLang 候选保持语义、融合与计时边界，AICPU/host 输入计算回退不在支持范围。

按 harness-resume 的节点契约记录 TASK/RESULT。新模板将机器字段放在头部 metadata 区块；@4 评测通过 harness-develop 的 evaluate.py 执行冻结适配器，自动生成证据。接收与最佳候选晋升分别执行。范围变化需要新任务时，先核对作业和资源，再安全 close-run。

同会话可连续执行；委派时传工作目录根和 TASK 路径，节点产物须支持新 agent 独立接续。并行 develop 使用独立 worktree，共享设备排队；协调者统一接收。授权内工作持续推进，实际缺失且影响当前动作的信息才补问。

status 只读汇报，next 执行一步，automatic 在目标和预算内连续协调。新 automatic 批次启用 searchctl control；runctl/evaluate 自动记录已配置的预留、作业观察和结算。未知作业先查询原请求。reconcile 汇报漂移与未结工作，STATE 保存摘要；check 做完整性扫描。

搜索树、机制历史和 hot/cooled 知识由 harness-resume 的 searchctl.py 管理；知识冷却保留交付时的真实人工批准，远端发布单独授权。冻结/归档使用 harness-archive-project，恢复使用 harness-restore-project；恢复后按记录选择下一步。旧 `.workspace/STATE.md` 显式按 legacy 读取并保留原路径。
<!-- /operator-harness -->
