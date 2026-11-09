<!-- operator-harness -->
## Operator Harness

先按 `.agents/skills/harness-new-project/references/project-management.md` 选择工作目录和项目。默认只发现 active/freezing 项目；多项目不猜测，不使用全局当前项目。继续算子开发、性能调优或分析时，先读 `.workspace/<project>/STATE.md`，再按其中链接读取目标、相关实验与证据。使用 `.agents/skills/harness-resume/SKILL.md` 中的接续与子 agent 交接流程；不要仅凭当前源码推断之前的测量结果。

新项目使用 `harness-new-project`，已有项目首次接入使用 `harness-onboard`。项目记录放在 `.workspace/<project>/`，可写源码和专属依赖使用该目录内的独立分支/worktree；共享环境描述放在 `.workspace/environments/`，项目只绑定具体修订。一个项目可以包含多个算子，暂不提供子项目。写入前核对 project.json 的状态与 owner，同项目其他会话必须明确交接。

PROJECT.md 按 `.agents/skills/harness-new-project/references/project-contract.md` 和对应模板保存集成规则、早期检查与交付条件；TASK 固定适用修订。首次正式设备验收前完成适用的低成本预检，交付时用 DELIVERY 对照发布版与实测版，不将静态等价当作新版本实卡证据。

实验前使用 `harness-prepare-environment` 多轮确定 Python、TileLang/torch/torch_npu 与 CANN 的选择，再准备并验证。共享上板描述放在 `.workspace/environments/onboard/<id>/`，项目 ENVIRONMENT.md 绑定确切修订和自身使用约定，不重复询问已有设置；simulator 新接入后续设计，旧仿真约定独立读取。环境就绪后，需要建立或更新正确性参考与性能标杆时使用 `harness-establish-baseline`，固定来源、case、协议和实测证据。标杆就绪且无适用 TileLang 候选时，先用 `harness-port-baseline` 移植并完成初版精度、性能及分析数据交付，再按 `analysis → develop → analysis …` 推进；已有候选跳过移植，analysis 专注当前 kernel 缺陷。无适用性能源码但独立参考与目标已明确时，固定初版方案并使用 develop 的 initial-and-validate。TileLang 候选不引入 AICPU，移植节点自行寻找保持语义、融合要求及计时边界的替代方案，不能通过 host/AICPU 回退绕过约束。develop 内完成实现、精度验证和任务要求的性能测量；精度失败或优化实现问题必须在本节点修复，不能接收；精度通过且证据确认优化思路问题时，才将性能退化交给下一 analysis；单独测试也使用 develop 的验证模式；达成目标或遇到停止条件就收尾。只对当前动作必需的环境变化或验证失败补问，离线分析不强制运行仿真。
允许同一会话连续完成多个节点；项目记录须足以让全新 agent 独立接续。按 `harness-resume` 的节点契约记录 TASK/RESULT；新任务使用 @4 结构化验收，适用运行先固定 evaluation_plan，接收与最佳候选晋升分开。无法继续的任务核实执行者、作业及资源后安全 close-run，保留失败记录；委派时仅传工作目录根和任务路径，子 agent 写所属产物，主 agent 汇总。单 agent 可自行完成执行与汇总。多个独立 analysis 方案可按并行开发契约分配独立 worktree 的 develop，每个成员实现并测试；协调者统一接收，共享设备按约定排队。上下文过长时仅建议新开会话，不强制切换或因没有隔离工具阻塞；automatic 使用当前宿主的主/子 agent 协调能力，无委派工具可明确 inline 执行。

查看进展与建议使用 `harness-status`（只读）；直接推进一个下一步使用 `harness-next`；按目标和预算连续推进使用 `harness-automatic`。automatic 遇到真实缺口时问询、遇到完成或上限时收尾，不重复提交状态未知的作业。新批次在安全边界启用 searchctl control 账本，提交前记录资源预留与 job 意图；先用 reconcile 核对未结工作和预算，再更新 STATE 摘要。已有有效授权继续复用。
算子探索的策略树与 hot/cooled 知识由 harness-resume 的 `references/search-knowledge.md` 和 `scripts/searchctl.py` 管理。策略节点包含多轮阶段执行，不替代现有 TASK/RESULT 或候选谱系；知识冷却须有交付时的真实人工批准，远端发布另行授权。
冻结或归档使用 `harness-archive-project`，恢复使用 `harness-restore-project`。恢复只激活项目，不自动恢复实验或 automatic。旧 `.workspace/STATE.md` 显式按 legacy 读取，禁止自动搬动历史或改写旧 TASK 路径。
<!-- /operator-harness -->
