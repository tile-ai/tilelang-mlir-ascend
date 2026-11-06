<!-- operator-harness -->
## Operator Harness

先按 `.agents/skills/harness-new-project/references/project-management.md` 选择工作目录和项目。默认只发现 active/freezing 项目；多项目不猜测，不使用全局当前项目。继续算子开发、性能调优或分析时，先读 `.workspace/<project>/STATE.md`，再按其中链接读取目标、相关实验与证据。使用 `.agents/skills/harness-resume/SKILL.md` 中的接续与子 agent 交接流程；不要仅凭当前源码推断之前的测量结果。

新项目使用 `harness-new-project`，已有项目首次接入使用 `harness-onboard`。项目记录放在 `.workspace/<project>/`，可写源码和专属依赖使用该目录内的独立分支/worktree；共享环境描述放在 `.workspace/environments/`，项目只绑定具体修订。一个项目可以包含多个算子，暂不提供子项目。写入前核对 project.json 的状态与 owner，同项目其他会话必须明确交接。

实验前使用 `harness-prepare-environment` 问清并验证环境。上板与仿真各自保存使用约定，后续从 `.workspace/<project>/ENVIRONMENT.md` 读取各自绑定，不重复询问已有设置。环境就绪后，需要建立或更新正确性参考与性能标杆时使用 `harness-establish-baseline`，固定来源、case、协议和实测证据。标杆就绪后先用 `harness-analysis` 分析实现思路，再按 `analysis → develop → device-test → analysis …` 推进；后续 analysis 通过测试反馈、仿真和流水分析定位当前 kernel 问题。`test` 对应 `harness-device-test`；达成目标或遇到停止条件就收尾。只对当前动作必需的环境变化或验证失败补问，离线分析不强制运行仿真。
允许同一会话连续完成多个节点；项目记录须足以让全新 agent 独立接续。按 `harness-resume` 的节点契约记录 TASK/RESULT；委派时仅传工作目录根和任务路径，子 agent 写所属产物，主 agent 汇总。单 agent 可自行完成执行与汇总。上下文过长时仅建议新开会话，不强制切换或因没有隔离工具阻塞；automatic 使用当前宿主的主/子 agent 协调能力，无委派工具可明确 inline 执行。

查看进展与建议使用 `harness-status`（只读）；直接推进一个下一步使用 `harness-next`；按目标和预算连续推进使用 `harness-automatic`。automatic 遇到真实缺口时问询、遇到完成或上限时收尾，不重复提交状态未知的作业。
算子探索的策略树与 hot/cooled 知识由 harness-resume 的 `references/search-knowledge.md` 和 `scripts/searchctl.py` 管理。策略节点包含多轮阶段执行，不替代现有 TASK/RESULT 或候选谱系；知识冷却须有交付时的真实人工批准，远端发布另行授权。
冻结或归档使用 `harness-archive-project`，恢复使用 `harness-restore-project`。恢复只激活项目，不自动恢复实验或 automatic。旧 `.workspace/STATE.md` 显式按 legacy 读取，禁止自动搬动历史或改写旧 TASK 路径。
<!-- /operator-harness -->
