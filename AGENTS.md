# Harness development

本仓库的开发限于 `harness` 分支及其派生分支。开发文档放在 `docs/`，中间产物放在 `.workspace/`。对外技能维护在 `assets/skills/`，`.agents/skills/` 通过软链接使用同一份内容。

<!-- operator-harness -->
## Operator Harness

继续算子开发、性能调优或分析时，先读 `.workspace/STATE.md`，再按其中链接读取目标、相关实验与证据。使用 `.agents/skills/harness-resume/SKILL.md` 中的接续与子 agent 交接流程；不要仅凭当前源码推断之前的测量结果。

新项目使用 `harness-new-project`，已有项目首次接入使用 `harness-onboard`。流程记录都放在 `.workspace/`，项目文档可直接编辑。

实验前使用 `harness-prepare-environment` 问清并验证环境。上板与仿真各自保存使用约定，后续从 `.workspace/ENVIRONMENT.md` 读取各自绑定，不重复询问已有设置。环境就绪后，需要建立或更新正确性参考与性能标杆时使用 `harness-establish-baseline`，固定来源、case、协议和实测证据。标杆就绪后先用 `harness-analysis` 分析实现思路，再按 `analysis → develop → device-test → analysis …` 推进；后续 analysis 通过测试反馈、仿真和流水分析定位当前 kernel 问题。`test` 对应 `harness-device-test`；达成目标或遇到停止条件就收尾。只对当前动作必需的环境变化或验证失败补问，离线分析不强制运行仿真。
允许同一会话连续完成多个节点；项目记录须足以让全新 agent 独立接续。按 `harness-resume` 的节点契约记录 TASK/RESULT；委派时仅传项目根和任务路径，子 agent 写所属产物，主 agent 汇总。单 agent 可自行完成执行与汇总。上下文过长时仅建议新开会话，不强制切换或因没有隔离工具阻塞；automatic 使用当前宿主的主/子 agent 协调能力，无委派工具可明确 inline 执行。

查看进展与建议使用 `harness-status`（只读）；直接推进一个下一步使用 `harness-next`；按目标和预算连续推进使用 `harness-automatic`。automatic 遇到真实缺口时问询、遇到完成或上限时收尾，不重复提交状态未知的作业。
<!-- /operator-harness -->
