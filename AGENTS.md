# Harness development

本仓库的开发限于 `harness` 分支及其派生分支。开发文档放在 `docs/`，中间产物放在 `.workspace/`。对外技能维护在 `assets/skills/`，`.agents/skills/` 通过软链接使用同一份内容。

<!-- operator-harness -->
## Operator Harness

继续算子开发、性能调优或分析时，先读 `.workspace/STATE.md`，再按其中链接读取目标、相关实验与证据。使用 `.agents/skills/harness-resume/SKILL.md` 中的接续与子 agent 交接流程；不要仅凭当前源码推断之前的测量结果。

新项目使用 `harness-new-project`，已有项目首次接入使用 `harness-onboard`。流程记录都放在 `.workspace/`，项目文档可直接编辑。
<!-- /operator-harness -->
