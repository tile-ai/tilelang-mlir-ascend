# 安装清单、升级与回滚

首次安装保持原入口。安装器保存 `.agents/.harness/manifest.json`，只登记随包技能文件和 Harness AGENTS 区块的摘要；额外文件和 AGENTS 区块外内容归用户所有。重复安装只在随包内容一致时成功，冲突在复制前拒绝。

没有清单的旧安装，先提供与已安装内容确切匹配的旧 harness 源码包，通过当前安装器登记归属；不依据目录名猜文件属于 harness：

```bash
python3 scripts/install_codex_skills.py --workspace /path/to/workspace --adopt-from /path/to/old-harness
```

有用户修改或旧包不匹配时拒绝，先审阅差异；不能拿当前新包强行认领旧文件。采用相同内容的新安装无需额外 adopt。

```bash
python3 scripts/install_codex_skills.py --workspace /path/to/workspace --upgrade-plan
python3 scripts/install_codex_skills.py --workspace /path/to/workspace --upgrade PLAN_ID
python3 scripts/install_codex_skills.py --workspace /path/to/workspace --resume TRANSACTION_ID
python3 scripts/install_codex_skills.py --workspace /path/to/workspace --rollback TRANSACTION_ID
```

upgrade-plan 只读输出受管文件前后摘要和稳定计划 ID。执行前重新检查源包及目标状态，变化即拒绝旧计划。只改未被用户修改的受管文件；移除旧资源也仅限旧清单确认为随包且摘要一致的文件，保留额外文件和目录。

每次升级在 `.agents/.harness/transactions/` 保存前后内容及清单，再逐文件原子替换，最后更新安装清单。中断后显式 resume，目标文件只能与计划前或计划后内容一致；发现第三种内容停止，不覆盖新编辑。rollback 同样先检查所有目标，恢复旧受管内容与清单。回滚中断后 resume 继续原回滚方向；不能越过后续不同版本强行回滚。

这些工具不迁移或修改 `.workspace/`、项目授权、TASK、RESULT 和外部作业。执行前由宿主核对活跃任务；@4 run 记录工具与资产摘要，升级会阻止其继续使用不同规则 accept。需要继续原 run 时使用对应旧包，或在安全关闭原任务后建立新任务；不通过重写旧摘要解决冲突。升级工具不认证人类批准、不代替项目交接，也不自动推送/发布。
