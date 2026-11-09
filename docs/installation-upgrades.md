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

升级前核对活跃任务。@4 run 固定评测 Python 代码身份；仅修改技能文档可继续接收，更改评测代码时使用原包完成在途任务，或安全关闭后重新登记。项目记录与外部作业保持原状。

源码仓库的 `.agents/skills/*` 软链接属于开发模式，直接随 assets/skills 和 Git 更新。安装器在写清单前明确拒绝这种目录，upgrade-plan 同样给出开发模式提示；受管安装使用独立目录中的文件副本。旧清单中已移除的文件由升级计划列出并删除；未登记的额外文件继续保留，无法根据文件名判断其是否为旧版残留。
