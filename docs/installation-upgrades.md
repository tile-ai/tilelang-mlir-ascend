# 安装清单、升级、回滚与卸载

Operator Harness 通过跨平台 Node.js CLI（`tilelang-harness`，要求 Node.js >= 18，支持 Windows / macOS / Linux）完成安装与生命周期管理，推荐直接通过 `npx` 从本仓库 `harness` 分支运行，无需预先克隆源码仓库。

**安装位置与运行时 `.workspace/` 解耦**：
- **安装阶段**只负责安装全部受管技能、宿主指令区块和受管清单（支持 **Global 全局安装一次** 或 **Local 项目本地安装**），不会创建 `.workspace/`，也不会静默修改任何仓库的 `.gitignore`。
- **运行阶段**由各节点（`harness-new-project` / `harness-onboard` / `projectctl.py`）按当前打开的算子仓库各自管理 `.workspace/`。首次在 Git 仓库（含 linked worktree）内建立 `.workspace/` 时，工作流会引导用户执行 `echo ".workspace/" >> "$(git rev-parse --path-format=absolute --git-path info/exclude)"` 做本地忽略，避免污染团队共享的 `.gitignore`（仅在团队统一使用本 harness 时才建议写入 `.gitignore`）。

安装、升级、恢复、回滚与卸载在写入前均会检查目标及安装元数据的祖先路径；指向工作区外、路径别名或悬空目标的软链接一律拒绝。安装器保存 `<metadataDir>/manifest.json`（`operator-harness/installation@2`，向下兼容旧版 `@1` 清单），只登记随包技能文件和 Harness 指令区块（`<!-- operator-harness --> ... <!-- /operator-harness -->`）的摘要；额外文件、指令区块外内容以及 `.workspace/` 归用户所有。重复安装只在随包内容一致时成功，冲突在写入前拒绝。

## 支持的宿主与安装路径

| `--host` | 别名 | Global（`--global`，推荐装一次） | Local（`--local`，单项目目录） |
| --- | --- | --- | --- |
| `codex`（默认） | - | `~/.codex/skills/` + `~/.codex/AGENTS.md` | `.agents/skills/` + `AGENTS.md` |
| `antigravity` | `agy` | `~/.gemini/config/skills/` + `~/.gemini/GEMINI.md` | `.agents/skills/` + `AGENTS.md` |
| `claude-code` | `claude` | `~/.claude/skills/` + `~/.claude/CLAUDE.md` | `.claude/skills/` + `CLAUDE.md` |
| `agent` | `.agent`, `generic` | `~/.agent/skills/` + `~/.agent/AGENTS.md` | `.agent/skills/` + `AGENTS.md` |

- 安装到 `global` 或非 `.agents/skills` 的 `local` 宿主时，随包指令区块与技能文档中引用本套件受管技能的路径会自动改写为对应路径（例如 `~/.codex/skills/harness-*`、`.claude/skills/harness-*`），外部仓库路径（如 TileLang 仓库内的 `.agents/skills/tilelang-ascend/...`）保持原样。
- **Global 模式下 4 个宿主路径完全互不重叠**，可同时全选安装（如 `--global --host codex,antigravity,claude-code,agent`）。
- Local 模式下，使用不同指令文件的宿主可共存（如 `--local --host codex,claude-code`）；共享同一指令文件但技能目录不同的组合（如 `codex` 与 `agent`）会在写入前被拒绝。

## 首次安装与检查

```bash
# 交互式安装向导（推荐：先选 Global / Local，再勾选宿主）
npx github:tile-ai/tilelang-mlir-ascend#harness

# 非交互式：全局安装一次（所有仓库通用）
npx github:tile-ai/tilelang-mlir-ascend#harness init --global --host codex,claude-code
npx github:tile-ai/tilelang-mlir-ascend#harness init -y

# 非交互式：仅安装到指定项目目录
npx github:tile-ai/tilelang-mlir-ascend#harness init --local --dir /path/to/repo --host codex

# 体检（默认同时检查本地与全局安装，以及当前 Git 仓库 .workspace/ 忽略状态）
npx github:tile-ai/tilelang-mlir-ascend#harness doctor

# 本地源码仓库调试可直接运行：
node bin/tilelang-harness.js
node bin/tilelang-harness.js doctor
```

> 若所用 npm 版本禁用了 Git 来源并提示 `EALLOWGIT`，加上 `--allow-git=all`（例如 `npx --allow-git=all github:tile-ai/tilelang-mlir-ascend#harness`）。

## 旧安装认领

没有清单的旧本地安装，先提供与已安装内容确切匹配的旧 harness 源码包，通过 `adopt` 登记归属；不依据目录名猜文件属于 harness：

```bash
npx github:tile-ai/tilelang-mlir-ascend#harness adopt --local --dir /path/to/repo --from /path/to/old-harness
```

有用户修改或旧包不匹配时拒绝，先审阅差异；不能拿当前新包强行认领旧文件。采用相同内容的新安装无需额外 adopt。

## 升级、中断恢复与回滚

```bash
npx github:tile-ai/tilelang-mlir-ascend#harness upgrade --plan --global --host codex
npx github:tile-ai/tilelang-mlir-ascend#harness upgrade --apply <PLAN_ID> --global --host codex
npx github:tile-ai/tilelang-mlir-ascend#harness resume <TRANSACTION_ID> --global --host codex
npx github:tile-ai/tilelang-mlir-ascend#harness rollback <TRANSACTION_ID> --global --host codex
```

本地项目安装将 `--global` 换为 `--local --dir /path/to/repo` 即可；仅存在单一安装时也可省略范围与 `--host` 自动发现。

- `upgrade --plan` 只读输出受管文件前后摘要和稳定计划 ID。执行前重新检查源包及目标状态，变化即拒绝旧计划。只改未被用户修改的受管文件；移除旧资源也仅限旧清单确认为随包且摘要一致的文件，保留额外文件和目录。
- 技能改名走同一机制：`ascend-mfu` 已整合为 `ascend-roofline`。升级到含新名称的版本时，旧目录下摘要一致的受管文件被删除、新目录被安装，用户在该目录留下的额外文件（如个人笔记）保留；旧文件被本地修改时计划失败并报告 `user-modified`，先审阅差异再决定重新 adopt 或手工迁移。安装器仍按旧名称识别受管路径，因此无需先删目录。
- 每次升级在 `<metadataDir>/transactions/` 保存前后内容及清单，再逐文件原子替换，最后更新安装清单。中断后显式 `resume`，目标文件只能与计划前或计划后内容一致；发现第三种内容停止，不覆盖新编辑。`rollback` 同样先检查所有目标，恢复旧受管内容与清单。回滚中断后 `resume` 继续原回滚方向；不能越过后续不同版本强行回滚。

升级前核对活跃任务。`@4` run 固定评测 Python 代码身份；仅修改技能文档可继续接收，更改评测代码时使用原包完成在途任务，或安全关闭后重新登记。项目记录与外部作业保持原状。

## 卸载

```bash
npx github:tile-ai/tilelang-mlir-ascend#harness uninstall --global --host codex --dry-run
npx github:tile-ai/tilelang-mlir-ascend#harness uninstall --global --host codex
```

`uninstall` 仅删除 `manifest.json` 登记的受管技能文件，并仅剥离指令文件中的 Harness 区块（若指令文件仅包含该区块则删除该文件），完整保留用户自定义文件、区块外说明、全局宿主配置目录（如 `~/.codex`）以及各仓库的 `.workspace/`。若检测到用户修改了受管文件或指令区块，默认拒绝卸载；确认覆盖时显式传入 `--force`。

## 源码仓库开发模式

源码仓库的 `.agents/skills/*` 软链接属于开发模式，直接随 `assets/skills/` 和 Git 更新。安装器在写清单前明确拒绝这种目录，`doctor` 会将其识别为 `source_linked_dev`；受管安装与升级必须使用文件副本。
