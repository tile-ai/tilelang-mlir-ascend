# 多项目定位、隔离与生命周期

## 三个位置

- `workspace_root`：用户选定的工作目录，安装一次 `.agents/skills/`。不因位于 Git 子目录就自动改成 Git 根。
- `project_id`：与用户确定的稳定目录名；一个项目可以包含多个算子，当前不提供子项目层级。
- `project_dir`：`<workspace_root>/.workspace/<project_id>`。记录、实验、专属源码、依赖和构建都归此目录；它不是源码仓库根。

TASK v2 的路径字段相对 workspace_root；Markdown 链接仍相对所在文件。实际命令总是写明具体仓库 cwd，不从记录目录猜源码入口。TASK v1 的 project_root 含义保持原样，只用于读取历史。

```text
workspace/
  .agents/skills/
  .workspace/
    environments/             # 共享环境描述，具体节点改造另行设计
    example/
      project.json            # 身份、状态、owner、仓库清单、生命周期事件
      PROJECT.md              # 目标、范围、仓库选择与环境引用
      STATE.md
      REQUIREMENTS.md
      ROADMAP.md
      DECISIONS.md
      EXPERIMENTS.md
      RESOURCE_USAGE.md
      AUTOMATIC.md
      repos/                  # 每仓库独立分支的 worktree
      sources/                # 必要时获取的项目专属依赖源仓库
      builds/                 # 项目自己的构建和可变缓存
      runs/
      search/
      knowledge/hot/
```

按需创建内容，不预填整套空文档。project.json 只放发现、归属和生命周期信息，STATE 保存开发进展，不在两处维护当前最佳候选。

## 选择项目

用户指定或 TASK 绑定优先，其次是当前会话已明确选择的项目；没有上下文且只有一个 active 项目时可选用并说明。多个 active 项目时列出简短目标让用户选，不按最新修改时间猜测。没有 active 项目时不能自动恢复 frozen/archived 项目。

默认列表仅展示 active 和正在收尾的 freezing；`list --all` 才包含 frozen/archived。不创建全局 CURRENT_PROJECT，避免并行会话互相改变选择。工作目录下只列含合法 project.json 的直接子目录，不递归扫描依赖源码。

目录名使用 1–64 位小写字母、数字、连字符或下划线；工具拒绝共享/旧布局保留名、路径穿越和项目目录软链接。展示名称可以中文。重名时先核对目标，明确继续原项目还是选择新名字，不能覆盖。目录改名或移动涉及引用与 Git worktree 元数据，本版不自动执行。

## 创建与仓库隔离

从工作目录运行；`SESSION` 代表宿主提供的实际会话/协调者标识，不是权限令牌：

```bash
python3 .agents/skills/harness-new-project/scripts/projectctl.py \
  --workspace . --project example --owner SESSION create --title '项目目标'
python3 .agents/skills/harness-new-project/scripts/projectctl.py \
  --workspace . --project example --owner SESSION add-repo \
  --name operator --source /path/to/existing/repo --revision selected-ref
```

source/revision 必须来自用户选择或已确认事实。工具解析具体 commit 并在 `repos/<name>` 创建新 worktree，默认分支 `codex/<project>/<repo>`，可显式 `--branch`；不同工作目录的同名项目共用源仓库时，分支冲突应另选唯一名称，不复用已检出的分支；不得绕过源仓库的分支约束。未提交的源仓库改动不会被 worktree 自动带入，应保留原处，并将需要的补丁/未跟踪输入作为明确的独立导入步骤。

依赖仓库同样创建项目独立 worktree。没有本地源码时，可在已授权的获取范围内 clone 到 `sources/<name>`，再用 add-repo 创建实际开发 worktree；工具本身不下载依赖、不重建编译器。构建目录、虚拟环境和可变缓存放本项目，不能借共享安装产生跨项目写入。worktree 共用 Git 对象/部分管理元数据，项目执行者不改共享配置或其他项目分支。

add-repo 在执行 Git 前登记 preparing。失败保留现场；恢复用原参数重试，已注册且身份匹配的 worktree 可补齐 ready。目的目录冲突或身份变化时停止检查，不自动删除目录或分支。基础 commit 已记录，恢复时使用该 commit，避免可变 ref 漂移。

## 会话归属

一个项目只有一个写入协调者；不同项目使用独立 worktree。project.json 中的 owner 是宿主事实，工具用文件锁串行化生命周期与搜索账本写入，不是认证服务，也不能拦截任意编辑器或 shell 写入。

执行者必须遵守 TASK 的 project_id、owner 和源码白名单。子 agent 在同一协调者管理下仅写其获准范围，不能作为第二个协调者抢占项目。只读查询不接管 owner。

新会话接管时，先确认旧执行者已停止写入，保存项目内 JSON 交接记录，然后使用：

```bash
python3 .agents/skills/harness-new-project/scripts/projectctl.py \
  --workspace . --project example --owner NEW_SESSION claim \
  --previous-owner OLD_SESSION --checkpoint lifecycle/handoff.json
```

交接记录含 `schema: operator-harness/project-checkpoint@1`、`project_id`、非空 `summary`、`previous_owner_stopped: true`，summary 引用实际核实依据。没有依据不得填写 true；超时不等于自动接管。接管 owner 不改变项目生命周期，也不终止或重投外部作业。

TASK 的 owner 是派发时归属快照。接管后通过 project.json 的 claim 事件与新检查点记录当前执行者，不改写冻结 TASK；接续时同时检查原范围和有效交接链，新派发 TASK 使用新 owner。

## 冻结、归档和恢复

- active：可以按任务授权继续工作。
- freezing：正在安全收尾；停止新增实验和源码开发，只允许保存检查点、接收已有结果与回收作业。
- frozen：暂停开发、保留恢复现场。
- archived：结束当前工作、默认不展开历史；不自动表示算子达标。

由 harness-archive-project 执行 `begin-close --target frozen|archived`，再生成真实检查点并 `finish-close --checkpoint ...`。检查点格式：

```json
{
  "schema": "operator-harness/project-checkpoint@1",
  "project_id": "example",
  "summary": "收尾证据及后续恢复入口",
  "writers_stopped": true,
  "jobs": [],
  "unsettled_resources": [],
  "open_runs": [],
  "paused_runs": {}
}
```

jobs/unsettled_resources/open_runs 列出尚未解决的事项，未知也必须列出，不能为了归档填空。安全暂停但尚未终态的搜索 run 可放 paused_runs，映射 run_id 到项目内 checkpoint 相对路径；其状态只能 interrupted/blocked/needs_input。工具核对这些字段与搜索树，但不能自行探测 SSH 作业或认证人的陈述，skill 必须先取得真实证据。

生命周期检查点使用不同文件名保留历史，不覆盖已关闭快照引用的文件。

关闭时保存仓库 HEAD、分支、状态、跟踪文件差异及未跟踪文件摘要。原 worktree、未提交改动、依赖和构建目录留在原位，不自动 commit/stash/reset/delete。忽略文件和依赖二进制不做完整备份；这不是灾备打包工具。项目记录/源码迁移与磁盘回收另行规划。

恢复先 `inspect` 核对源码与关闭快照，再检查依赖、环境适用性、证据链接、暂停 run、作业和资源。保存包含相同 schema/project_id/summary、空 jobs/unsettled_resources/open_runs 和 `reviewed_changes: true` 的真实检查点。调用 restore 仅把状态改为 active，记录仓库是否变化；不自动重新验证、执行节点或恢复 automatic。缺失仓库/分支错位不激活，失效环境或测量另标 stale/not_verified。新会话先按上述规则接管 owner。

## 共享信息与旧布局

共享环境描述位于 `.workspace/environments/`；每项目的 ENVIRONMENT.md 仍只绑定本项目选择的角色/ID/修订与能力。共享环境不继承另一个项目的许可或设备预算。本轮只改位置边界，不新增跨项目环境发布、额度调度或探针策略。项目 RESOURCE_USAGE 与已有物理设备锁仍分别生效。

项目私有知识放 `knowledge/hot/`。共享 cooled 按现有人工批准与发布规则生成到工作目录 `knowledge/cooled/<project>/`，不把另一个项目的内部证据路径当成自己的候选来源。

发现旧 `.workspace/STATE.md` 或 `.workspace/search/tree.json` 时明确标为 legacy，保留原位。searchctl 的 `--legacy show/check/review` 只读旧树；不得把旧 project_root 解释成新 project_dir。继续新工作时与用户确定项目名，创建新项目和 worktree，在 PROJECT/STATE 中显式引用旧资料，逐项判断证据是否仍适用。不要批量搬动历史、改写冻结 TASK/hash 或自动迁移正在运行的任务。本版不提供自动历史迁移。
