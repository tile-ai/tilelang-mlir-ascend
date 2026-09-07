# 合入策略：分级治理、delta 操作、冲突消解、预算与 git 快照

> 本文件是 evolver Phase 3/4 的执行标准。核心思想（源自业界 ACE / Voyager / Mem0 实践，见 docs/developer/conductor-self-evolution-design.md §1）：**增量 delta 而非重写、写入须验证门、冲突须消解、文件须有预算、进化须可回滚**。

## 目录

- [1. 写权限矩阵（Tier 分级）](#1-写权限矩阵tier-分级)
- [2. evolver 的写范围](#2-evolver-的写范围)
- [3. 冲突消解](#3-冲突消解)
- [4. 五种合法 delta 操作](#4-五种合法-delta-操作)
- [5. 预算与 consolidate](#5-预算与-consolidate)
- [6. queue 生命周期](#6-queue-生命周期)
- [7. git 快照规则](#7-git-快照规则)

---

## 1. 写权限矩阵（Tier 分级）

| Tier | 对象 | distill 模式下的写入者 | 验证门 | 审批 | 回滚 |
|------|------|----------------------|--------|------|------|
| **Tier 0** | D 类（数据条目）、C 类（案例索引行） | evolver 直接写 pattern-library | 证据三件套齐备（缺一降级 Tier 1）+ **合入前机械检查**（D2）：引用路径逐一存在性核验（断链降级）、条目文本无指令性祈使句（启发式 lint）、条目带 `origin_task`（来源 task_id） | 无需（事实陈述，可被后续实测自动纠正） | git revert |
| **Tier 1** | P 类（模式方法条目） | evolver（计数达 2/2 后合入） | 证据链齐全入队 + **2 次独立证据**（须来自不同任务） | 无需（二次确认即 ExpeL 式投票） | git revert |
| **Tier 2** | R 类（流程规则：SKILL.md / `.opencode/agents/*.md` / AGENTS.md / conductor 规则 / 其他 skill references 的流程性内容） | evolver 仅生成 diff 提案入队；**apply 模式**经用户批准后写入 | 结构化 diff 提案（锚文本 + old/new + 动机 + 证据） | **人工**（用户在 Primary 上下文批准 → conductor 调度 `mode=apply`） | git revert / 拒绝提案 |

理由：D/C 类是"事实陈述"——错误成本低，且版本戳机制使其可被后续实测自动纠正；R 类改变所有后续任务的行为——错误成本高且无自动纠正机制，必须人工门。

> **边界情况**：一个建议同时含数据与流程（如"新增 BP 模式 + 修改 SKILL.md 触发时机"）→ 拆成两条分别走各自 Tier。

## 2. evolver 的写范围

**允许写**（distill 模式）：

| 文件 | 允许动作 |
|------|---------|
| `.agents/skills/tilelang-op-optimize/references/pattern-library.md` | Tier 0：§1/§2/§4 的 add/update/negate/deprecate + 预算触发的 consolidate |
| `.agents/skills/tilelang-op-optimize/references/bottleneck-patterns.md` | Tier 1：P 类合入（计数达 2/2） + consolidate |
| 其他 skill 的 `references/` 下**数据/模式类**文件（P 类 target_doc） | Tier 1：合入（计数达 2/2） |
| `.agents/evolution/queue.md`、`.agents/evolution/stats.md` | 全权维护 |

**apply 模式额外允许**：Tier 2 提案中 `target_doc` 指向的流程文件（按提案文本执行，禁止扩大）。

**任何模式下禁止写**：

- 任务算子工件（`DESIGN.md` / `{op}.py` / `REVIEW.md` / `opt_log.md` / `perf_feedback.md` / `integration_log.md` / `RETROSPECTIVE.md` / `history_version/`——**只读**）；
- `.stage_state.json` / `.migration_state.json`（conductor 专属，只读）；
- `.task_timeline.jsonl`（statectl 事件流，只读——失败根因链一手输入）；
- `docs/`、`examples/`、`testing/`、`src/`（仓库本体代码与文档）；
- `SKILL.md` 文件本体（任何 skill 的主流程文档，包括 optimize 的——optimize SKILL.md Phase 4 对调优 Agent 的"pattern-library 例外授权"同理适用于 evolver，但同样只覆盖数据文件，不覆盖 SKILL.md）。

> **conductor 文件（`.opencode/agents/tilelang-op-conductor.md`）与各 `SKILL.md` / `AGENTS.md` 属 Tier 2 写域**：distill 模式绝对只读；仅 apply 模式下可按**用户已批准**的提案文本写入（含 conductor 文件本身——用户批准即授权）。

## 3. 冲突消解

新条目与现有条目（或 queue pending 条目）结论矛盾时，**不得并存堆放**：

1. **先核对可比性**：双方工具链版本戳、测量口径（msprof Task Duration vs NPU event）、workload 上下文是否可比——口径不同的"矛盾"不是矛盾，各自成立（在条目中互相注明适用口径）。
2. **可比时预判胜者**：① 实测 > 推演；② 新版本戳 > 旧版本戳；③ 更严格的证伪协议（按 pattern-library §2 协议得出的结论）> 普通结论。
3. **败者处理**：`deprecate`（保留条目与"被谁取代"标注，**不删除**——失效知识本身是信息）。
4. **无法判定** → queue 记 `conflict`，报告中列出双方证据，留人工裁决。
5. **Tier 0 D 类冲突**（新实测推翻旧实测）：可直接执行"新 add + 旧 deprecate"（版本戳机制自愈）；Tier 1 以上或涉及 P/R 类语义的冲突 → 必须留人工。

## 4. 五种合法 delta 操作

| 操作 | 语义 | 约束 |
|------|------|------|
| `add` | 在目标章节**末尾追加**新条目（不改动其他内容） | 追加前必须查重；同主题已有条目时禁用（改用 update） |
| `update` | 原位更新既有条目（保留条目标识，如 BP id / 章节号） | 更新须保留旧信息中有价值的部分（如旧版本戳下的数字可注明"截至 {戳}"） |
| `consolidate` | 合并同类条目 / 压缩冗长条目 / 按主题重排 | **逐条目操作**；保留全部信息密度，禁止摘要式压缩；negate 类条目**不参与合并**，只可 deprecate |
| `negate` | 添加负面条目（"此路不通 + 原因 + 证据"） | 与 add 同规范（含三件套）；负面信息密度最高，优先保留 |
| `deprecate` | 标记条目失效：条目头部加 `> [已失效 {日期}，被 {新条目/原因} 取代]` | 不删除原文；失效原因必须可追溯 |

**禁止整文件重写**——所有操作都是对具体条目的局部编辑（ACE 防坍缩原则：迭代重写会逐代丢失细节）。

## 5. 预算与 consolidate

| 文件 | 预算 | 触发动作 |
|------|------|---------|
| `pattern-library.md` §1-§3（知识区） | 300 行 | 超限时本轮执行 consolidate（合并同类、压缩措辞、表格化 negate 条目） |
| `pattern-library.md` §4（案例索引） | 150 行 | 同上；长期零引用的案例行可 deprecate |
| `bottleneck-patterns.md` | 400 行 | 同上 |
| 其他 P 类 target_doc | 300 行/文件 | 同上 |

规则：

1. **update 优先于 add**（同主题已有条目时禁止新开条目——这是控制膨胀的第一道闸）。
2. consolidate 是**触发式**的（超预算或语义重复明显时），不是每次蒸馏都做。
3. consolidate 也消耗蒸馏预算：一次蒸馏最多 consolidate 一个文件，避免报告失焦。
4. 预算检查在 Phase 4 执行；合入（Phase 3）导致的超限允许存在到下一次触发——不为了预算阻塞当次合入。

## 6. queue 生命周期

| 状态 | 含义 | 迁移 |
|------|------|------|
| `pending` | 待确认（Tier 1 计数未满）或待审批（Tier 2） | Tier 1：新证据来自**不同任务** → `confirmations+1` → 达 2/2 → 下次蒸馏合入 → `merged`；Tier 2：用户批准 → `mode=apply` 执行 → `merged` |
| `verified` | Tier 1 计数已满、待合入（可跳过，直接 merged） | 合入后 → `merged` |
| `merged` | 已写入目标文件 | 终态；记录 target 锚点 |
| `rejected` | 人工或 evolver 否决 | 终态；**必须记录拒绝原因**（防重复提案） |
| `expired` | 创建超 90 天仍 pending | 终态；过期不删除（保留供参考） |
| `conflict` | 与现有条目矛盾且无法自动裁决 | 人工裁决后 → merged / rejected |

维护规则：

- merged/rejected/expired/conflict 条目移入 `## Decided` 归档区，`## Pending` 区只保留活跃提案。
- 同一主题的重复提案：合并入已有 pending 条目（证据链追加），不新开 proposal_id。
- proposal_id 分配：`VP-{YYYY}-{NNNN}`，NNNN 按年递增，不复用已决条目的 id。

## 7. git 快照规则

目的：每次进化一个 commit，可 review、可回滚、可 diff（git 即进化的版本控制）。

1. **快照范围**：仅 add 本次进化实际写入的文件——`pattern-library.md` / `bottleneck-patterns.md` 等 target 文件 + `.agents/evolution/queue.md` + `.agents/evolution/stats.md`。**不包含**任务算子工件（那些属于用户的任务提交）。
2. **提交信息**：`evolution: {一句话摘要} (task: {task_id})`；apply 模式：`evolution: apply {proposal_id 列表} (approved by user)`。
3. **前置守卫（目标文件脏时不静默跳过——快照是进化的回滚保障，D2 修订）**：写入前检查 `git status --porcelain -- <目标文件>`——目标文件在本次进化写入之前已有未提交改动 → **优先 stash 方案**：`git stash push -- <目标文件>`（暂存用户改动）→ commit 进化快照 → `git stash pop` 恢复用户改动（pop 冲突时保留 stash 并在报告中说明，用户可 `git stash list` 找回）；stash/pop 不可用（非 git 仓 / 无权限）→ 降级为在 `.agents/evolution/stats.md` 变更日志追加一行记录（文件路径 + skip 原因 + 日期），并在报告中提示"文件 {path} 进化前已有未提交改动，快照未覆盖用户改动段，请人工 review"。
4. **commit 失败**（如 hooks 拒绝、无 git 身份配置）：写入已生效时如实报告 `[EVOLVE_FAIL]` + `git_snapshot: skipped(原因)`，**不回滚文件写入**（写入本身经过验证门，回滚反而丢失价值点）。
5. **禁止** `git add -A` / `git add .`；禁止 push。
