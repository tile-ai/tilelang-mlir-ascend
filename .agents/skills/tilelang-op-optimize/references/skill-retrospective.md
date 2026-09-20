# 调优复盘与模式沉淀

## 用途

在算子性能调优完成后读取本文件。目标是复盘本次调优过程是否暴露出 skill 流程问题，并判断是否需要提出新的 `BP_xxx` 瓶颈模式。

本文件只用于生成复盘记录和改进建议，不自动修改 `SKILL.md`、`iteration-diagnosis.md`、`profile-collection.md`、`bottleneck-patterns.md` 或 `autotune.md`。只有用户明确要求时，才把建议合入 skill 文档。

---

## 输入

复盘前必须具备：

- `perf_opt/opt_log.md`
- final `perf_opt/{op}.py`
- baseline 与 final 的 `msprof op` 数据
- 所有实验分支的 `improved / config_no_gain / family_no_gain / invalid / blocked / defer` 记录
- 本次最终 `stop_reason`

如果 `perf_opt/opt_log.md` 缺少关键实验记录，先补齐日志，再做复盘。

---

## 价值点分类（vp_type，所有产出表必填）

复盘产出按四类标注（判定口径：**有数字的是 D，有方法的是 P，改流程的是 R，可参考的是 C**；canonical 定义见 `tilelang-skill-evolution` skill 的 [references/distillation-rules.md](../../tilelang-skill-evolution/references/distillation-rules.md) §3）：

| vp_type | 定义 | 本次调优的典型来源 | 归宿 |
|---------|------|------------------|------|
| **D 实测数据** | 性能数字、代价常数、编译器/运行时陷阱实证、API 行为实证（含证伪更正） | 新实测代价、实验裁决实测出的未知常数、新陷阱实证 | 已按 SKILL.md Phase 4 第 6 条回写 pattern-library 的在表中记录**回写位置**（防蒸馏双写）；未回写的由终态蒸馏（`tilelang-skill-evolver`）合入 |
| **P 模式方法** | 瓶颈模式 BP_xxx、调试手法、**结构优化点**（慢→快关键更改） | 新 BP 候选、有效/无效的结构实验结论 | 提案入队 `.agents/evolution/queue.md`，2 次独立证据后合入；**结构优化点须附最小代码证据**（慢→快关键更改的最小 diff + 优化见解摘要，见「代码证据」节） |
| **R 流程规则** | skill 流程、conductor 路由/门禁、Agent 契约修改建议 | Skill Flow Issues 中 suggested_doc_change 指向流程文件的行 | 提案入队，人工审批后合入 |
| **C 案例索引** | 值得参考的算子目录（正/反例） | 倍率 > 2x 的结构性优化案例、终态失败档案 | pattern-library/cases.md 案例索引 |

**证据三件套（ED-A 语义，D 类必填）**：**provenance**（origin_task + 出处描述，如 `opt_log.md#round-N`——允许失效，过程文件不一定随 agent 合入主干）+ 工具链版本戳（tilelang build/commit + 设备 + CANN 版本）+ **repro**（知识域自包含最小代码——见下节）。缺任一时 evolver 会降级处理，但源头写全可减少往返。

**代码证据（经验与过程文件解耦——需要代码时用最少的代码表达）**：凡价值点以代码行为为依据（陷阱实证 / 代价常数 / 结构优化点），其代码证据是**知识域自包含的最小代码**，不是对 `perf_opt/{op}.py`、opt_log 或探针文件的路径引用（那些是 provenance，允许失效；甚至最终调优版本也不一定合入主干）：

- 陷阱/常数类（D）：自包含 + 断言的最小复现（慢/错误形态或绕法/标定形态），落 `examples/{project}/{op}/repro/<ID>.py` 供 evolver 机械转正；
- **结构优化点（P，正向/反向）**：**慢→快的关键代码更改**——从实验分支/探针/参考 kernel 裁出的最小 diff（完整 before/after 对照，或效应只在完整 kernel 规模显现时的 delta 骨架）**+ 优化见解摘要**（LLM 提炼的一般化洞察：为何该更改慢→快，机制归因到硬件常数/编译器行为）——目标形态见 pattern-library [repro/README.md](pattern-library/repro/README.md)（PATT-twophase-restructure.py 为范例）；无法当场最小化的在 repro 字段标 `repro-missing`。

---

## Step 1：复盘 Skill 流程

从 `perf_opt/opt_log.md` 回看本次调优过程，检查 skill 流程是否有问题。

重点检查：

- 性能采集是否覆盖 benchmark 的全部非 smoke、可运行 workload，并记录 smoke/skip 排除原因。
- profile 口径是否清楚，是否误采到框架小算子。
- `Task Duration` 复测是否稳定；候选差异落入噪声阈值时是否先解决测量分辨力再排序。
- 当前现象分析是否足够解释候选优化点。
- 候选优化点是否遗漏明显方向。
- 实验分支是否都从同一个 current best 派生。
- 是否有分支一次混入多个主要优化点，导致无法归因。
- 是否把单个 `config_no_gain` 错误扩大成 `family_no_gain`。
- `T.serial / num_cores / pipeline` 等结构候选是否完成足够配置覆盖；`num_cores` 是否分析了任务并发不足、甜点区、派发开销过高和整除性。
- `bottleneck-patterns.md` 是否缺少某类现象或动作。
- `autotune.md` 的触发时机是否正确。
- stop_reason 是否有 profile 和实验记录支撑。
- 日志模板是否足够复现 winner 选择。

复盘结论必须绑定本次 workload、profile 和实验记录，不要把单次偶然现象泛化成通用规则。

---

## Step 2：提取新的 BP_xxx 候选

只有满足以下条件，才提出新的 `BP_xxx`：

- 它解释了本次真实 profile 现象。
- 现有 [bottleneck-patterns.md](bottleneck-patterns.md) 不能很好覆盖。
- 它能对应明确的优化动作。
- 它有可验证指标。
- 它不是单个算子的偶然实现细节。

如果只是对现有模式的补充，优先提出"更新现有 BP_xxx"的建议，不要强行新增模式。

> **新增**：非 BP 类价值点同样提取——实测数据（D）、流程问题（R）、案例索引（C）按「价值点分类」表标注 vp_type 填入 Value Point Proposals 表；`family_no_gain` / `config_no_gain` 类**负向结论**若带实验记录佐证，按 P 类（negate 倾向）记录，防后续任务重蹈覆辙。

---

## Step 3：写入 opt_log.md

把复盘写入 `perf_opt/opt_log.md` 的 `Skill Retrospective` 章节。

最小模板：

```markdown
## Skill Retrospective

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| none | none | none | none | - |

### Value Point Proposals（含 BP_xxx）

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| none | - | - | - | - | - |
```

字段说明：

- `area`：`Profile-collection / Iteration-diagnosis / Bottleneck-patterns / Autotune / SKILL / logging / stop_condition`
- `issue`：本次暴露的问题，没有则写 `none`
- `evidence`：来自 `opt_log.md`、profile 或实验分支的证据
- `suggested_doc_change`：建议修改哪个文档，以及怎么改
- `vp_type`：`D / P / R / C`（见「价值点分类」表；流程问题行指向流程文件时标 R，指向数据/模式文件时标 P/D）
- `title`：一句话，含可检索关键词；更新已有 BP 时写 `update BP-xxx: {一句话}`
- `evidence`（价值点表）：溯源路径（`opt_log.md#round-N` / `profiles/` 路径）；D 类若已回写 pattern-library，此处写回写位置（如 `pattern-library §1.x（已回写）`）
- `repro`：知识域自包含最小代码路径（`repro/<ID>.py`，优化点为慢→快关键更改 + 见解摘要）或 `repro-missing`（待同族任务补）；纯流程建议写 `none`
- `toolchain_stamp`：tilelang build/commit + 设备 + CANN 版本；无法确定如实标注
- `target_doc`：通常为 `bottleneck-patterns.md`；D 类为 `pattern-library/` 主题文件（含 `constants.md`），R 类为具体流程文件

---

## Step 4：最终报告

最终报告必须包含：

```markdown
- skill_retrospective: {none_or_summary}
- value_point_proposals: {none_or_list}   # 含 BP_xxx 与 D/P/R/C 各类，逐条带 vp_type
```

如果提出了 `BP_xxx`，只作为 proposal 返回，不直接修改 skill 文档。

> **与终态蒸馏的衔接**：本文件产出的复盘表由 conductor 在任务终态调度 `@tilelang-skill-evolver`（`tilelang-skill-evolution` skill）统一蒸馏与分级合入——D/C 类（Tier 0）自动合入 pattern-library（Phase 4 第 6 条的任务内回写除外，防双写），P 类（Tier 1）入队等 2 次独立证据，R 类（Tier 2）入队等人工审批。复盘表的字段（vp_type / evidence / repro / toolchain_stamp）就是蒸馏的输入，请按 schema 写全。
