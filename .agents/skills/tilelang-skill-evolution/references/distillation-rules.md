# 蒸馏标准：信号 → 价值点映射、分类与证据规则

> 本文件是 evolver Phase 1/2 的执行标准。目标：把任务工件中的原始信号转化为**可合入、可检索、可追溯**的价值点，同时把单任务噪声挡在门外。

## 目录

- [1. 价值点四分类（canonical）](#1-价值点四分类canonical)
- [2. 信号 → 价值点映射（按来源）](#2-信号--价值点映射按来源)
- [3. vp_type 判定树](#3-vp_type-判定树)
- [4. 防过拟合红线](#4-防过拟合红线)
- [5. 证据三件套规范](#5-证据三件套规范)

---

## 1. 价值点四分类（canonical）

| 类型 | 定义 | 判定核心 | 典型目标文件 |
|------|------|---------|------------|
| **D 实测数据** | 有数字的事实：性能数字、代价常数（µs/倍率/KB）、编译器/运行时陷阱实证、API 实际行为实证（含证伪更正） | **有数字或有"实证/证伪"结论** | `pattern-library.md` §1/§2 |
| **P 模式方法** | 可复用的方法：瓶颈模式（BP_xxx）、设计候选模式（某类算子的布局候选）、调试手法、检视检查项 | **有方法且对应明确动作** | `bottleneck-patterns.md`、design/review/develop/error-fixer 的 references |
| **R 流程规则** | 改变后续任务行为的规则：skill 流程修改、conductor 路由/门禁/重试规则、Agent 交互契约、AGENTS.md 路由 | **修改的是"怎么做事"而非"事本身"** | `SKILL.md`、`.opencode/agents/*.md`、`AGENTS.md` |
| **C 案例索引** | 值得参考的算子目录：正例（同类问题的成功解）与反例（典型失败档案） | **路径真实存在 + 有明确触发条件** | `pattern-library.md` §4 |

一句话判定：**有数字的是 D，有方法的是 P，改流程的是 R，可参考的是 C**。一个复盘可同时产出多类。

## 2. 信号 → 价值点映射（按来源）

### 2.1 `RETROSPECTIVE.md`（Stage 1/2/3/5 复盘，schema 见 [retrospective-schema.md](retrospective-schema.md)）

| 信号位置 | 提取规则 | 常见 vp_type |
|---------|---------|-------------|
| Value Point Proposals 表行 | 直接为候选（已含 vp_type/evidence/repro/toolchain_stamp 字段），做查重与分级校验即可 | D/P/R/C 全有 |
| Skill Flow Issues 表行 | `suggested_doc_change` 指向流程文件的 → R 类候选；指向数据/模式文件的 → P/D 候选 | R 为主 |
| Transferable Lessons 小节 | 逐条判定：含数字/实证 → D；含可复用手法 → P；仅本任务有效（如某函数的 UB 笔误）→ 丢弃 | D/P |
| Stage 1 revision 章节的存在本身 | 说明设计判断被推翻——从 `design_error_summary` 与 design_v{N} 差异提取"哪类判断错了、正确依据是什么" | P/R（新检查项候选）/D（若推翻依据是实测） |

### 2.2 `perf_opt/opt_log.md`（Stage 4）

| 信号 | 提取规则 | 常见 vp_type |
|------|---------|-------------|
| Skill Retrospective 的 BP Proposals 表行 | 同 2.1 Value Point Proposals | P |
| Skill Retrospective 的 Skill Flow Issues 表行 | 同上 | R |
| 实验数据：新代价常数 / 新模式实测收益 / rollback 归因 | 未被任务内 pattern-library 回写覆盖的 → D 类（含三件套） | D |
| 证伪更正（"误判根因 + 合法形态 + 新数据"） | **最高优先级 D 类**：同步 pattern-library §2 状态列（旧条目 deprecate + 新条目 add 或 update） | D |
| Final Summary 的 stop_reason 与改进倍率 | 倍率 > 2x 的结构性优化 → 模式化候选（P）+ 案例索引（C） | P/C |

### 2.2a `perf_opt/perf_feedback.md`（`[DESIGN_LIMIT]` 设计层发现，存在时必读）

| 信号 | 提取规则 | 常见 vp_type |
|------|---------|-------------|
| 触发判定（设计层归因 + 结构性加速估计） | **D 类优先**：归因到的 DESIGN 假设被实测推翻（含 msprof 数字与 >2x 估计依据）→ pattern-library §1/§4 候选；同族算子后续设计任务的候选模式 | D/P/C |
| 反馈结论与建议路由 | 已回流（附录补记/路径 C 修订）→ 提取"哪类设计假设错了、实测依据是什么"（与 design_v{N} 差异交叉核对）；未回流（不处理）→ 反例档案候选 | D/C/R |

### 2.3 `integration_log.md`（Stage 5）

| 信号 | 提取规则 | 常见 vp_type |
|------|---------|-------------|
| 调试历史 attempt 链 | 成功修复路径 → P（集成调试手法）；修复 2 次以上仍失败 → 根因档案（C 反例 + R 候选） | P/C/R |
| 集成脚本 `[error]`/`[warn]` 模式 | 脚本缺陷 → R（add-npu-op skill / integrator agent 的已知修复目录候选） | R |
| bench 数值 | 与同类算子量级异常（如 launch 开销主导） → D | D |

### 2.4 `.stage_state.json` / `.migration_state.json`（只读）

| 信号 | 提取规则 | 常见 vp_type |
|------|---------|-------------|
| `retry_count` / `stage_retry_count` 分布 | 高重试 Stage 的失败子类型 → 若属新失败形态（conductor 失败子类型路由表未覆盖）→ R；已有对应条目仍复发 → stats 复发率计数 + 检索注入缺口报告 | R |
| `failure_reason=BLOCKED_*` | 终态失败根因链 → C（反例档案）+ D（若有实证） | C/D/R |
| `stage3_failure_breakdown` | precision_fail 主导 → 精度手法候选（P）；runtime_fail 主导 → 陷阱候选（D） | P/D |

### 2.5 `history_version/` 修订与调试链

| 信号 | 提取规则 | 常见 vp_type |
|------|---------|-------------|
| design_v{N} 之间的差异 | 被推翻的设计判断 + 推翻依据 → P（若可泛化为检查项/候选模式）/ R（若依据是文档条款缺失，需要补约束参考文件） | P/R |
| {op}_impl_s3_attempt{N} 之间的差异 | 调试路径中的 API 行为实证 → D；有效的调试手法 → P | D/P |

### 2.6 `REVIEW.md` 不通过原因

| 信号 | 提取规则 | 常见 vp_type |
|------|---------|-------------|
| 不通过问题列表 | 问题本身已随修订解决，价值在于**检视维度是否早该拦住**——维度缺口 → R（review skill 检查项候选） | R |

### 2.7 `.task_timeline.jsonl`（statectl 事件流，只读——失败根因链一手输入）

> 每次状态迁移一行（`ts / action / stage / subagent / mode / verdict / duration_s`，由 statectl 机械追加，不经 LLM 手写）；`statectl timeline-summary --dir ...` 可直接产出机械汇总（dispatches / 各 Stage attempts 与耗时 / 失败事件链）。定位输入：时间线给**顺序与计数**，下钻工件（§2.1–2.6）给**原因**。

| 信号 | 提取规则 | 常见 vp_type |
|------|---------|-------------|
| fail 事件链（stage → subagent → verdict → duration_s，按 ts 排序） | 失败根因链的**一手骨架**：先读时间线定位「哪个环节、第几次 attempt、多久后、以何种 verdict 失败」，再下钻对应 Stage 工件取细节；evolver 不得凭时间线单独下结论（无原因文本） | 定位输入（vp_type 视下钻结果） |
| 同一 Stage 反复 start→fail 的 attempt 序列 | 与 `.stage_state.json` 计数交叉核对（§2.4）：计数一致但复盘工件未覆盖的 attempt → 复盘缺口候选（R）；duration_s 显著偏离同任务其他 attempt → 效率异常候选（P/R，须绑定原因） | R/P |
| complete 事件的 duration_s | 跨任务同族算子的 Stage 耗时对比（需 ≥2 任务数据才可比较，绑定 workload 与版本戳）→ 调度成本画像（D 类候选，缺三件套则降级） | D |
| verdict=BLOCKED_* 的 fail 事件 | 终态失败反例档案入口：与 §2.4 `failure_reason` 交叉印证，时间线补充失败前的完整 attempt 上下文 → C（反例档案） | C/D |

## 3. vp_type 判定树

```
候选价值点
├─ 含实测数字 / 实证结论（含证伪）？
│   ├─ 是 → D（目标 pattern-library §1/§2；无三件套 → 降级 Tier 1 入队）
│   └─ 否 ↓
├─ 是"某个算子目录值得参考"（路径存在 + 触发条件明确）？
│   ├─ 是 → C（目标 pattern-library §4）
│   └─ 否 ↓
├─ 修改对象是流程/规则/契约（SKILL.md / agents md / AGENTS.md / 路由表）？
│   ├─ 是 → R（Tier 2，只出 diff 提案）
│   └─ 否 ↓
└─ 是可复用的方法/模式/手法（有明确动作 + 可验证指标）？
    ├─ 是 → P（Tier 1，2 次独立证据）
    └─ 否 → 丢弃（记录到进化报告 issues，不入库）
```

分类错误的代价不对称：**宁可把 D 判成 P（多等一次证据）也不要把 P 判成 D（无验证直接合入）**；把 R 判成 P 是最危险的（流程修改绕过了人工门）——判不准时一律就高不就低（D<P<R 保守序）。

## 4. 防过拟合红线

1. **单任务偶然现象不得泛化**：某优化点在当前 workload 失败只能记录为该 workload / 工具链版本下的实测结论（与 optimize skill 的现有规则一致）。
2. **P 类须绑定可复现证据链**：来源案例 + 复现路径，二者缺一不入队。
3. **无证据的主观抱怨不入库**："skill 流程太绕" 这类无具体建议的反馈只进进化报告 issues。
4. **重复不是新发现**：与 pattern-library 已实测条目结论一致的"新发现"直接丢弃（查重命中）。
5. **矛盾先核对再冲突**：新候选与现有条目矛盾时，先确认双方的工具链版本戳与测量口径（msprof vs event）是否可比——口径不同的"矛盾"不是矛盾。
6. **绑定 workload 的数字必须带 workload 上下文**：裸数字（"21.6µs"）不可入库，必须含 shape/dtype/dispatch 上下文。

## 5. 证据三件套规范（D 类合入门槛）

| 要素 | 规范 | 缺失后果 |
|------|------|---------|
| **溯源路径** | 指向本仓库真实存在的工件：`examples/{project}/{op}/perf_opt/opt_log.md#round-3`、`testing/npuir/...`、`examples/.../DESIGN.md#§x` | 降级 Tier 1 |
| **工具链版本戳** | `tilelang build/commit 时间 + 设备型号 + CANN 版本`；来源任务工具有记录时直接引用；无记录时标注「版本戳缺失，来源任务 {task_id}」 | 降级 Tier 1 |
| **复现命令** | 可直接执行的命令（`python examples/{project}/{op}/perf_opt/{op}.py --level L0`）或复现条件描述（shape/dtype/dispatch） | 降级 Tier 1 |

> 三件套与 pattern-library §2 证伪协议同源：一切编译器/运行时结论绑定版本戳，工具链变更后自动待重验。
