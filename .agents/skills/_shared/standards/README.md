# 跨 Agent 共享标准（单一事实源）⭐

> 本目录是编排 Agent（conductor）、各 Stage Subagent、skills、README 之间**共享规则块的唯一权威维护点**（改进项 #17，对应弱点 U10「规则冗余、文档漂移、上下文膨胀」）。
>
> **维护规则（强制）**：
> 1. 本目录内的每个标准文件是对应规则块的**唯一完整版本**（single source of truth）。修改规则只改此处，再由消费者引用同步。
> 2. 消费者（conductor.md / 各 Subagent .md / 各 SKILL.md / README.md）**只引用不复制**：可保留一句话摘要 + 指向本目录的路径引用，禁止整段内联标准文本。
> 3. 每次修改标准文件后必须执行 `python3 .agents/tools/standards_check.py update` 重新生成 `standards.lock.json`（SHA256 清单），再提交——哈希未更新的修改会被 `standards_check.py check` 拦截。
> 4. 漂移检查：`python3 .agents/tools/standards_check.py check` 校验 ① 标准文件哈希与 lock 一致；② 标准文本指纹未被消费者内联复制；③ 消费者对本目录的引用全部可解析。`tilelang-review-skill` 在 PR 涉及 `.opencode/agents/`、`.agents/skills/` 时执行本检查。
> 5. 机械门禁（`statectl gate` / `gate_lint.py`）检查的是本目录标准的**可机械判定子集**，两者同步演进：改标准时同步改 lint 规则，改 lint 规则时同步改标准。

## 标准文件索引

| 标准文件 | 内容 | 主要消费者 |
|----------|------|-----------|
| [core-split-strategy.md](core-split-strategy.md) | 分核策略三要素（逻辑核数 / 物理核数实查 / 规模判定与分核方案）+ 各角色要求文本 + 失败信号路由 | conductor（Stage 1/2/4 透传与门禁）、designer、reviewer、developer、optimizer、gate_lint |
| [algorithm-research.md](algorithm-research.md) | 算法调研要求（调研四问、调研对象、深度分级、负向断言纪律、落位与下游约束） | conductor（Stage 1 透传）、designer、reviewer |
| [negative-claim-evidence.md](negative-claim-evidence.md) | 弃选论证证据规则（负向断言举证责任、API 先枚举后检索、未文档化假设标注） | conductor（Stage 1/2 透传）、designer、reviewer、gate_lint |
| [gate-and-retry.md](gate-and-retry.md) | 阶段门禁总表 + 重试与中止规则 + 任务级预算默认值 + 统一结束态（BLOCKED_* 映射） | conductor、statectl、README |
| [signal-registry.md](signal-registry.md) | 信号与契约注册表：完成/失败/通过信号 token、Subagent 调度 mode 枚举、状态字段枚举、`perf_iteration` 与 `perf_records.jsonl` 字段契约、关键术语执行定义 | conductor、各 Stage Subagent、statectl、gate_lint、README |
| [stage3-routing.md](stage3-routing.md) | Stage 3 调度模型、四出口路由、运行失败子类型路由 | conductor、developer、README |
| [perf-feedback.md](perf-feedback.md) | TUNING→DESIGN 受控逆向反馈（[DESIGN_LIMIT] 信号契约、触发条件、perf_feedback.md 固定 schema、三路由与防抖动） | conductor（Stage 4 路由）、optimizer、tilelang-op-optimize skill、gate_lint、README |
| [final-report-template.md](final-report-template.md) | 任务最终输出报告模板 | conductor（终态输出）、README |
