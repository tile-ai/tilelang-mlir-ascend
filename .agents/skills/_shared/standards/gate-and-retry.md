# 阶段门禁与重试标准 ⭐

> **单一事实源**：本文件是「门禁总表 / 重试与中止规则 / 统一结束态（BLOCKED_* 映射）/ 典型 `[DESIGN_ERROR]` 场景」的权威版本。机械可判定子集由 `statectl gate N`（`gate_lint.py`）执行；计数器累加、上限判定与 `BLOCKED_*` 自动路由由 `statectl fail/start/complete` 内建。README 只链接摘要，不复制本表。

## 1. 门禁总表

> **失败类型**：所有 Stage 都可能产生两类失败——
> - **门禁失败**：`statectl complete N` 内置的机械门禁未通过（`gate.failures` 非空：产物缺章节 / 结论字面量缺失 / 占位符 / 死链 / AST 结构违规等），按 conductor「门禁失败处理流程」处理。表中标准里的一致性、独立性、跑测结果等**执行型与推断型判定**仍由 conductor 依据 Subagent 返回信号把关（机械下界 + conductor 判定都通过才 complete）。
> - **执行失败**：Subagent 已返回结果但运行/精度等不达标，按各 Stage 自身路由处理。

| Stage | 必需工件 | 门禁校验标准 | 执行失败类型 | 失败路由 |
|-------|---------|-------------|---------|---------|
| 0 | `op_name` + `gpu_repo_root` | TileOPs 7 文件存在 + Tier 1 通过（import / manifest / collect-only）+ `.migration_meta.json` 字段完整（含 ≥1 个 extracted_functions） | 结构校验失败 / GPU 无实现 / repo 缺失 | 结构 → `fail_stage(0)` 重试；无 TileLang 实现 → `BLOCKED_SPEC`；repo 缺失 → `BLOCKED_ENVIRONMENT`；超限 → `BLOCKED_SCAFFOLD` |
| 1 | 用户需求（迁移任务：另含 `source_op_path`） | `DESIGN.md` 含算子名、I/O 规格、编程模式、算法调研与优化分析（**§1.6.0 算法调研（调研四问：等价化简公式/在线算法/复杂度/硬件亲和，候选表含基线、复杂度四口径、选定结论有依据，「无更优替代」写明调研范围）+ §1.6 数学等价优化（更少计算量/访存量，逐项含原式→优化后→等价论证→收益）+ 向量化替代分析（循环/标量点全覆盖，不可替代有充分理由），且 §1.4/§3.1/§6 与 §1.6.0 选定算法及优化后公式一致**；**§1.6.1 否决项与 §1.6.3 弃选行必须含 `docs/`/`testing/`/`examples/` 佐证路径或『未文档化』显式标注——grep 机械核对，缺证即门禁失败**）、API 映射、tiling 策略（**含分核策略三要素**，见 [core-split-strategy.md](core-split-strategy.md)）、内存层级、同步策略、验证方案（含 L0 计划）、技术约束检测结论；**迁移任务另须含 §0**（0.1 语义 / 0.3 算法解读 / 0.4 优化手段 / 0.5 耦合性判定 / 0.6 NPU 重设计，且 §1–§7 与迁移决策一致、golden 独立） | 必须字段缺失 / 用户中途取消 / 迁移源码不可读 | `fail_stage(1)` → 重试 Stage 1（计 `stage_retry_count`）；迁移源码不可读 → `BLOCKED_SPEC` |
| 2 | `DESIGN.md`（迁移任务：另含 `source_op_path`） | `REVIEW.md` 存在且含明确 `结论: 通过` 或 `结论: 不通过`；迁移任务维度数 = 9（含维度 0 源算子理解与迁移分析，且结论附源码核对证据），非迁移 = 8（含维度 8 算法优化分析，且结论附独立推演证据）；**维度 8 含弃选论证前提核对证据（逐条负向论断的 API 文档核对结论，缺即门禁失败）与调研结论独立复核证据（负向断言复核 + 复杂度复算结论，缺即门禁失败）** | 检视不通过 | 设计修订循环（路径 A，计 `retry_count`） |
| 3 | `DESIGN.md`（检视通过）| 真实跑测完成四出口判定，且 **L0/L1 全过**（`[PRECISION_PASS]`）才视为门禁通过；L2/Boundary 告警不影响门禁 | 编译/运行/精度失败 / `[DESIGN_ERROR]` | 分类路由（见 [stage3-routing.md](stage3-routing.md)） |
| 4 | `{op}.py`（精度通过） + 用户调优信息（optimize 场景：kernel 路径 + 回归入口） | 单轮性能迭代完成；optimize 场景额外要求 `perf_opt/{op}.py` 回归 L0+L1 通过；`perf_opt/perf_feedback.md` 存在时须过 `S4-PERF-FEEDBACK-*` schema 校验（[perf-feedback.md](perf-feedback.md) §2）；`perf_opt/perf_records.jsonl` 存在时须过 `S4-PERF-RECORDS-*` 记录校验 + winner 对账与 `S4-OPTLOG-COMPTABLE` 对比表存在性（[signal-registry.md](signal-registry.md) §5） | 性能不足 / 回归失败 / `[DESIGN_LIMIT]` 工件不合规 | Stage 4 内继续迭代（参数级不足调优 Agent 自完成，不回退）；optimize 回归失败 → `mode=precision_fix` 重调度（只跑回归修复，不重走已完成轮次）；`[DESIGN_LIMIT]` 设计层天花板 → 受控逆向反馈路由（[perf-feedback.md](perf-feedback.md) §3：附录补记不耗预算 / 设计修订路径 C 共享 `retry_count` / 不处理） |
| 5 | 全函数 Stage 3 通过 + `.migration_meta.json` | 集成包存在（kernel + 每函数 `{func}_DESIGN.md`）+ wrapper 双 import 切换块已生成（baseline 激活）+ TileOPs pytest smoke+全量真实通过 + bench 已记录 | 集成前置失败 / 精度失败 / `[DESIGN_ERROR]` / 环境 | `[INTEGRATE_FAIL]` → 重调度 integrator（≤2 次）；`[DESIGN_ERROR]` → 该函数设计修订后重集成；超限 → `BLOCKED_INTEGRATION` |

## 2. 重试与中止规则

| Stage | 上限（`stage_retry_count`） | 超限后状态 |
|-------|------|------------|
| 0 | 3 次（结构问题重试） | `BLOCKED_SCAFFOLD`（"GPU 无实现"直接 `BLOCKED_SPEC`，repo 缺失直接 `BLOCKED_ENVIRONMENT`，不耗重试） |
| 1 | 3 次 | `BLOCKED_DESIGN`（门禁失败类） |
| 2 | 不适用（检视不通过走 `retry_count` 修订循环） | `retry_count >= max_retry` → `BLOCKED_DESIGN` |
| 3 | 5 次 Subagent 调度（运行失败 + 精度失败合并累计；`[DESIGN_ERROR]` 触发修订不计入） | 因运行失败超限 → `BLOCKED_IMPL`；因精度失败超限 → `BLOCKED_ACCURACY` |
| 4 | 10 轮迭代（optimize 场景含回归 `precision_fix` 重调度） | `SUCCESS`（附中止原因） |
| 5 | 2 次重调度（integrator 内部另有 5 次调试闭环，预算独立） | `BLOCKED_INTEGRATION` |
| 设计修订（`retry_count`） | `max_retry`（默认 3；harness 迁移为**全 op 共享预算**，跨函数累计） | `BLOCKED_DESIGN` |

计数器累加与上限判定已内建于 `statectl fail N`（含 Stage 3 `--fail-type` 区分、Stage 4 上限置 `DONE` 附 abort 原因、修订重入清零下游）。

### 任务级预算（`budget` 字段，E1.3）

`.stage_state.json` 的 `budget` 字段（枚举与写入方式见 [signal-registry.md](signal-registry.md) §3）为**任务级开销上限**，与上表的 Stage 级重试预算正交：

| 维度 | 默认 | 语义 |
|------|------|------|
| `max_wallclock_s` | `null`（不限） | 任务墙钟上限（秒） |
| `max_subagent_dispatches` | `null`（不限；建议值 40） | Subagent 调度总次数上限（timeline `start` 事件计数） |
| `max_stage4_experiments` | `null`（不限；建议值 30，与 optimizer `max_experiments=30` 对齐） | Stage 4 实验分支总数上限（调度 optimizer 时透传） |

执行方式：`statectl start` 每次返回预算水位 `budget`（`dispatches_used` / `wallclock_s` / `exceeded`，advisory 不拦截）；`budget.exceeded` 非空 → conductor **停止调度新 Subagent**，向用户报告水位并请求决策（经 `statectl set --budget-json` 上调后继续，或以现有最优产物收束 / 标记失败）。最终报告「成本」段粘贴水位数据（见 [final-report-template.md](final-report-template.md)）。

## 3. 统一结束态（BLOCKED_* 映射）

| `phase` | `failure_reason` | 含义 |
|---------|------------------|------|
| `DONE` | — | Stage 4 按中止条件完成 **或** 精度通过后用户表示不需要性能调优 **或** harness 迁移 Stage 5 集成通过 **或** optimize 调优+回归通过 |
| `FAILED` | `BLOCKED_DESIGN` | Stage 1 门禁超限 或 设计修订 `retry_count` 超限 |
| `FAILED` | `BLOCKED_IMPL` | Stage 3 运行失败超限 |
| `FAILED` | `BLOCKED_ACCURACY` | Stage 3 精度失败超限 |
| `FAILED` | `BLOCKED_SCAFFOLD` | Stage 0 脚手架结构校验重试超限（仅 harness） |
| `FAILED` | `BLOCKED_INTEGRATION` | Stage 5 集成验证重调度超限（仅 harness） |
| `FAILED` | `BLOCKED_ENVIRONMENT` | 环境问题阻塞（torch / torch_npu / CANN 版本不达标、GPU repo 缺失、子模块修复失败等） |
| `FAILED` | `BLOCKED_SPEC` | 用户拒绝提供必需字段，或 GPU 侧无 TileLang 实现不可迁移 |

## 4. 典型 `[DESIGN_ERROR]` 场景（设计修订触发识别）

> conductor / reviewer / developer 共用的设计层错误识别清单（识别信号由 Stage 2 维度 8 / 维度 0 或 Stage 3 Developer 产出；conductor 只按 `[DESIGN_ERROR]` 标记路由到设计修订路径 B，本表帮助各方对齐什么构成设计层错误）。

| 场景 | 识别信号 |
|------|----------|
| 设计选用的 API 实际不可用 | Developer 报告"API 在 `tilelang/language/` 中无导出 / lowering 未实现" |
| Tiling 策略导致 L0C 溢出 | 编译期或运行期报 L0C 超限 |
| 分核策略违反静态边界约束 | 核内串行（persistent）任务数或循环边界依赖动态 shape / 运行时核数，与 Ascend 静态循环边界约束冲突 |
| 分核策略导致串行调度开销 | 逻辑核数远超物理核数被运行时串行调度（核启动风暴），Stage 3 跑测显著超时或性能异常 |
| 内存层级路径无法实现 | 设计要求 GM→L0 直接搬运 |
| 同步策略与编程模式冲突 | Developer 模式下要求手动 set_flag/wait_flag |
| 设计的 loop 结构依赖动态边界 | 与 Ascend "只支持静态循环边界" 约束冲突 |
| 精度调试多次后定位到根因是设计 | Stage 3 多次精度调试后 Developer 报告"修复实现层无解" |
| 公式优化破坏语义（等价论证不成立） | Stage 2 维度 8 判 fail（§1.6.1 变形实际改变语义 / 累加顺序 / dtype 而无论证），或 Stage 3 精度失败根因指向优化后公式 |
| 算法调研缺失或负向断言被推翻 | Stage 2 维度 8 判 fail（§1.6.0 四问缺失 / 候选表无基线 / 复杂度表缺口径，或"无在线变体/无化简公式"断言与参考表、源码 online 证据矛盾，或复杂度算术复算错误，或调研选 A 设计做 B） |
| 标量/循环未向量化且无理由 | Stage 2 维度 8 判 fail（§1.6.2 缺分析或保留理由不充分，如逐元素标量循环、标量累加可用向量归约替代而未替代） |
| 迁移：源算子语义理解偏差 | Stage 2 维度 0 判 fail（§0.1/0.2 与源码不一致，如累加顺序 / 中间 dtype / 输出 shape 错误） |
| 迁移：照搬源硬件方案 | Stage 3 编译失败根因为三维 Kernel / GPU 专用 API / warp shuffle 类未重设计结构 |
| 迁移：NPU 重设计不可行 | Stage 2 维度 0 判 fail（§0.6 重设计违反分形 / 容量 / 一维 Kernel 约束，或语义保持论证缺失） |
