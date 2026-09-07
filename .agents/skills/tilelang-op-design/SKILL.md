---
name: tilelang-op-design
description: "根据算子需求生成 TileLang-NPUIR 算子设计文档（DESIGN.md）。涵盖算法调研（Phase R 调研四问：等价化简公式替代、在线算法、复杂度对比、硬件亲和性——先于公式优化，输入公式/源算法只是候选之一）、编程模式选型（Developer/Expert/混合）、算法级优化（数学等价的公式优化：更少计算量/访存量 + 循环/标量计算的向量化替代分析，替代不了的须给出充分理由）、API 映射、内存层级规划、Tiling 策略、循环结构、同步策略、验证方案等。迁移场景先做源算子三问解读（语义/算法/优化手段）→ 算法调研 → 硬件耦合性判定与 NPU 算法重设计。触发：设计算子、生成 DESIGN.md、算子方案设计、新算子开发、算子实现方案、迁移算子设计。"
---

# TileLang-NPUIR 算子设计文档生成

## 1. 目标

根据算子需求信息，生成一份完整的 TileLang-NPUIR 算子设计文档（`DESIGN.md`），涵盖以下核心决策：

- **算法调研** ⭐（设计第一优先级的前置，先于一切算法 / Tiling 决策）：对**同一数学语义的算法族**做系统调研——有没有**等价化简公式**、有没有**在线算法**、**复杂度**是否过高、算法是否**硬件亲和**（调研四问）；输入公式 / 源算法只是候选之一（Phase R 强制执行，落入 §1.6.0；迁移任务的调研结论另作为 Phase M1 判定与重设计的输入）
- **算法级优化** ⭐（设计第一优先级）：先在保证数学等价的前提下优化公式，使计算量更少或访存量更少；再审视实现方案中的循环 / 标量计算，能用向量操作等价替代的全部替代，替代不了的必须给出充分理由；最后枚举核内布局与向量化轴候选并量化取舍（I/O layout 是契约、核内布局是设计变量）（Phase 2 强制执行，分析对象为 §1.6.0 调研选定算法，落入 §1.6）
- **编程模式选型**：Developer / Expert / 混合模式
- **API 映射**：将数学公式拆解为 TileLang DSL 原语组合
- **内存层级规划**：GM → L1/UB → L0 的数据搬运路径
- **Tiling 策略**：Block 划分与 Tile Shape 设计
- **循环结构**：T.Parallel / T.serial / T.Pipelined / T.Persistent 的选择
- **同步策略**：自动同步 vs 手动同步标志
- **验证方案**：与 Golden 函数输出作比对

**迁移场景（提供了迁移算子路径）额外目标**：先彻底读懂源算子（语义 / 实现算法 / 优化手段），再做算法调研（Phase R：源算法只是候选之一），再判定算法与优化手段的硬件耦合性，对不能直接用在 NPU 上的部分完成 NPU 算法重设计（重设计可直接承接调研选定的算法族）——迁移决策全部记录在 `DESIGN.md` §0，并驱动 §1–§11 的所有设计决策。

---

## 2. 输入要求

### 必需信息

| 字段 | 说明 |
|------|------|
| **项目名称** | 项目分组名，决定 `examples/{project}/` 项目目录；由 conductor 解析，无明确项目名时与算子名相同 |
| 算子名称 | 如 `softmax`、`layer_norm`、`flash_attention`，决定 `examples/{project}/{op}/` 算子目录及 `{op}.py` 文件名 |
| 数学公式 | 算子的数学表达，如 $\text{softmax}(x_i) = e^{x_i} / \sum e^{x_j}$ |
| 输入张量规格 | shape、dtype |
| 输出张量规格 | shape、dtype |
| 编程模式偏好 | Developer / Expert / 混合 |
| **迁移算子路径** ⭐ | 原算子文件路径（迁移时必需），用于分析原始实现及 实现 golden 函数 |
| **输出形状** ⭐ | 原算子输出 shape（迁移时必需），如 `(N, M)` 或 `(M, N)` |

**迁移算子时必须提供原算子路径和输出形状**，否则无法证明迁移正确性。Golden 实现一致性要求详见 [tilelang-op-develop SKILL.md](../tilelang-op-develop/SKILL.md)。

> **迁移任务的字段来源**：迁移场景下，数学公式 / 输入规格 / 输出规格**不向用户提问**，由本 skill 在 Phase M0 解读源算子代码后自行得出（与 conductor「迁移执行规则」第 4 条一致）；编程模式默认 `developer`（用户显式指定时以用户为准）。

**提问规则（必须严格遵守）**：
1. **优先使用调用方传入的字段**：若调用方（如 `@tilelang-op-conductor` 通过 designer 传入 `op_requirements` 结构）已经提供了字段值，**全部跳过提问**，直接进入技术约束检测和 DESIGN.md 生成
2. **每次只询问一个字段**：使用 `question` 工具时，`questions` 数组中只包含一个元素
3. **按表格顺序依次询问**：算子名称 → 数学公式 → 输入张量规格 → 输出张量规格 → 编程模式偏好
4. **已提供的字段跳过**：如果用户在初始请求中已提供某个字段的值，跳过该字段继续下一个
5. **示例**：
   - 第 1 次询问：只问"数学公式"
   - 用户回答后，第 2 次询问：只问"输入张量规格"
   - 以此类推

**⚠️ 当被 conductor → designer Subagent 链路调度时**：
- designer 会把 conductor 在 Primary 上下文预检收集到的 `op_requirements` 完整传入
- 此时 5 个必需字段应当全部已 provided，跳过整个提问环节
- 若 skill 仍发现字段歧义或缺漏，**不要**在当前 Subagent 上下文调用 `AskUserQuestion`（透传不到真实用户），而是让 designer 返回 `partial_input` + 缺失字段名给 conductor，由 conductor 在 Primary 上下文追问

### 推荐信息

| 字段 | 说明 |
|------|------|
| 典型配置 | 常用的 shape 组合与优先级 |
| 参考实现 | PyTorch / NumPy 参考代码 |
| 性能目标 | 目标吞吐量或延迟 |
| 动态轴说明 | 哪些维度在运行时变化 |

若用户未提供**必需信息**中的任一项，通过提问补全后再继续。

---

## 3. 技术约束（必须遵守）

本项目为 TileLang-NPUIR （后端为华为昇腾 NPU），与 GPU 版 TileLang 有显著差异。外部参考实现不可直接使用，必须转换为 Ascend 兼容方案。

**生成 DESIGN.md 前必须执行强制检测**：三维 Kernel、GPU 专用 API、GEMM 非整除、L0C 溢出等。

详细已知限制清单、强制检测规则、警告输出模板见 [references/ascend-constraints.md](references/ascend-constraints.md)。

### 资源分层加载（上下文预算控制）⭐

references 与 templates 按**任务类型与调研深度条件加载**，禁止启动时全量预读（目标：非迁移轻量任务的指令栈 ≤90KB）：

| 资源 | 加载条件 |
|------|---------|
| [references/migration-analysis.md](references/migration-analysis.md)（17KB） | **仅迁移任务加载**（Phase M0/M1 前）；非迁移任务禁止读取 |
| [references/algorithm-research.md](references/algorithm-research.md)（14KB） | 轻量调研（单步逐元素 / 纯搬运类，Phase 1 判定）只读 **§1–§4**（执行位置 / 深度分级 / 调研四问 / 结论规范）；完整调研（规约 / 统计 / 窗口 / 矩阵 / 多步 / 融合类）读全文（含 §5 参考表） |
| [references/ascend-constraints.md](references/ascend-constraints.md) | Phase 3 技术约束检测时读 |
| [references/decision-tree.md](references/decision-tree.md) | 编程模式 / API 映射决策时读 |
| [references/info-sources.md](references/info-sources.md) | Phase 3 信息收集时读 |
| [references/quality-checklist.md](references/quality-checklist.md) | Phase 7 自检时读 |
| [templates/design-template.md](templates/design-template.md)（25KB） | **按需取章节**：生成对应 DESIGN.md 章节时再读该节模板，禁止全文预读 |
| [templates/report-template.md](templates/report-template.md) | Phase 8 输出报告时读 |

---

## 4. 工作流程

> **双轨工作流**：迁移任务（必需信息含迁移算子路径）按 **Phase M0 → Phase R → Phase M1** 执行（源算子三问解读 → 算法调研 → 硬件耦合性判定与 NPU 算法重设计——调研结论是 M1 判定与重设计的输入），产出迁移决策后再进入通用设计流程；非迁移任务从 Phase 1 开始，按 **Phase 1 → Phase R → Phase 2** 执行（算法调研先于算法级优化设计）。两轨的算法决策（M1 / Phase 2）都必须以 Phase R 调研结论为输入。

### Phase M0：源算子深度解读（迁移任务必执行）⭐

按 [references/migration-analysis.md](references/migration-analysis.md) 的三问框架执行：

1. **Read 源算子代码全文**（含 host 侧逻辑与 pass_configs），按 §3.1 结构化阅读协议逐层读：入口签名 → Kernel 结构 → TIR 原语 → 内存分配 → 循环调度 → host 逻辑 → 编译配置。
2. **第一问（语义）**：解读数学语义、I/O 契约、规约语义（累加顺序）、dtype 语义、边界语义 → 产出 §0.1 / §0.2。完成标准：能写出与源算子数值一致的 golden 函数。
3. **第二问（算法）**：产出计算步骤分解表、数据流图（源硬件视角）、循环与并行结构、host 侧逻辑 → §0.3。完成标准：源码每个计算语句都归入某步骤、无臆造步骤。
4. **第三问（优化手段）**：逐项识别源码中的优化手段（SMEM tiling / coalesced / mma / warp shuffle / 异步流水 / swizzle / register 累加 / persistent / online 算法……），逐项记录其目的、机制、依赖的源硬件特性 → §0.4。**识别不出来 ≠ 不存在**，未识别的优化会被静默丢弃。

### Phase R：算法调研（所有任务必执行）⭐

> **调研先于优化**：算法族的选择（两遍还是单遍、materialize 还是 online、直接法还是变换域）对性能的影响通常大于后续 tiling / 流水优化，必须在任何算法决策（M1 / Phase 2）与 Tiling 设计之前完成。执行标准与常见算子族参考表见 [references/algorithm-research.md](references/algorithm-research.md)。执行位置：**迁移任务在 Phase M0 之后、Phase M1 之前**（调研结论作为 M1 判定与重设计的输入）；**非迁移任务在 Phase 1 之后、Phase 2 之前**（Phase 1 的算子类别与复杂度级别判定决定调研深度）。本地信息源（参考表 / pattern-library / examples / 源码）未覆盖时可辅以**互联网检索**（webfetch，若环境可用）补充候选——只取算法思路、来源可溯、API 存在性与性能代价仍须本地佐证/实测，纪律与降级处理见 algorithm-research.md §6。

1. **明确调研对象**：同一数学语义的算法族——所有能算出相同结果（数学等价或容差内等价）的算法结构。输入公式 / 源算法（迁移任务取自 M0 解读）**只是基线候选之一**，不得当作唯一算法。
2. **调研四问（每问都必须有明确结论）**：
   - **R1 等价化简公式**：有没有数学等价（或容差内等价）的化简公式 / 结构重排？对照 algorithm-research.md §5 参考表命中行 + 恒等变形自查 + examples/pattern-library 同类实现所用公式；候选表含基线，每候选一行等价性初判与收益方向（正式等价论证与收益量化在 §1.6.1 完成，不重复）。
   - **R2 在线算法**：有没有单遍 / 流式（online）变体？查参考表 + 迁移任务查 §0.4（源算法已用 online 手段是直接证据）+ 结构判据（可分解为 running 统计量的算子原则上存在在线变体）；收益口径 = 扫描遍数、中间缓冲、UB 驻留可行性。结论必须是明确的「有」（纳入对比）或「无」（写结构依据）。
   - **R3 算法复杂度**：候选间四口径量化对比——FLOPs / 访存 Bytes / 数据扫描遍数 / 中间缓冲峰值（附加：可并行度、跨核归并代价）；基线候选必须在表中；口径写清含什么不含什么。带宽受限算子上访存/遍数是主导项，不得只比 FLOPs。
   - **R4 硬件亲和性**：逐候选对照检查清单——计算单元匹配（Cube MAC 密集 / Vector 逐元素规约）、片上容量（UB 192KB / L1 512KB / L0C 128KB）、对齐整除（尾轴 32B / 分形 ≥16 / 向量宽度）、静态边界、流水可融合性（T.Pipelined / CV 融合）、跨核结构；**负向淘汰与弃选论证证据规则同口径**（附 docs/testing/examples 佐证或显式「未文档化假设 + 估算依据」，代价类论断先查 pattern-library）。
3. **调研深度分级**：单步逐元素 / 纯搬运类（Phase 1 / M0 判定）可轻量调研（四问各一行结论 + 依据）；规约 / 统计 / 窗口 / 矩阵 / 多步 / 融合类必须完整调研（R1 候选表 + R3 对比表 + R4 逐候选评估）。
4. **产出与下游约束**：
   - 调研结论落入 `DESIGN.md` **§1.6.0**：选定算法族 + 关键依据 + 与基线的结构差异；「无更优替代」必须写明调研范围（查过的参考表条目 / examples / pattern-library / 结构分析），禁止空白；
   - §1.4 算法描述、§1.6.1–§1.6.3 的分析对象 = **调研选定算法**（迁移任务：经 M1 重设计落地）；不得回退到未调研的输入公式 / 源算法直译；
   - 迁移任务：M1 的四态处置与 §0.6 重设计对照调研结论——源算法被替换时，§0.4 已识别优化手段的意图承接逐项说明（algorithm-research.md §4），不得静默丢弃；
   - R3 并列（判定裕度小）的候选 → 交 §1.6.3 布局分析 / 实验裁决模式决胜，不得纸面拍板。

### Phase M1：硬件耦合性判定与 NPU 算法重设计（迁移任务必执行）⭐

1. 按 [references/migration-analysis.md](references/migration-analysis.md) §5 的三层模型（语义层/算法层/优化层）+ 四态处置（**保留 / 等价替换 / 重新设计 / 舍弃**）逐项判定：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？
   - 每项判定必须写明依据（映射表条目 / `examples/` 佐证 / `docs/` 条目）。
   - 等价替换优先查 §5.3「GPU → Ascend 能力映射表」。
   - **判定输入含 Phase R 调研结论（§1.6.0）**：源算法与调研候选同台比较——调研选定更优算法族时，其采纳经 §6.2「算法替代」模式进入重设计（§0.6 引用 §1.6.0），源算法的优化意图承接规则不变；源算法未被替换时，调研结论同样约束重设计方向（如源算法两遍结构在调研后确认无可在线化替代，须引用该结论而非默认照抄）。
2. 对判定为「重新设计」的条目，按 §6 重设计模式库（并行结构 / 归约 / 矩阵计算 / 数据搬运 / 跨阶段融合 / 算法替代 / host 内移）设计 NPU 新算法，**每个重设计项必须给出语义保持论证**（数学等价或容差内等价 + 边界语义核对）→ §0.5 / §0.6。
3. **迁移决策成为后续设计的输入**：Phase 1–4 基于**重设计后的 NPU 算法**展开（可能与源算法结构不同），不得再照抄源方案；其中 Phase 2 在 §0.6 重设计结果之上继续做数学等价优化与向量化替代分析，**不得止步于源公式的直译**。

### Phase 1：输入解析与算子特征分析

1. 解析算子名称与数学公式（迁移任务：取自 Phase M0 的语义解读 §0.1，不重复推断）
2. 验证必需字段是否完整
3. 分析算子特征（迁移任务：分析对象是 Phase M1 产出的 NPU 算法，而非源算法）：
   - **计算类型判定**：
     - 纯 Vector（element-wise / reduction）→ 仅需 UB
     - 纯 Cube（仅 matmul）→ 需要 L1 + L0A/L0B/L0C
     - 混合（matmul + element-wise 后处理）→ 核间流水线，需要 CV 融合
     - **Host 预处理**：如 im2col 等预处理步骤（迁移任务以 §0.5 的处置为准：内移 kernel 或保留 host），标明在 DESIGN.md 的 §1 和 §4 中
   - **复杂度级别**：
     - 单步（如 element-wise add）→ 无循环、单次搬运
     - 多步（如 softmax = max + sub + exp + sum + div）→ 多次计算、可能需要中间缓冲
     - 融合（如 flash attention = GEMM + softmax + GEMM）→ 核间协作、流水线
   - **动态 shape 判定**：是否存在运行时才确定的维度
4. **非整除场景预判**：检查输入 shape 是否可能不被 block size 整除。GEMM 类算子的 `M // block_M` 和 `N // block_N` 在 `M < block_M` 或 `N < block_M` 时产生零 block 或不完整 tile，必须在设计中明确处理策略（host 侧 zero-padding + crop，或 Kernel 内动态 block size）
5. **分核策略预判（物理核数适配）**⭐：先实际查询目标设备物理核数（`NPUUtils.get().get_aicore_num()` 实查；查询失败即无 NPU 环境时明确报告环境异常、不得猜测数值），再按 block 初步取值计算逻辑核数 `ceil(M/block_M) × ceil(N/block_N)` 并与物理核数比较，按标准文件 §1 要素③ 做规模判定与分核方案三选一（无需适配依据 / 中等规模对齐物理核整数倍 / 极大规模核内 `T.serial` 串行且循环边界静态）。**权威标准文本**：`.agents/skills/_shared/standards/core-split-strategy.md`（Read 后按 §1 三要素表与 §2.1 设计要求执行；依据 docs/开发指南.md §3.3）。

### Phase 2：算法级优化设计 ⭐（所有任务必执行）

> **设计的第一优先级是算法层**，先于任何 API / Tiling 决策：⓪ 算法族选型已在 Phase R 完成（§1.6.0 调研结论——本节分析对象即调研选定、（迁移任务）经 M1 重设计落地的算法，不得回退到未调研的输入公式/源算法直译）；① 在保证数学等价的前提下优化公式，使**计算量更少或访存量更少**；② 审视实现方案中的**循环与标量计算**，能用向量操作等价替代的全部替代，**替代不了的必须给出充分理由**；③ 回答"在**哪个轴**上向量化"——向量化轴与核内数据布局是**设计变量而非 I/O 契约**，必须枚举候选布局并量化取舍（阻塞级）。产出落入 `DESIGN.md` §1.6，并作为 §3.1 公式拆解与 §6 循环结构的输入。

1. **数学等价优化（公式级 → §1.6.1）**：对计算公式做恒等变形与结构调整，逐项评估、择优采纳：

   | 优化类别 | 典型做法 | 示例 |
   |----------|----------|------|
   | 恒等变形降代价 | 除法转乘倒数；sqrt+div 转 rsqrt；log/exp 换底缩放 | `x/s` → `r=1/s; x·r`（N 次除法 → 1 次倒数 + N 次乘法）；layer_norm 的 `x/sqrt(var+ε)` → rsqrt 一次 + 向量乘 |
   | 稳定化变形（数学等价且数值更稳） | softmax / sigmoid 减 max；log-sum-exp 改写 | `e^{x_i}/Σe^{x_j}` → `e^{x_i-m}/Σe^{x_j-m}`，m = max(x) |
   | 公共子表达式消除 | 多处出现的子式只算一次，结果驻留 UB 复用 | rmsnorm 中 `x·rms` 与后续缩放共享同一中间量 |
   | 算子降代换 | 高代价函数换低代价等价函数（须佐证 API 存在与精度影响） | exp → exp2·log2e 缩放 |
   | 访存量削减 | 中间结果驻留 UB/L0C 复用、不落 GM；cast 次数最小化；重复读取的输入 tile 只搬一次 | max 与 sum 共享同一次 GM→UB 搬入 |
   | 归约结构优化 | 多轮逐元素扫描合并为单轮多目归约 | softmax 的 max 与 sum 合并扫描（可行时） |

   每个采纳项必须给出**四要素**：原式 → 优化后公式 → **等价性论证**（数学恒等，或容差内等价 + fp16/bf16 舍入影响评估，与 §8.2 精度标准一致）→ **收益量化**（减少的计算量（op 数 / FLOPs）或访存量（Bytes）的估算）。
   **无优化空间时必须显式写明结论与依据**（如"单次逐元素映射，无公共子表达式与高代价算子"），不得留空、不得跳过。

2. **向量化替代分析（循环 / 标量消除 → §1.6.2）**：盘点实现方案中**全部**循环与标量计算点，逐项判定能否用向量操作等价替代：
   - 逐元素标量循环 → v-prefix 逐元素 API（vadd / vmul / vexp / vcast …）或 `T.Parallel` 向量化；
   - 标量累加 / 多轮逐元素扫描循环 → 一次向量归约（`T.reduce_sum/max/min` 等）；
   - 标量广播 / 按行填充 → `T.vbrc`；
   - 逐元素条件计算 → 向量掩码 / 比较选择（须有 API 佐证）；
   - 替代方案所用 API 必须以 `examples/` 或 `docs/Tilelang.language/` 佐证真实存在（禁止凭记忆猜 API；查证按 Phase 3 信息源流程执行）。

   **无法替代的点必须逐项给出充分理由**（须具体到本算子，不得泛泛而谈）。可接受的理由类别：
   - block 级索引 / 任务映射计算（每 block 少量标量，非逐元素热点）；
   - host 侧 shape / stride 等元数据计算（不在 kernel 内）；
   - tile 级顺序依赖（如 online 更新、跨 tile 累加的循环携带依赖，且已论证无法用归约 / 双缓冲消除）；
   - 依赖动态 shape 的边界处理（且已说明为何不能静态化或 padding）；
   - 所需向量 API 在本项目确不存在（附查证过程与佐证）。

   仅以"实现简单"、"照源码写"为由保留标量循环 **不构成充分理由**。

3. **向量化轴与数据布局决策（→ §1.6.3，阻塞级）**：I/O layout 是契约，但**核内布局与 lane 映射是自由变量**。对任何具有 ≥2 个非 batch 维度的算子，**按算子类别从下列清单枚举候选**（类别内必选候选齐全；跨类别组合算子枚举各类别候选的笛卡尔组合中有意义者）：
   - **逐元素 / 广播类**（add/mul/激活/bias/residual）：连续轴选择 × **广播轴与向量轴对齐**（如 bias 沿 C 广播：C-last 使广播退化为标量 broadcast、C-first 退化为逐行标量加载——广播代价随轴选择反转）；
   - **窗口 / 池化类**：必选候选「I/O 原生布局 + 最内连续轴」与「核内重排布局 + 高整除性轴（如 C 轴）」；重排路径区分核内融合转置（T.transpose 二轴交换链，实测 µs 级，见 pattern-library §1.1）与 host permute（百 µs 级，通常净亏）；
   - **规约类**（softmax/layernorm/rmsnorm/logsumexp/mean-var）：**水平归约（lane 映射到归约轴，用向量 reduce）vs 垂直扫描（lane 映射到独立行，串行步进归约维）**——两者可用同一条张量轴但 lane 语义不同，是"选哪条轴"之外的**正交决策**，须分别枚举；同时评估两遍 vs online 单遍算法与 lane 映射的交互；
   - **卷积类**：直接跨步窗口（strided load）vs UB 内 im2col 重排（连续 load + 重排代价）vs implicit GEMM（Cube 路径，交 cube-skill 细化但候选矩阵须列出并给接口）；
   - **Cube / MixCV 类**：fractal/NZ 布局、load_nd2nz 即时重排 vs 显式转置、epilogue 归属侧（Cube vs Vector）——细节按 cube/mixcv skill，但 §1.6.3 须记录决策与依据；
   - **gather / scatter / 索引类**：连续批量 gather 粒度与向量轴对齐、逐元素索引 vs 分段拷贝；
   - 已验证模式与实测代价**优先查** `tilelang-op-optimize` skill 的 [references/pattern-library.md](../tilelang-op-optimize/references/pattern-library.md) §1（含核内融合转置链、C 轴切片累加、H-collapse、tiling 启发式等已验证形态与实测数字）——**禁止凭先验（尤其 GPU 直觉，如"transpose = GM 级重排"）否决候选**；核内重排候选的 repack 代价必须落到具体 API 及其文档——T.transpose 见 `docs/Tilelang.language/创建操作/T.transpose.md`（须亲自打开核对：UB 级执行、permutation 限两轴交换且 >2 轴可分解为相邻轴交换链、dtype 矩阵 fp16 ✓/fp32 ✓/bf16 ×/整型 ×），**禁止把核内重排写成 GM 级全量重排的稻草人**（正确口径：GM 流量不变，代价在 VTransposeOp 向量管线开销与 UB 容量）；
   - **逐候选评分**：轴长对向量宽度（fp16/bf16 ×8、fp32 ×4）的整除性与尾 lane 浪费率、累加链形态（串行累加维的跨步系数是否阻碍向量化）、repack 代价、UB 容量影响（重排布局的驻留缓冲预算对照 §4.5）；
   - **弃选必须给量化理由**（禁止"未考虑"；纯 GEMV/单轴算子可写明"仅一个候选轴"豁免）；
   - **弃选论证证据规则（防未检索先否定）**：凡负向论断（"API 不支持 / 代价高 / 无先例 / 无链支撑"），必须附已亲自核对的 `docs/`、`testing/` 或 `examples/` 路径并引用具体限制条款；确无文档的必须显式标注「未文档化假设 + 估算依据」。**弃选候选所依赖的 API 必须先枚举出名字，再在 `docs/Tilelang.language/` 全部子目录检索**（含 AGENTS.md 关键词路由未覆盖的目录：`创建操作/`、`索引与元素操作/`、`条件操作/`、`排序操作/`、`逻辑操作/`、`原子操作/`）——禁止在检索前凭先验（尤其 GPU 直觉，如"transpose = GM 级重排"）否决候选；
   - 迁移任务：GPU 源码的并行轴选择（thread/warp 映射）是**输入而非结论**——GPU 轴与 NPU Vector 轴不对应，须对照 §0.3 的源码轴选择独立评估；
   - **决策落地约束**：主选轴与布局必须同步写入 §3.3 伪代码与 §6 循环结构——累加循环的内层向量维必须是本节主选轴；
   - **实验裁决模式（判定裕度依赖未实证常数时强制）**：当主选与弃选候选的判定裕度落在任一未文档化/未实证常数（如向量管线吞吐比、跨步 UB 访问代价、转置指令效率、水平归约指令效率、gather 逐元素代价等；pattern-library §1/§2 已实测的量不算未实证——先查再判）的不确定区间内时，**不得仅凭先例或下界估算纸面单选**——必须产出三件套：① **主选方案**：默认实现（进 Stage 3），仍须是当前证据下的最优判断；② **备选方案**（≤1 个）：结构完整可实现——buffer 形状与 UB 预算、循环结构、repack/转置链在 kernel 内的位置、dtype 路径、分核三要素按新任务粒度重算，达到 Stage 4 可直接实现的深度（弃选论证 ≠ 备选设计，代价模型用于排序而非替代结构设计）；③ **实验裁决计划**：代表 shape 清单、测量指标（latency + 未知常数的实测/反解方法）、**判定阈值**（如备选 latency < 主选 × (1−ε) 才翻转）、裁决结果回写路径（实测数据触发设计修订回写 §1.6.3）。备选方案随 DESIGN.md 冻结进入 Stage 4 调优的 A/B 清单；**shape 特化工厂（lru_cache 等）的算子，实验裁决计划必须评估按 shape 分派主选/备选的可行性**（按负载分派达成总体最优，而非全局二选一）。

4. **产出与下游约束**：
   - §1.6.1 的**优化后公式**是 §3.1 公式拆解的唯一输入；
   - §1.6.2 的向量化结论必须与 §6 循环结构一致——判定向量化的计算点，§6 不得再出现对应的逐元素标量循环；
   - §1.6.3 的布局决策必须与 §3.3 伪代码、§4 内存规划（驻留缓冲形状/预算）、§6 循环结构三方一致——伪代码与循环结构中的 buffer 形状和向量维不得与选定的布局方案矛盾；
   - 实验裁决模式下，§3.3/§4/§6 只承载主选方案；备选方案的结构设计与分核参数独立成节（§1.6.4 或 §1.6.3 内小节），不与主选方案的结构章节混排。

### Phase 3：信息收集

**必须执行强制步骤 0：搜索本项目同类实现**。详细工具调用、信息收集步骤、禁止行为见 [references/info-sources.md](references/info-sources.md)。

**必须执行强制步骤 0.5：检索实测性能模式库**——读 `tilelang-op-optimize` skill 的 [references/pattern-library.md](../tilelang-op-optimize/references/pattern-library.md) §1（已验证向量化轴/布局模式与实测代价）、§2（编译器/运行时陷阱，**注意版本戳**——tilelang 重编译后旧陷阱结论自动待重验，勿引用已标注"已失效/已推翻"的条目）与 §4（案例索引——同类算子正/反例参考，命中时优先精读对应算子目录，优先于盲目 Glob）。该库是任务实测的积累（如核内融合转置链实测代价、C 轴切片累加已验证形态、跨步系数阻碍向量化案例），其优先级高于 docs 规格与 examples 先例（见 info-sources.md 优先级表）；§1.6.3 的候选枚举与代价模型必须先对照该库，库内已有的实测数字不得当作"未实证常数"重新假设。

> 迁移任务的额外信息源：Phase M0/R/M1 的源算子解读、算法调研结论与迁移决策（优先级介于 `examples/` 同类实现与外部参考实现之间——迁移决策界定"算法该怎么设计"，`examples/` 界定"API 怎么用"）。

### Phase 4：生成 DESIGN.md

基于 [templates/design-template.md](templates/design-template.md) 模板，填充所有章节：

0. 源算子解读与迁移分析（**迁移任务必填**：语义 / 算法 / 优化手段 / 硬件耦合性分析 / NPU 重设计；非迁移任务删除本章节）
1. 概述（迁移任务：描述**迁移决策后的 NPU 侧算法**，与源算法有差异时注明；**必含 §1.6 算法调研与优化分析——1.6.0 算法调研（调研四问，产出见 Phase R）+ 1.6.1 数学等价优化 + 1.6.2 向量化替代分析 + 1.6.3 向量化轴与数据布局决策**，产出见 Phase 2；§1.4 算法描述与 §1.6.0 选定算法一致）
2. 编程模式选型
3. API 映射设计
4. 数据规格与内存规划
5. Tiling 策略（**必含：非整除时 padding+crop 策略，或 Kernel 内动态 block 方案；分核策略三要素——逻辑核数计算、物理核数依据、规模判定与分核方案（对齐物理核整数倍 / 核内串行），见 Phase 1 第 5 项**）
6. 循环与调度结构
7. 同步策略
8. CV 融合设计（详见 design-template.md §8.2）
9. 验证方案（Golden + **L0 门槛测试计划**；完整分层套件 L1/L2/Boundary 交由 `tilelang-op-develop`，不在此枚举）
10. 风险点与注意事项
11. 交付清单

### Phase 5：质量自检

按照 [references/quality-checklist.md](references/quality-checklist.md) 中的自检清单逐项检查，确保文档质量。**所有任务必须通过算法级检查项（算法调研完整（§1.6.0 调研四问齐全、结论有依据）/ 数学等价优化分析完整 / 向量化替代分析完整）**；**迁移任务必须额外通过清单中的迁移专属检查项**（源算子解读完整性 / 耦合性判定有依据 / 重设计有语义保持论证 / §1–§7 与迁移决策一致）。

### Phase 6：针对性修订

仅修正未通过自检的项目。信息确实不足的标注为「待确认」并说明原因。

### Phase 7：输出

将 `DESIGN.md` 输出到 `examples/{project}/{op}/` 算子目录（`{project}` 为项目名称、`{op}` 为算子名称，均由调用方传入；无明确项目名时与算子名相同）。若文件已存在，询问是否覆盖。

### Phase 8：任务复盘（Retrospective，自进化钩子）⭐

DESIGN.md 输出后（`first_design` 与 `revision` 每次执行均写），向算子目录 `examples/{project}/{op}/RETROSPECTIVE.md` **追加**复盘章节（先 Read 既有内容，整文件写回，只追加不覆盖历史章节）：

- 章节模板与字段规范（canonical）：`tilelang-skill-evolution` skill 的 [references/retrospective-schema.md](../tilelang-skill-evolution/references/retrospective-schema.md)。两个标准表（**Skill Flow Issues**：area / issue / evidence / suggested_doc_change / vp_type；**Value Point Proposals**：title / vp_type / evidence / repro / toolchain_stamp / target_doc）+ **Transferable Lessons（可迁移教训）**小节。
- **revision 模式价值最高**：从 `design_error_summary` 与上一版设计差异中提取"哪类设计判断被推翻、正确依据是什么"（如 API 误判、tiling 不可行、内存层级估算错误、弃选论证前提与文档矛盾），逐条记入 Value Point Proposals 并附证据定位（DESIGN.md v{N} 章节 / REVIEW.md 维度 / 文档路径）。
- 质量红线：无则如实写 `none`；单任务偶然现象不得写成通用规则；D 类（实测/实证）须带 evidence + repro + toolchain_stamp 三件套；负面发现（"此路不通 + 原因 + 证据"）优先记录。

---

## 5. 算子特征分析决策树

详细决策树（Ascend 版）、平台识别、API 映射规则、NPU 硬件约束（分形限制 / 对齐要求 / 存储大小上限）见 [references/decision-tree.md](references/decision-tree.md)。

---

## 6. 信息源优先级

信息源优先级表与冲突处理原则见 [references/info-sources.md](references/info-sources.md)。

---

## 7. 错误处理

| 场景 | 处理方式 |
|------|----------|
| 用户未提供数学公式 | 提问补全，给出常见算子公式作为参考 |
| 必需字段缺失 | 列出缺失项，逐一提问 |
| 迁移算子路径不存在 / 源码不可读 | 返回 `source_missing` 给调用方，不得在未读源码的情况下臆测语义生成设计 |
| 源码解读后仍有语义歧义（如累加顺序无法确定） | 在 §0.1 标注「待确认」，列入风险点；conductor 链路下返回 `partial_input` 由 Primary 上下文追问 |
| API 查询无结果 | 标注为「需扩展」，在风险点中说明 |
| 目标文件已存在 | 询问用户是否覆盖或另存 |
| 算子过于复杂 | 建议拆分为多个子算子分别设计 |
| 调研负向断言（无在线变体 / 无化简公式）无结构依据 | 在 §1.6.0 如实标注「待确认」并列入 §9 风险点，交由 Stage 2 检视复核裁定；禁止无依据断言「无」 |
| 调研选定候选的等价性无法论证 | 放弃该候选（回退基线或其他候选），在 §1.6.0 记录放弃原因；禁止无论证采纳；采纳项必须进 §1.6.1 完成四要素论证 |
| 调研候选与源算子语义约束冲突（如在线变体改变累加顺序而语义要求固定顺序） | 在 §1.6.0 记录冲突与取舍依据；语义约束优先于性能收益 |
| 优化手段无法判定耦合性 | 按「重新设计」处理并在 §0.5 标注不确定项与依据缺口 |
| 公式优化的等价性无法论证 | 放弃该项优化，按原公式设计，并在 §1.6.1 记录放弃原因；禁止无论证采纳 |
| 循环 / 标量点找不到向量替代且理由不充分 | 在 §1.6.2 如实标注「待确认」并列入 §9 风险点，交由 Stage 2 检视裁定；不得默认保留标量循环 |
| 向量替代所需 API 查证不存在 | 该计算点按"不可替代（API 缺失）"记录理由与查证过程，可另选算法路线绕开该 API |

---

## 8. 完成报告

文档生成完成后，按 [templates/report-template.md](templates/report-template.md) 输出报告（迁移任务含源算子解读结论、耦合性判定统计与重设计项清单）。

---

## 9. 生成算子

完成报告后，询问用户是否根据此报告生成对应算子代码。

---

## 子目录索引

- [references/algorithm-research.md](references/algorithm-research.md) — 算法调研方法论：调研四问（等价化简公式 / 在线算法 / 复杂度 / 硬件亲和）、调研深度分级、常见算子族替代算法参考表、复杂度四口径、硬件亲和检查清单
- [references/migration-analysis.md](references/migration-analysis.md) — 迁移方法论：三问解读（语义/算法/优化手段）、硬件耦合性判定（保留/等价替换/重设计/舍弃）、GPU→NPU 能力映射、NPU 算法重设计模式库
- [templates/design-template.md](templates/design-template.md) — DESIGN.md 完整模板
