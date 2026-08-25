---
name: tilelang-op-design
description: "根据算子需求生成 TileLang-NPUIR 算子设计文档（DESIGN.md）。涵盖编程模式选型（Developer/Expert/混合）、API 映射、内存层级规划、Tiling 策略、循环结构、同步策略、验证方案等。迁移场景先做源算子三问解读（语义/算法/优化手段）、硬件耦合性判定与 NPU 算法重设计。触发：设计算子、生成 DESIGN.md、算子方案设计、新算子开发、算子实现方案、迁移算子设计。"
---

# TileLang-NPUIR 算子设计文档生成

## 1. 目标

根据算子需求信息，生成一份完整的 TileLang-NPUIR 算子设计文档（`DESIGN.md`），涵盖以下核心决策：

- **编程模式选型**：Developer / Expert / 混合模式
- **API 映射**：将数学公式拆解为 TileLang DSL 原语组合
- **内存层级规划**：GM → L1/UB → L0 的数据搬运路径
- **Tiling 策略**：Block 划分与 Tile Shape 设计
- **循环结构**：T.Parallel / T.serial / T.Pipelined / T.Persistent 的选择
- **同步策略**：自动同步 vs 手动同步标志
- **验证方案**：与 Golden 函数输出作比对

**迁移场景（提供了迁移算子路径）额外目标**：先彻底读懂源算子（语义 / 实现算法 / 优化手段），再判定算法与优化手段的硬件耦合性，对不能直接用在 NPU 上的部分完成 NPU 算法重设计——迁移决策全部记录在 `DESIGN.md` §0，并驱动 §1–§11 的所有设计决策。

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

---

## 4. 工作流程

> **双轨工作流**：迁移任务（必需信息含迁移算子路径）先执行 **Phase M0 / M1**（源算子三问解读 → 硬件耦合性判定与 NPU 算法重设计），产出迁移决策后再进入通用设计流程；非迁移任务直接从 Phase 1 开始。

### Phase M0：源算子深度解读（迁移任务必执行）⭐

按 [references/migration-analysis.md](references/migration-analysis.md) 的三问框架执行：

1. **Read 源算子代码全文**（含 host 侧逻辑与 pass_configs），按 §3.1 结构化阅读协议逐层读：入口签名 → Kernel 结构 → TIR 原语 → 内存分配 → 循环调度 → host 逻辑 → 编译配置。
2. **第一问（语义）**：解读数学语义、I/O 契约、规约语义（累加顺序）、dtype 语义、边界语义 → 产出 §0.1 / §0.2。完成标准：能写出与源算子数值一致的 golden 函数。
3. **第二问（算法）**：产出计算步骤分解表、数据流图（源硬件视角）、循环与并行结构、host 侧逻辑 → §0.3。完成标准：源码每个计算语句都归入某步骤、无臆造步骤。
4. **第三问（优化手段）**：逐项识别源码中的优化手段（SMEM tiling / coalesced / mma / warp shuffle / 异步流水 / swizzle / register 累加 / persistent / online 算法……），逐项记录其目的、机制、依赖的源硬件特性 → §0.4。**识别不出来 ≠ 不存在**，未识别的优化会被静默丢弃。

### Phase M1：硬件耦合性判定与 NPU 算法重设计（迁移任务必执行）⭐

1. 按 [references/migration-analysis.md](references/migration-analysis.md) §5 的三层模型（语义层/算法层/优化层）+ 四态处置（**保留 / 等价替换 / 重新设计 / 舍弃**）逐项判定：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？
   - 每项判定必须写明依据（映射表条目 / `examples/` 佐证 / `docs/` 条目）。
   - 等价替换优先查 §5.3「GPU → Ascend 能力映射表」。
2. 对判定为「重新设计」的条目，按 §6 重设计模式库（并行结构 / 归约 / 矩阵计算 / 数据搬运 / 跨阶段融合 / 算法替代 / host 内移）设计 NPU 新算法，**每个重设计项必须给出语义保持论证**（数学等价或容差内等价 + 边界语义核对）→ §0.5 / §0.6。
3. **迁移决策成为后续设计的输入**：Phase 1–3 基于**重设计后的 NPU 算法**展开（可能与源算法结构不同），不得再照抄源方案。

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
5. **分核策略预判（物理核数适配）**⭐：按 block 初步取值计算逻辑核数 `ceil(M/block_M) × ceil(N/block_N)`，与目标设备物理核数（A2 系列 Cube 核约 20~24 个、Vector 核数量翻倍；优先通过设备接口获取，采用文档假设时必须显式标注）比较：
   - **逻辑核数 ≤ 物理核数**：结论"无需适配"，给出依据；
   - **中等规模**：通过增大 block_M/block_N 减少内核总数，使其接近物理核数整数倍（如 20/40/60），避免负载不均（如启动 21 个内核将导致其中一个物理核执行两倍任务）；
   - **极大规模**（无法通过调整分块缩减核数）：固定启动内核数 = 物理核数，每个物理核内 `T.serial` 串行处理多个逻辑块任务（`num_local_tasks = T.ceildiv(num_logical_kernels - kernel_id, num_physical_kernels)`），摊薄核启动开销；循环边界必须为静态值。
   - 依据：docs/开发指南.md §3.3「物理核数限制与分核策略优化」。

### Phase 2：信息收集

**必须执行强制步骤 0：搜索本项目同类实现**。详细工具调用、信息收集步骤、禁止行为见 [references/info-sources.md](references/info-sources.md)。

> 迁移任务的额外信息源：Phase M0/M1 的源算子解读与迁移决策（优先级介于 `examples/` 同类实现与外部参考实现之间——迁移决策界定"算法该怎么设计"，`examples/` 界定"API 怎么用"）。

### Phase 3：生成 DESIGN.md

基于 [templates/design-template.md](templates/design-template.md) 模板，填充所有章节：

0. 源算子解读与迁移分析（**迁移任务必填**：语义 / 算法 / 优化手段 / 硬件耦合性分析 / NPU 重设计；非迁移任务删除本章节）
1. 概述（迁移任务：描述**迁移决策后的 NPU 侧算法**，与源算法有差异时注明）
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

### Phase 4：质量自检

按照 [references/quality-checklist.md](references/quality-checklist.md) 中的自检清单逐项检查，确保文档质量。**迁移任务必须额外通过清单中的迁移专属检查项**（源算子解读完整性 / 耦合性判定有依据 / 重设计有语义保持论证 / §1–§7 与迁移决策一致）。

### Phase 5：针对性修订

仅修正未通过自检的项目。信息确实不足的标注为「待确认」并说明原因。

### Phase 6：输出

将 `DESIGN.md` 输出到 `examples/{project}/{op}/` 算子目录（`{project}` 为项目名称、`{op}` 为算子名称，均由调用方传入；无明确项目名时与算子名相同）。若文件已存在，询问是否覆盖。

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
| 优化手段无法判定耦合性 | 按「重新设计」处理并在 §0.5 标注不确定项与依据缺口 |

---

## 8. 完成报告

文档生成完成后，按 [templates/report-template.md](templates/report-template.md) 输出报告（迁移任务含源算子解读结论、耦合性判定统计与重设计项清单）。

---

## 9. 生成算子

完成报告后，询问用户是否根据此报告生成对应算子代码。

---

## 子目录索引

- [references/migration-analysis.md](references/migration-analysis.md) — 迁移方法论：三问解读（语义/算法/优化手段）、硬件耦合性判定（保留/等价替换/重设计/舍弃）、GPU→NPU 能力映射、NPU 算法重设计模式库
- [templates/design-template.md](templates/design-template.md) — DESIGN.md 完整模板
