# 分核策略标准（物理核数适配）⭐

> **单一事实源**：本文件是「分核策略三要素」的权威版本。conductor / designer / reviewer / developer / optimizer / README 只引用本文件，不复制本文文本（可留一句话摘要）。
>
> **背景**（依据 [docs/开发指南.md](../../../../docs/开发指南.md) §3.3「物理核数限制与分核策略优化」）：昇腾 NPU 的 AI Core 物理核数有限，实际数目必须通过 `NPUUtils.get().get_aicore_num()` 接口实查获取（Cube/混合算子直接使用返回值；纯 Vector 算子核数翻倍，即 `get_aicore_num() * 2`），禁止以文档假设或经验值（如 20~24）替代实查。运行时虽允许下发大量逻辑内核（如 65535），但超出物理核数的部分会被**串行调度**，引入额外的核启动开销；内核总数非物理核数整数倍还会造成负载不均（如启动 21 个内核将导致其中一个物理核执行两倍任务）。因此 Tiling 策略是**双重维度**：Block 尺寸（片上缓存容量 / 32B 尾轴对齐）+ **分核策略（逻辑核数与物理核数适配）**。

## 1. 分核策略三要素（Stage 1 门禁核对项）

DESIGN.md 的 Tiling 策略章节（§5）必须同时包含：

| 要素 | 内容 | 缺失判定 |
|------|------|---------|
| ① 逻辑核数计算 | `num_logical_kernels = ceil(M/block_M) × ceil(N/block_N)`（按算子实际输出网格） | 无核数计算 → 门禁失败 |
| ② 物理核数依据 | 目标设备物理核数为 `NPUUtils.get().get_aicore_num()` 实查值（Cube/混合算子直接使用返回值；纯 Vector 算子核数翻倍，即 `get_aicore_num() * 2`），须记录查询代码与实际返回值，禁止文档假设/经验值替代 | 无物理核数、无查询记录或使用假设值 → 门禁失败 |
| ③ 规模判定与分核方案 | 三选一并给出依据：**逻辑核数 ≤ 物理核数**——结论"无需适配"及依据；**中等规模**——通过调整 block_M/block_N 减少内核总数，使其接近物理核数整数倍（按实查核数取 1×/2×/3×），说明对齐取值；**极大规模**——无法通过调整分块缩减核数时，固定启动内核数 = 物理核数，核内 `T.serial` 串行处理多个逻辑块任务（`num_local_tasks = T.ceildiv(num_logical_kernels - kernel_id, num_physical_kernels)`），摊薄核启动开销，且循环边界必须为静态值 | 无判定或无方案 → 门禁失败 |

机械核对由 `statectl gate 1`（`gate_lint.py` 规则 `S1-CORES-LOGICAL` / `S1-CORES-PHYSICAL` / `S1-CORES-PLAN`）执行存在性检查；数值自洽与正确性由 Stage 2 维度 3 检视。

## 2. 各角色要求文本（调度 prompt 透传 / 自检用）

### 2.1 Stage 1 — designer 设计要求（透传文本）

> Tiling 策略必须含分核策略三要素：逻辑核数计算、物理核数依据（`NPUUtils.get().get_aicore_num()` 实查——Cube/混合直接用返回值、纯 Vector 算子核数翻倍；记录查询代码与实际返回值，禁止文档假设/经验值替代）、规模判定与分核方案（逻辑核数 ≤ 物理核数给『无需适配』依据 / 中等规模对齐物理核整数倍 / 极大规模核内串行）；核内串行循环边界必须为静态值。参考 docs/开发指南.md §3.3。

### 2.2 Stage 2 — reviewer 检视要点（透传文本）

> 维度 3（Tiling 策略）须核对分核策略三要素齐全、物理核数为 NPUUtils.get().get_aicore_num() 实查值（纯 Vector 算子翻倍）、核数与 block 取值自洽、核内串行边界静态。

### 2.3 Stage 3 — developer 提醒

按 DESIGN.md §5 分核方案实现（对齐或核内串行），不得擅自改回逻辑核超发。

### 2.4 Stage 4 — optimizer 调优提示（透传文本）

> 分核策略调优是可选优化策略：调整 block 使核数对齐物理核整数倍（消除负载不均）/ 极大规模下核内串行 persistent 化（摊薄核启动开销）。

另：DESIGN.md §1.6 含实验裁决三件套（主选+备选+裁决计划）时，A/B 实测为调优必做项——按裁决计划对代表 shape 实测主选 vs 备选（perf_opt/ 下产出备选变体，基准不动），实测出裁决所依赖的未知常数，按判定阈值裁决；备选胜出（全局或按 shape 分片）则采纳备选变体，并按 [perf-feedback.md](perf-feedback.md) 产出 `perf_feedback.md` + `[DESIGN_LIMIT]` 信号交 conductor 受控路由（附录补记回写 DESIGN.md 或设计修订路径 C）；主选胜出则以实测数字经附录补记固化 §1.6.3 判定依据。

## 3. 分核相关失败信号路由（conductor）

| 信号 | 来源 | 路由 |
|------|------|------|
| DESIGN.md 缺分核策略要素 | Stage 1 门禁（statectl gate） | 门禁失败 → `fail_stage(1)` 重试（计 `stage_retry_count[1]`） |
| 维度 3 分核策略 fail | Stage 2 `REVIEW.md` | 设计修订路径 A（计 `retry_count`） |
| `[DESIGN_ERROR]` 分核类（核内串行边界依赖动态值 / 逻辑核数远超物理核数致串行调度开销剧增） | Stage 3 Subagent | 设计修订路径 B（计 `retry_count`） |
| 分核参数性能不达标 | Stage 4 | Stage 4 内迭代（参数级不逆向反馈），不回退 Stage 1/3；判定为设计层天花板时按 [perf-feedback.md](perf-feedback.md) 受控路由 |
