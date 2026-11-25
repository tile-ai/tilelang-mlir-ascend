---
id: pattern/cv-explicit-ring-pipeline
layer: pattern
arch: [Ascend950]
kind: hw-pattern
status: active
verified_scope: tilelang-ascend
---

# 显式 CV 流水：多槽 L1 环、双缓冲 L0 与细粒度事件拆分

## 适用症状与触发条件

- 复杂混合 Cube/Vector 算子（特别是多 GEMM 交错的反向算子，如 Attention Backward 的 `QK / dP / dV / dK / dQ`）在自动调度下出现 `PrepareMultiBuffer` 失败、lowering 超时，或流水线中存在大量 Cube 小空泡、MTE2 补装延迟与 `FIXP` 写回阻塞。
- 小分片周期级仿真显示：Cube 忙时远低于核总跨度，存在明显 `FIXP → Scalar → MTE2/MTE1 → Cube` 串行空档，或跨 head / 跨段切换时流水被强行排空重启。

## 核心组合手法（按典型收益顺序）

1. **跨逻辑边界展平主循环（`flat-head`）**
   - 当同一驻留块（如同一 KV tile）下依次服务多个 GQA Q head 时，不要按“外层 head 循环 + 内层 Q tile 循环”导致每次切 head 都做 epilogue 排空与 prologue 重启；将其展平为单一连续 tile 流，彻底消除 head 边界空泡。
2. **`L1` 多槽环形预取（如 `l1-ring4`）+ 提前释放槽位**
   - 为流式输入（如 `Q`/`dO`）在 `L1` 分配多槽环形缓冲，在最后一个消费该槽的 GEMM 读完后立刻释放 `L1` 槽供 `MTE2` 提前补装，拉大供数提前量。
3. **满配双缓冲 `L0A / L0B / L0C` 与提前装载 `L0B`**
   - 为 `L0A/L0B` 分配双物理槽，让下一组操作数的 `MTE1` 搬运与当前 `MMAD` 重叠；在等待 Vector 产出 `P`/`dS` 的间隙，先把已就绪的操作数从 `L1` 搬入 `L0B`。
   - 对独立累加器（如 `QK` 与 `dP`，或 `dQ` 双缓冲 `dq_c[2]`），在 `L0C` 容量允许时拆成独立双槽，解除 `FIXP` 写回对下一轮 `MMAD` 的阻塞。
4. **拆分就绪事件与尾段提前写回**
   - 将捆绑的就绪信号拆开（例如 `Q` 就绪即可启动 `QK`，不必等 `dO` 也到达；`dP` 在消费 `dO` 前再等 `dO`），压缩首尾等待。
   - 在最后一轮 `dV`/`dK` 累加结束后即发起 `FIXP` 写回，使其与后续 Cube 计算重叠。

## 关键风险与防坑约束

- **跨界同步漏加导致数据竞争**：为 Vector 元数据或 `L1` 槽引入跨轮预取时，必须在所有边界切换处（如 head 切换或尾段复用前）补齐 `V → MTE2` 等反向通知/等待；务必用包含**跨 head/多段边界**的小分片仿真复现校验。
- **单/双缓冲分支一致性**：当主路径启用双槽（`slots=2`）而大 `D` 通用回退分支退回单槽（`slots=1`）时，必须严查通用分支索引（禁止对单槽 buffer 使用 `prev` 等双槽下标导致越界）。
- **静态容量逐域复核**：每增加一个 `L1` 槽或 `L0C` 槽，须按 [writing-paradigms.md](../foundations/writing-paradigms.md) 重新核算全并发存活占用。

## 验证清单

- 先在本地小分片仿真中验证三核（AIC/AIV0/AIV1）完整 END、边界空泡归零且梯度/输出与父版逐字节一致。
- 再上真机核对同批配对时延、`msprof` Cube 占空比与小空泡缩减幅度，以及全 `dtype × head_dim` 分支回归。
