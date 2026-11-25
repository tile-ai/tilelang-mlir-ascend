---
id: pattern/softmax-col-fp32
layer: pattern
arch: [Ascend950]
kind: hw-pattern
status: active
verified_scope: tilelang-ascend
---

# `S_trans` 列向 FP32 Softmax 与硬件 N-split `FIXPIPE`

## 适用症状与触发条件

- 混合 Cube/Vector 算子（典型如 Attention Prefill Forward）在 `msprof` 中表现为 **`aiv_vec` 占空比接近饱和（如 >95%）而 `aic_mac` 被迫等待**，整体呈轻度或重度 Vector-bound。
- 常见诱因：传统行向 Softmax 在 Vector 侧需要大量跨 lane 归约、标量广播、反复转置或额外 UB 缓冲搬运，导致 Vector 执行时间超过同 tile 的 Cube `QK + PV` 耗时。

## 核心物理机理

1. **列向组织消除行内归约开销**：将 score 块以转置形态 `S_trans` 组织，使原本沿行方向的 max/sum/scale/exp 归约转为沿连续向量寄存器列向推进，充分发挥 SIMD 256B 宽向量吞吐并减少跨 lane shuffle。
2. **硬件 N-split `FIXPIPE` + `L1→L0A` 转置**：利用 Cube `FIXPIPE` 硬件拆分写回与 `asc_copy_l12l0a_transpose`（或等价硬件转置搬运）承担布局变换，把软件侧在 UB 上做的显式转置开销转移到专用数据通路，同时使用 `S.vmadd` 等融合乘加压缩向量指令数。
3. **反转瓶颈归属**：压缩 Vector 关键路径耗时至低于 Cube 窗口后，算子从 Vector-bound 彻底反转为严格 Cube-bound（典型实测可使 `aic_mac` 达 98%–99%、频修 MFU 突破 95%）。

## 实现要点与约束

- **精度保持**：Softmax 累加、差值、exp 与归一化状态（`m`、`l`、`O` 缩放）保持 FP32，仅在送入 `PV` 前按目标格式转换为 BF16/FP16。
- **尾块与全屏蔽行处理**：列向布局下仍须严格屏蔽无效列，处理首个全屏蔽 tile、中间空 tile 与整行全屏蔽时的 `-inf - (-inf)` 问题，防止产生 `NaN`。
- **多维度泛化检查**：检查 `D=32..256` 及 `BF16/FP16/FP32` 下的 UB/L1 物理占用；`FP32` 路径关闭非预期 HF32 以保证 IEEE 精度。

## 不适用边界

- 当 `head_dim` 很大（如 `D=192/256`）且算子已经转为 **`mte2` 访存受限（`mte2` 占空比 >95%）** 时，进一步压榨 Vector Softmax 指令数对端到端耗时收益有限，应转而优化 K/V 分块与访存复用。
- 若目标绑定版本的 TileLang 不支持对应转置搬运 intrinsic，需先查绑定版本文档核实替代通路，不硬抄未支持符号。

## 验证清单

- 核对生成 ASC 中是否存在多余 UB 软件转置或退化的 `SimtVF`。
- 核对 `msprof`：`aiv_vec` 占空比显著低于 `aic_mac`，且 `aic_mac` 成为主导瓶颈。
- 跑通含全屏蔽行、尾块、`NaN` 哨兵预填与非连续输入的完整精度矩阵。
