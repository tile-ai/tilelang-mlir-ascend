---
id: family/conv
layer: family
arch: [Ascend950]
kind: playbook
status: active
verified_scope: tilelang-ascend
---

# Convolution / Unfold 算子族 Playbook

覆盖分组 1D 卷积（NCL）、Im2Col / Tiled DMA Unfold 及跨步/空洞卷积变体。

## 1. 核心组织原则

1. **隐式/显式打包转为规整 Cube GEMM（参考 `example/convolution/conv1d.py`）**：
   - 按 `K = Ci_g * Kw` 的实际大小自适应选择归约块 `kb`（例如小 `K` 用 `16`、中等用 `128`、满足对齐的大 `K` 用 `256`），将 `K / Co / L_out` 按硬件对齐粒度（16 / 64 / 128）规整填充；
   - 保持 Cube 内 FP32 累加，在最终写回前仅执行一次 cast，减少中间格式转换开销与精度损失。
2. **跨步/多维 Unfold 采用“分块 DMA + 向量抽样”（参考 `example/convolution/unfold.py`）**：
   - 面对 `stride > 1` 或多维滑窗，不要逐元素发起细碎标量搬运；先按行/列计算规整 `raw_width` 与 `raw_pitch`（16 对齐）做连续大块 DMA 读入 UB，再用 `simd` 向量指令在 UB 内做 stride decimation（抽样）并按 128 对齐写出。
   - 严格按 UB 容量上限反推每次处理的 `rows`/`cols` 上界，避免大 stride 或大通道时触发片上存储溢出（见 [writing-paradigms.md](../../foundations/writing-paradigms.md)）。

## 2. 常见陷阱与验证要点

- **分组与空洞边界**：覆盖 `groups > 1`、非零 `padding`、`stride > 1`、`dilation > 1` 与非对齐输出长度，确认越界 padding 区域严格补零且不越界读取输入。
- **多阶段融合边界**：若卷积拆为 `unfold + GEMM` 两个 kernel，性能计时必须记录**完整算子端到端耗时**，不得只报 GEMM 阶段时长。
- **通用评测铁律**：执行 [eval-invariants.md](../../foundations/eval-invariants.md) 中的 `NaN` 哨兵预填与独立 FP32 参考核对。
