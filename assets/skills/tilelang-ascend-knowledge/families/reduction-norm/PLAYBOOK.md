---
id: family/reduction-norm
layer: family
arch: [Ascend950]
kind: playbook
status: draft
verified_scope: tilelang-ascend
---

# Reduction & Normalization 算子族 Playbook（骨架）

覆盖独立 Softmax、RMSNorm、LayerNorm、TopK/归约类向量算子。本文件为待实战扩充的初始骨架；交付首个代表性算子后按 `SKILL.md` 的 Distill 流程补充实测模式。

## 1. 起点方法论

1. **以带宽利用率（BWU）为核心指标**：绝大多数独立归约/归一化算子属于强 Memory-bound，开局先用 `ascend-roofline` 算出理论最小 GM 读写字节数（理想状态为单次读入、单次写出，避免两趟遍历 GM）。
2. **优先满宽 256B `SimdVF`**：参考 [simd-vf-256b-fusion.md](../../patterns/simd-vf-256b-fusion.md)，在 UB 容得下整行时做单 pass 融合（读入 → FP32 归约 → 归一化/缩放 → cast 写出）；超长行再做分块在线归约（Online Softmax / Welford）。
3. **数值稳定性**：归约累加（sum / sum-of-squares / max）全程保持 FP32，注意极大/极小值防溢出与全零方差除零保护。

## 2. 待沉淀条目（开放占位）

- [ ] 超长隐藏维（超出单核 UB）的两级归约与分块策略
- [ ] RMSNorm + Residual Add + Quantize 融合 epilogue 的最优 UB 布局
- [ ] 非对齐尾维的满宽 SIMD 掩码归约样例
