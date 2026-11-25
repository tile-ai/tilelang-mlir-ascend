---
id: family/gemm
layer: family
arch: [Ascend950]
kind: playbook
status: active
verified_scope: tilelang-ascend
---

# GEMM / BMM / 量化 GEMM 算子族 Playbook

覆盖标准 GEMM、Batched GEMM（BMM）以及混合精度/量化 GEMM（如 W4A8）。

## 1. 核心组织原则

1. **按物理核数规划持久任务与 L1/L0 分块**：
   - 常规 BMM/GEMM 可优先使用 TileLang 的 `T.Pipelined(..., num_stages=2)` 与规整 `128×128×128` 级分块，通过 `get_num_cube_cores()` 对齐物理 Cube 核数（仓内参考样例：`example/gemm/bmm.py`）。
   - 先用 `ascend-roofline` 核算当前 shape 是 Cube-bound 还是 GM/L2 带宽受限，再决定优化重点是 K 维流水深度、L2 swizzle 还是尾块处理。
2. **量化 GEMM（如 W4A8）：把“向量解包”与“纯 Cube 乘法”按硬件优势解耦**：
   - 仓内参考样例：`example/gemm/w4a8.py` 展示了典型的三段式组织思路——
     - **Pass 1（Vector 解包重排）**：将分组缩放的有符号 INT4 权重在 Vector 单元解包、乘以 group scale 并舍入到目标 FP8（`float8_e4m3fn`），避免在 Cube 主循环内穿插重型解包拖慢流水；
     - **Pass 2（原生 FP8 Cube GEMM）**：Cube 直接读取规整 FP8 权重与激活并做 FP32 累加，保持硬件满速 FP8 吞吐；
     - **Pass 3（Vector 后处理缩放）**：施加 token/channel scale。
   - **取舍边界**：多 Pass 会增加中间权重/激活的写回与重读流量；在权重可跨多批/多步复用或解包本身极重时收益显著。若后续面临小 batch 带宽瓶颈，需对照 Roofline 评估是否融合部分解包或 epilogue。

## 2. 常见陷阱与验证要点

- **精度与累加口径**：明确输出 dtype 与累加 dtype（FP32 累加 vs 低精度、FP32 场景是否关闭 HF32），量化算子须核对舍入模式与参考实现逐项一致。
- **非对齐尾块与小矩阵**：核对 `M/N/K` 不能被 tile 整除时的边界 mask、`NaN` 哨兵预填与零填充，防止越界写回污染相邻 batch。
- **动态/缓存键检查**：`@functools.lru_cache` 编译缓存键必须包含全部影响生成代码的 shape、dtype、转置与量化分组参数。
