---
id: falsified/2d-zigzag-atomic-contention
layer: falsified
arch: [Ascend950]
kind: falsified-direction
status: active
verified_scope: tilelang-ascend
---

# 反例：2D Zig-Zag 调度导致共享累加器原子写冲突暴增

## 原本设想

在带不规则 mask 或负载均衡需求的反向 Attention 中，直觉上认为采用 **2D Zig-Zag** 网格遍历（交错分配不同 KV/Q 区间给各物理核）可以同时兼顾核间负载均衡与缓存局部性。

## 为什么失败（定量物理原因）

- 在 KV-outer 反向实现中，多个并行的 KV 任务需要对同一份 `dQ` tile 执行 **FP32 原子累加写回**。
- 2D Zig-Zag 会让大量物理核在同一时间窗口内恰好撞上相同的 `Q` 区间，导致同步 `dQ` 原子写冲突次数数量级暴增（离线 32 核并发模拟测得中等序列下冲突放大 **~11×**）。
- 原子写序列化开销远大于 Zig-Zag 带来的微弱均衡改善。

## 何时可能重新成立（推翻本反例的条件）

- 算法彻底消除了跨核对同一输出块的原子累加（例如每个输出块由单一物理核独占写回，或改用两阶段无冲突归约且额外流量可被隐藏）；
- 或者目标硬件在 L2/GM 原子累加吞吐上发生架构级变化。

## 替代方案

- 采用 **按有效 tile 前缀和均分 + KV-head 优先排布 + 自适应 L2 分段**（见 [l2-adaptive-sectioning.md](../l2-adaptive-sectioning.md)），在不放大跨核原子写冲突的前提下同时兼顾负载均衡与 L2 复用。
