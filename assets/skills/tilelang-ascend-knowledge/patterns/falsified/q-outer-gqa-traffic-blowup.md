---
id: falsified/q-outer-gqa-traffic-blowup
layer: falsified
arch: [Ascend950]
kind: falsified-direction
status: active
verified_scope: tilelang-ascend
---

# 反例：高 GQA 比下反向改用 Q-outer 遍历导致访存与原子写倍增

## 原本设想

Attention Backward 中 `KV-outer` 需要对 `dQ` 做原子累加；直觉上如果改为 **`Q-outer`** 遍历，可以让 `dQ` 常驻片上无冲突写回，似乎能简化同步并减少原子操作。

## 为什么失败（定量物理原因）

- 在高 GQA 组比场景（例如 `Hq:Hkv = 8:1`）下，同一个 KV head 对应 8 个 Q head：
  - `KV-outer` 驻留一份 `K/V` 即可连续服务同组 8 个 Q head，GM 读流量最小，只需对 `dQ` 原子写；
  - 改为 `Q-outer` 后，每个 Q head 都要重复从 GM 读取同一份 `K/V`，且 8 个 Q head 都需要对同一份 `dK` 和 `dV` 执行原子写。
- 在典型 `64×128` 分块与 `GQA=8` 下，流量模型定量显示 `Q-outer` 会使 **GM 读流量放大 ~3.0×、原子写流量放大 ~4.0×**，在离线建模阶段即可直接证伪淘汰，无需浪费真机机时。

## 何时可能重新成立（推翻本反例的条件）

- **标准 MHA（`Hq == Hkv`，GQA=1）** 且序列极短/`Sq << Sk`、使得 `K/V` 重复读代价远低于 `dQ` 写回代价的特定场景（需重新做流量核算，不能无条件外推）。

## 替代方案

- 在 GQA 场景坚持 `KV-outer` 驻留 `K/V`，配合 [cv-explicit-ring-pipeline.md](../cv-explicit-ring-pipeline.md) 的 `flat-head` 展平与双缓冲 `dq_c[2]` 来完全隐藏 `dQ` 写回开销。
