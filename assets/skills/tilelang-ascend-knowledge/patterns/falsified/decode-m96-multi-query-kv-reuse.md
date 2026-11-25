---
id: falsified/decode-m96-multi-query-kv-reuse
layer: falsified
arch: [Ascend950]
kind: falsified-direction
status: active
verified_scope: tilelang-ascend
---

# 反例：Decode 盲目扩大 M 维合并多 Query（如 M96）复用 KV 反而变慢

## 原本设想

在多 Query / MTP 投机 Decode（例如 `Sq=4, Hq=24`）中，直觉上把 `M` 从 `48`（每次 2 个 query）翻倍到 `96`（4 个 query 共用同一份 `KV/RoPE` 搬运），可将逻辑 `GM → L1` 的 KV 有效载荷减半（−50%），似乎能直接换来接近 2× 的带宽加速。

## 为什么失败（定量物理原因）

在匹配输入、编译选项、核数与频率的完整单 kernel 仿真对照中，`M96` 相比 `M48` **三组对照全部回退（+5.94%、+56.73%、+57.55%）**：

1. **逻辑流量减半 ≠ 实测时延减半**：当核内流水受限于计算、片上缓冲重排与每行因果 mask 的标量控制依赖时，减少 GM 载荷字节并不能直接转化为整核时间缩短。
2. **Split 与 Workspace 归并代价翻倍**：为填满物理核数，四 query 合并后 `split` 数往往被迫翻倍，FP32 workspace 占用与最终 `merge` 归并工作量同步翻倍（3.2 MB → 6.5 MB），在完整单 kernel 计时中抵消甚至反超 KV 复用收益。
3. **片上压力陡增挤占流水深度**：`M96` 使每核 `UB / L1 / L0C` 占用逼近上限，压缩了双缓冲与隐藏延迟的空间。

## 何时可能重新成立（推翻本反例的条件）

- 真实超长序列下 HBM 带宽成为绝对唯一瓶颈（且 workspace 归并占比可忽略），同时通过外提逐行因果 mask 消除了标量控制流依赖，并且**在匹配核数的实卡上**跑出端到端正收益。

## 替代方案

- 保持适中 `M` 打包（如 `M48` 匹配双 AIV 均衡切分），通过 [decode-split-kv-fused-barrier.md](../decode-split-kv-fused-barrier.md) 中的 `N128 PV`、双 `L0C/UB`、下一页 KV 预取与两行输出展开压榨核内 DFU。
