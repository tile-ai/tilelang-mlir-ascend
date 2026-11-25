---
id: falsified/local-ub-or-early-sync-micro-tweaks
layer: falsified
arch: [Ascend950]
kind: falsified-direction
status: active
verified_scope: tilelang-ascend
---

# 反例：仅减局部 UB 读写或仅拆首轮就绪同步，未动稳态关键路径时无收益甚至倒退

## 原本设想

两类在源码审查阶段极易产生的“局部直觉优化”：
1. **单遍 Softmax 改写**：把两遍 Softmax 强行压成一遍，从字节账上看减少了 UB 读写次数，直觉上认为一定更快；
2. **仅拆分首轮输入就绪信号（如 Decode `split_ready` 拆分 `K` 与 `RoPE` 就绪事件）**：让 `NoPE QK` 不必等 `RoPE` 搬完就提前发射，直觉上认为能缩短整核耗时。

## 为什么失败（定量物理原因）

1. **局部 UB 字节数下降 ≠ 整核跨度缩短**：在多组小分片全核流水仿真对照中，三种压缩 UB 读写的 Softmax 改写全部导致整核模型跨度**退化 +0.30% 至 +2.39%**。原因是在 CV 高度重叠的流水中，Vector 段的局部改动打破了原有指令发射/寄存器排布或引入了额外标量/依赖等待，而真正卡住整核的是 Cube/FIXPIPE 队首阻塞而非 UB 局部读写字节。
2. **提前首轮启动 ≠ 提升稳态吞吐**：拆分 `K/RoPE` 就绪同步确实让第一轮 `NoPE QK` 提早开始，但在长序列多页稳态迭代中，后续每页的迭代节拍仍受制于下游 `PV / FIXP / L0C` 复用环路，稳态每页周期基本不变。

## 何时可能重新成立（推翻本反例的条件）

- 当完整流水图（而非单段代码估算）证明 **Vector UB 带宽本身已是唯一关键路径**（例如无 mask 内部块取消无用写回、且全核仿真显示跨度确实下降），或**首轮启动延迟占总核时长很高**（极短序列）时，再针对性采纳。

## 替代方案

- 先按 [roofline-sim-workflow.md](../../foundations/roofline-sim-workflow.md) 审视**完整核内流水与原始指令配对**，优先消除真正的队首阻塞（见 [cube-fixpipe-hol-reordering.md](../cube-fixpipe-hol-reordering.md)），再用全核跨度（而非局部字节数或局部启动点）做消融裁决。
