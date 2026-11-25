---
id: pattern/cube-fixpipe-hol-reordering
layer: pattern
arch: [Ascend950]
kind: hw-pattern
status: active
verified_scope: tilelang-ascend
---

# 消除 Cube/FIXPIPE 队首阻塞与事件号独占防死锁

## 适用症状与触发条件

- 在本地仿真或 `msprof` 流水图中，用原始指令 ID 配对核查发现：**`QK` 的 `MMAD` 早已完成，但其结果 `score FIX` 却被延迟数百至数千周期才发射**；
- 原因是 `FIX` 或 `MTE2` 队列被上一轮尚未就绪的 `PV` 等待卡住（队首阻塞，Head-of-Line Blocking），导致 Vector 侧迟迟拿不到新 score，整条 CV 流水被迫拉长。
- 或者在大长度/多阶段特化中，出现设备同步超时（如 `507014` 错误码），排查看似无逻辑死循环但硬件事件卡死。

## 核心物理机理

1. **前置 `score FIX` 发射，解除 `PV` 等待的队首阻塞**
   - 当本轮 `QK` 已经算完但上一轮 `PV` 仍在等待 Vector 产出的 `P` 时，若按顺序把 `score FIX` 排在旧 `PV` 之后，就会造成无谓的队首阻塞，形成“Cube 等 Vector、Vector 等 Cube `score FIX`”的假性串行。
   - 重排发射顺序，让当前 `QK` 完成后**立即发射 `score FIX`**，同时**优先保证旧 `V` 的 `MTE2` 供数**，并严格维护 buffer credit 握手（实战中可将 `QK → score FIX` 发射间隔从 >2,200 周期压缩至个位数周期，Cube/FIXPIPE 区间重叠率由 ~64% 提升至 ~79%）。
2. **排查流水应看“整体队首阻塞”，不要先抠单个 Vector 小段**
   - 当 Cube/FIXPIPE 存在上千周期的队首阻塞时，微调 Softmax 内部指令往往毫无收益；必须先通过完整流水图与原始指令配对识别主阻塞链。
3. **屏障与 `FIX` 流水事件号严格独占**
   - 在显式事件流水中，若把同一组事件 ID 复用于不同阶段的屏障或 `FIX` 握手，在长序列/特定调度时序下可能产生事件错配导致硬件同步超时（`507014`）；须依据编译后的实际二进制/PC 核对事件分配，为关键握手保留独占事件槽。

## 不适用边界

- 提前发射 `score FIX` 需要目标 `L0C` 与下游 `UB` 接收槽位具备独立的双槽/信用保护；若 `score` 接收缓冲仍为单槽且被旧消费占用，强行提前只会触发数据竞争或更早的同步等待。

## 验证清单

- 用小分片仿真核对原始指令配对下的 `QK → score FIX` 间隔变化与三核完整 `END`。
- 在大长度（如 8K/16K）与全 grid 融合配置上复测无同步超时，同时检查 `msprof` 中 Cube 与 FIXPIPE 的重叠占比是否显著提升。
