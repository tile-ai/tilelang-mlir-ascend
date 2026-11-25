---
id: pattern/attention-prefix-window-fusion
layer: pattern
arch: [Ascend950]
kind: hw-pattern
status: active
verified_scope: tilelang-ascend
---

# Attention 变体融合：双 L1 Ring 零拷贝、滑窗边界拆分与入口不变量缓存

## 适用症状与触发条件

当基础 Dense Attention 扩展到真实推理/训练变体（Prefix Caching、Shared RoPE、Sliding Window、Attention Sinks、Causal 分块）时，最常见的性能杀手是**在主循环外做了多余 GM `cat` 拼接，或在主循环内对每个 tile 做重复 mask / 重复小字节搬运**。

## 核心可组合模式

### 1. Prefix + Current 与 K-NoPE + Shared-RoPE 双重零拷贝融合
- **痛点**：传统做法在 kernel 外先做 `cat([k_nope, k_rope], dim=-1)` 再做 `cat([prefix_k, current_k])`，产生大量额外 GM 读写与显存占用。
- **解法**：在 kernel 内部使用**两个独立 `L1` ring**（例如 128 列 `k_nope` + 64 列全 head 共享的 `[P, 1, 64]` `k_rope`，总 `L1` 占用与原 192 列单 ring 相同），分别处理尾行补零，并在 `QK` 阶段按 `128 + 64` 两段乘加直接累加到同一个 `L0C`，彻底消除前缀拼接与 RoPE expand 的 GM 张量（独立端到端 benchmark 可带来约 10% 净收益）。

### 2. Sliding Window 按 KV 块精确拆分上下界
- **痛点**：滑窗（如 `window=128`）若在整条 KV 序列上走通用逐元素 mask 判定，或在边界 tile 内保留大量冗余分支，耗时远高于有效工作量。
- **解法**：在块循环边界直接按每个 Q tile 对应的有效滑窗区间拆分 **下界部分块 / 内部全可见块 / 上界部分块**，跳过窗口外全部 KV 块，并仅对上下界少数块执行 mask（实测 8K 滑窗整核耗时可骤降 **~44%**，追平官方库）。

### 3. 无 Mask 内部块免除中间 Score 的 UB 写回
- **痛点**：对完全可见的内部 KV 块仍沿用带 mask 分支，把缩放后的 score 先写回 UB 再二次读出求 max，白白打满 UB 写带宽。
- **解法**：将无 mask 内部块独立特化，在单个 SIMD VF 内完成共同缩放与行 max 归约，消除缩放后 score 的 UB 写回（实测可将 Vector UB 写带宽压降 ~26%）。

### 4. 逐 Head 不变量（如 Attention Sink）外提到 AIV 入口缓存
- **痛点**：每个 Q tile 尾部都发一次 4B 同步 `MTE2` 读 `sink` 标量，在全核造成大量碎小访存与等待。
- **解法**：在每个 AIV 核入口一次性缓存本核负责的 1–2 个 head 的 `sink` 标量到片上，全核 `MTE2` 指令数可从几十次降至个位数，端到端稳定降时。

### 5. Causal 静态任务顺序均衡
- 含右下 Causal mask 的 prefill 若按线性顺序分配 Q tile，前部核空转、后部核过载；调整因果任务静态配对顺序（首尾配对或动态负载均衡）可一次性砍掉 30%–47% 的长尾延迟。

## 不适用边界

- 双 L1 ring 拆分 `128 + 64` 会增加搬运指令条数，仅当外部 `cat` / `expand` 的 GM 流量开销大于新增搬运发射成本时有净收益，须同设备对比“外部 `cat` + 旧核”与“融合新核”的完整范围耗时。

## 验证清单

- 覆盖零 prefix、页尾、仅 RoPE / 仅 NoPE 边界、多请求变长 TND 隔离（确认请求间零串扰）、双 head 缓存槽区分（验证多 head 共用同一 AIV 时不会串用同一 `sink` 槽）。
- 多请求批处理验证线性扩展性：同核数下测 1 / 2 / 3 个独立请求耗时，检查 `T2 ≈ 2×T1`、`T3 ≈ 3×T1`，排除调度非线性退化。
