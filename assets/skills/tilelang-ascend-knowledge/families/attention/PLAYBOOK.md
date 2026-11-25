---
id: family/attention
layer: family
arch: [Ascend950]
kind: playbook
status: active
verified_scope: tilelang-ascend
---

# Attention 算子族 Playbook

覆盖 Prefill Forward、Prefill Backward、稀疏/滑窗/Sink/Prefix 融合变体以及 FP8 MLA / GQA Decode。

## 1. 高起点黄金骨架（不要从零重走低 MFU 弯路）

编写新的 Attention 算子或衍生变体（Causal、滑窗、稀疏 Mask、Bias、RoPE 融合等）时，**直接从下列自包含高起点骨架起步**，在对应 `# HOOK:` 处叠加业务语义，再按实测 `msprof` 症状按需叠加第 2 层模式卡：

| 子类 | 仓内可运行高起点骨架 | 骨架已具备能力与规格 | 实测遇瓶颈后按需叠加的模式卡 |
| --- | --- | --- | --- |
| **Prefill Forward（Dense / 衍生起点）** | `example/attention/prefill_fwd_dense.py` | 单二进制真动态 `seq_q`/`seq_k`、`BF16/FP16 D=128`、BSHD + GQA、`S_trans` 列向 FP32 256B `SimdVF` Softmax + 硬件 N-split `FIXPIPE` + `L1→L0A` 转置 PV + KV-head-major 持久调度（实卡验证 ~95%–96% 频修 MFU、`aic_mac ~0.99`，预留 mask/bias/epilogue `# HOOK:`） | 遇 `aiv_vec` 饱和：[softmax-col-fp32.md](../../patterns/softmax-col-fp32.md)；需滑窗/Prefix/Sink：[attention-prefix-window-fusion.md](../../patterns/attention-prefix-window-fusion.md)；需调流水队首阻塞：[cube-fixpipe-hol-reordering.md](../../patterns/cube-fixpipe-hol-reordering.md) |
| **Prefill Backward（Dense / 衍生起点）** | `example/attention/prefill_bwd_dense.py` | 单二进制真动态 `seq_q`/`seq_k`、`BF16/FP16 D=128`、BHSD + GQA、`KV-outer` + `flat-head` + `l1-ring4` + **`128×128` 常驻 `dK/dV` + 2 槽 `tmp_c[2]` 复用 + `dQ` 前置 + L0 奇偶交替** + 全 256B `SimdVF` 融合 `P/dS` + 自适应 L2 `G_sec` 分段（实卡验证 **97.9%–99.3% MFU**、`aic_mac 0.996–0.997`、`aic_fixpipe 0.632–0.654`，预留 `QRange`/mask/epilogue `# HOOK:`） | L0C 双槽复用与 `dQ` 前置：[l0c-pingpong-dq-first-128x128-bwd.md](../../patterns/l0c-pingpong-dq-first-128x128-bwd.md)；调核内流水空泡：[cv-explicit-ring-pipeline.md](../../patterns/cv-explicit-ring-pipeline.md)；调向量占空比：[simd-vf-256b-fusion.md](../../patterns/simd-vf-256b-fusion.md)；调长序列 L2 命中：[l2-adaptive-sectioning.md](../../patterns/l2-adaptive-sectioning.md) |
| **GQA Prefill 变体（Sink / 滑窗 / Prefix / Shared-RoPE）** | 基于上述 Prefill Fwd 骨架按 `# HOOK:` 扩展 | Query-major 复用 + 滑窗 KV 块上下界拆分 + 双 `L1` Ring 零拷贝融合 Prefix/RoPE + 入口缓存不变量（如 `sink`） | [cube-fixpipe-hol-reordering.md](../../patterns/cube-fixpipe-hol-reordering.md)、[attention-prefix-window-fusion.md](../../patterns/attention-prefix-window-fusion.md) |
| **FP8 MLA / GQA Decode** | 待抽自包含独立骨架（当前按模式卡组织） | Split-KV + 单次 MIX 全核屏障融合（`grid` 严对并发物理核数，`B × splits` 整除核数）；非 2 的幂 head 精确打包（如 `Hq24→M48` 不补 32） | [decode-split-kv-fused-barrier.md](../../patterns/decode-split-kv-fused-barrier.md) |

> 上游 TileLang 仓内的 `examples/ascend/flash_attention/core.py` 展示了 `T.Stage` + AutoSchedule 写法，可与仓内 `example/attention/prefill_fwd_dense.py`（显式 intrinsic + `S_trans` 路径）对照阅读，用于按 [writing-paradigms.md](../../foundations/writing-paradigms.md) 评估范式选型。

## 2. 常见陷阱与全矩阵泛化要点

1. **Decode 全核屏障的物理核数死锁红线**：
   - 凡是内部使用 `SyncAll` 全核屏障的融合 Decode，**严禁在物理核数更少的卡上发射更大 `grid`**（例如 28 核卡跑 `grid32` 会直接死锁），且不同核数（28 核 vs 32 核）必须各自独立验收，不得互相外推。
2. **不要忽视全 `dtype × head_dim` 与动态/尾块泛化**：
   - 主场景（如 `BF16/D128`）打满后，避免将其余 `BF16/FP16`（`D32..D256`）全部丢进极慢的保守小分块；按维度分级设计统一快路径（如 `D<=128` 用 `64×128`、`D=160/192/256` 用 `64×64`），同时用生成 ASC 的 **SHA-256 逐字节一致** 锁定已达标主分支。
   - Scaled FP8 / Sink / RoPE 等融合变体须检查：RoPE 表容量上限、空形状输入、非默认 Stream、输入保护以及事件号独占（防 `507014` 同步超时）。
3. **明确已证伪方向，不再重复试错**：
   - 反向禁用 2D Zig-Zag 调度：[2d-zigzag-atomic-contention.md](../../patterns/falsified/2d-zigzag-atomic-contention.md)
   - 高 GQA 比反向禁用 Q-outer 遍历：[q-outer-gqa-traffic-blowup.md](../../patterns/falsified/q-outer-gqa-traffic-blowup.md)
   - 反向禁止为扩 `128×128` 而放弃 `dK/dV` L0C 常驻、改为每 tile 原子写回 GM：[non-resident-dkdv-atomic-writeback.md](../../patterns/falsified/non-resident-dkdv-atomic-writeback.md)（正确做法是用 2 槽 `tmp_c[2]` 复用保持 `dK/dV` 常驻）
   - Decode 盲目合并多 Query（如 `M96`）复用 KV 却使 split/workspace 翻倍：[decode-m96-multi-query-kv-reuse.md](../../patterns/falsified/decode-m96-multi-query-kv-reuse.md)
   - 仅凭局部 UB 字节减少或首轮 `split_ready` 做微调而未动稳态关键路径：[local-ub-or-early-sync-micro-tweaks.md](../../patterns/falsified/local-ub-or-early-sync-micro-tweaks.md)
4. **稀疏 / 动态 Mask 专项校验**：
   - 必须覆盖全空/全屏蔽行、尾块、非对齐 `Sq/Sk`、原位改写 mask 的缓存失效校验与 `NaN` 哨兵预填（见 [eval-invariants.md](../../foundations/eval-invariants.md)）。
