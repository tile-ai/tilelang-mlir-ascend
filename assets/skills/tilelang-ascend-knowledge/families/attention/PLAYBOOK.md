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

## 1. 架构级选型速查

| 子类 | 推荐主循环组织 | 典型瓶颈演进路线 | 核心参考模式 |
| --- | --- | --- | --- |
| **Prefill Forward（含稀疏/Mask）** | 显式 CV 常驻主循环（如 `128×128` 档），单任务完成一个 Q tile 归约与写回；动态 `Sq/Sk` 与稀疏区间跳块 | 架构低效（~25% MFU）→ 重写 NPU 亲和主循环（~91% MFU，轻微 Vector-bound）→ 列向 Softmax 反转为严格 Cube-bound（>95% MFU） | [softmax-col-fp32.md](../../patterns/softmax-col-fp32.md)、[writing-paradigms.md](../../foundations/writing-paradigms.md) |
| **Prefill Backward** | 高 GQA 坚持 `KV-outer` 驻留 `K/V`，设备侧按有效 tile 前缀和均分给持久物理 block | 负载不均/Vector 慢 → `SimdVF` + 提前送 `P` → 显式 CV 双槽/4槽环 + `flat-head` + `dq_c[2]` → 自适应 L2 分段（长序列稳定 80%+ MFU） | [cv-explicit-ring-pipeline.md](../../patterns/cv-explicit-ring-pipeline.md)、[simd-vf-256b-fusion.md](../../patterns/simd-vf-256b-fusion.md)、[l2-adaptive-sectioning.md](../../patterns/l2-adaptive-sectioning.md) |
| **GQA Prefill 变体（Sink / 滑窗 / Prefix / Shared-RoPE）** | Query-major 复用 + 滑窗 KV 块上下界拆分 + 双 `L1` Ring 零拷贝融合 Prefix/RoPE + 入口缓存不变量（如 `sink`） | 先修静态 causal 负载不均（-30%~47%）→ 消除 Cube/FIXPIPE 队首阻塞（重叠 64%→79%）→ 无 mask 内部块免 UB score 写回 → 滑窗块拆分（-44%） | [cube-fixpipe-hol-reordering.md](../../patterns/cube-fixpipe-hol-reordering.md)、[attention-prefix-window-fusion.md](../../patterns/attention-prefix-window-fusion.md) |
| **FP8 MLA / GQA Decode** | Split-KV + 单次 MIX 全核屏障融合（`grid` 严格等于并发物理核数，`B × splits` 整除核数）；非 2 的幂 head 精确打包（如 `Hq24→M48` 不补 32） | 对齐物理核数去尾波 → 单核融合屏障去 launch 间隙（-36%）→ 精确 head 打包 → `N128 PV` + 双 `L0C/UB` + 交错 Softmax/输出更新 + UnitFlag + 下一页 KV 预取 + 两行展开（可达官方 FIA 2×+ 提速） | [decode-split-kv-fused-barrier.md](../../patterns/decode-split-kv-fused-barrier.md) |

## 2. 常见陷阱与全矩阵泛化要点

1. **Decode 全核屏障的物理核数死锁红线**：
   - 凡是内部使用 `SyncAll` 全核屏障的融合 Decode，**严禁在物理核数更少的卡上发射更大 `grid`**（例如 28 核卡跑 `grid32` 会直接死锁），且不同核数（28 核 vs 32 核）必须各自独立验收，不得互相外推。
2. **不要忽视全 `dtype × head_dim` 与动态/尾块泛化**：
   - 主场景（如 `BF16/D128`）打满后，避免将其余 `BF16/FP16`（`D32..D256`）全部丢进极慢的保守小分块；按维度分级设计统一快路径（如 `D<=128` 用 `64×128`、`D=160/192/256` 用 `64×64`），同时用生成 ASC 的 **SHA-256 逐字节一致** 锁定已达标主分支。
   - Scaled FP8 / Sink / RoPE 等融合变体须检查：RoPE 表容量上限、空形状输入、非默认 Stream、输入保护以及事件号独占（防 `507014` 同步超时）。
3. **明确已证伪方向，不再重复试错**：
   - 反向禁用 2D Zig-Zag 调度：[2d-zigzag-atomic-contention.md](../../patterns/falsified/2d-zigzag-atomic-contention.md)
   - 高 GQA 比反向禁用 Q-outer 遍历：[q-outer-gqa-traffic-blowup.md](../../patterns/falsified/q-outer-gqa-traffic-blowup.md)
   - Decode 盲目合并多 Query（如 `M96`）复用 KV 却使 split/workspace 翻倍：[decode-m96-multi-query-kv-reuse.md](../../patterns/falsified/decode-m96-multi-query-kv-reuse.md)
   - 仅凭局部 UB 字节减少或首轮 `split_ready` 做微调而未动稳态关键路径：[local-ub-or-early-sync-micro-tweaks.md](../../patterns/falsified/local-ub-or-early-sync-micro-tweaks.md)
4. **稀疏 / 动态 Mask 专项校验**：
   - 必须覆盖全空/全屏蔽行、尾块、非对齐 `Sq/Sk`、原位改写 mask 的缓存失效校验与 `NaN` 哨兵预填（见 [eval-invariants.md](../../foundations/eval-invariants.md)）。
