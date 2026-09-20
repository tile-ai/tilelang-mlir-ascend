# 向量化轴与布局模式（优先级最高的检索项）

> 本文件是 pattern-library 主题文件之一（入口与预算见 [INDEX.md](INDEX.md)）。条目带 front-matter（id/kind/family/apis/status/origin_task/toolchain/repro——schema 见 INDEX.md §3）；`repro: repro-missing` 表示待回填最小复现代码（存量条目允许，新 D 类条目缺 repro 降级 Tier 1 入队）。

**诊断触发器**：msprof 显示算子热点段**标量执行占比 > 50%** 时，强制触发"换向量化轴/换布局"分析——不要在原轴上微调（实测案例：窗口累加 98.3% 标量源于 `j·sW`（sW≠1）跨步系数阻碍向量化，原轴微调收益为 0，换轴后 5.6x）。

---
id: PL-1.1-transpose-chain
kind: pattern
family: [pooling, window, layout]
apis: [T.transpose, T.copy]
dtype: [fp16, fp32, bf16]
device: 910B2C
status: verified
origin_task: AvgPool2dFwdOp-optimize-2026-08
toolchain: 2026-08 build（Ascend910B2C）
repro: repro-missing
---

### 1.1 核内融合转置链（布局重排的正确形态）✅ 已验证

- **模式**：I/O 保持原生契约（如 NCHW），核内用 **T.transpose 二轴交换链**重排到目标布局（如 NHWC）。3D 置换 = 两次二轴交换：
  ```
  (CH, Hi, Wi) -[1,0,2]-> (Hi, CH, Wi) -[0,2,1]-> (Hi, Wi, CH)   # 输入侧
  (BH, WO, CH) -[0,2,1]-> (BH, CH, WO) -[1,0,2]-> (CH, BH, WO)   # 输出侧
  ```
- **实测代价**：4 次融合转置合计 **~5.4µs**（远低于算子本体）——"转置很慢"是错误印象，勿因直觉弃用。
- **硬约束**：① `T.transpose` 的 permutation **仅支持二轴交换**（如 `[1,0,2]`/`[0,2,1]`），**不支持 3-cycle**（`[1,2,0]`）——3D 置换必须拆成两次交换链；② **reshape→transpose 链会 mis-compile（数据错乱）**——必须始终在自然形状 buffer 之间做 transpose；③ 转置须在**自然形状** shared/UB buffer 间进行。
- **佐证**：`testing/npuir/broken/test_slice_transpose_dev.py`；AvgPool2d perf_opt v3a（`examples/TileOPs/tileops/kernels/pool/avg_pool2d/avg_pool2d_kernel/perf_opt/`——任务工作区，未上库，provenance 允许失效；模式与实测数字已自包含于本条目）。

---
id: PL-1.2-caxis-accum
kind: pattern
family: [pooling, window, elementwise]
apis: [T.vadd, T.serial]
dtype: [fp16, fp32]
device: 910B2C
status: verified
origin_task: AvgPool2dFwdOp-optimize-2026-08
toolchain: 2026-08 build（Ascend910B2C）
repro: repro-missing
---

### 1.2 C 轴切片累加（channel-batched vector accumulation）✅ 已验证

- **模式**：`T.vadd(acc[i, j, :], in_f32[i*SH+ki, j*SW+kj, :], acc[i, j, :])`——整 C 轴切片作为向量操作数，**i/j 用 T.serial**（不是 Parallel），ki/kj serial 外层。
- **适用**：C 恒为向量宽度整数倍的 NCHW 类算子（C=64/96/128；96 用 CH=48/32/16 分块）。
- **实测收益**：AvgPool2d kernel 级 1.98–13.2x（vs 同代 H-collapse 版本）。

---
id: PL-1.3-host-permute
kind: pattern
family: [layout]
apis: []
dtype: [fp16, fp32, bf16]
device: 910B2C
status: verified
origin_task: AvgPool2dFwdOp-optimize-2026-08
toolchain: 2026-08 build（Ascend910B2C）
repro: repro-missing
---

### 1.3 host permute 路线 ❌ 通常净亏（已量化证伪）

- 两次 host permute（NCHW↔NHWC）实测 **106–147µs** > 多数中等算子本体耗时；仅当算子本体远大于此（如 >500µs）且无法核内重排时才值得复测。

---
id: PL-1.4-tiling-heuristic
kind: pattern
family: [pooling, window]
apis: []
dtype: [fp16, fp32]
device: 910B2C
status: verified
origin_task: AvgPool2dFwdOp-optimize-2026-08
toolchain: 2026-08 build（Ascend910B2C）
repro: repro-missing
---

### 1.4 tiling 启发式（C 轴向量化形态）✅ 已验证

- **BH=1 + 最宽 CH 最优**：CH64 21.6µs < CH32 26.7µs < BH2/CH16 44.8µs——更宽 C 向量 + 更少 block 胜过更高空间并行度。
- **UB 约束**：CH×空间 tile 驻留超 192KB 时收缩 CH（实测 vis-3x3 的 CH=64 需 252.8KB 被排除，CH=32 落地）；buffer 生命周期不重叠时可探索 aliasing 复用（预估再省 10–20%，未验证）。
- **回退分发**：C%16≠0 或 shape 越界时回退到非重排路径（工厂层静态判定，无运行时开销）——保持全 shape 兼容。

---
id: PL-1.5-quickref
kind: pattern
family: [elementwise, pooling, general]
apis: [T.vmul, T.ceildiv, T.Pipelined]
dtype: [fp16, fp32, bf16]
device: 910B2C
status: verified
origin_task: mixed（AvgPool2d 2026-08 / lerp_tensor 2026-09-07）
toolchain: 2026-08 build；lerp 行为 tilelang-mlir-dev dev root build 2026-09-07
repro: repro-missing
---

### 1.5 其他已验证模式速查

| 模式                  | 一句话                                                                                                                                                                                               | 实测参考                                                                                                           |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| 乘编译期常数倒数      | divisor 恒定时`T.vmul(acc, 1/d, acc)`，省逐元素除法                                                                                                                                                                                                                | AvgPool2d fast path                                                                                                |
| H-collapse            | 把 kH 折叠进向量管道，消除一层 serial 累加                                                                                                                                                                                                                           | AvgPool2d v2（对 sW 跨步问题的 W 轴解法）                                                                          |
| fp32 求和序匹配       | kernel 累加序与 golden（F.avg_pool2d 等）一致时解锁高精度快路径                                                                                                                                                                                                      | AvgPool2d fp32 13.2x                                                                                               |
| launch 开销主导判定   | 小 shape 与大 shape 耗时同量级 → 固定开销主导；拆 pad-only/padded-in 定位                                                                                                                                                                                           | bench_runner`--decompose`                                                                                        |
| host 层编译期常量折叠 | `@T.prim_func` 外的 `T.ceildiv(int,int)`/`min(IntImm,int)` 对 Python int 常量折叠为 IntImm（复现：`T.ceildiv(4097,4096)`→`2 <IntImm>`）——persistent 设计的 num_logical/num_kernels/num_local_tasks 可在工厂层直接求编译期常量，无需改写 Python 整数算术 | lerp_tensor 迁移 2026-09-07（REVIEW.md 附#29；origin_task: lerp_tensor-_make_lerp_tensor_kernel-20260907T010433Z） |

---
id: PL-1.15-broadcast-perf-tax
kind: pattern
family: [vector, layout]
apis: [T.vsub, T.vmul, T.vbrc]
dtype: [fp16, fp32]
device: 910B2C
status: verified
origin_task: ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z（Stage 4；2026-09-17 蒸馏 D2 溯源归位——原回写误标 20260917T0855Z，实际 task_id 以 .stage_state.json 为准）
toolchain: tilelang 0.1.2+1990aa9fe4 / CANN 8.5.0 / Ascend910B2C / 2026-09-17
repro: repro/PL-1.13-aiv-dup-subid-split.py
---

### 未文档化列广播形态「可编译可算对但有性能税」（vsub [M,N]−[1,bs] 实测 +12~16%；vsub dst=src2 alias 形态 +17~20%——同族第二实证）

- **形态一（列广播）**：`T.vsub` 的列广播 `[M,N] − [1,N]`（T.vsub.md §2.2.2 未列——文档仅同 shape/行广播 [M,N]−[M,1]/退化 [1,N]−[1,1]/标量四形）在当前工具链**可编译、L0 精度正确**，但实测 **w2 +12.2% / w3 +16.0%**（msprof op，vs `vbrc([1,N]→[M,N]) + 同 shape vsub` 的文档化两步形态）。
- **形态二（dst=src2 alias，2026-09-17 第二轮 R7 v11 第二实证）**：`T.vsub(a, b, b)`（diff 落到第二操作数原地址，元素级原地安全）同样可编译可算对，但实测同口径 **w2 +19.6% / w3 +17.0% / w4 +2.0%**——原语境为 vbrc hoist（把循环不变的 [M,1]→[M,N] 广播提出来需 diff 重定向到 alias 目标）。**AIV 接近关键路径的 workload（w2 瞬态/w3）受害最大，AIV 远离临界的稳态（w4）仅 +2%**——税的大小与该引擎的空闲余量负相关。
- **结论**：广播/alias 等「非典型形态」的「合法化」（编译通过 + 数值正确）不等于「可用」（无性能税）——证伪协议须含性能步；列因子在 vsub/vmul 上仍应走 vbrc 展开 + 同 shape 运算；循环不变量 hoist 若被迫引入 alias 形态，宁可保留循环内重算（与 Stage-2 review 的广播形态速查口径互补：形态合法性看文档 §2.2.2，可用性看实测）。
- 溯源：`examples/ssd_chunk_scan/_ssd_chunk_scan_fwd_kernel/perf_opt/`（形态一 opt_log R2 v4_p2 行 / perf_records round 2；形态二 opt_log R7 v11 行 / perf_records round 7 同口径对比 v9_recheck）。
