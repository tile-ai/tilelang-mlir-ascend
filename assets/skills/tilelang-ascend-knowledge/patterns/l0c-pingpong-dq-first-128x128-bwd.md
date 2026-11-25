---
id: pattern/l0c-pingpong-dq-first-128x128-bwd
layer: pattern
arch: [Ascend950]
kind: hw-pattern
status: active
verified_scope: tilelang-ascend
---

# L0C 双槽复用 + `dQ` 前置 + L0 奇偶交替（`128×128` 常驻 `dK/dV` 反向调度）

## 适用症状与触发条件

- 多 GEMM 融合反向算子（如 Attention Prefill Backward）因常驻累加器占去一半 L0C（`dk_c 64 KiB + dv_c 64 KiB = 128 KiB`），被迫把 `block_q` 压到 `64`（半宽 `64×128×128` 甚至拆分的 `64×64×128` MMAD）。
- 实卡 `msprof` 显示 `aic_mac_ratio` 已达 ~`0.94`，但频率修正 MFU 仍停在 ~`82%`（`MFU / aic_mac ≈ 0.87`，明显低于满尺寸 `128×128×128` GEMM 的 ~`0.97`），且 `aiv_vec_ratio`（~`0.40`）和 `aic_fixpipe_ratio`（~`0.57`）仍有充足余量。
- 实测或预算显示 `L1`（≤`512 KiB`）与 `UB`（≤`256 KiB`）尚能容纳 `128×128`（例如 `480 KiB` L1、`~193.5 KiB` UB），真正卡住扩容的只是 L0C 账本。

## 核心物理机理

1. **按生命周期复用 L0C，无需为每个非驻留 GEMM 单独留槽**：
   - 传统写法为 `QK`、`dP`、`dQ[2]` 各自独立分配 L0C，扩到 `128×128` 时总计 `dk_c` (64K) + `dv_c` (64K) + `qk_c` (64K) + `dp_c` (64K) + `dq_c[2]` (128K) = `384 KiB > 256 KiB`。
   - 实际上 `QK`、`dP`、`dQ` 全部经由同一条顺序 `FIXPIPE` 排空，任意时刻最多只有两个非驻留输出处于活跃（一个写、一个排空），只需 **2 槽 ping-pong `tmp_c[2]`（`128 KiB`）** 即可复用给三者：`dk_c` (64K) + `dv_c` (64K) + `tmp_c[2]` (128K) = `256 KiB` 刚好装满且零溢出。
2. **`dQ` 前置隐藏 `FIXPIPE -> GM` 写回**：
   - 将每步的 5 个满尺寸 `128×128×128` MMAD 排布为 `QK(i) → dP(i) → dQ(i-1) → dV(i-1) → dK(i-1)`。
   - `dQ(i-1)` 完成后立即发起 `FIXPIPE` 原子写回，随后连续执行的 **`dV(i-1)`、`dK(i-1)`、`QK(i+1)` 三个满尺寸 MMAD** 都只写 `dv_c`、`dk_c` 与对侧槽 `tmp_c[1-b0]`，为 `dQ` 写回提供整整 **3 个满尺寸 MMAD（~`0.98 us`）** 的隐藏窗口，完全覆盖实卡单次 `64 KiB` `dQ` 原子写回延迟（~`0.70 us`）。
   - 同时 `dQ` 写回事务数减半（单次连续 `64 KiB` 替代两次 `32 KiB`），且 `dK`/`dV` 保持 L0C 常驻（整块仅尾步写一次），无任何额外 `FIXPIPE` 流量。
3. **`b0 = step % 2` 奇偶翻转消除奇数 GEMM 跨步气泡**：
   - 5 个 GEMM 为奇数；若固定 `L0` 槽号 `0, 1, 0, 1, 0`，尾部的 `dK` 与下步 `QK` 撞槽引入 `MTE1 -> Cube` 气泡。
   - 用 `b0 = step % 2, b1 = 1 - b0` 同步索引 `a_l0/b_l0` 与 `tmp_c`（本步 `b0 → b1 → b0 → b1 → b0`，下步翻转为 `b1 → b0 → ...`），使包括跨迭代边界在内的全部相邻 MMAD 严格交替 `L0A/L0B`，实现零 `MTE1` 气泡。

## 实现要点与约束

- 仓内已验证实现见 `example/attention/prefill_bwd_dense.py`（Ascend 950 实卡 8K–32K BF16 与 8K FP16 实测 `97.9%–99.3%` MFU，`aic_mac = 0.996–0.997`，`aic_fixpipe = 0.632–0.654`，较 `64×128` 基线提升 `1.16×–1.21×` / `+16.3~+17.0 pp` MFU）。
- 同步令牌平衡：
  - `dQ` 前置后须在 `dQ` 前等待 `ds_l1[prev]`（`PIPE_MTE1` 标志 `6 + prev`），并在 `dV` 前消费 `p_l1[prev]`（标志 `4 + prev`）；因 Vector 侧按序先发 `P` 后发 `dS`，无死锁风险。
  - `q_l1`/`do_l1` 与 `p_l1`/`ds_l1` 均在 `dK` 完成 `MTE1` 读入后统一释放（同在 `PIPE_MTE1` 保序）。
  - 须覆盖 `total_local == 1`（单 Q-block）边界，确认 `FIX_M 0/1`、`M_MTE1 0/1`、`MTE1_MTE2 0..3` 与跨核标志在序言/尾声严格 1:1 闭合。
- 对齐约束：`block_q = 128, block_k = 128` 要求 `seq_q % 128 == 0` 且 `seq_k % 128 == 0`（与 Prefill Forward `128×128` 对齐契约一致）。
- 仿真解读注意：部分 CAModel 版本在 `FIXP Dst:OUT` 与 `MTE2` 并发时对 GM 原子写回的建模比真机悲观 3–4×，可能导致 `tmp_c[2]` 在仿真里出现假性 `FIXP` 空洞；应同时核对各子单元（`CUBE`/`MTE1`/`MTE2`/`FIXP`/`RVECEX`）净 busy 下降及无 `MTE2` 尾步 II，并上实卡裁决（见 [roofline-sim-workflow.md](../foundations/roofline-sim-workflow.md)）。

## 不适用边界

- 当 `L1` 或 `UB` 无法容纳 `128×128`（例如 `head_dim > 128` 或带复杂大状态掩码导致 `L1 > 512 KiB` / `UB > 256 KiB`）时，不可强行扩容，需退回 `64×128` 或 `64×64`。
- 当实卡 `dQ` 写回在极端多核 L2 争抢下耗时持续超过 3 个满尺寸 MMAD 窗口时，需先配合 [l2-adaptive-sectioning.md](l2-adaptive-sectioning.md) 收敛 Q 侧工作集。

## 验证清单

- ASC 审计确认恰好 **5 个** `asc_mmad`（无 `dQ` 拆分）、`simt_vf == 0`、`simd_vf == 2`、动态 `seq_q`/`seq_k` 存在。
- 必测边界：`total_local == 1`（如 `Sq = 128`）、单二进制动态复用（`Sq = 256 → 1024` 同 SHA-256）、强制 `l2_budget_bytes = 1` 分段原子写回、矩形 `Sq != Sk`、`BF16` + `FP16` 与非默认 Stream。
- 真机 `msprof` 核对 `aic_mac_ratio` 逼近 `0.99+`、`aic_fixpipe_ratio` 处于 `~0.63–0.66` 且 `aic_mte1`/`aic_mte2`/`aic_scalar` 同步下降。
