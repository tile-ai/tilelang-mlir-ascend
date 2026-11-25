---
id: falsified/non-resident-dkdv-atomic-writeback
layer: falsified
arch: [Ascend950]
kind: falsified-direction
status: active
verified_scope: tilelang-ascend
---

# 反例：释放 L0C 常驻 `dK/dV`，每 Q-tile 用 `FIXPIPE` 原子写回 GM 以换取更大分块

## 原本设想

在 `64×128` Attention 反向中，`dk_c`（`64 KiB`）与 `dv_c`（`64 KiB`）常驻 L0C 占去了一半容量，看似阻碍 `block_q` 从 `64` 扩到 `128`。由于 `64×128` 基线的实卡 `aic_fixpipe_ratio` 仅约 `0.57`（看似尚有余量），直觉上可能会设想：去掉 `dk_c`/`dv_c` 的 L0C 常驻、改为每个 Q-tile 算完即通过 `FIXPIPE` FP32 atomic-add 累加到 GM，从而腾出 `128 KiB` L0C 把全部 5 个 GEMM 升级为满尺寸 `128×128×128`。

## 为什么失败（定量物理原因）

1. **`FIXPIPE -> GM` 原子写回流量膨胀 3×，直接反超 Cube 成为绝对瓶颈**：
   - 常驻方案每 `128×128` 等效 Q-tile 只需 1 次 `64 KiB` 的 `dQ` GM 原子写回（`dK`/`dV` 整 KV-block 仅写 1 次，摊销近零）；非常驻方案每步必须串行写回 `dQ + dK + dV = 192 KiB`（3 次 `64 KiB` FP32 原子写），再加上 `QK/dP -> UB`（`128 KiB`），总 `FIXPIPE` 字节涨至 `320 KiB`（`1.67×`），且慢速 GM 原子写部分暴涨 **`3×`**。
2. **仿真与真机流量模型双重证伪**：
   - 在 Ascend 950 CAModel 同工作量单核对比中，虽然纯 Cube `MMAD` 总时长确实下降了 `6.1%`，但 `FIXP Dst:OUT` 服务时长从 `21.6 us` 飙升至 `40.1 us`（`+86%`），整核稳态周期从 `2.70 us` 暴涨至 `5.51 us`（**慢 `2.04×`**），整核总时长慢 **`1.77×`**。
   - 即便按真实 950DT 硬件实测带宽反推（排除 CAModel 的 GM 悲观偏差）：基线 `128×128` 等效 `FIXP` 占 `1.07 us`（其中 `QK/dP -> UB` 约 `0.37 us`，`dQ` GM 原子写约 `0.70 us`），放大 3 倍后仅 `FIXP` 就需要 **`~2.47 us`**，不仅远超 `128×128` Cube 的 `~1.63 us`，甚至比 `64×128` 基线整步（`1.89 us`）还慢 ~30%，且同核多个 Q-tile 连续原子写同一块 `dK`/`dV` L2 cache line 还会加剧冲突。

## 何时可能重新成立（推翻本反例的条件）

- 仅当未来硬件的 `FIXPIPE -> L2/GM` FP32 atomic-add 带宽达到当前 3× 以上（使 `192 KiB` 原子写 + `128 KiB` UB 写总计低于 5 个满尺寸 MMAD 耗时），且同时无法在 L0C 内通过非驻留缓冲复用容纳常驻累加器时，才值得重估。

## 替代方案

- **不要牺牲 `dK`/`dV` 常驻**，而是把非驻留输出（`QK`/`dP`/`dQ`）共用到 2 槽 L0C ping-pong `tmp_c[2]` 并采用 `dQ` 前置调度，在不增加任何 `FIXPIPE` 流量的前提下直接实现 `128×128` 常驻 `dK`/`dV`（实卡实测 `97.9%–99.3%` MFU）：见 [l0c-pingpong-dq-first-128x128-bwd.md](../l0c-pingpong-dq-first-128x128-bwd.md)。
