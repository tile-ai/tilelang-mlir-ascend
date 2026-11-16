# 950DT SIMT FP32 算子吞吐实测（2026-11-16）

本文件是 [counts.md](../references/counts.md) “实例：SIMT 实测分母”一节的数据来源。
数值只对下列设备、版本和指令路径成立，不是通用硬件默认值。

## 环境与口径

| 项 | 值 |
|---|---|
| 设备 | Ascend950DT_9592，36 AIC / 72 AIV，Memory 96 GiB，L2 128 MiB |
| 频率 | Current = Rated = 1650 MHz（平台 INI 写 1800，属配置值） |
| CANN | 9.3.0（V100R001C12B116） |
| TileLang | dev `de44ac0a`（`/home/sym/dyw/tilelang`），`T.SimtVF(threads=1024)` |
| 计数路径 | `aiv_vec_simt_vf_ratio` ≈ 0.96–0.99；`aiv_vec_sfu_ratio` **全程 0.000** |
| 生成代码 | `expf` / `logf` / `sqrtf` / `rsqrtf`，`1/x` 为除法，sigmoid 为 `1/(1+expf(-x))` |

方法（`bench_vector_ops.py`）：

- 每元素每轮 1 次 FMA 递推 `v = fmaf(v, 0.5, 0.5)`，把 `v` 锁在 `[0.75, 1.25]`，避免 denormal/NaN 慢路径；
- 每轮把 `op(v)` 累加进活跃寄存器，使所有轮次的 op 都不能被 DCE/CSE 或 hoist；
- **控制组**用完全相同的骨架，只把 `op(v)` 换成 `v`；
- 用每 block 的平均 `aiv_total_cycles` 做差（不用 Task Duration），避开 block 间调度与冷启动偏差；
- 6 个算子均先做单轮数值校验（`--verify`，对 torch 逐元素 `rtol=2e-6`），全部通过。

## 校准：FMA 基线

| 配置 | Task Duration | vec_ratio | element-fma/cycle/AIV | 实测 |
|---|---:|---:|---:|---|
| rounds=2048 | 185.19 us | 0.978 | 110.6 | **26.28 TFLOPS** |
| rounds=512 | 51.02 us | 0.922 | 102.1 | 24.27 TFLOPS |

名义 FP32 Vector 峰值 30 TFLOPS（36/72 档）= `128 lane × 2 flop × 1.65 GHz × 72 AIV`，
与 1650 MHz 自洽；实测 26.28 TFLOPS ≈ **名义的 87.6%，128 lane 的 86.4%**。
与既往独立采集（26.11 TFLOPS / 185.06 us）相差 0.6%，设备与工具链可复现。

## 结果（rounds=128，边际成本）

`cycles/op` 与 `op/cycle/AIV` 为控制组相减后的**边际**值；`Tops/s` = `op/cycle × 72 × 1.65 GHz`。

| 操作 | cycles/op | op/cycle/AIV | cycles per 128 元素 | 实测 Tops/s | 相对 FMA 成本 |
|---|---:|---:|---:|---:|---:|
| FMA（基线） | 0.00904 | 110.6 | 1.16 | 26.28 TFLOPS | 1.0 |
| `sqrtf` | 0.0438 | 22.84 | 5.60 | 2.71 Tops/s | 4.8 |
| `rcp`（1/x） | 0.0439 | 22.76 | 5.62 | 2.70 Tops/s | 4.9 |
| `expf` | 0.0468 | 21.35 | 5.99 | 2.54 Tops/s | 5.2 |
| `rsqrtf` | 0.0996 | 10.04 | 12.75 | 1.19 Tops/s | 11.0 |
| `logf` | 0.1711 | 5.85 | 21.90 | 0.70 Tops/s | 18.9 |
| `sigmoid`（复合） | 0.1981 | 5.05 | 25.36 | 0.60 Tops/s | 21.9 |

`sigmoid` 由 `exp → add → div` 串行组成，4 路基址 ILP 下受延迟限制，
把它当上界使用前需要提高 ILP 重测（当前值是**下界**）。

## 可靠性

| 检查 | 结果 |
|---|---|
| 单轮数值 | 6/6 与 torch 一致（`rtol=2e-6`） |
| 重复运行（exp, rounds=128） | 0.0468 vs 0.0471 cycles/op，差 **0.6%** |
| 第二几何（tiles 8，两倍每 block 工作量） | 0.0470 vs 0.0468，差 **0.4%** |
| rounds 扫描（exp 64/128/256） | 18.65 / 21.35 / 22.75 op/cycle，仍随 rounds 上升 |
| 管道占用 | 被测 kernel `vec_ratio` 0.96–0.99；控制组 0.87–0.93 |

读数结论：**±10%**。rounds 依赖来自控制组与被测 kernel 的固定开销不等，
按 `a + b/rounds` 外推 exp 约为 0.041 cycles/op（≈24 op/cycle，2.9 Tops/s）；
本表统一取 rounds=128 的保守值。

## 与“只做一个 exp 的算子”对比

同目录上级的 `simt-math-950dt` 用“搬入-算一次-搬出”的朴素 kernel：56.6M 元素、132.98 us，
`aiv_vec_ratio` 仅 0.673，逻辑流量 453 MB / 133 us = **3.41 TB/s**（HBM 4 TB/s 的 85%），
msprof Roofline 判 `latency bound: memory caused`——它测的是主存带宽，不是 exp。

差别只在**每元素重复次数**：朴素 kernel 的 AI = 1 op / 8 B = 0.125 op/B，
而 exp 的拐点是 `2.54 Tops/s ÷ 4 TB/s ≈ 0.64 op/B`，低 5 倍，所以由带宽屋顶决定；
本探针把每元素重复 128 次，AI 抬到 16 op/B 以上才进入计算受限。

## 追加：vexpdif 与表达式敏感性（同日第二次采集）

需求是补测融合指令 `vexpdif`（softmax 里的 `exp(x - max)`）。两条路径都试了：

**SIMD 路径（融合指令本体）未能隔离。** `T.simd.vexpdif` 只在 `T.SimdVF` 里可用，指令宽度 64 个 fp32。
用 4 路寄存器链、大 UB tile、内层 `T.serial` 反复调用，control / `vexp` / `vexpdif` 三次采集为
**455.10 / 455.11 / 455.13 us**——差值在噪声内。原因是每 chunk 的 `vld`/`vsts` 与同步开销主导
（约 3.5 us/chunk，而算子本身只有几十个周期），与算子无关。此路径本轮不产出数字。

**SIMT 等价路径可用。** 在 `bench_vector_ops.py` 里加 `expdif`（`acc += expf(v - 0.5)`，以常量充当 per-row max），
与同一轮的 control 相减：

| 写法 | rounds=128 Task Duration | 边际 op/cycle/AIV | 实测 |
|---|---:|---:|---:|
| control（无算子） | 32.81 us | — | — |
| `acc += exp(v)` | 118.23 us（3 次） | 21.3 | 2.54 Tops/s |
| `acc += exp(v-0.5)` | **69.47 us**（2 次） | **37.4** | **4.45 Tops/s** |
| `acc += exp(v-1)` | 69.11 us | 37.3 | 4.44 Tops/s |
| `acc += exp(v)-0.5` | 56.90 us | 52.8 | 6.27 Tops/s |

**同一语义、三种写法、2.5 倍差异**，说明这类 libm 序列的实测值对生成代码的调度很敏感
（`exp` 单独一行时反而最慢，多一个 ±0.5 会改变调度）。因此：

- `vexpdif` 取 **37.4 op/cycle、4.45 Tops/s**（与 `exp(v-1)` 相互印证）；
- `exp` 行的 21.4 是**同一写法的下界**，跨写法套用会差 2 倍，报告里必须写清写法。

## 边界

- 只覆盖 **SIMT VF + libm 序列**；`vexp/vln` 等 SIMD 指令路径未测，`aiv_vec_sfu_ratio` 可作判据。
- 数值是**本机、本版本**的实测值（`basis=measured`），不覆盖官方标称峰值；
  换设备、换 CANN/TileLang、换频率都需重测。
- 控制组本身未完全饱和（0.87–0.93），相减后的 op 成本系统性偏高约 5–10%。
- 目录重跑的入口：`bench/run_bench.sh full`，汇总：`python3 bench/summarize.py`。
