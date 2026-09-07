# _make_lerp_tensor_kernel Stage 4 性能调优日志（opt_log.md）

- 算子: `_make_lerp_tensor_kernel`（LerpTensorFwdOp，Tensor-weight lerp：`out = a + w·(b−a)`，(N,)×3 进 (N,) 出，fp16/bf16/fp32）
- 基准: `../_make_lerp_tensor_kernel.py`（Stage 3 精度通过版本，只读，永不修改）
- 模式: Developer（保持不变）, target="npuir", `@tilelang.jit(out_idx=[3])`
- 硬件: Ascend 910B2C（本机实查 `NPUUtils.get().get_aicore_num()`=24 → 纯 Vector 翻倍 48 vector cores），UB 192KB/core, 1800MHz
- 性能目标: best_effort（用户"尽力调优"），噪声阈值 3%，max_rounds=10
- 主指标: `msprof op` Task Duration(us)（唯一 kernel 时延口径；不采 NPU event / 端到端）
- 测试 shape（manifest workloads，10 组合）: smoke-1m (2^20) fp32/fp16/bf16、elementwise-16m (2^24) fp16/bf16/fp32、elementwise-64m (2^26) fp16/bf16、elementwise-256m (2^28) fp16/bf16
- 对照基线: torch.lerp NPU（conductor 提供 msprof 口径：1M 15.2–16.6us / 16M 166.7–203.5us / 64M ~691us / 256M ~2827us）
- 环境: python=`/home/tilelang/miniconda3/envs/zuo/bin/python`, msprof (cann-8.5.0), tilelang 0.1.2+ed787bb（2026-09-07 build）
- 运行目录: `examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/`
- 采集协议: `msprof op --kernel-name=main --launch-count=15 --warm-up=5 --dump=off --aic-metrics=BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,MemoryUB,MemoryL0,L2Cache,ResourceConflictRatio python bench.py --impl {impl} --dtype {d} --N {n} [--block-size {bs}] --check`；bench 4-set 输入轮换；每次 msprof 后用 `extract_msprof.py` 取 15 launch 的 median；branch 校验含 `--check`（golden 对比）+ 分支文件自身 `--level L0`。

## Dispatch Path 分析

工厂内 Python 层编译期分支 `use_fp32_transit = dtype in ("float16", "bfloat16")`（真实代码分支；fp16/bf16 共用同一 transit 分支体，不视为两条 path）：

| dispatch_path | 触发 | kernel 结构 | target_kernel_name |
|---|---|---|---|
| fp32_transit | dtype ∈ {float16, bfloat16} | GM→UB(同dtype) ×3 → vcast(rint)→fp32 ×3 → vsub/vmul/vadd(fp32 原地链) → vcast(rint)→原dtype → UB→GM | main |
| fp32_native | dtype == float32 | GM→UB ×3 → vsub/vmul/vadd(原地链) → UB→GM（无 cast） | main |

代表 workload（10 组合，全部必测）：

| workload_id | dispatch_path | N | dtype |
|---|---|---|---|
| smoke-1m/fp16 · smoke-1m/bf16 · smoke-1m/fp32 | transit/transit/native | 2^20 | fp16/bf16/fp32 |
| elementwise-16m/fp16 · bf16 · fp32 | transit/transit/native | 2^24 | fp16/bf16/fp32 |
| elementwise-64m/fp16 · bf16 | transit | 2^26 | fp16/bf16 |
| elementwise-256m/fp16 · bf16 | transit | 2^28 | fp16/bf16 |

## Performance Test Data (Baseline, round 0)

默认 block_size：fp32=2048 / fp16·bf16=4096；persistent `num_kernels=min(num_logical, 48)` + `T.serial` grid-stride + if-guard + 动态尾块 `T.min`。

| dispatch_path | workload_id | target_kernel | captured_op | task_duration_us (median/15) | block_dim | MTE2 ratio | MTE2 active bw (GB/s/core) | vec ratio | scalar ratio | GM→UB (KB/core) | profile_status | raw_profile_dir |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| fp32_transit | smoke-1m/fp16 | main | main | **7.30** | 48 | 0.72 | 33.9 | 0.47 | 0.19 | 128 | valid | profiles/baseline/1m_fp16 |
| fp32_transit | smoke-1m/bf16 | main | main | **7.52** | 48 | 0.70 | 33.4 | 0.47 | 0.21 | 128 | valid | profiles/baseline/1m_bfloat16 |
| fp32_native | smoke-1m/fp32 | main | main | **11.14** | 48 | 0.84 | 32.2 | 0.17 | 0.15 | 256 | valid | profiles/baseline/1m_fp32 |
| fp32_transit | elementwise-16m/fp16 | main | main | **71.36** | 48 | 0.96 | 30.2 | 0.56 | 0.08 | 2048 | valid | profiles/baseline/16m_float16 |
| fp32_transit | elementwise-16m/bf16 | main | main | **72.46** | 48 | 0.95 | 30.7 | 0.60 | 0.08 | 2048 | valid | profiles/baseline/16m_bfloat16 |
| fp32_native | elementwise-16m/fp32 | main | main | **165.17** | 48 | 0.98 | 25.1 | 0.15 | 0.04 | 4096 | valid | profiles/baseline/16m_float32 |
| fp32_transit | elementwise-64m/fp16 | main | main | **388.24** | 48 | 0.98 | 20.9 | 0.40 | 0.05 | 8192 | valid | profiles/baseline/64m_float16 |
| fp32_transit | elementwise-64m/bf16 | main | main | **389.42** | 48 | 0.98 | 20.9 | 0.42 | 0.05 | 8192 | valid | profiles/baseline/64m_bfloat16 |
| fp32_transit | elementwise-256m/fp16 | main | main | **1710.07** | 48 | 0.99 | 18.4 | 0.36 | 0.04 | 32768 | valid | profiles/baseline/256m_float16 |
| fp32_transit | elementwise-256m/bf16 | main | main | **1714.95** | 48 | 0.99 | 18.6 | 0.38 | 0.04 | 32768 | valid | profiles/baseline/256m_bfloat16 |

L0: baseline `--level L0` 全 PASS（fp16 max_diff 1.953e-03 在 atol+rtol 容差内、bf16 bit-exact、fp32 4.768e-07；logs/round0/baseline_l0.log）。

### Baseline 现象（对照 torch.lerp msprof 口径）

- P1 **kernel 已全档快于 torch.lerp**（Stage 5 events 口径的"1M 慢于 torch"是端到端派发开销失真，与本 Stage kernel-only 口径无关）：1M 7.30 vs 15.2（2.1x）、16M 71.4 vs 166.7–203.5（2.3–2.8x）、64M 388 vs ~691（1.8x）、256M 1710 vs ~2827（1.65x）。
- P2 **MTE2（加载）全面主导**：≥16M 时 mte2_ratio 0.95–0.99；vector 链（transit 7 pass）大部分被 auto multi-buffer 隐藏在 MTE2 窗口内（16M fp16：串行模型 64.8+38.0+14.8+5.4=123us，实测 71.4us → tile 间 copy/compute 重叠确实存在，与 mish Round-2 "完全串行" 结论不同——lerp 的 7 个短生命周期 UB buffer 使 auto multi-buffer 生效）。
- P3 **MTE2 有效带宽随 N 退化**（DRAM 侧饱和）：33.9 GB/s/core @1M（4-set 轮换下含 L2 成分）→ 30.2 @16M → 20.9 @64M → 18.4 @256M（聚合读 ~0.88 TB/s + 写 ~0.30 TB/s ≈ 1.18 TB/s 混合流量，2.15GB IO / 1710us = 1.26 TB/s）。
- P4 **1M 档 scalar 开销显著**：scalar_ratio 0.15–0.21（每核 ~1us；动态尾块 `T.min` + if-guard + 地址计算，5–6 tile/core 时无法摊薄），且 mte2_ratio 仅 0.70–0.84（流水浅、排空占比高、5/6-tile 负载不均）。
- P5 **MTE3（写出）效率高于 MTE2**：mte3_active_bw 19.8–45.4 GB/s vs mte2 18.4–33.9，store 流量仅 1/4 且与 MTE2 部分并发（mte3_ratio 0.19–0.31）。

## Iteration 1

### Diagnostic Context

- current_best: baseline（persistent nc=48 + T.serial grid-stride + 动态尾块；默认 bs fp32=2048 / fp16·bf16=4096）
- Profile facts: 见 Baseline 表——≥16M 时 mte2_ratio 0.95–0.99（MTE2 主导）；1M 档 scalar_ratio 0.15–0.21、mte2_ratio 仅 0.70–0.84
- Theory estimate: copy-only 地板探针（probe_copy.py，同结构 3 载入 + 2 vadd 保活 + 1 写出）：

| workload | probe_copy_us | lerp_us | delta | 结论 |
|---|---:|---:|---:|---|
| smoke-1m/fp16 bs4096 | 6.86 | 7.30 | +0.44 (+6.4%) | 计算链少量暴露 |
| elementwise-16m/fp16 bs4096 | 71.02 | 71.36 | +0.34 (+0.5%) | **已贴搬运地板** |
| elementwise-64m/fp16 bs4096 | 387.10 | 388.24 | +1.14 (+0.3%) | 已贴搬运地板 |
| elementwise-256m/fp16 bs4096 | 1714.05 | 1710.07 | −4.0 (−0.2%) | 已贴搬运地板（噪声内） |
| elementwise-16m/fp32 bs2048 | 165.93 | 165.17 | −0.76 (−0.5%) | 已贴搬运地板 |

→ **transit 计算 7 个向量 pass 在 ≥16M 全部被 auto multi-buffer 隐藏在 MTE2 窗口内**；≥16M 的唯一剩余杠杆是提升 MTE2/DRAM 搬运效率本身（burst 粒度/访问模式）。1M 档残余 0.44us 计算暴露 + ~1us scalar 开销。

- Code structure: 每 tile 4 copy + 7 v-pass（transit）/ 3 v-pass（native）；动态尾块 `T.min` 每 tile 1 次标量比较 + 切片长度运行时化；if-guard 每 tile 1 次
- Known conclusions: 无（首轮）

### Current Phenomena

- P1: ≥16M 全 dtype 已贴 copy-only 地板（delta ≤1.1us），MTE2 有效带宽 30.2→18.4 GB/s/core 随 N 退化（DRAM 混合读写饱和，256M 全流量 ~1.26 TB/s）
- P2: 1M 档 scalar_ratio 0.15–0.21（动态尾块 + guard + 地址计算，5–6 tile/core 摊不薄），mte2_ratio 0.70–0.84（流水浅、排空占比高、256/48=5.33 不均）
- P3: fp32 默认 bs=2048 的 MTE2 burst 仅 8KB，且 16M fp32 mte2_active_bw 25.1 GB/s/core 低于 fp16 的 30.2

### Candidate Optimization Points

| opt_id | 目标现象 | 优化点 | 判断依据 | 具体改法 | 验证指标 | 状态 |
|---|---|---|---|---|---|---|
| OP1 | P1/P3 MTE2 burst 粒度 | block_size 扫描（参数级，不改代码） | 更大 tile → 更大单次 MTE2 burst（8KB→16/32KB）+ 更少 tile/core 摊薄 scalar | bench --block-size：fp16/bf16 {2048, 6144, 8192}、fp32 {4096, 8192, 12288} | Task Duration + 非回退 | candidate |
| OP2 | P1 地板标定 | copy-only 探针（诊断，非候选） | 量化各 size 距搬运地板的剩余空间，决定结构方向 ROI | probe_copy.py 同结构探针 | probe vs lerp delta | done（见上表） |

### Experiment Branches

| branch | base | opt_id | major_change | correctness | task_duration_us (median/15) | profile_status | result |
|---|---|---|---|---|---|---|---|
| baseline bs=8192 (fp16) | baseline | OP1 | MTE2 burst 16KB | --check PASS | 1M **7.68** / 16M 73.02 / 64M 383.40 / 256M 1707.47 | valid | 1M **回退 +5.2%**；64M −1.2%（<3%）；256M −0.15%（噪声）→ config_no_gain |
| baseline bs=6144 (fp16 16M) | baseline | OP1 | 中点参照 | --check PASS | 71.24 | valid | config_no_gain（−0.2%） |
| baseline bs=2048 (fp16 1M) | baseline | OP1 | 深流水假设（更多 tile/core） | --check PASS | 8.06 | valid | **回退 +10.4%** → per-tile 开销主导，证伪深流水方向 |
| baseline bs=8192 (bf16) | baseline | OP1 | 同 fp16 | --check PASS | 16M 71.70 / 256M 1708.03 | valid | config_no_gain（−1.0%/−0.4%） |
| baseline bs=4096 (fp32) | baseline | OP1 | burst 16KB | --check PASS | 1M 11.14 / 16M **159.33** | valid | 16M −3.5% |
| baseline bs=8192 (fp32) | baseline | OP1 | burst 32KB | --check PASS | 1M **10.90** / 16M **156.51** | valid | 16M **−5.2% improved**；1M −2.2%（噪声内不回退） |
| baseline bs=12288 (fp32 16M) | baseline | OP1 | burst 48KB | 编译失败（bishengir-compile: 4-buffer×192KB 静态 + auto-multi-buffer 膨胀超 UB） | na | invalid | **blocked**（复现 mish BP_multi_buffer_ub_budget：fp32 4-buffer 的 bs 上限=8192；解锁需削 buffer → Iteration 2 OP1） |

### 候选 vs current best 对比表（Round 1 主读数，msprof Task Duration median/15）

| 候选 (dispatch/workload) | current_best_us | 候选_us | Δ | MTE2 ratio | vec ratio | L0/精度 | 结论 |
|---|---:|---:|---:|---|---|---|---|
| bs8192 / smoke-1m fp16 | 7.30 | 7.68 | +5.2% | 0.72(基线) | 0.47(基线) | --check PASS | 回退，拒 |
| bs8192 / 16m fp16 | 71.36 | 73.02 | +2.3% | 0.96 | 0.56 | --check PASS | 回退，拒 |
| bs8192 / 64m fp16 | 388.24 | 383.40 | −1.2% | 0.98 | 0.40 | --check PASS | 噪声内，不单独采纳 |
| bs8192 / 256m fp16 | 1710.07 | 1707.47 | −0.15% | 0.99 | 0.36 | --check PASS | 噪声内 |
| bs8192 / 16m bf16 | 72.46 | 71.70 | −1.0% | 0.95 | 0.60 | --check PASS | 噪声内 |
| bs8192 / 1m fp32 | 11.14 | 10.90 | −2.2% | 0.84 | 0.17 | --check PASS | 噪声内不回退 |
| **bs8192 / 16m fp32** | **165.17** | **156.51** | **−5.2%** | 0.98 | 0.15 | --check PASS | **improved，采纳** |
| bs4096 / 16m fp32 | 165.17 | 159.33 | −3.5% | 0.98 | 0.15 | --check PASS | 被 bs8192 支配 |

### 必测 Dispatch 检查（候选 winner：fp32 默认 bs 2048→8192，其余 dtype 不变）

| dispatch_path | workload | current_best_us | branch_us | status |
|---|---|---:|---:|---|
| fp32_native | smoke-1m/fp32 | 11.14 | 10.90 | pass（−2.2%） |
| fp32_native | elementwise-16m/fp32 | 165.17 | 156.51 | pass（−5.2%） |
| fp32_transit | 全部 8 组 | 不变（fp16/bf16 维持 bs=4096） | 不变 | pass |

### Iteration Winner

- winner: baseline 结构 + **fp32 默认 block_size 8192**（fp16/bf16 维持 4096）
- reason: fp32 两必测组 −5.2%/−2.2% 无回退；fp16/bf16 的 bs 扫描全部落入噪声或回退（4096 已是甜点）
- new_current_best: （参数级采纳，待最终文件落地；代码层面 = baseline + fp32 默认 8192）
- rollback_branches: fp16/bf16 bs8192/6144/2048（config_no_gain 或回退）
- blocked: fp32 bs12288（UB+multi-buffer，Iter 2 以 3-buffer 结构解锁重试）

## Iteration 2

### Diagnostic Context

- current_best: baseline 结构 + fp32 默认 bs=8192（Round 1 采纳）；fp16/bf16 bs=4096
- Profile facts: fp32 16m bs8192：mte2_ratio 0.98、vec_ratio 0.15（MTE2 主导）；fp32 bs12288（4-buffer）blocked（UB 膨胀 245760bits 需求 > 1572864bits... 实测 12288 时 requires 1966080 bits vs 1572864 available，膨胀 ~1.7x）
- Theory estimate: fp32 4-buffer 16B/elem，bs>8192 静态即 >128KB，multi-buffer 膨胀后超 192KB；削掉 out_ub（3-buffer 12B/elem）理论解锁 bs 12288–16384
- Known conclusions: Round 1 已证 ≥16M 贴 copy 地板；fp32 收益来自 MTE2 burst 粒度而非结构

### Candidate Optimization Points

| opt_id | 目标现象 | 优化点 | 判断依据 | 具体改法 | 验证指标 | 状态 |
|---|---|---|---|---|---|---|
| OP1 | fp32 bs>8192 被 4-buffer UB 阻塞 | fp32 3-buffer 原地链（去 out_ub，vadd dst 别名 b_ub） | mish v2_op3 先例（削 buffer 解锁大 bs）；mte2 burst 32→48/64KB | v2_op1 分支：fp32 路径 16→12B/elem，UB cap 16384 | 16m fp32 bs{8192,12288,16384} Task Duration | candidate |
| OP2 | 1M 档 scalar_ratio 0.15–0.21（T.min 动态尾块 + 动态长度 copy） | 编译期 exact-multiple 尾块特化（tail 折叠为 Python 常量，copy 静态长度化） | manifest 全部 N=2^k 整除；动态长度 copy 走 memref 动态 shape 路径（T.copy.md §3） | v2_op2 分支：`tail = block_size if exact else T.min(...)`，非整除 N 保留动态路径（L0-3 覆盖） | 1m/16m/256m fp16 Task Duration + L0 | candidate |

### Experiment Branches

| branch | base | opt_id | major_change | correctness | task_duration_us (median) | profile_status | result |
|---|---|---|---|---|---|---|---|
| _opt_v2_op1.py bs8192 (16m fp32) | R1 best | OP1 | 3-buffer 原地链 | L0 PASS + --check | 157.27 | valid | config_no_gain（vs 4-buffer 156.51，+0.5% 噪声内；结构本身无收益） |
| _opt_v2_op1.py bs8192 (1m fp32) | R1 best | OP1 | 同上 | L0 PASS + --check | 10.90 | valid | tie |
| _opt_v2_op1.py bs12288/16384 (16m fp32) | R1 best | OP1 | 3-buffer 解锁大 bs | 编译失败（`ub overflow, requires 1966080 bits while 1572864 bits available`，multi-buffer 膨胀 ~1.7x） | na | blocked | **blocked**（3-buffer 也无法过 8192；fp32 bs 上限=8192 封顶，family 级结论） |
| _opt_v2_op2.py bs4096 (1m fp16) | baseline | OP2 | 静态尾块 | L0 PASS + --check | 7.62 (n15) / **7.30 (n30 复测)** | valid | tie（复测翻转，run 级双态；合并 ≈ 基线 7.30/7.48） |
| _opt_v2_op2.py bs4096 (16m fp16) | baseline | OP2 | 静态尾块 | L0 PASS + --check | 70.80 (n15) / 72.20 (n30 复测) | valid | tie（合并 ≈ 基线 71.36/71.78） |
| _opt_v2_op2.py bs4096 (256m fp16) | baseline | OP2 | 静态尾块 | L0 PASS + --check | 1700.19 | valid | −0.6%（< 3% 不采纳） |
| baseline 复测锚点 (1m/16m fp16, n30) | — | — | — | — | 7.48 / 71.78 | valid | 双态校准（与 round0 的 7.30/71.36 合并成带状分布） |

### 候选 vs current best 对比表（Round 2）

| 候选 (dispatch/workload) | current_best_us | 候选_us | Δ | MTE2 ratio | vec ratio | L0/精度 | 结论 |
|---|---:|---:|---:|---|---|---|---|
| v2_op1 3-buffer / 16m fp32 (bs8192) | 156.51 | 157.27 | +0.5% | 0.98 | 0.15 | L0 PASS | tie，结构不采纳 |
| v2_op2 static tail / 1m fp16 | 7.30–7.48 | 7.30–7.62（两 run） | ±0（带状重叠） | 0.70–0.72 | 0.47 | L0 PASS | tie，不采纳 |
| v2_op2 static tail / 16m fp16 | 71.36–71.78 | 70.80–72.20（两 run） | ±0 | 0.95–0.96 | 0.56 | L0 PASS | tie，不采纳 |
| v2_op2 static tail / 256m fp16 | 1710.07 | 1700.19 | −0.6% | 0.99 | 0.36 | L0 PASS | < 3%，不采纳 |

### Iteration Winner

- winner: current_best 保持（baseline 结构 + fp32 bs8192）
- reason: OP1 结构无收益且大 bs 全 blocked；OP2 全档 tie/亚阈值（scalar 在 ≥16M 已被饱和 MTE2 隐藏——mte2_ratio 0.95–0.99，v2_op2 实测佐证：T.min 消除仅 -0.6%）
- rollback_branches: v2_op1、v2_op2（均不采纳；代码保留于 perf_opt/ 供复查）
- 机制沉淀：run 级双态（1M fp16 同一 kernel 跨 run 7.30–7.66，±4%）再次实证 mish BP_run_state_bimodality；<3% 结论必须多 run 合并中位数

## Iteration 3

### Diagnostic Context

- current_best: 不变（baseline 结构 + fp32 bs8192）
- 剩余未验证结构杠杆：grid-stride 任务映射 → 连续分块（DRAM 局部性）；guard-free epilogue 预判 <0.3% 被机制证据否决（mte2_ratio 0.99 = scalar 已隐藏 + v2_op2 实测 -0.6% 佐证，记 defer 不再分支）；T.Pipelined / Expert 手动流水按 mish 先例否决（auto multi-buffer 已生效且平台无真并发，BP_manual_pipeline_developer_broken）

### Candidate Optimization Points

| opt_id | 目标现象 | 优化点 | 判断依据 | 具体改法 | 验证指标 | 状态 |
|---|---|---|---|---|---|---|
| OP1 | 256M MTE2 有效带宽 18.4 GB/s/core（grid-stride 窗口式 48 流交织） | 连续分块映射（每核独占连续 tile 段，DRAM row 局部性） | 访存模式唯一未测结构变量 | v3_op2 分支：`block_id = cid * num_local_tasks + i` | 1m/16m/256m fp16 Task Duration | candidate |

### Experiment Branches

| branch | base | opt_id | major_change | correctness | task_duration_us (median) | profile_status | result |
|---|---|---|---|---|---|---|---|
| _opt_v3_op2.py bs4096 (1m fp16, n30) | baseline | OP1 | 连续分块 | L0 PASS + --check | 7.58 | valid | tie |
| _opt_v3_op2.py bs4096 (16m fp16, n30) | baseline | OP1 | 连续分块 | L0 PASS + --check | 71.50 | valid | tie |
| _opt_v3_op2.py bs4096 (256m fp16) | baseline | OP1 | 连续分块 | L0 PASS + --check | **1765.01** | valid | **回退 +3.2%，拒** |

### 候选 vs current best 对比表（Round 3）

| 候选 (dispatch/workload) | current_best_us | 候选_us | Δ | MTE2 ratio | vec ratio | L0/精度 | 结论 |
|---|---:|---:|---:|---|---|---|---|
| v3_op2 contig / 1m fp16 | 7.30–7.48 | 7.58 | +2%（噪声带内） | 0.72 | 0.47 | L0 PASS | tie |
| v3_op2 contig / 16m fp16 | 71.36–71.78 | 71.50 | ±0 | 0.96 | 0.56 | L0 PASS | tie |
| v3_op2 contig / 256m fp16 | 1710.07 | 1765.01 | **+3.2%** | 0.99 | 0.36 | L0 PASS | **回退，拒** |

### Iteration Winner

- winner: current_best 保持
- reason: 连续分块在大 N 明确回退（+3.2%）——grid-stride 的"48 核聚集在 384KB 移动窗口"访问模式对 DRAM 更友好（fewer open pages / 行缓冲命中），证伪连续流假设；小 N tie。family 级结论：任务映射维持 grid-stride。
- rollback_branches: v3_op2
- 至此结构候选空间穷尽（bs 扫描 / 3-buffer / 静态尾块 / 连续分块 / guard-free[defer+机制证据] / T.Pipelined / Expert 流水[平台先例] / 跨 dtype copy 融合[文档证伪]），进入收敛验证

## Iteration 4（Final 验证 + fp32 采纳项 A/B 复核）

### Diagnostic Context

- current_best: baseline 结构 + fp32 默认 bs=8192，落地为 `perf_opt/_make_lerp_tensor_kernel.py`（相对基准仅改 `_DEFAULT_BLOCK["float32"]`、docstring、新增 L1 fp32 bs8192 用例；结构/dispatch/UB 预算/工厂契约字节级一致）
- 关键疑点：Round 1 的 fp32 -5.2%（165.17→156.51）是否为 run 级双态伪影（两条腿各捕获相反状态）

### fp32 采纳项交错 A/B/A/B 复核（16m fp32）

| run | 配置 | median_us (n=15) | raw_dir |
|---|---|---:|---|
| B1 | final（bs8192 默认） | 158.75 | profiles/final/final_16m_fp32 |
| A1 | baseline impl bs2048 | 163.65 | profiles/final/ab_base_16m_fp32_bs2048 |
| B2 | final（bs8192 默认） | 158.59 | profiles/final/ab_final_16m_fp32_r2 |
| A2 | baseline impl bs2048 | 162.27 | profiles/final/ab_base_16m_fp32_bs2048_r2 |

- 配对结论：两次 A 均显著慢于两次 B（A 中位 163.65/162.27 vs B 158.75/158.59），次序稳定、分布不重叠（A min 158.2–158.4 vs B max 161.9–162.3 边缘交叉但中位差 4.0–5.1us）
- 合并全部 run（含 round1）：baseline {165.17, 163.65, 162.27} 中位 **163.65**；final(bs8192) {156.51, 158.75, 158.59} 中位 **158.59** → **-3.09%（过 3% 门槛）**
- 单轮 -5.2% 确系双态膨胀（两条腿各捕获快/慢态）；真实增益 -3.1%，按多 run 协议（mish BP_run_state_bimodality：≥3 独立 run + 交替次序 + 合并中位）**维持采纳**
- 1m fp32 复核：final {11.16, 11.22} vs baseline 11.14 → 持平（噪声内，无回退）

### Final 全组合验证（final 文件默认参数，10 组合）

| dispatch_path | workload | final_us (median/15) | baseline_us (round0) | Δ | vs torch.lerp (msprof) | raw_dir |
|---|---|---:|---:|---:|---:|---|
| fp32_transit | smoke-1m/fp16 | 7.66 / 7.32（两 run） | 7.30 | 噪声内（配置未变，同 kernel 跨 run 7.30–7.66 双态带） | 2.0–2.3x | profiles/final/final_1m_fp16{,_r2} |
| fp32_transit | smoke-1m/bf16 | 7.44 | 7.52 | -1.1%（噪声内） | ~2.1x | profiles/final/final_1m_bf16 |
| fp32_native | smoke-1m/fp32 | 11.16 / 11.22 | 11.14 | 持平（噪声内） | ~1.4x | profiles/final/final_1m_fp32{,_r2} |
| fp32_transit | elementwise-16m/fp16 | 72.32 | 71.36 | +1.3%（噪声内，同 kernel 双态） | ≥2.3x | profiles/final/final_16m_fp16 |
| fp32_transit | elementwise-16m/bf16 | 72.16 | 72.46 | -0.4%（噪声内） | ≥2.3x | profiles/final/final_16m_bf16 |
| fp32_native | elementwise-16m/fp32 | **158.59**（合并中位，3 run） | 163.65（合并中位，3 run） | **-3.1%（采纳项）** | 1.05–1.28x | profiles/final/{final_16m_fp32, ab_final_16m_fp32_r2} |
| fp32_transit | elementwise-64m/fp16 | 388.28 | 388.24 | ±0 | ~1.78x | profiles/final/final_64m_fp16 |
| fp32_transit | elementwise-64m/bf16 | 385.76 | 389.42 | -0.9%（噪声内） | ~1.79x | profiles/final/final_64m_bf16 |
| fp32_transit | elementwise-256m/fp16 | 1709.33 | 1710.07 | ±0 | ~1.65x | profiles/final/final_256m_fp16 |
| fp32_transit | elementwise-256m/bf16 | 1710.95 | 1714.95 | -0.2%（噪声内） | ~1.65x | profiles/final/final_256m_bf16 |

精度回归：`python _make_lerp_tensor_kernel.py --level all` 全 PASS（L0 六项 / L1 含新增 fp32 bs8192 用例与 E2 双配置复用 / L2 / Boundary；fp16 max_diff 9.766e-04、bf16 2.441e-04、fp32 4.768e-07，均在容差内；logs/final/final_level_all_v2.log）。

## Final Summary

- best 版本: `perf_opt/_make_lerp_tensor_kernel.py`（= 基准字节级结构 + `_DEFAULT_BLOCK["float32"]: 2048→8192`；fp16/bf16 维持 4096；工厂 E1/E2 契约、UB guard、persistent nc=48 + T.serial grid-stride + 动态尾块全部不变）
- **final_latency: 158.59 us**（elementwise-16m/fp32，final 文件默认参数，3 次独立 run 合并中位；对应 perf_records round4 ab_final_16m_fp32_r2 行）
- 总提升（msprof Task Duration，合并中位）:
  - elementwise-16m/fp32: 163.65 → 158.59 us（**-3.1%**，唯一采纳项；单轮首测 -5.2% 系双态膨胀）
  - smoke-1m/fp32: 持平（11.14 → ~11.2，噪声内无回退）
  - fp16/bf16 全部 8 组合: 配置未变，性能等同（final 验证 run 全部落在基线双态带内）
- vs torch.lerp（conductor msprof 口径）: 1M 2.0–2.3x / 16M fp16·bf16 ≥2.3x / 16M fp32 1.05–1.28x / 64M ~1.78x / 256M ~1.65x（Stage 5 events 口径的"1M 慢于 torch"为端到端派发开销失真，kernel-only 口径下 baseline 即已全档快于 torch）
- 有效优化点（采纳）:
  1. fp32 默认 block_size 2048→8192（32KB MTE2 burst；16M fp32 -3.1% 合并中位，A/B/A/B 4 run 次序稳定）
- 无效/回退/阻塞优化点（放弃，全部留档）:
  1. fp16/bf16 bs 扫描 {2048, 6144, 8192}：2048 +10.4%（per-tile 开销主导，证伪深流水）、8192 小 N +5.2% / 大 N 噪声内 → 4096 即甜点
  2. fp32 bs12288/16384（4-buffer 与 3-buffer 均 blocked）：auto-multi-buffer 膨胀 ~1.7x 超 192KB UB
  3. fp32 3-buffer 原地链（v2_op1）：结构本身 tie（157.27 vs 156.51），不采纳
  4. 静态尾块特化（v2_op2）：全档 tie/-0.6%（scalar 已被饱和 MTE2 隐藏），不采纳
  5. 连续分块映射（v3_op2）：256M **+3.2% 回退**（grid-stride 聚集窗口对 DRAM 更优），拒
  6. guard-free epilogue：机制证据 defer（mte2_ratio 0.99 → scalar 隐藏；v2_op2 实测佐证 <0.6%）
  7. T.Pipelined / Expert 手动 rs·flag 流水：mish 先例（auto multi-buffer 已生效、平台同步发射无真并发、Developer tensor-compile 破坏手动流水）+ 本 kernel 16M 探针 delta 0.34us 证 auto-MB 已充分
  8. 跨 dtype T.copy 融合加载+升精度：文档证伪（T.copy.md §3 Developer GM→UB lowering 插入 hivm::VCastOp = 隐藏 vcast，不减少 vector pass 且舍入语义不可控）
- 中止原因: **plateau + 候选穷尽** —— 结构候选（bs/3-buffer/静态尾块/连续分块/guard-free/T.Pipelined/Expert 流水/跨 dtype copy）全部验证或以机制证据+平台先例关闭；**≥16M 全 dtype 已贴 copy-only 地板**（probe_copy 同结构 3 载入+2vadd 探针 delta ≤1.1us；256M 混合读写 ~1.26 TB/s 为该 3:1 R/W 流量组合的设备级地板，grid-stride 已是实测最优访问模式）；1M 档 MTE2 主导（0.70–0.84）且残余计算暴露仅 0.44us。best_effort 目标下无 ≥3% 的剩余候选。
- [DESIGN_LIMIT] 判定: **不触发** —— 剩余差距（~1.26 vs 参考 1.8 TB/s）属设备混合读写带宽属性而非设计层假设错误：DESIGN.md roofline（bytes=4N·elem_bytes 为 I/O 下界、访存主导）与实测一致；persistent nc=48、UB 预算、fp32 transit 路径假设全部被实测确认；无 >2x 结构性候选（copy 探针 = 当前 kernel）。参数级收敛，不产出 perf_feedback.md。

## Skill Retrospective

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Iteration-diagnosis | 1M 级小 kernel（<10us）的 run 级双态（同配置跨 run ±3–5%）使单 run median-of-15 不足以支撑 3% 门槛判断；本轮两次踩坑（v2_op2 首测 +4.4% 复测翻转为 tie；fp32 采纳项单轮 -5.2% 复测收缩为 -3.1%） | opt_log Iteration 2/4：1m fp16 同 kernel 跨 run 7.30–7.66；A/B/A/B 4 run 次序稳定 | iteration-diagnosis.md Step 6 补充：kernel <20us 或逼近带宽上限时，采纳判断强制 ≥3 独立 run + 交错次序 + 合并中位（mish P.3 协议的规则化） | R |
| SKILL/调度 | conductor 透传的 Stage 5 events 基线与 msprof kernel-only 口径方向相反（1M events 182.83us "慢于 torch" vs msprof 7.30us 快于 torch 2.1x），若直接当 headroom 依据会误导调优方向（本任务因 skill 强制 msprof 唯一口径而免误） | DESIGN §11 表 vs 本日志 Baseline 表 | 调度透传的 events 数据须标注"仅趋势参考、禁止作为 headroom 依据"；pattern-library §2 已弃用 event 口径的结论应同步到 conductor 调度模板 | R |
| Bottleneck-patterns | 缺"标量削减类优化在 MTE2 饱和区无效"的判定式，导致 v2_op2（静态尾块）实验虽可预判仍花费两轮复测确认 | mte2_ratio 0.95–0.99（≥16M）+ v2_op2 全档 tie/-0.6% | BP_ub_traffic_floor 或新增条目补判定式：标量/分支削减优化前先查 mte2_ratio，≥0.95 时标量已被搬运隐藏 | P |
| Profile-collection | 无阻塞（10 组合必测 dispatch 全覆盖，无框架小算子误采，captured_op=main 全部可追溯） | 本日志 Performance Test Data 表 | none | - |
| stop_condition | plateau 判定证据充分：结构候选 8 项全部验证/关闭 + copy-only 地板探针标定 + 无 ≥3% 剩余候选 | Final Summary 无效优化点 1–8 + probe 表 | none | - |

### Value Point Proposals（含 BP_xxx）

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| elementwise 多输入 kernel ≥16M 贴 copy 地板：auto multi-buffer 将 7-pass fp32-transit 链完全隐藏于 MTE2（16M delta 0.34us；探针标定法） | D | `opt_log.md#iteration-1` probe 表 + `profiles/round1/probe_*`（已回写 pattern-library §1.6） | `msprof op ... python bench.py --impl ./probe_copy.py --dtype float16 --N 16777216` | tilelang 0.1.2+ed787bb (2026-09-07) / Ascend910B2C / CANN 8.5.0 | pattern-library.md |
| MTE2 有效带宽随总流量退化曲线：33.9→30.2→20.9→18.4 GB/s/core（1M→16M→64M→256M，4-set 轮换）；3:1 R/W 混合 ~1.26 TB/s 为该设备地板，"~1.8 TB/s 峰值参考"对混合流量不可达 | D | `opt_log.md` Baseline 表 MTE2 active bw 列（已回写 pattern-library §1.6） | 同上（--N 1048576/16777216/67108864/268435456） | 同上 | pattern-library.md |
| run 级双态第二例证：同 kernel 跨 run ±3–5%；单轮 -5.2% 双态膨胀 → 交错 A/B/A/B 合并中位 -3.1%（update BP_run_state_bimodality，mish 后第二次独立证据，满足 Tier 1 合入条件） | P | `opt_log.md#iteration-4` A/B 表 + `profiles/final/ab_*` | 16m fp32：final vs baseline(bs2048) 交替各 2 run | 同上 | bottleneck-patterns.md |
| BP_grid_stride_vs_contig_chunk（新增候选）：带宽饱和区（256M）连续分块映射 +3.2% 回退 vs grid-stride 聚集窗口，DRAM open-page 直觉在 HBM 混合流下反向；小 N tie | P | `opt_log.md#iteration-3` + `profiles/round3/v3op2_256m_fp16_bs4096`（1765.01 vs 1710.07） | `--impl ./_make_lerp_tensor_kernel_opt_v3_op2.py --dtype float16 --N 268435456` | 同上 | bottleneck-patterns.md |
| auto-multi-buffer UB 膨胀 ~1.7x 超限第二例证（fp32 4-buffer 与 3-buffer 的 bs12288/16384 均 blocked；update mish BP_multi_buffer_ub_budget，第二次独立证据） | D | `logs/round1/base_16m_fp32_bs12288.log`、`logs/round2/v2op1_16m_fp32_bs12288.log`：`ub overflow, requires 1966080 bits while 1572864 bits available` | `_make_lerp_tensor_kernel(16777216,"float32")(12288)` | 同上 | pattern-library.md（已回写 §2） |
| 标量削减优化在 MTE2 饱和区无效的判定式（静态尾块 tie / guard-free defer） | P | `opt_log.md#iteration-2` v2_op2 全档 tie | `--impl ./_make_lerp_tensor_kernel_opt_v2_op2.py --dtype float16 --N 1048576` | 同上 | bottleneck-patterns.md |
| lerp_tensor perf_opt 案例入索引（elementwise 3 输入 persistent + copy-floor 标定 + 交错 A/B 协议完整示范） | C | 本目录 | `python perf_opt/_make_lerp_tensor_kernel.py --level all` | 同上 | pattern-library.md §4（已回写） |

### 复盘要点

- skill 流程有效性：Phase 1 强制 msprof 唯一口径直接避免了被 events 基线误导（"1M 慢于 torch"实为端到端失真，kernel-only 全档快于 torch）；copy-only 探针一步把 8/10 组合判定为贴地板，节省了在计算链方向的无效轮次；每轮重分析使 v2_op1（3-buffer）在 fp32 大 bs blocked 后立即转向收敛而非继续扫参。
- 主要教训：小 kernel（<10us）+ 逼近带宽上限的组合下单 run median-of-15 的置信度不足——本次 v2_op2 与 fp32 采纳项各消耗一轮复测才校正归因；交错 A/B 协议应前置到首次出现 <5% 差异时而非采纳后复核。
- 版本戳声明：本日志全部结论绑定 tilelang 0.1.2+ed787bb（2026-09-07 build）/ Ascend910B2C / CANN 8.5.0；工具链变更后 §2 引用条目自动待重验。
