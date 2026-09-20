# opt_log.md — `_ssd_chunk_scan_fwd_kernel` Stage 4 调优日志

> 项目：`ssd_chunk_scan`；算子：`_ssd_chunk_scan_fwd_kernel`（Mamba-2 SSD chunk scan fwd，Expert-mode MixCV）。
> 工具链：tilelang `0.1.2+1990aa9fe4`（dev root build）/ CANN 8.5.0 / Ascend910B2C（24 AIC + 48 AIV，UB 192KB/AIV，L1 512KB，L0C 128KB）/ npu-smi 26.0.rc1。
> 主指标：**`msprof op` Task Duration(us)**（launch-count=20 / warm-up=5 / median of 20；`--kernel-name=main` 过滤，captured op = `main_mix_aic`，Block Dim 24 / Mix Block Dim 48）。唯一 kernel 时延口径；不采 NPU event / 端到端时间。
> 性能目标：**best_effort**（无硬性目标数值；噪声阈值 3%；max_rounds=10、max_experiments=30）。
> 模式约束：**Expert**（Developer + persistent + gemm 模式级不兼容，Stage 3 实证；本轮全程 Expert）。

## 0. 上下文与首轮必查项（Phase 0）

- 算子类型：**mix**（Cube 双 GEMM 路径 + Vector 因子链）；结构：persistent 24 核一维 `T.Kernel(24)`，task = (b,c,h)，Vector 产因子写 GM ws（ws_c / ws_lcb）→ task 级双 flag 握手 → Cube 消费（history `T.gemm(b_transpose)` + intra 因果块 gemm）。
- pattern-library 消费：PL-1.11（算术惩罚掩码，status verified——baseline 已采纳，Expert 形态同门，无需复证）；**PL-1.12**（Cube 深度 2 任务流水 / flag 双槽 / `2×nk_total ≤ 15`——本轮 v1_pipe 的直接蓝本）；CONST-flag-id-budget（≤15/核）；CONST-store-fixpipe-gm-only（L0C 只能回 GM——输出行缩放类方案被此约束封死）；CONST-vector-launch-overhead（~0.5µs/op 串行口径，实测有效摊薄后 ~55-75ns/op，见 §R4）；TRAP-UB-dst-align / v-op 多维切片拒绝（PL-1.9-hardlimits——v3 的 task 级行 buffer 失败与此同族）。
- kb_stale_check 口径：stale_count=83 系工具链版本戳差异；引用条目前核对 front-matter `status`（本任务引用的 PL-1.11 / PL-1.12 均 `verified`）。
- 首轮布局重估：核内布局 = 契约直读 + 尾轴向量化（DESIGN §1.6.3 主选），profile 现象（无标量热点、瓶颈在结构串行与指令数）与该决策不矛盾，不触发换轴。
- **DESIGN §1.6.3 实验裁决执行**（见 §7 裁决记录）：掩码 A/B 由 PL-1.11（Expert verified）直接定局（惩罚主选保留）；bl=128 变体实测 UB 溢出否决。

## 1. Performance Test Data（Phase 1 baseline）

dispatch path：**单一 default 路径**（fp16/bf16 仅 dtype 参数化，无代码分支；Q/N 差异不触发分支）。目标 kernel：`main`（captured `main_mix_aic`）。

| dispatch_path | workload_id | target_kernel | captured_op | task_duration_us (median of 20) | profile_status | raw_profile_dir |
|---|---|---|---|---:|---|---|
| default | smoke (1,2,64,4,64,32,1) fp16 | main | main_mix_aic | 31.11 | valid | `perf_opt/profiles/baseline/smoke/OPPROF_20260917075813_WPXSFTCZMNQEEFQZ` |
| default | w2-780m-s4k (1,16,256,48,64,128,1) fp16 | main | main_mix_aic | 608.15 | valid | `perf_opt/profiles/baseline/w2-780m-s4k/OPPROF_20260917075909_YTVQFRDXTOCBRDVV` |
| default | w3-2p7b-s2k (4,8,256,80,64,128,1) bf16 | main | main_mix_aic | 1990.47 | valid | `perf_opt/profiles/baseline/w3-2p7b-s2k/OPPROF_20260917080000_ZKJRGHLPGHZEHTMX` |
| default | w4-1p3b-s32k (2,128,256,64,64,128,1) fp16 | main | main_mix_aic | 11127.69 | valid | `perf_opt/profiles/baseline/w4-1p3b-s32k/OPPROF_20260917080052_JOSYGBDBMTLELBWS` |

采集命令模板：`msprof op --kernel-name=main --output=perf_opt/profiles/{stage}/{case} --launch-count=20 --warm-up=5 --dump=off --aic-metrics=BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,MemoryUB,MemoryL0,L2Cache,ResourceConflictRatio python perf_opt/bench.py --kernel {file} --case {case} --iters 30`（实验分支同理，`--kernel` 换成分支文件）。

### Baseline 瓶颈画像（w2，block0，~583µs 壁钟）

- **Cube**：cube(gemm) 5.5%（34µs）/ scalar 11.0%（69µs）/ mte1 9.0% / **mte2 31.2%（196µs）** / fixpipe 6.0%——L2 读命中 94.1%。
- **Vector（每 AIV）**：vec 35.5%（222µs）/ scalar 21.8%（136µs）/ mte2 32.4%（203µs，active bw 仅 22.4GB/s——小块延迟受限）/ mte3 17.0%（106µs，41.3GB/s）。
- 诊断：**非 HBM 带宽瓶颈**（聚合 ~0.6TB/s ≪ 1.26TB/s 地板）；**task 级 ping-pong 串行**（Vector(T) 产完 → Cube(T) 消费 → Vector(T+1)……两引擎逐任务交替零重叠）+ 小块搬运指令延迟受限。
- **设计估算 vs 实测偏差行（D-2 回填）**：DESIGN §1.6.0 估算下界 = max(流量 42–83µs, 发射(重叠后)~数百µs, Cube 20.7µs)。实测 608µs —— 落在「发射/向量链主导」预言区间（DESIGN 判定向量链为第一疑似瓶颈 **被证实**）；偏差项 = task 级 ping-pong 零重叠（设计未建模的握手串行）+ MTE 小块指令延迟（CONST-mte2 退化曲线的指令维度），二者本轮根治/缓解。

## 2. Iteration Log

> 每轮：现象 → 候选优化点 → 实验分支（单一优化点）→ L0 → msprof → 对比表（数据来自 perf_records.jsonl）→ winner/rollback。

### Round 1 — 解除 task 级 ping-pong 串行（+ 搬运/发射两个探针分支）

**现象**：baseline 画像如上——Vector/Cube 逐任务严格交替（壁钟 ≈ Σ(V_task + C_task)），所有管线利用率 <36%。

**候选优化点**：
1. 深度 2 任务流水（PL-1.12 家族）：ws 加槽维 ×2 + 4 flag（ready/cons × slot0/slot1）+ **Cube 前导 set**（两槽初始"空闲"，使 Vector T=0/1 的 slot-free wait 立即通过，免除运行时分支）——Vector(T+1) 与 Cube(T) 全重叠。
2. band 搬运合并（v2_band）：x 任务级 L1 驻留 + ws_lcb band 布局 + 每 lt 单条 K=(l0+tl) band gemm。
3. 向量链发射削减（v3_hoist）：dA_l 广播 per-lt hoist + dA/dt 行任务级装载。

**分支结果**（L0 全过分支方进 msprof；v3 编译失败未测）：

| 候选 vs baseline | smoke | w2 | w3 | w4 | L0 | w2 AICore 利用率（cube gemm / aiv vec） | w2 memory（cube mte2 忙比 / aiv mte2 忙比·active bw） | 备注 |
|---|---:|---:|---:|---:|---|---|---|---|
| baseline | 31.11 | 608.15 | 1990.47 | 11127.69 | PASS | 5.5% / 35.5% | 31.2% / 32.4%·22.4GB/s | — |
| **v1_pipe（深度 2 流水）** | 30.84 | **367.46 (−39.6%)** | **1092.60 (−45.1%)** | **6813.71 (−38.8%)** | PASS | 9.5% / 56.5% | 79.6% / 56.8% | **winner** |
| v2_band（band 布局） | 31.03 | 622.05 (+2.3%) | 1851.01 (−7.0%) | 11018.40 (−1.0%) | PASS | 4.6% / 35.5% | 25.0% / 42.4%（aiv mte3 0.17→0.35，106→220µs） | rollback：Vector MTE3 跨步写 3×（band 行 stride 512B），抵消 Cube 收益 |
| v3_hoist（发射削减） | 31.13 | compile FAIL | — | — | PASS | — | — | task 级 [1,Q] 行 buffer 在 s_blk 循环内被动态偏移 subview 消费，auto-multi-buffer pass 产出非支配 IR（"operand #2 does not dominate this use"，Q≥128，首版）；回退为纯 hoist 后 L0 过、smoke 采到 31.13，但 Q=256 编译 UB 溢出 3.2KB（195.1KB > 192KB，实测报错文本）——该 family 双重 blocked |

**结论**：v1_pipe 胜出成 current best。v2_band 教训：**ws 写侧必须保持块连续布局**（MTE3 跨步写 ~3× 慢）；v3 家族 blocked（编译器 dominance 缺陷 + UB 容量）。

### Round 2 — Cube 搬运合并（保留块连续写）+ P-2 广播探针

**现象**（v1_pipe, w2, ~350µs 双引擎平衡）：Cube mte2 79.6%（281µs，~29GB/s 有效——16 条/任务 nd2nz 指令延迟受限）；Vector vec 203µs + mte2 204µs + mte3 117µs。

**候选优化点**：
1. v5_xband：**Cube 侧** band 组装（ws_lcb 仍块布局、逐块连续读入 L1 band 列偏移区）+ x 任务级 [Q,bp] L1 缓存（消 2.5× 跨 lt 重读）+ 每 lt 单条 band gemm（gemm 10→4/任务）——Vector 侧不动。
2. v4_p2：P-2 探针——`T.vsub` 列广播 `[bl,bs] − [1,bs]`（文档未列形态）+ dA_l hoist（省每块 2 次 vbrc）。

| 候选 vs v1_pipe | smoke | w2 | w3 | w4 | L0 | w2 AICore 利用率（cube gemm / aiv vec） | w2 memory（cube mte2 忙比 / aiv mte2 忙比） | 备注 |
|---|---:|---:|---:|---:|---|---|---|---|
| v1_pipe (best) | 30.84 | 367.46 | 1092.60 | 6813.71 | PASS | 9.5% / 56.5% | 79.6% / 56.8% | — |
| v4_p2（vsub 广播） | 31.93 | 412.45 (+12.2%) | 1267.83 (+16.0%) | 6956.56 (+2.1%) | PASS | — | — | rollback：**未文档化广播形态可编译可算对但有性能税**（w2/w3 显著回退）|
| **v5_xband（Cube band 组装）** | 31.23 | **360.78 (−1.8%)** | **1068.88 (−2.2%)** | **6518.38 (−4.3%)** | PASS | 8.3% / 67.7% | 69.6% / 54.1% | **winner**（小幅但全域一致；非必测 dispatch 无回退）|

### Round 3 — n-block 配置 + fp16 c 链

**候选优化点**：
1. v6_bn128：`block_n = 128`（N=128 时 n-loop 单块：Cube 省 4 mte2 + 4 gemm/任务；N<128 由 N_tiles/tn 钳位自然回退——后经 L0-5/B-q96 发现需工厂层 `bn=min(bn,N)` 防分形 stale 带，见 §6 修复记录）。
2. v7_cf16：c_scaled 链 fp16 化（省 2 次 [64,128] vcast/vmul + 释放 32KB UB）。

| 候选 vs v5_xband | smoke | w2 | w3 | w4 | L0 | w2 AICore 利用率（cube gemm / aiv vec） | w2 memory（cube mte2 忙比 / aiv mte2 忙比） | 备注 |
|---|---:|---:|---:|---:|---|---|---|---|
| v5_xband (best) | 31.23 | 360.78 | 1068.88 | 6518.38 | PASS | 8.3% / 67.7% | 69.6% / 54.1% | — |
| **v6_bn128** | 31.83 | **323.01 (−10.5%)** | **1015.29 (−5.0%)** | **5493.78 (−15.7%)** | PASS | 8.2% / 66.0% | 50.0% / 54.6% | **winner**（配置项） |
| v7_cf16（fp16 c 链） | 30.87 | 315.26 (−2.4%) | **FAIL(bf16)** | 5572.11 (+1.4%) | PASS | — | — | rollback：`T.vmul` bf16 ×（RETROSPECTIVE 已知约束）需 dtype 分派；且 w2 −2.4% / w4 +1.4% 均 <5% 未过 T-2 协议、收益不稳，弃置 |

### Round 4 — AIV 工作分片（最大单项）

**现象**（v6, w2, ~310µs）：Mix Block Dim 48 = 每 block 2 AIV；两 AIV 子块指标**完全相同**（vec_ratio 均 0.66 而非分摊的 0.33）→ **Vector 程序在 2 个 AIV 上重复执行**（每 AIV 全量算 + 全量写 ws——GQA "dual-producer" 同 flag 语义的副产品）。AIV 产能 2× 浪费。

**候选优化点**：v8_aivsplit——GQA v11 `subid` 边界表达式先例：`for i in T.serial(lt_count): lt = T.min(i*2+subid, L_tiles-1)`（lt 交错分配，退化 L 折叠为重复 lt=0——bit-identical 良性；无 if 解析器风险）。ws 按 lt 天然不相交；双 AIV 仍同 set 同 wait（本 kernel 现行结构已证 set/wait 语义安全）。

| 候选 vs v6_bn128 | smoke | w2 | w3 | w4 | L0 | w2 AICore 利用率（cube gemm / aiv vec） | w2 memory（cube mte2 忙比 / aiv mte2 忙比） | 备注 |
|---|---:|---:|---:|---:|---|---|---|---|
| v6_bn128 (best) | 31.83 | 323.01 | 1015.29 | 5493.78 | PASS | 8.2% / 66.0%（双 AIV 重复） | 50.0% / 54.6% | — |
| **v8_aivsplit** | 31.24 | **211.85 (−34.4%)** | **651.30 (−35.9%)** | **3958.35 (−27.9%)** | PASS | 12.1% / aiv0 39.0%·aiv1 53.2%（分片后不均 4/6） | 71.9% / aiv0 37.0%·aiv1 52.2% | **winner**（w2/w4 bench --check 同 max_diff 通过）|

### Round 5 — AIV 蛇形均衡（tie，按决胜规则采纳）

**现象**（v8, w2, ~212µs）：AIV1（lts {1,3}，6 s-block + 2 c 链）vec 112µs vs AIV0（{0,2}，4+2）82µs——交错分配不均衡；Cube 成第一 critical（mte2 156µs）。

**候选**：v9_snake——蛇形均衡分配 {0,L−1,2,…}/{1,L−2,3,…}，通用仿射式 `lt = i + subid + (i%2)·(L_tiles − 2i − 2·subid)` + 双向 clamp（L=4 时恰为 {0,3}/{1,2}，5+5 s-block 均分；奇/退化 L 折叠为良性重复）。

| 候选 vs v8_aivsplit | smoke | w2 | w3 | w4 | L0 | w2 AICore 利用率（cube gemm / aiv vec） | w2 memory（cube mte2 忙比 / aiv mte2 忙比） | 备注 |
|---|---:|---:|---:|---:|---|---|---|---|
| v8_aivsplit (best) | 31.24 | 211.85 | 651.30 | 3958.35 | PASS | 12.1% / aiv0 39.0%·aiv1 53.2% | 71.9% / aiv0 37.0%·aiv1 52.2% | — |
| v9_snake | 31.15 | 220.58 (+4.1%*) | 635.63 (−2.4%) | 3885.39 (−1.8%) | PASS | （final 复测画像）10.8% / aiv0 43.5%·aiv1 43.5%（**蛇形均衡达成**） | 77.5% / ~52%·~52% | 单次 run 差异落入 run-state 双态区 |

*w2 的 +4.1% 单次差异经 **ab_test.py 交错协议**（3 对 A/B/A/B，launch-count=15）裁决：pair 中位 A 214.64/219.42/220.92 vs B 219.46/213.69/216.11，合并中位 **A 219.42 vs B 216.11，sign test p=1.0 → verdict: tie**（单次 211.85/220.58 差异系双态噪声）。按 skill 决胜规则（打平→负载均衡优先）+ w3/w4 方向一致 + 结构均衡性（7/7 vs 6/8 单元），**采纳 v9_snake**（`profiles/ab_round5_w2/`）。

### Round 6 — 最终收束与回归修复（winner 复测）

最终产物 `perf_opt/_ssd_chunk_scan_fwd_kernel.py` = v9_snake + tuned 默认（`TUNED_DEFAULT_CONFIG` block_n=128）+ 两处正确性修复（见 §6）。全量 `--level all`：**L0 5/5 + contract / L1 6/6 / L2 4/4 / Boundary 4/4 全过**；w2/w3/w4/smoke bench --check 全过（max_diff 与 baseline 同分布）。

| final vs baseline | smoke | w2 | w3 | w4 | L0 | w2 AICore 利用率（cube gemm / aiv vec） | w2 memory（cube mte2 忙比 / aiv mte2 忙比） |
|---|---:|---:|---:|---:|---|---|---|
| baseline | 31.11 | 608.15 | 1990.47 | 11127.69 | PASS | 5.5% / 35.5%（双 AIV 重复执行） | 31.2% / 32.4%·22.4GB/s active |
| **final_v9_tuned** | 32.04 (+3.0%†) | **217.65 (−64.2%)** | **642.28 (−67.7%)** | **3886.86 (−65.1%)** | PASS（level all 全过） | 10.8% / aiv0 43.5%·aiv1 43.5%（均衡分片） | 77.5% / ~52%·~52% |

†smoke（8 任务欠载域，固定开销主导）：跨轮复测摆动 30.84–32.04（±2%），+3.0% 恰在噪声阈内（未超过 3%），非必测回退判据不触发；主判别 workload（w2–w4）全域 −64~−68%。

**跨 dispatch 几何平均加速比**（baseline→final，perf_records 对账）：(608.15/217.65 × 1990.47/642.28 × 11127.69/3886.86)^(1/3) = (2.794 × 3.098 × 2.862)^(1/3) ≈ **2.913×**。

### Round 7 — 取景框轮换：w4/w3 稳态画像（第二轮 full，续跑）

> 本轮为第二轮 full 调优（续跑语义），用户指令：以非主判别 workload（w4/w3/smoke）为瓶颈分析取景框。R7 起的工具链状态：tilelang `0.1.2+1990aa9fe4` / CANN 8.5.0 / 910B2C 不变。

**现象（w4 取景框，current best final profile 深度解析——第一轮从未做过非 w2 画像）**：

- w4（3886.86µs，16384 任务，每核 683 串行——纯稳态）：Cube **mte2 85.0% 忙比**（3318µs，12978 条指令，~256ns/条 = 64 段 × ~4ns/128B，段传输主导）+ scalar 63.5%；**cube_wait 0.919 / mte1_wait 0.878**（Cube 计算与 L1→L0 装载 9 成时间在等数据供应）；GM_to_L1 带宽利用率仅 18%（非带宽墙，段数/指令墙）。AIV：vec 62.2% + scalar 46.5% + mte2 47.7%（wait 0.62）+ mte3 24.7%——**非关键路径**。
- w4 稳态 mte2 85% vs w2 瞬态 71.9%：w2 的 32 任务/核处于流水爬坡区，**w2 低估了 Cube mte2 的真实占比**。
- w3（642.28µs，107 任务/核，bf16）：与 w4 高度同构（Cube mte2 81%、scalar 60%；AIV vec 62%）——无 bf16 特有结构瓶颈。
- 段数核算（每任务 Cube mte2）：ws_lcb band 重组 640 段（Σ(lt+1)=10 块）+ x 256 + ws_c 128 + prev 128 ≈ 1216 段。GQA 读放大（H/G=64/80/48）由 L2 吸收（cube read_hit 91%、AIV 99%）。

**候选优化点**（基于各自 shape 现象，不复用 w2 结论）：

| opt_id | 目标现象 | 优化点 | 判定 |
|---|---|---|---|
| OP1 | Cube mte2 85%（band 重组 640 段冗余） | 增量 band 组装：band 嵌套包含 → L1 跨 lt 累积，每 lt 只搬 diag 块（640→256 段） | **invalid（理论错误）**：band 行绑定 l-tile（块 (lt,s_blk) 内容 = lcb[l0+i, s0+j]，不同 lt 行内容不同），band(lt) 与 band(lt−1) 列前缀内容不同、无公共前缀可累积——L_tiles=1 全过 / L_tiles≥2 全挂（L0-2/L0-5 FAIL max_diff 5.9e-3/7.9e-3）证实。v10 分支 |
| OP2 | AIV vec 62%（dA_l 广播每 s_blk 重做） | dA_l_mat vbrc hoist + vsub 输出重定向 dA_s_mat（原地） | **rollback**：vsub dst=src2 alias 形态税——同口径（bn=64）w2 +19.6% / w3 +17.0% / w4 +2.0%（w2/w3 AIV 接近临界时炸、w4 稳态 AIV 远离临界时轻）。与 R2 v4_p2 广播形态税同族：**Bisheng v-op 非典型形态（广播/alias）可编译可算对但有税**。v11 分支 |
| OP3 | mte2 85% 忙比 vs 稳态理论 ~95%（~10% 任务间气泡假设） | 深度 3 任务流水（ws 三槽 + 6 flag ≤15） | **config_no_gain**：同口径 w2 +0.1% / w3 −2.7% / w4 +1.4% 平区。副作用发现：3 任务 in-flight 的 ws 工作集 9.2→13.8MB 使 mte2 每条 256→346ns（busy 85%→93% 但更慢）——**mte2 指令时间对内存系统压力敏感，深度 2 是 L2 局部性甜点**。v12 分支 |
| OP4 | flag wait 期间 mte2 空闲 | x 预取：x 不依赖 ws flag，pp=0 的 x copy 提到 wait 前（pp range 展开） | **config_no_gain**：同口径 w2 −0.7% / w3 −5.2%（单次）——**ab_test 交错协议裁决 tie**（+0.48%，p=0.25；A/B 各 3 run 中位 768.36 vs 772.08，`ab_round7_w3/`）。wait 间隙在稳态下 ≈0（AIV 快于 Cube）。v13 分支 |

**候选 vs current best 对比表（R7，msprof op Task Duration(us)，median of 20；⚠ 本轮四分支为 bn=64 口径——见下方口径断裂记录——对比基准用同 session 同口径的 v9_recheck）**：

| 候选 | smoke | w2 | w3 | w4 | L0 | w4 AICore（cube mte2 / aiv vec） | w4 memory（cube mte2 忙比·每条ns） | 结论 |
|---|---:|---:|---:|---:|---|---|---|---|
| v9_recheck（bn=64 基准） | 30.77 | 258.86 | 808.12 | 4695.16 | PASS | 14.5% / 62.2% | 85.0%·256ns | — |
| v10_incband | — | — | — | — | **FAIL** | — | — | **invalid**（OP1 理论错误） |
| v11_hoistda | 31.88 | 309.55 (+19.6%) | 945.54 (+17.0%) | 4788.32 (+2.0%) | PASS | 13.2% / ~39%* | 93.6%·339ns（mte2 每条劣化） | **rollback**（alias 税） |
| v12_pipe3 | 32.17 | 259.18 (+0.1%) | 785.98 (−2.7%) | 4762.92 (+1.4%) | PASS | 13.0% / ~35%* | **93.0%·346ns（L2 压力劣化）** | config_no_gain |
| v13_xprefetch | 31.96 | 256.99 (−0.7%) | 766.15 (−5.2%*) | 4737.23 (+0.9%) | PASS | ~14% / ~62% | ~85%·~256ns | config_no_gain（*ab_test tie） |

*v11/v12 的 aiv vec 与 w2 非同类项（w4 口径）；v13 w3 的 −5.2% 经 ab_test 交错协议（3 对 A/B/A/B，launch-count=15）裁决为 tie——单次差异是 run 状态摆动。

**⭐ 口径断裂发现与修复（R7 最重要产出）**：本轮续跑走 `run_experiments.py` 默认参数（不传 `--block-n`），bench.py 旧默认回落 `bn=min(64,N)`（=64），而第一轮 R3 起 winner 采集全部**显式传 `--block-n 128`**（opt_log §1 采集命令模板漏记该参数——模板与实际执行不符的文档缺陷）。后果：v9_recheck（bn=64）= 258.86µs vs final 记录（bn=128）= 217.65µs，差 +18.9%——**初判为"设备漂移 ±20%"，实为配置口径断裂**（证据：修复后重测 v9@bn=128 = 219.23µs 与早上记录 217.65µs 同分布 <1%；smoke 两种口径均在 30.75–32.17 平区）。**修复**：bench.py 的 bn 解析改为 `显式 --block-n > kernel 模块 TUNED_DEFAULT_CONFIG["block_n"]（交付配置）> min(64,N)（baseline 兼容）`，杜绝续跑/翻转后口径再断。**量化**（w2，同 session 探针 `profiles/round7_bn_check/`）：bn=64 → 261.55µs，bn=128 → 226.26µs（n-loop 2 块的 2 条 mte2 + 1 gemm/任务代价 −13.5%）。

**R7 校准基准（current best 在交付配置 bn=128 下的当前 session 数据，round 7 追加 4 行 `v9_bn128tuned`）**：

| workload | final 记录（round 6, bn=128） | v9_bn128tuned（round 7, bn=128） | 一致性 |
|---|---:|---:|---|
| smoke | 32.04 | 30.75 | 平区摆动 |
| w2-780m-s4k | 217.65 | 219.23 | +0.7% 同分布 |
| w3-2p7b-s2k | 642.28 | 631.45 | −1.7% 同分布 |
| w4-1p3b-s32k | 3886.86 | 3878.39 | −0.2% 同分布 |

**必测 Dispatch 非回退检查**：本轮无 winner——四分支均未通过（1 invalid / 1 rollback / 2 no_gain），current best 保持 v9 结构 + TUNED_DEFAULT_CONFIG 不变。

**R7 结论**：w4/w3 取景框暴露的稳态瓶颈（Cube mte2 段数墙）在四个方向上被实测否决——①band 段数削减被数学事实否决（band 行绑定 lt，无嵌套复用）；②深度 3 被 L2 局部性否决（ws 工作集膨胀劣化每条 mte2）；③AIV 减负无墙钟收益（AIV 非关键路径）且 alias 形态有税；④x 预取无间隙可填（稳态 wait ≈0）。**mte2 剩余 idle（15%）的最后假设成因 = L1 单 buffer 串行依赖（gemm 期间无法预取下一块）→ R8 做软件流水双缓冲验证**。预算：R7 用 4 实验分支（4/20）、1 轮（1/4）。

### Round 8 — L1 双缓冲软件流水（mte2 idle 的最后假设成因）

**现象**（承接 R7）：mte2 剩余 ~15% idle 的最后假设成因 = L1 单 buffer 串行依赖（band copy(lt) → gemm(lt) → copy(lt+1) 顺序执行，gemm 期间 mte2 无法预取）。

**候选优化点（OP5）**：v14_l1dbuf——l1_c/l1_state/l1_lcb 双槽化（_a/_b 独立 buffer，L1 占用 96→160KB < 512KB），lt 循环 Python range 展开（编译期常量）+ prologue 装 lt=0 → 循环内先发射 lt+1 的全部 GM→L1 载入（写另一槽）、后执行 lt 的 history/band gemm（读本槽）——MTE2 与 CUBE 管线在数据流上无依赖，静态顺序允许硬件并行。tuned 配置断言 N_tiles==1（bn≥N 恒成立）。Vector 侧不动。

**候选 vs current best 对比表（R8，msprof op Task Duration(us)，median of 20，bn=128 交付口径）**：

| 候选 | smoke | w2 | w3 | w4 | L0 | w4 AICore（cube / scalar / mte1 / mte2） | w4 memory（mte2 忙比·传输时间） | w4 冲突（mte2_wait） | 结论 |
|---|---:|---:|---:|---:|---|---|---|---|---|
| v9_bn128tuned (best) | 30.75 | 219.23 | 631.45 | 3878.39 | PASS | 14.7% / 61.5% / 22.6% / 83.5% | 83.5%·3217µs | 0.864 | — |
| v14_l1dbuf | 30.46 | 245.31 (+11.9%) | 744.55 (+17.9%) | 4606.02 (+18.8%) | PASS（level all 19/19） | 12.3% / 49.5% / 18.0% / **66.3%** | **66.3%·3056µs（传输确有减少）** | **0.976** | **rollback（family blocked）** |

**机制归因**（v14 vs v9，w4 cube0）：双缓冲确实压缩了 mte2 的纯传输时间（3217→3056µs，idle 被填），但 **mte2_wait 从 0.864 恶化到 0.976、cube_wait cycles +18%**——prefetch 的 MTE2 写（L1 _b 槽）与 gemm 操作数装载的 MTE1 读（L1 _a 槽）**在 L1 端口/带宽层竞争**，两条管线互相拖慢的代价（~+700µs）超过空闲回收收益（~160µs）。**判定：Ascend910B2C 此 BiSheng 调度下 L1 双缓冲软件流水 family blocked（硬件/调度层约束），非代码正确性问题**。

**必测 Dispatch 非回退检查**：v14 在 w2/w3/w4 全部显著回退（+12~19%）→ rollback，current best 不变。

**R8 结论**：L1 串行假设实测否决。**w4/w3 取景框的五个候选方向至此全部实测穷尽**（band 段数削减=数学否决 / 深度 3=L2 否决 / AIV 减负=非关键+alias 税 / x 预取=无间隙 / L1 双缓冲=端口竞争）。**Cube mte2 段数墙（1216 段/任务 × ~4ns/128B ≈ 3217µs @w4，85% 忙比）为当前冻结设计族（Expert 单 kernel 全融合 + Vector→GM ws→Cube 中继 + band 行绑定 lt）的实测地板**。smoke 欠载域复核：24 核 8 任务场景下空核提前退出不拖尾部、深度 2 流水无收益也无害（prologue set 立即通过）、NUM_KERNELS 动态化的 Task Duration 收益 ≈ 0（执行时间由每核 1 任务决定）——按用户指令如实记录、不强行优化。

## 3. Autotune Log

未使用 autotune（结构性优化主导；唯一参数维 block_n 经 v6 分支直接 A/B 实测裁决，bl/bs/bp 受 UB 容量锁定 64——见 §7；R7/R8 附加：深度 3 / x 预取 / L1 双缓冲 / band 增量组装 / vbrc hoist 等结构候选见 §5 清单与 R7/R8 记录）。

## 4. 结构演化链（获胜路径）

```
baseline (ping-pong 串行, 608µs w2)
  └─ v1_pipe: 深度2任务流水 (ws双槽+4flag+Cube前导set)      → 367µs  (−39.6%)
      └─ v5_xband: Cube band 组装 + x 任务级 L1 缓存        → 361µs  (−1.8%)
          └─ v6_bn128: block_n=128 (n-loop 单块)            → 323µs  (−10.5%)
              └─ v8_aivsplit: subid 分片 (消 AIV 重复执行)   → 212µs  (−34.4%)
                  └─ v9_snake: 蛇形均衡分配 (ab_test tie 采纳) → 218µs* (tie)
                      └─ final_v9_tuned (+tuned默认+尾块修复) → 217.65µs (复测)
                          └─ R7-R8 二轮调优: 五方向实测穷尽, 结构保持 → 219.23µs (bn128 校准复测)
```
*v9 单次 w2 记录 220.58µs；final 复测 217.65µs（同分布）。各轮 winner 均通过必测 dispatch 非回退检查（smoke 全程噪声平区）。

## 5. 残余瓶颈与 blocked 候选清单（Phase 3 天花板证据）

final 画像（w2, ~218µs）：Cube mte2 ~156µs（16 nd2nz 指令/任务，~4ns/128B 段，段数 = ws_lcb 640 + ws_c 256 + x 256 + prev 64 段/任务）为第一项；Cube scalar（地址算术+自旋）~120µs；AIV1 侧 vec+mte2 ~110µs 次临界。

**第二轮（R7/R8，w4/w3 取景框）稳态画像补充**（w4, 3878µs）：Cube mte2 83.5% 忙比（3217µs，1216 段/任务 × ~4ns —— **段数墙**；w2 瞬态 72% 低估了该占比）；cube_wait 0.919 / mte1_wait 0.908（数据供应饥饿）；AIV vec 62% 非关键。已识别但 **blocked** 的候选：

| 候选 | 阻塞点 | 证据 |
|---|---|---|
| bl=128 大 l-tile（DESIGN §1.6.3 变体 B） | **UB 容量** | 编译实测：bs=128 需 4,769,792 bits（582KB）、bs=64 需 2,929,664 bits（357KB）vs 192KB 可用（`logs/bl128_probe.log` / `logs/bl128_bs64_probe.log`）——容量否决 |
| [64,128] 成对向量链（s-block 数减半，同时降 Vector 发射与 Cube ws 读段数） | **UB 容量** | 手工 liveness 峰值 ~199KB > 192KB（cb_16x 16 + cb_f32x 32 + dA_l_src 32 + dA_s_mat 32 + dt_mat 32 + lcb_16 16 + pen_mix 32 + 杂项 ~7）；v3 实测证明 Bisheng UB 记账与手工核算偏差 <2%（+16KB 手工 → 195.1KB 实测报错），无通胀余量 |
| task 级 dA/dt 行装载（省 12 次 tiny GM 读/任务） | **编译器** | 动态偏移 UB subview 进嵌套循环 → auto-multi-buffer pass 产出非支配 IR（v3 复现，`logs/round1/v3_hoist_L0.log`）|
| 输出行缩放消 ws_c 中继（L0C→UB→GM 路径） | **硬件/API** | CONST-store-fixpipe-gm-only：L0C 跨引擎仅 GM 往返；中继改道引入等量 GM 流量 + 第三握手相位，净亏 |
| history M=256 合并 gemm（1 读 1 gemm/任务） | **API** | gemm dst 不支持 L0C region 写（单 [Q,bp] acc 与 per-lt band gemm 的 [64,64] dst 不可共用；两 L0C acc 相加无原语）|
| vsub/vmul 列广播形态 | **性能税** | v4_p2 实测 w2 +12.2% / w3 +16.0%（可编译可算对但回退）|
| fp16 c 链（v7） | **dtype 契约** | `T.vmul` bf16 ×；需 trace 分派；且收益 <5% 未过 T-2、w4 方向翻转——弃置 |
| host 侧因子预折（如 prev·exp(dA[0])） | **口径纪律** | 将 kernel 工作外移到未测量 host 算子（指标 gaming），不采纳 |
| **band 增量组装**（嵌套包含 → L1 跨 lt 累积，640→256 段）〔R7-v10〕 | **数学事实** | **band 行绑定 l-tile**：块 (lt,s_blk) 内容 = lcb[l0+i, s0+j]，不同 lt 的行内容不同（dA_l 依赖 l），band(lt) 与 band(lt−1) 列前缀无公共可复用内容——L0-2/L0-5 FAIL（max_diff 5.9e-3/7.9e-3），L_tiles=1 全过恰好证实 |
| **vsub dst=src2 alias 形态**（vbrc hoist 的代价）〔R7-v11〕 | **性能税** | 同口径实测 w2 +19.6% / w3 +17.0% / w4 +2.0%（alias 形态生成次优代码，与 v4_p2 广播税同族：Bisheng v-op 非典型形态可编译可算对但有税） |
| **深度 3 任务流水**（ws 三槽 + 6 flag）〔R7-v12〕 | **L2 局部性** | 同口径 w2 +0.1% / w3 −2.7% / w4 +1.4% 平区；3 任务 in-flight 的 ws 工作集 9.2→13.8MB 使 mte2 每条 256→346ns（busy 85%→93% 但更慢）——**深度 2 是 L2 局部性甜点** |
| **x 预取**（x copy 提到 flag wait 前）〔R7-v13〕 | **无间隙可填** | 同口径 w2 −0.7% / w4 +0.9%（ab_test 交错协议裁决 w3 tie：+0.48%, p=0.25）——稳态下 AIV 快于 Cube，wait ≈ 0 |
| **L1 双缓冲软件流水**（_a/_b 槽 + lt 展开 + prefetch 先行）〔R8-v14〕 | **L1 端口竞争** | 同口径 w2 +11.9% / w3 +17.9% / w4 +18.8%：mte2 纯传输确有减少（3217→3056µs）但 **mte2_wait 0.864→0.976**——prefetch 的 MTE2 写（_b 槽）与 gemm 操作数装载的 MTE1 读（_a 槽）在 L1 端口层互拖，代价 +700µs > 收益 ~160µs |

**判定：剩余候选均因精度/编译/容量/API/数学/性能税/L2·L1 端口/纪律阻塞 → stop_reason = blocked**（两轮累计 budget：15 实验、8 轮均未耗尽；R7/R8 连续 2 轮无 >3% valid 提升，w4/w3/smoke 取景框候选全部实测穷尽——含 band 段数削减的全部数学变体与 mte2 idle 的全部三个假设成因）。

## 6. 最终版本正确性修复记录（收束期发现）

1. **L1 band 组装尾块越界（B-q96-bf16 回归）**：band 组装 copy 的 dst 列区间写 `s0:s0+bs`，Q%bs≠0 时越出 `l1_lcb [bl,Q]` 缓冲（Q=96 末块 [64:128] vs 缓冲 96 列）——越界写污染相邻 L1 分配。fp16 同 shape 靠 L1 布局运气通过（L0-5 PASS），bf16 布局命中关键 buffer（max_diff 7.8e-3）。**修复 = 列宽按 `ts = min(bs, Q−s0)` 裁剪**（src/dst 同步）。该缺陷自 v5_xband 起潜伏；manifest workload 全整除（Q=256/64 % 64 = 0）故全部 perf 数据不受影响；修复后全量 level=all 通过。
2. **block_n 分形 stale 带（L0-5, N=48/bn=128）**：`l1_c/l1_state` 直接按 bn=128 分配而 K 实宽 48——stale 分形列带泄漏进 gemm（max_diff 7.4e-3）。**修复 = 工厂层 `bn = min(block_n, N)`**（N=128 仍单块快路径；N<128 回退基线行为）。注：首轮修复后因 tilelang **磁盘缓存**命中未重编译（同 max_diff 复现）——清 `~/.tilelang/cache` 后生效（排障留痕）。

## 7. DESIGN §1.6.3 实验裁决记录（Stage 4 必做项）

| 裁决项 | 判定 | 证据 |
|---|---|---|
| ① 掩码变体 A/B（惩罚掩码 vs vselect） | **惩罚掩码维持主选** | 裁决计划针对"Developer 模式 vcmp 是否标量化待复证"；本任务为 Expert 模式，PL-1.11 的 4.4–14.2× 实测即为 Expert 形态（GQA），直接适用，无需复证。全程未引入 vcmp/vselect，零标量化税。 |
| ② bl=128 tiling 变体 | **容量否决（不采纳）** | 编译实测 UB 需求 582KB（bs=128）/ 357KB（bs=64）vs 192KB——Expert 直差分链结构下 bl=128 不可行；DESIGN §1.6.3 的"UB 膨胀系数未定"以实测常数定格（显式 alloc 结构实际 ≈ 手工核算，无 1.7× 通胀）。 |
| 裁决常数回填 | — | bl=64 链手工峰值 ~179KB（v3 实测反推）≈ 192KB 上限的 93%；`×1.10–1.12` 通胀系数在本显式 alloc 结构不必然成立（与 PL-1.12 r9 结论一致）。 |

## 8. Final Summary

> 含第二轮（R7–R8，w4/w3/smoke 取景框）结果。本轮结构未变（五方向实测穷尽全败），final = 第一轮 v9 结构 + TUNED_DEFAULT_CONFIG；三列对比口径：baseline（Stage 3 原版）/ 第一轮后 v9（round 6 `final_v9_tuned`，bn=128）/ 本轮 final 校准复测（round 7 `v9_bn128tuned`，bn=128，同 session；与 round 6 同分布 <±1.7%）。

- **best 版本**：`perf_opt/_ssd_chunk_scan_fwd_kernel.py`（= v9_snake 结构 + TUNED_DEFAULT_CONFIG(block_l=64, block_p=64, block_n=128, block_s=64, num_stages=2) + §6 两修复）——第二轮未变更。
- 结构变更四件套（第一轮）：①深度 2 任务流水（ws 双槽 + 4 flag + Cube 前导 set）；②Cube 侧因果 band 组装 + x 任务级 L1 驻留 + 每 lt 单 band gemm（ws 写保持块连续）；③AIV `subid` 蛇形分片（消双 AIV 重复执行）；④tuned block_n=128。第二轮新增：⑤bench 口径修复（bn 解析对齐 TUNED_DEFAULT_CONFIG，防续跑口径断裂）。
- **三列对比表（全部 msprof op Task Duration(us)，median of 20，perf_records 可对账）**：

| workload | baseline（round 0） | 第一轮后 v9（round 6, bn=128） | 本轮 final（round 7 `v9_bn128tuned`, bn=128） | 本轮 ratio vs baseline | 本轮 vs 第一轮 v9 |
|---|---:|---:|---:|---:|---:|
| smoke (1,2,64,4,64,32,1) fp16 | 31.11 | 32.04 | 30.75 | 1.01×（欠载域平区） | −4.0%（噪声摆动 30.75–32.17） |
| w2-780m-s4k (1,16,256,48,64,128,1) fp16 | 608.15 | 217.65 | **219.23** | **2.77×** | +0.7%（同分布） |
| w3-2p7b-s2k (4,8,256,80,64,128,1) bf16 | 1990.47 | 642.28 | **631.45** | **3.15×** | −1.7%（同分布） |
| w4-1p3b-s32k (2,128,256,64,64,128,1) fp16 | 11127.69 | 3886.86 | **3878.39** | **2.87×** | −0.2%（同分布） |
| **几何平均（3 真实 workload）** | 1.00× | 2.91× | **2.93×** | — | +0.4%（结构未变） |

- final_latency: 219.23 us（主判别 workload w2-780m-s4k 口径，msprof op Task Duration，median of 20，bn=128 交付配置；与 perf_records.jsonl round 7 `v9_bn128tuned` 的 w2 行精确对账，偏差 0%）
- **final_latency（跨 workload 汇总口径，每 workload 一行 duration + ratio）**：
  - smoke: 30.75 us（1.01×）
  - w2-780m-s4k: 219.23 us（2.77×）
  - w3-2p7b-s2k: 631.45 us（3.15×）
  - w4-1p3b-s32k: 3878.39 us（2.87×）
  - **几何平均（3 真实 workload）加速比: 2.93×**；（4 workload 含 smoke 几何平均时长 358.45 us / 几何平均加速比 2.25×，仅作汇总参考——smoke 欠载域拉低均值）
  - 各 workload 独立可对账 round 7 `v9_bn128tuned` 四行，偏差 0%；主判别 workload 参考（承接第一轮口径）：w2-780m-s4k = 219.23 us（round 6 记录 217.65 us，同分布）
- 精度：level=all 全过（L0 5 + contract / L1 6 / L2 4 / Boundary 4，v14 分支亦 19/19）；w2/w3/w4 bench --check max_diff 与 baseline 同分布（1.3e-4 / 1.2e-3(bf16) / 1.9e-4 vs atol 1e-3/2e-3）。
- **中止原因：blocked**——两轮累计：第一轮 §5 清单（UB 容量 / 编译器 dominance / API 硬件 / 广播税 / 口径纪律）+ 第二轮五方向实测穷尽（band 增量组装=数学否决【band 行绑定 lt】、深度 3=L2 局部性劣化、AIV vbrc hoist=alias 税 + AIV 非关键路径、x 预取=稳态无间隙、L1 双缓冲=L1 端口竞争倒贴）。budget 余量充足（两轮累计 8/10+4 轮、15+7/30 实验）。**w4 稳态画像确认 Cube mte2 段数墙（1216 段/任务 × ~4ns/128B，83.5% 忙比）为当前冻结设计族（Expert 单 kernel 全融合 + Vector→GM ws→Cube 中继）的实测地板**——参照结构 diff（T-1）：separable 两-kernel（intra/inter 分离）可减少 ws 中继段数，但属设计层结构变更（推翻 DESIGN 融合决策）且两 kernel 总时延不在单 kernel msprof 口径内可比，作为设计层观察记录、不触发 perf_feedback（结构性加速 >2x 无实证支撑——separable 方案的因子计算仍需 Vector 中继，收益不明确）。
- **按 shape 分派建议（S4-5 出口）**：单一 trace 全域占优（无分档收益反转），无需 wrapper 层 shape 分派表；bench.py 已修复为读 kernel 的 TUNED_DEFAULT_CONFIG（翻转 wrapper 后测量口径自动对齐交付配置）。
- **第二轮产出清单**：①bench 口径断裂发现与修复（R7，w2 bn=64 vs 128 实测 +18.9%——续跑/翻转场景的口径风险）；②w4/w3 稳态画像与 mte2 段数墙定量（1216 段/任务、~4ns/128B、L2 91% 命中）；③五个结构候选的实测否决证据（§5 新增五行的机制归因）；④smoke 欠载域无可做优化（如实记录）。

## 9. Skill Retrospective

### 流程观察

1. **「双 AIV 重复执行」是本轮最大单点发现，但发现它花了 4 轮**——前 3 轮我把 vector0/vector1 的相同指标读作"分摊或镜像"而未深究；直到 R4 才用"若分摊则 ratio 应减半"的反证确认重复。**建议**：MixCV 族的 Phase 1 画像步骤增加一条机械检查——`Mix Block Dim = 2×Block Dim` 时核对两 AIV 子块指标是否相同（相同 ⟹ 重复执行 ⟹ subid 分片是头号候选）。〔vp_type: R〕
2. **L0 分支验证覆盖不足导致 band 尾块越界潜伏 4 轮**：v5 引入的 `s0:s0+bs` 越界仅在 bf16+Q=96（Boundary 用例）触发；分支验证只跑 L0（fp16 Q=96 恰好靠布局运气通过）。**建议**：实验分支的精度回归应至少包含一个 bf16 × 非整除 shape 的组合（或直接跑 Boundary 层）。〔vp_type: R〕
3. **tilelang 磁盘缓存掩盖修复生效**：同参数同源码路径的修复在缓存命中下不重编译，复现"修复无效"假象。**建议**：分支验证脚本涉及 kernel 源变更时先清 `~/.tilelang/cache`（或 bump cache key）。〔vp_type: R〕
4. ab_test.py 的 msprof 输出目录权限问题（group-writable 拒采）需要预创建 pair 目录——工具可自行 `chmod 700`。〔vp_type: R〕

### 价值点（vp_type + 证据三件套，供 evolver 蒸馏；D/C 类已任务内回写 pattern-library，见回写位置）

| # | 价值点 | vp_type | evidence | repro | toolchain_stamp | 回写位置 |
|---|---|---|---|---|---|---|
| 1 | **Mix kernel 双 AIV 默认重复执行 Vector 程序；`subid` 边界表达式分片（GQA v11 形态推广到 l-tile 交错/蛇形）实测 −28~−36%**——AIV 产能 2× 释放 | D | 本 log R4/R5 + perf_records round 4/5；profile 反证（两 AIV ratio 相同 vs 分摊应减半） | `repro/PL-1.13-aiv-dup-subid-split.py` | tilelang 0.1.2+1990aa9fe4 / CANN 8.5.0 / 910B2C | pattern-library/attention.md PL-1.13 |
| 2 | **深度 2 任务流水的"消费侧前导 set"技巧**：Cube 循环前 set 双槽 cons flag（表达"初始空闲"），免去 Vector 侧 T<2 运行时分支——4 flag 严格交替、无死锁（比 PL-1.12 的 prologue 重排更轻量的同族形态） | P | 本 log R1；v1_pipe 代码 `perf_opt/_ssd_chunk_scan_fwd_kernel.py` Cube 前导 | 同上 repro | 同上 | pattern-library/attention.md PL-1.13 补充 |
| 3 | **ws 中继布局铁律（MTE3 侧）**：Vector→Cube 因子中继的 GM ws 必须保持块连续布局；band 化（行 stride 512B）使 AIV MTE3 时间 3×（106→220µs），Vector 侧损失吞掉全部 Cube 侧收益 | D | 本 log R1 v2_band 行；profile 对比 v2_band vs baseline aiv_mte3 | 同上 repro | 同上 | pattern-library/elementwise.md（MTE3 跨步写条目） |
| 4 | **vsub/vmul 未文档化列广播形态可编译可算对但有 ~12–16% 性能税**（w2/w3 实测）——证伪协议反向案例：合法化不等于可用 | D | 本 log R2 v4_p2 行；perf_records round 2 | `repro/PL-1.13-aiv-dup-subid-split.py`（含 vsub 广播 A/B 微基准段） | 同上 | pattern-library/layout.md（广播形态速查补遗） |
| 5 | **task 级 UB 行 buffer + 嵌套循环动态偏移 subview → BiSheng auto-multi-buffer 非支配 IR**（"operand does not dominate this use"）——UB→UB 切片消费需避开该形态 | D | 本 log R1 v3 行；`logs/round1/v3_hoist_L0.log` | 同上 repro | 同上 | pattern-library/traps-compiler.md |
| 6 | **L1 band 组装 dst 列区间必须按尾块裁剪**（`s0:s0+ts` 而非 `s0:s0+bs`）——越界写相邻 L1 的触发依 L1 布局而变（fp16 靠运气通过、bf16 必现），属布局敏感潜伏缺陷 | D | 本 log §6.1；B-q96-bf16 修复前后 | 同上 repro | 同上 | pattern-library/traps-runtime.md |
| 7 | MTE 指令代价口径：nd2nz GM→L1 ~300ns/指令、~4ns/128B 段（w2 标定）；AIV 重复执行时双 AIV 指标镜像可用于判定执行模式 | D | 本 log §5 段数核算 + 各轮 profile | 同上 repro | 同上 | pattern-library/constants.md |
| 8 | BN 流程：分支 L0 回归应含 bf16×非整除 shape 组合；tilelang 磁盘缓存与源变更的失效纪律 | R | 本 log §9.2/9.3 | — | 同上 | proposal（不改 skill 文档） |

### BP proposal

- **BP_aiv_duplication_check**（新）：MixCV（Mix Block Dim = 2× Block Dim）画像出现两 AIV 子块指标相同 → 判定 Vector 程序重复执行 → subid 分片为第一候选（本轮 −34% 单点）。证据：本 log R4；建议入 bottleneck-patterns.md。

### 第二轮（R7–R8）追加流程观察

1. **续跑口径断裂是本轮最大流程风险**：第一轮 opt_log 的采集命令模板漏记 `--block-n 128`（实际执行的显式参数），第二轮续跑走 runner 默认（bn=64）后所有分支数据系统性偏慢 +19~21%，初判为"设备漂移 ±20%"，经对照实验（bn=64/128 同 session A/B）才定位为配置口径断裂。**建议**：①调优日志的采集命令模板必须与实际执行完全一致（含全部显式参数）；②bench harness 的默认配置应从 kernel 模块的 TUNED 常量解析（本轮已修复 bench.py），使"交付配置=测量配置"成为机械保证而非纪律约定；③跨 session/续跑对比前先做 current best 的同 session 重测校准（本轮 v9_recheck/v9_bn128tuned 两组基准行即此用途）。〔vp_type: R〕
2. **取景框轮换的价值实证**：以 w4（683 任务/核纯稳态）为取景框暴露了 w2（32 任务/核瞬态）不可见的 mte2 83.5% 段数墙与 AIV 非关键事实；但五个候选方向全部实测否决也说明——**稳态画像暴露的"新瓶颈"不必然存在"可打的空间"**（段数是中继结构的物理流量）。结构候选的可行性受数学事实（band 行绑定）、L2 局部性（工作集）、L1 端口（读写竞争）三重硬件/数学约束夹击，逐一实测是唯一裁决方式。〔vp_type: R〕
3. **v10 教训（L0 抓住理论错误）**："band 嵌套包含"的数学直觉在 L_tiles=1 case 全过、L_tiles≥2 全挂——**结构性改动前的逐 case 数学走查（尤其退化/尾块维度）不可省**；L0 分层用例的形状覆盖（Q=64/96/128/256）恰好构成对 L_tiles 语义的自然枚举。〔vp_type: R〕

### 第二轮价值点（vp_type + 证据三件套，供 evolver 蒸馏；D 类已任务内回写 pattern-library，见回写位置）

| # | 价值点 | vp_type | evidence | repro | toolchain_stamp | 回写位置 |
|---|---|---|---|---|---|---|
| 9 | **bench 口径断裂实测**：续跑走 runner 默认 bn=64 vs 交付配置 bn=128，w2 同 session 实测 261.55 vs 226.26µs（+15.6%）；修复 = bench 默认从 kernel TUNED_DEFAULT_CONFIG 解析（baseline 无常量时 fallback 兼容） | D | 本 log R7 口径断裂段 + `profiles/round7_bn_check/`；perf_records round 7 v9_recheck（bn=64）vs v9_bn128tuned（bn=128）两组基准行 | repro-missing（双 bn A/B 最小化待回填） | tilelang 0.1.2+1990aa9fe4 / CANN 8.5.0 / 910B2C | pattern-library/traps-runtime.md TRAP-BENCH-CONFIG-CALIBRATION |
| 10 | **band 行绑定 l-tile 的数学事实**：chunked SSD 因果 band (lt, s_blk) 块内容 = lcb[l0+i, s0+j]（行内容随 lt 变化），band(lt) 与 band(lt−1) 列前缀无公共可复用内容——"嵌套包含增量组装"类优化对该结构数学不可行（L0 实测 L_tiles≥2 全挂） | D | 本 log R7 v10 行；`logs/round7/v10_incband_level_all.log`（L0-2/L0-5 FAIL 5.9e-3/7.9e-3，L_tiles=1 全过） | repro-missing（分支文件 `_opt_v10_incband.py` 为完整对照） | 同上 | pattern-library/attention.md PL-1.18 |
| 11 | **L1 双缓冲软件流水倒贴**（_a/_b 槽 + prefetch 先行 + lt 展开）：mte2 纯传输 3217→3056µs（idle 确被填）但 mte2_wait 0.864→0.976——prefetch 的 MTE2 写与 gemm 操作数的 MTE1 读在 L1 端口层竞争，+700µs 代价 > ~160µs 收益，w2-w4 全回退 +12~19% | D | 本 log R8 v14 行；`profiles/round8/v14_l1dbuf_w4-1p3b-s32k/`（PipeUtilization + ResourceConflictRatio 对比 v9_bn128tuned） | repro-missing（分支文件 `_opt_v14_l1dbuf.py` 为完整对照） | 同上 | pattern-library/attention.md PL-1.18 |
| 12 | **深度 3 任务流水的 L2 局部性劣化**：ws 三槽使 in-flight 工作集 9.2→13.8MB，mte2 每条 256→346ns（busy 85%→93% 但总时间更长）——任务流水深度存在 L2 局部性甜点（本结构深度 2 最优） | D | 本 log R7 v12 行；`profiles/round7/v12_pipe3_w4-1p3b-s32k/`（mte2_t 4491µs vs v9 3217µs） | repro-missing（分支文件 `_opt_v12_pipe3.py` 为完整对照） | 同上 | pattern-library/attention.md PL-1.18 |
| 13 | **vsub dst=src2 alias 形态税**（+17~20% w2/w3）：vbrc hoist 需 vsub 落到 alias 目标——Bisheng v-op 非典型形态（广播/alias）可编译可算对但有税（与 v4_p2 广播税同族，第二实证；税与引擎空闲余量负相关） | D | 本 log R7 v11 行；perf_records round 7（同口径对比 v9_recheck） | repro-missing（并入既有 v-op 形态 A/B 段） | 同上 | pattern-library/layout.md PL-1.15（update 并入） |
| 14 | **稳态 vs 瞬态画像差异**：persistent 任务流水 kernel 的短任务串（32/核）低估 Cube mte2 占比（72%），长任务串（683/核）暴露真实稳态（83.5%）；跨 workload 诊断须以最长任务串的画像为准 | D | 本 log R7 现象段（w2/w4 PipeUtilization 对比） | —（profile 数据即证据，`profiles/final/` w2 vs w4） | 同上 | pattern-library/attention.md PL-1.18（条目内方法论段） |
| 15 | BN 流程：续跑/翻转场景的测量口径机械对齐（bench 读 TUNED 常量）；跨 session 对比先做同 session 基准校准 | R | 本 log §9 第二轮观察 1 | — | 同上 | proposal（不改 skill 文档） |

### 第二轮 BP proposal

- **BP_bench_config_calibration**（新）：perf_opt 采数 harness 的 block/tile 默认值必须从被测 kernel 模块的 TUNED 常量解析，禁止 harness 侧硬编码 fallback 成为事实默认；续跑轮次的第一步是 current best 的同 session 重测校准（双配置 A/B 探针优先于"设备漂移"假设）。证据：本 log R7（+19~21% 假回退）；建议入 profile-collection.md。
- **BP_steady_state_frame**（新）：persistent/任务流水 kernel 的瓶颈诊断应取**最长任务串 workload** 的画像为取景框（短串处于流水爬坡瞬态，引擎占比失真）；候选否决证据（L2/L1 端口约束）只在稳态画像下可观测。证据：本 log R7 现象段 + R8 机制归因；建议入 iteration-diagnosis.md。

## 10. 工件清单

- `perf_opt/_ssd_chunk_scan_fwd_kernel.py`（final，tuned 默认内嵌，两轮共用）
- `perf_opt/_ssd_chunk_scan_fwd_kernel_opt_v{1..14}_*.py`（实验分支，按轮留档；v10_incband 为 L0 失败的理论错误分支，留档供复盘）
- `perf_opt/perf_records.jsonl`（append-only，round 0–8，64 行：第一轮 40 行（baseline 4 + v1–v9/final 各分支）+ 第二轮 24 行（round 7：v11/v12/v13 各 4 + v9_recheck 4（bn=64 校准基准）+ v9_bn128tuned 4（bn=128 交付口径基准）；round 8：v14 4；v10 L0 失败未采集无行））
- `perf_opt/profiles/{baseline,round1..round5,final,round7,round8,round7_bn_check}/`（raw msprof op 数据；round 7 的 v11/v12/v13/v9_recheck 行为 bn=64 口径、v9_bn128tuned 为 bn=128 口径，见 R7 口径断裂记录）
- `perf_opt/profiles/ab_round5_w2/`、`perf_opt/ab_round7_w3/`（T-2 交错协议原始数据）
- `perf_opt/logs/`（L0 / level-all / probe / 校准探针 / ab_test 日志，含 `logs/round7/`、`logs/round8/`）
- `perf_opt/bench.py`（R7 修复：bn 解析 = 显式参数 > TUNED_DEFAULT_CONFIG > min(64,N)）/ `perf_opt/run_experiments.py`（采集与批量 runner）
