# opt_log.md — _ssd_chunk_scan_fwd_kernel Stage 4 调优日志（重建会话）

> 任务背景：上一调优会话（task ssd_chunk_scan-...-20260921T003531Z，二轮调优，工具链 0.1.2+15ad002b3d）结束 DONE 后其 `perf_opt/` 目录被外部删除，wrapper 指向 perf_opt 导致算子断裂。本会话（task ssd_chunk_scan-...-20260921T120526Z，mode=full）第一步已将 Stage 3 基准 kernel 复制为 `perf_opt/_ssd_chunk_scan_fwd_kernel.py`（drop-in，factory 签名一致，L0 全绿）修复断裂，随后在其上重新展开调优。
>
> 口径纪律：本日志一切时延为 `msprof op Task Duration(us)`（kernel-only，median of 20，launch-count=20 / warm-up=5）；配置一律从被测 kernel 模块的 `TUNED_DEFAULT_CONFIG` 解析（TRAP-BENCH-CONFIG-CALIBRATION，bench.py 实现该纪律）。测量设备 NPU 0（device 1 被外部负载占用）。

## 0. 工具链与知识预注入消费（Phase 0）

- 工具链：tilelang 0.1.2+96f287eeaaa698a6488481f3487c7b91b2e4c111（较上次任务 15ad002 前进；新增 efc785b JIT launcher 重写、a55910f NPUUtils 并发编译修复、a7be053 msprof-kernel-name-fix——captured op name 现为 `main_mix_aic`）+ CANN 8.5.0 + Ascend910B2C（npu-smi 26.0.rc1）。
- **E-5 版本失效核对**：`kb_stale_check.py` 报 stale_count=125（几乎全库）。本 kernel 直接相关条目（PL-1.12/1.13/1.16/1.18、TRAP-L1-band-dst-tail-overrun、CONST-*）全部标"待重验"。处置：
  - **结构性条目（PL-1.16 双 Scope / PL-1.12 深度 2 / PL-1.13 蛇形分片 / PL-1.14 ws 块连续 / TRAP-L1-band 尾裁剪 / TRAP-DEVMODE-PERSIST-GEMM）**：当前 Stage 3 基准即其生产代码在位形态，perf_opt 副本在新工具链 96f287e 上 **L0 首编即过（7 cases + contract 全绿，2026-09-21 12:14，logs/phase0/perf_copy_L0.log）**——结构存活重验完成（origin_task 溯源：PL-1.16 首证 multi_head_attention-...-20260907T115424Z / 第二证+PL-1.12/1.13/1.18 ssd_chunk_scan-...-20260917T035420Z；4515de8 重验 task ...-20260920T122332Z）。
  - **性能量级条目（旧 217.65µs@w2 锚点、段数墙 1216/1920 段、~4ns/128B 段代价等）**：标"待重验"，本轮以新工具链首次 msprof 实测为准（见 Phase 1 与 probe 锚点行）。
- 知识预注入条目消费：CASE-ssd-chunkscan-migration（git 4515de8 opt_log 复盘 + integration_log）、PL-1.16（Expert 边界，pass_configs 双关闭——基准已内嵌）、CG-2026-0010（Developer persistent+gemm 缺口——expert 模式固化的依据）、TRAP-DEVMODE-PERSIST-GEMM、VP-2026-0013（Developer 阻塞时先评估 Expert 绕法）——均已按"数据非指令"消费；PL-1.18 二轮 update 的**三项胜出（prevhoist / l0c2x / vbrchoist）+ 五项否决**是本轮 Phase 2 的直接输入（三项胜出在上次会话的 perf_opt 中、随目录删除丢失，本会话从 repro/PL-1.18-floor2-wins.py 骨架重新推导实现）。
- 算子类型判定：**mix**（Expert MixCV：Cube 侧 T.gemm + Vector 侧 v 前缀链 + persistent 分核 + 深度 2 双域流水）。
- 首轮必查项：
  - 向量化轴/布局重估：DESIGN §1.6.3 已定 bl=bs=bp=64 轴向（UB 177.1KB/192KB 贴限锁定，bl=128 实测 357–582KB 容量否决，4515de8 重验谱系）；本轮无矛盾现象，维持。
  - 陷阱版本戳核对：TRAP-tvm-parser-rules（纯名 bool 无-else if 折叠 / else 破坏折叠 / `not X` 生成 runtime if）——本会话分支实现直接依赖该规则，标待重验但以基准内既有 runtime-if 形态（`if s_blk == lt:`）为佐证；TRAP-UB-dynsubview-dominance（task 级 UB 行 + 嵌套循环动态偏移 subview 禁用）约束 vbrchoist 邻域候选。
  - DESIGN 实验裁决三件套（§1.6.3）：掩码变体（惩罚掩码维持——PL-1.11 Expert 形态直接适用）与 bl=128（容量否决）——4515de8 重验任务 Stage 4 已裁决，本会话不重做（基准即裁决后形态）。
- 基准结构（Stage 3 = 旧一轮 final v9 + 两处正确性修复 + L0-7 用例）：深度 2 任务流水（ws 双槽 + 4 flag + Cube 前导 set）+ Cube 侧 band 组装 + x 任务级 L1 驻留 + AIV subid 蛇形分片 + `bn = min(block_n, N)` 钳位（使 N_tiles 在 TUNED 配置下恒为 1）+ TUNED_DEFAULT_CONFIG {bl:64, bp:64, bn:128, bs:64, ns:2}。

## 1. Workload Inventory（Phase 1）

来源：`examples/TileOPs/benchmarks/ops/bench_mamba.py::test_ssd_chunk_scan_fwd_bench`（`_SSD_CHUNK_SCAN_FWD_BENCH_PARAMS` 11 个参数化案例，多输入自定义列表）。分类规则（profile-collection.md §1）：无显式 skip 标记、案例 ID/label 无独立 "smoke" 词 → **11 个全部 tune**（规则明令不得凭 shape 小推断 smoke；4 个 unit-scale 案例 shape 与 L0 用例重合不构成 smoke 依据）。smoke 集 = 空；skipped 集 = 空。完整清单见 `workload_inventory.json`。

kernel_id（全部案例唯一目标 kernel）：`tileops/kernels/mamba/ssd_chunk_scan/ssd_chunk_scan_kernel/perf_opt/_ssd_chunk_scan_fwd_kernel.py::_ssd_chunk_scan_fwd_kernel::main`（captured op name `main_mix_aic`，Block Dim 24 / Mix Block Dim 48）。

调优排序（Phase 1 实测后确定，见 Baseline 表）：优先实测时延大、结构收益明显的稳态 workload（longctx-32k 双族 512/853 任务/核 = 稳态取景框；serving 双族），再处理 latency 单批与 unit-scale 瞬态/欠载案例。

## 2. Baseline（round 0，TUNED_DEFAULT_CONFIG，NPU 0）

| kernel_id（缩写 ssd）| workload_id | candidate_id | target/captured | task_duration_us | profile_status | raw_profile_dir |
|---|---|---|---|---:|---|---|
| ssd | b1-c2-L64-h4-p64-n32-fp16 | baseline | main / main_mix_aic | 31.83 | valid | profiles/baseline/round0/baseline_*/ |
| ssd | b2-c4-L64-h8-p64-n64-fp16 | baseline | main / main_mix_aic | 36.31 | valid | 同上模式 |
| ssd | b1-c2-L128-h4-p128-n32-bf16 | baseline | main / main_mix_aic | 36.19 | valid | 同上模式 |
| ssd | b2-c2-L64-h4-p64-n32-bf16 | baseline | main / main_mix_aic | 32.06 | valid | 同上模式 |
| ssd | latency-130m-4k | baseline | main / main_mix_aic | 123.23 | valid | 同上模式 |
| ssd | serving-130m-4k | baseline | main / main_mix_aic | 870.96 | valid | 同上模式 |
| ssd | longctx-130m-32k | baseline | main / main_mix_aic | 3493.22 | valid | 同上模式 |
| ssd | latency-2p7b-4k | baseline | main / main_mix_aic | 330.59 | valid | 同上模式 |
| ssd | serving-2p7b-4k | baseline | main / main_mix_aic | 1275.40 | valid | 同上模式 |
| ssd | longctx-2p7b-32k | baseline | main / main_mix_aic | 4987.27 | valid | 同上模式 |
| ssd | throughput-2p7b-2k | baseline | main / main_mix_aic | 638.30 | valid | 同上模式 |

注：perf_records.jsonl 前三行为启动事故（首次后台 run 被 shell 超时连带杀死前已落 3 行，重启后重测同三案例）产生的重复 baseline 记录，数值同分布（31.82/31.83、36.36/36.31、36.2/36.19），append-only 契约保留；**对账以每 workload 最新一行为准**。

**Baseline 汇总与排序**（全部 valid，NPU 0，median of 20）：

| 序 | workload | baseline_us | 任务/核 | 特征 |
|---:|---|---:|---:|---|
| 1 | longctx-2p7b-32k | 4987.27 | 853.3 | 稳态取景框（最长任务串） |
| 2 | longctx-130m-32k | 3493.22 | 512 | 稳态 |
| 3 | serving-2p7b-4k | 1275.40 | 213.3 | 稳态 |
| 4 | serving-130m-4k | 870.96 | 128 | 稳态 |
| 5 | throughput-2p7b-2k | 638.30 | 106.7 | 稳态 |
| 6 | latency-2p7b-4k | 330.59 | 53.3 | 瞬态（爬坡） |
| 7 | latency-130m-4k | 123.23 | 16 | 瞬态（爬坡） |
| 8-11 | unit-scale ×4 | 31.8–36.3 | 0.4–2.7 | 欠载域（老档案"欠载域平区"复现） |

**排序理由**：结构收益集中在稳态 workload（PL-1.18 方法论：persistent 任务流水的瓶颈诊断以最长任务串画像为取景框——longctx-2p7b-32k 的 baseline 画像：Cube mte2 4313µs / 86.4% 忙比（段数墙）、cube_wait 0.934 / mte1_wait 0.896（数据饥饿）、AIV vec 0.594 非关键——与旧 w4 稳态画像同构）；先处理模型级稳态，再瞬态，最后欠载域。

**probe-w2 锚点重验（E-5）**：`probe-w2-780m-s4k`（B1·C16·L256·H48，非 inventory 成员，bench.py probe case）= **217.58µs**（median of 20，profiles/phase1/probe_w2_anchor/）vs 旧一轮 final 217.65µs（round 6）/ 219.23µs（round 7 复测）——**−0.03%，无工具链漂移**；旧 PL-1.18 量级锚点（217µs@w2 / 二轮 final ~204µs）在新工具链 96f287e 上存活，可作为本会话增益目标参照。

**设计估算 vs 实测偏差行（D-2 回填）**：DESIGN §1.6.0 估算下界 ~170–280µs@w2 vs 实测 217.58µs——**落在区间内，偏差 <1.3×，无失准项**（段数+scalar 主导的判断成立；本会话 baseline 画像 mte2 86.4% 忙比与设计段数墙归因一致）。

**口径对照注**：今晨 tileops 集成报告（20260921_024753，wrapper default bn=min(64,N)/ns=3 口径）latency-2p7b-4k = 476.36µs；本调优口径（TUNED bn=128/ns=2）= 330.59µs——wrapper 默认配置较 tuned 慢 ~44%（H=80 时 N_tiles=2 走多 n-loop 慢路径 + prev 重读），wrapper 切换块成对引用 TUNED_DEFAULT_CONFIG 的对齐收益由 conductor 翻转时兑现，不在本 Stage 口径内。

## 3. Iteration Log

### Round 1（base = baseline = Stage 3 基准拷贝）

#### Diagnostic Context
- kernel: ssd（上文 kernel_id），全部 11 tune workload 同测（分支全量测量制）
- base profile（稳态 longctx-2p7b-32k）：Task Duration 4987µs；Cube mte2 4313µs/86.4%（段数墙：ws_lcb band 640 + x 256 + ws_c 128 + prev×4 512 ≈ 1536+…段/任务，PL-1.18 段数记账二轮更正口径 prev lt 循环 ×4 重读）；cube_wait 0.934 / mte1_wait 0.896；AIV vec 0.594 非关键、aiv_mte2_wait 0.617。
- 已知结论（PL-1.18 二轮 update，重推导依据）：prevhoist（prev 装载 lt 不变却逐 lt 重读，mte2 字节 −20%）/ l0c2x（单 L0C acc 的 gemm 链 WAR 串行）/ vbrchoist（vbrc(dA_l) lt 不变逐 s-block 重做 + alias 税）三胜出；band 增量组装（数学否决）/ 深度 3（L2 劣化）/ x 预取（无间隙）/ L1 双缓冲（端口竞争）/ 运行时 if（调度毒）五否决。

#### Current Phenomena
- P1：Cube mte2 段数墙主导（86.4% 忙比）——prev_states 每 (task,pp) 重读 L_tiles 次（Q=256 → 4×16KB 中 48KB 冗余/任务），对应旧档案 prevhoist 胜出点。
- P2：单 l0_acc 使相邻 lt 的 hist→band→out gemm 链经同一 L0C buffer 串行（对应旧档案 l0c2x 胜出点）。
- P3：Vector 侧 vbrc(dA_l_col→[bl,bs]) 每 s-block 重做（对应旧档案 vbrchoist 胜出点）。

#### Candidate Optimization Points

| opt_id | 目标现象 | 优化点 | 判断依据 | 具体改法 | 验证指标 | 状态 |
|---|---|---|---|---|---|---|
| round1_prevhoist | P1 段数墙 | prev_states 装载提升出 lt 循环（N_TILES_ONE 快路径，纯名 bool 无-else if 折叠；N_TILES_MULTI 保原结构） | PL-1.18 update①（mte2 字节 −20%，指令 19→16/任务）；基准代码 prev 拷贝索引只含 task/pp/n_blk 维 | `_ssd_chunk_scan_fwd_kernel_round1_prevhoist.py` | Task Duration / mte2_time / L0 | improved |
| round1_vbrchoist | P3 AIV 链 | vbrc hoist 干净形态（fresh dst + diff_mat 死后复用，净零 UB） | PL-1.18 update③（−0.3~−3.5%）；repro/PL-1.18-floor2-wins.py 骨架 | `_ssd_chunk_scan_fwd_kernel_round1_vbrchoist.py` | Task Duration / aiv_vec_ratio | **H 族分裂（见结论）** |

#### Experiment Branches（候选 vs current best 对比表，B2；数据 = perf_records round 1，median of 20）

| branch | b1c2L64fp16 | b2c4L64fp16 | b1c2L128bf16 | b2c2L64bf16 | lat130m | srv130m | lctx130m | lat2p7b | srv2p7b | lctx2p7b | thr2p7b | L0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline（current best） | 31.83 | 36.31 | 36.19 | 32.06 | 123.23 | 870.96 | 3493.22 | 330.59 | 1275.40 | 4987.27 | 638.30 | PASS |
| round1_prevhoist | 31.65 (+0.6%) | 36.10 (+0.6%) | 35.98 (+0.6%) | 31.91 (+0.5%) | 118.90 (+3.5%) | 841.95 (+3.3%) | 3475.19 (+0.5%) | 327.76 (+0.9%) | 1241.09 (+2.7%) | 4876.97 (+2.2%) | 616.06 (+3.5%) | PASS |
| round1_vbrchoist | 31.76 (+0.2%) | 35.28 (+2.8%) | 37.13 (−2.6%) | 33.09 (−3.2%) | 114.69 (+6.9%) | 814.19 (+6.5%) | 3098.85 (+11.3%) | 344.59 (−4.2%) | 1360.74 (−6.7%) | 4939.26 (+1.0%) | 666.49 (−4.4%) | PASS |

（+% = 相对 baseline 提速；AICore 利用率/内存指标见下述机制分析。）

#### 机制分析（vbrchoist H 族分裂）
- profile diff（vector0+cube0，baseline vs round1_vbrchoist）：
  - longctx-130m-32k（H=24，获益 +11.3%）：AIV mte2_time 2426→1651µs（−32.0%）、Cube mte2_time 3523→2651µs（−24.8%）、aiv_vec_wait 0.474→0.302；
  - serving-2p7b-4k（H=80，回退 −6.7%）：AIV mte2_time 644→827µs（+28.4%）、Cube mte2_time 1055→1289µs（+22.2%）。
- 判读：效应集中在**双引擎 mte2 传输速度（L2 命中敏感）**而非向量发射数——G=1 下 cb 块被 H 个任务复读（H=80 读放大 3.3×于 H=24），vbrchoist 改变 AIV 生产节奏后 L2 局部性在 H=24 族改善、H=80 族劣化。旧任务 w3（=本族 throughput 形状，bf16）当时在 prevhoist+l0c2x 之上测得"不响应"——**累积序可能改变该分支的效应方向**。
- 决策：prevhoist 立即合入（全域无回退 + 2 workload >3% + 双独立测量同向）；vbrchoist 留待 Round 2 在 prevhoist（+l0c2x）之上复测（复现旧任务累积序），若 H 分裂仍在 → Round 3 评估核内 trace-time H 条件路径。

#### 合并检查（merged_r1_prevhoist = 最终文件复测，第二独立测量）

| merged_candidate | workload | baseline_us | merged_us(1st) | merged_us(2nd) | precision | status |
|---|---|---:|---:|---:|---|---|
| merged_r1_prevhoist | b1-c2-L64-h4-p64-n32-fp16 | 31.83 | 31.65 | 31.65 | L0 PASS | 平区无回退 |
| merged_r1_prevhoist | b2-c4-L64-h8-p64-n64-fp16 | 36.31 | 36.10 | 36.30 | L0 PASS | 平区无回退 |
| merged_r1_prevhoist | b1-c2-L128-h4-p128-n32-bf16 | 36.19 | 35.98 | 35.99 | L0 PASS | 平区无回退 |
| merged_r1_prevhoist | b2-c2-L64-h4-p64-n32-bf16 | 32.06 | 31.91 | 31.95 | L0 PASS | 平区无回退 |
| merged_r1_prevhoist | latency-130m-4k | 123.23 | 118.90 | 121.16 | L0 PASS | +1.7%（第二测）|
| merged_r1_prevhoist | serving-130m-4k | 870.96 | 841.95 | 852.87 | L0 PASS | +2.1% |
| merged_r1_prevhoist | longctx-130m-32k | 3493.22 | 3475.19 | 3410.03 | L0 PASS | +2.4% |
| merged_r1_prevhoist | latency-2p7b-4k | 330.59 | 327.76 | 326.91 | L0 PASS | +1.1% |
| merged_r1_prevhoist | serving-2p7b-4k | 1275.40 | 1241.09 | 1246.19 | L0 PASS | +2.3% |
| merged_r1_prevhoist | longctx-2p7b-32k | 4987.27 | 4876.97 | 4838.21 | L0 PASS | +3.0% |
| merged_r1_prevhoist | throughput-2p7b-2k | 638.30 | 616.06 | 625.47 | L0 PASS | +2.0% |

smoke 集：空（无 smoke workload）。两次独立测量全部同向（model-scale +0.9~+3.5%），无任何可信回退。

#### Iteration Winner
- winner: **prevhoist**（合入最终文件 `_ssd_chunk_scan_fwd_kernel.py`，candidate_id=merged_r1_prevhoist）
- rollback_branches: round1_vbrchoist（暂缓，Round 2 复测）
- new_current_best: `_ssd_chunk_scan_fwd_kernel.py`（= prevhoist 形态）

### Round 2（base = merged_r1_prevhoist）

#### Diagnostic Context
- base：merged_r1_prevhoist（prevhoist 已合入）。base 稳态画像（round1 prevhoist 分支 longctx-2p7b-32k）：4877µs，mte2 段数墙仍主导但 prev 重读已消（段/任务 1920→1536）。
- 本轮候选：l0c2x（Cube 侧剩余串行链：单 l0_acc 使相邻 lt 的 hist initC→band 累加→out fixpipe 经同一 L0C buffer WAR 串行）+ vbrchoist 累积复测（Round 1 的 H 分裂是否在 prevhoist 之上消失）。
- 已知否决（不重试，无工具链漂移证据）：band 增量组装（数学）、深度 3（L2）、x 预取（无间隙）、L1 双缓冲（端口）、运行时 if（调度毒）。

#### Current Phenomena
- P1：l0c2x 目标——gemm 链 L0C WAR 串行（对应旧档案 cube_wait 0.897→0.836 的改善空间）。
- P2：vbrchoist 目标——AIV 链 vbrc 逐 s-block 重做 + alias 税（Round 1 已证单点有效但 H=80 回退；本轮验证累积序假设）。

#### Candidate Optimization Points

| opt_id | 目标现象 | 优化点 | 判断依据 | 具体改法 | 验证指标 | 状态 |
|---|---|---|---|---|---|---|
| round2_l0c2x | P1 L0C WAR 串行 | L0C acc 乒乓配对循环（lt=2j→acc_a / lt=2j+1→acc_b；ONE_AND_ODD 尾块纯名预组合；N_TILES_MULTI 保原结构单 acc） | PL-1.18 update②（−1.7~−4.6%，cube_wait 0.897→0.836） | `_ssd_chunk_scan_fwd_kernel_round2_l0c2x.py` | Task Duration / cube_wait | improved |
| round2_vbrchoist | P2 AIV 链 | vbrc hoist 干净形态（同 round1 形态，base 换为 prevhoist 合入版） | PL-1.18 update③ + Round 1 累积序假设 | `_ssd_chunk_scan_fwd_kernel_round2_vbrchoist.py` | Task Duration / aiv 指标 | improved（H 分裂消失） |

#### Experiment Branches（候选 vs current best 对比表，B2；perf_records round 2，median of 20）

| branch | b1c2L64fp16 | b2c4L64fp16 | b1c2L128bf16 | b2c2L64bf16 | lat130m | srv130m | lctx130m | lat2p7b | srv2p7b | lctx2p7b | thr2p7b | L0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| merged_r1_prevhoist（current best） | 31.65 | 36.30 | 35.99 | 31.95 | 121.16 | 852.87 | 3410.03 | 326.91 | 1246.19 | 4838.21 | 625.47 | PASS |
| round2_l0c2x | 31.58 (+0.2%) | 36.27 (+0.1%) | 35.72 (+0.8%) | 31.92 (+0.1%) | 119.80 (+1.1%) | 791.30 (+7.2%) | 3173.67 (+6.9%) | 317.93 (+2.7%) | 1195.11 (+4.1%) | 4784.00 (+1.1%) | 604.29 (+3.4%) | PASS |
| round2_vbrchoist | 31.87 (−0.7%) | 35.14 (+3.2%) | 35.53 (+1.3%) | 33.05 (−3.4%) | 113.08 (+6.7%) | 765.80 (+10.2%) | 3100.19 (+9.1%) | 313.42 (+4.1%) | 1213.97 (+2.6%) | 4732.13 (+2.2%) | 611.01 (+2.3%) | PASS |

（+% = 相对 current best 提速。unit-scale 欠载域 ±3% 为平区噪声（老档案同判）；model-scale 全部正向。）

#### 结论与机制
- **累积序假设证实**：vbrchoist 在 prevhoist 之上 H=80 族回退消失（round1: lat2p7b −4.2%/srv2p7b −6.7%/thr2p7b −4.4% → round2: +4.1%/+2.6%/+2.3%）。机制：prevhoist 削减 Cube mte2 压力后 L2/总线交互改变，vbrchoist 的 AIV 生产节奏调整不再与 H=80 的 cb 读放大（G=1 下 H 任务复读同一 cb 块）冲突。**Round 1 的"H 族分裂"是单点叠加顺序的伪象，非 vbrchoist 形态本身的缺陷**——单点验证结论依赖 base 状态（对 skill 的教训：跨引擎耦合结构中，AIV 侧优化的单点效应受 Cube 侧状态调制，见复盘）。
- 两分支互补（Cube/Vector 各一），model-scale 互有胜负（vbrchoist 5/7 胜，l0c2x 在 srv2p7b/thr2p7b 胜 +1.6%/+1.1%）→ 均为有效单点，进入组合验证（Round 3，两单点均已独立证明有效，符合"组合优化只在单点证明有效后再做"纪律）。
- bn=64 multi-path 探针（probe_bn64.py，覆盖 l0c2x 重构的 N_TILES_MULTI 区域 + Q=96 尾块 + P=128 双 pp + 双 dtype）：4/4 PASS（wrapper default 配置路径正确性保持）。

#### Iteration Winner
- winner: round2_vbrchoist 与 round2_l0c2x 并列有效（不同引擎、互补）；组合验证移交 Round 3。
- new_current_best: 维持 merged_r1_prevhoist（待 Round 3 组合分支非回退检查后一次性更新）。

### Round 3（base = merged_r1_prevhoist；组合分支 = prevhoist + l0c2x + vbrchoist）

#### 依据
Round 2 两单点（l0c2x：Cube 侧；vbrchoist：Vector 侧）已各自在相同 base 上独立证明有效（model-scale 全正向）——符合"组合优化只在单点证明有效后再做"的前置条件；组合分支只叠加两个已证明点，不引入新优化点。

#### Experiment Branches（候选 vs current best 对比表，B2；perf_records round 3，median of 20）

| branch | b1c2L64fp16 | b2c4L64fp16 | b1c2L128bf16 | b2c2L64bf16 | lat130m | srv130m | lctx130m | lat2p7b | srv2p7b | lctx2p7b | thr2p7b | L0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| merged_r1_prevhoist（current best） | 31.65 | 36.30 | 35.99 | 31.95 | 121.16 | 852.87 | 3410.03 | 326.91 | 1246.19 | 4838.21 | 625.47 | PASS |
| round2_l0c2x（亲代1） | 31.58 | 36.27 | 35.72 | 31.92 | 119.80 | 791.30 | 3173.67 | 317.93 | 1195.11 | 4784.00 | 604.29 | PASS |
| round2_vbrchoist（亲代2） | 31.87 | 35.14 | 35.53 | 33.05 | 113.08 | 765.80 | 3100.19 | 313.42 | 1213.97 | 4732.13 | 611.01 | PASS |
| **round3_combined** | 31.57 (+0.3%) | 35.22 (+3.0%) | 35.49 (+1.4%) | 32.67 (−2.3%) | 112.74 (+6.9%) | 754.98 (+11.5%) | 2952.64 (+13.4%) | 313.11 (+4.2%) | 1211.48 (+2.8%) | 4666.30 (+3.6%) | 598.36 (+4.3%) | PASS |

（+% = 相对 current best（merged_r1_prevhoist）提速。组合版在全部 7 个 model-scale workload 上同时优于两个单亲分支；b2c2L64bf16 −2.3% 落在欠载域 ±3% 噪声带内（该 case 在 round2_vbrchoist 亦 −3.4%，老档案同判欠载域平区）。）

#### 稳态画像变化（baseline → round3_combined）
- longctx-2p7b-32k（853 任务/核）：总 4987→4616µs；Cube mte2 4313→3820µs（−11.4%，仍 83% 忙比=段数墙物理流量）；Cube scalar 3104→2594µs（−16.4%）；cube_wait 0.934→0.889 / mte1_wait 0.896→0.844；fixpipe 1356→1411µs（+4.1%，乒乓重叠的合理代价）。
- longctx-130m-32k（512 任务/核）：总 3896→2864µs（−26.5%）；AIV mte2 2428→1707µs（−29.7%）/ mte3 1100→757µs（−31.2%）；aiv_vec_wait 0.475→0.322——H=24 族获益主因是 AIV 侧等待大幅解除后双引擎 mte2 传输速度（L2 局部性）改善。

#### 合并检查（merged_r3_combined = 最终文件第二独立测量）

（见下方合并复测表——数据落 perf_records phase=merged round 3；probe_bn64 4/4 PASS 在合并前完成。）

#### Iteration Winner
- winner: **round3_combined**（合入 `_ssd_chunk_scan_fwd_kernel.py`，candidate_id=merged_r3_combined）
- new_current_best: `_ssd_chunk_scan_fwd_kernel.py`（prevhoist + l0c2x + vbrchoist 全形态）

### Round 4 / 候选穷尽分析（stop 判定）

#### 合并复测（merged_r3_combined = 最终文件第二独立测量，vs baseline）

| merged_candidate | workload | baseline_us | r3_branch(1st) | merged(2nd) | precision | status |
|---|---|---:|---:|---:|---|---|
| merged_r3_combined | b1-c2-L64-h4-p64-n32-fp16 | 31.83 | 31.57 | 31.62 | L0 PASS | 平区 |
| merged_r3_combined | b2-c4-L64-h8-p64-n64-fp16 | 36.31 | 35.22 | 35.11 | L0 PASS | +3.3% |
| merged_r3_combined | b1-c2-L128-h4-p128-n32-bf16 | 36.19 | 35.49 | 35.44 | L0 PASS | +2.1% |
| merged_r3_combined | b2-c2-L64-h4-p64-n32-bf16 | 32.06 | 32.67 | 32.63 | L0 PASS | −1.8%（<3% 噪声带，欠载域）|
| merged_r3_combined | latency-130m-4k | 123.23 | 112.74 | 111.25 | L0 PASS | +9.7% |
| merged_r3_combined | serving-130m-4k | 870.96 | 754.98 | 731.62 | L0 PASS | +16.0% |
| merged_r3_combined | longctx-130m-32k | 3493.22 | 2952.64 | 3053.21 | L0 PASS | +12.6%（双测 −12.6/−15.5%）|
| merged_r3_combined | latency-2p7b-4k | 330.59 | 313.11 | 303.45 | L0 PASS | +8.2% |
| merged_r3_combined | serving-2p7b-4k | 1275.40 | 1211.48 | 1245.49 | L0 PASS | +2.3%（双测 −2.3/−5.0%，run 间 ±2.8%）|
| merged_r3_combined | longctx-2p7b-32k | 4987.27 | 4666.30 | 4663.48 | L0 PASS | +6.5%（双测一致）|
| merged_r3_combined | throughput-2p7b-2k | 638.30 | 598.36 | 597.59 | L0 PASS | +6.4%（双测一致）|

smoke 集：空。model-scale 双独立测量全部同向正向；唯一负值 b2-c2-L64-bf16 −1.8% 稳定但 <3% 噪声阈值（欠载域，老档案同类判例），不构成可信回退。

#### probe-w2 锚点延续性
`probe-w2-780m-s4k`：baseline 217.58µs → **final 200.18µs（−8.0%）**（profiles/phase1/probe_w2_final/）；旧二轮 final 204.33µs——本会话重推导的三胜出组合**复现并略超**旧增益（旧 −8.6%，本 −8.0%，同分布）。

#### 剩余候选穷尽清单（stop 依据）

**A. 旧档案五否决 carry-over（不重测的依据）**：band 增量组装（数学：band 行绑定 lt）/ 深度 3 任务流水（L2 局部性）/ x 预取（稳态无间隙）/ L1 双缓冲（MTE2 写-MTE1 读端口竞争）/ 运行时 if 进热循环（BiSheng 跨迭代调度毒）。**carry-over 论证**：`git diff --stat 15ad002..96f287e` 显示本会话工具链与旧二轮任务工具链的 delta 仅为 7 个 `.agents/skills` 文档文件（evolution commit），**tilelang 源码零改动**；CANN 8.5.0 / BiSheng / 910B2C 硬件均未变；probe-w2 锚点复现（217.58 vs 217.65，−0.03%） triple 佐证。五否决的机制均绑定硬件常数（L2/L1 端口）或数学事实，与 host 侧 launcher 无关。

**B. 本会话新评估并否决的候选**：

| 候选 | 阻塞点 | 证据/机制 |
|---|---|---|
| H-fold / h_tile=2 任务折（削 H=80 cb 读放大） | 机制 + L2 风险 | 2.7B 族的墙在 Cube mte2（ws 段数），不在 AIV cb 流量；ws 工作集 ×2 与深度 3 否决同族 L2 风险；全折任务饥饿为 DESIGN R1 #10 否决 |
| task 级 dA/dt 行装载（省 ~20 tiny GM 读/任务） | 编译器 | TRAP-UB-dynsubview-dominance（旧 v3 在 pass 关闭下复现）——动态偏移 UB subview 非支配 IR |
| bl/bs/bp=128 配置邻域 | UB 容量 | 357–582KB > 192KB（4515de8 谱系编译探针） |
| per-lt 细粒度握手（flag 数 > 任务级 4 个） | flag 预算 + 结构族 | 2×L_tiles×slot + 4 > 15（CONST-flag-id-budget）；与深度 3/L1 双缓冲同属"更深流水"族（已否决） |
| vbrchoist-lite（fresh dst 去 dead-reuse） | UB 容量 | s_blk 循环活跃集 +16KB（旧 v3：+16KB 手工 → 195.1KB 实测溢出）；全形态已验证有效，无必要 |

**C. 参照结构 diff 检查（T-1 morph ladder 轻量版）**：同族参照 = 旧任务 final（结构与本 current best 一致，无 diff 可打）/ mamba_ssm Triton GPU 源（fragment 流水，GPU 专有形态，非 NPU 可迁移）/ GQA attention Expert（不同算法族）。**族内无可行动结构差异**。

**D. 残余瓶颈归因**：合并后稳态画像（longctx-2p7b）Cube mte2 仍 83% 忙比 = 1152 段/任务（ws_lcb band 640 + x 256 + ws_c 128 + prev 128）——「Vector 产因子 → GM ws 中继 → Cube 消费」DESIGN 冻结结构的物理流量。**[DESIGN_LIMIT] 判定：不触发**——触发条件②不满足（无 >2x 结构性替代实证：separable 两-kernel 收益不明确，与旧任务判定一致；且 baseline 217.58 与 final 200.18 均落在 DESIGN §1.6.0 估算区间 170–280µs 内，无设计假设矛盾）。按 perf-feedback.md §1 两门槛（须同时满足），禁止产出 perf_feedback.md。

**stop_reason = blocked**（剩余候选全部因数学/容量/编译器/flag 预算/机制证据阻塞；11 个 tune workload 全部 winner_merged；best_effort 结构最优已达成，w2 锚点 200.18µs 优于旧 final 204.33µs）。预算余量充足（3 调优轮 + 1 final 轮 / max_rounds~10；5 实验分支 / 30）。

## 4. Autotune Log

未使用 autotune（结构性优化主导）。参数维度现状：block_l/block_p/block_s 由 UB 容量锁定 64（bl/bs=128 需 357–582KB > 192KB，4515de8 谱系编译探针）；block_n=128 经 `min(bn, N)` 钳位后 N_tiles 恒 1（唯一有效值）；num_stages 为遗留形参（深度 2 任务流水硬编码于 slot 结构，深度 3 已实测否决）。无合法搜索空间可扫。

## 5. Final Summary

（数据见下方 Final Performance Test Data 表——phase=final 记录采集自最终文件 `_ssd_chunk_scan_fwd_kernel.py`（sha256 见 perf_records），第三独立测量。）

- **最终候选**：`perf_opt/_ssd_chunk_scan_fwd_kernel.py`（= Stage 3 基准 + prevhoist + l0c2x + vbrchoist；TUNED_DEFAULT_CONFIG {bl:64, bp:64, bn:128, bs:64, ns:2} 模块级常量暴露，供 wrapper 切换块成对引用）
- 结构变更四件套（相对 Stage 3 基准）：①prevhoist（prev_states 装载提升出 lt 循环，N_TILES_ONE 快路径 + N_TILES_MULTI 原结构保持）；②l0c2x（L0C acc 乒乓配对循环 + ONE_AND_ODD 尾块，N_TILES_MULTI 保持单 acc 原结构）；③vbrchoist（vbrc hoist 干净形态：fresh dst + diff_mat 死后复用，净零 UB）；④docstring/TUNED 注释更新（交付标识）。
- 正确性：L0 全绿（7 cases + contract）；bn=64 multi-path 探针 4/4（probe_bn64.py，覆盖 wrapper 默认配置路径）；合并后每轮 L0 通过；最终文件 `--level all` 全量回归（见下）。

### Final Performance Test Data

> 数据 = perf_records.jsonl phase=final 行（candidate_id=final，artifact = `perf_opt/_ssd_chunk_scan_fwd_kernel.py`，sha256 一致；median of 20，NPU 0，TUNED_DEFAULT_CONFIG）。**记录清理披露**：final 阶段原有 13 行，其中 2 行（b1-c2-L64-fp16=31.85µs / b2-c4-L64-fp16=35.25µs，sha 5eab56c1）来自一次被手动中止的采集（docstring 修订前误启动，进程组被杀时已落 2 行，其 raw profile 目录已删除、所测文件版本已被取代）——按"被中止会话的孤儿记录"清理，数值在此留痕（与保留值同分布）。另 baseline 阶段前 3 行为首次采集被 shell 超时连带杀死前的重复记录（同分布，见 §2 注），append-only 保留。

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| tileops/kernels/mamba/ssd_chunk_scan/ssd_chunk_scan_kernel/perf_opt/_ssd_chunk_scan_fwd_kernel.py::_ssd_chunk_scan_fwd_kernel::main | b1-c2-L64-h4-p64-n32-fp16 | 31.83 | 31.52 | final |
| tileops/kernels/mamba/ssd_chunk_scan/ssd_chunk_scan_kernel/perf_opt/_ssd_chunk_scan_fwd_kernel.py::_ssd_chunk_scan_fwd_kernel::main | b2-c4-L64-h8-p64-n64-fp16 | 36.31 | 35.21 | final |
| tileops/kernels/mamba/ssd_chunk_scan/ssd_chunk_scan_kernel/perf_opt/_ssd_chunk_scan_fwd_kernel.py::_ssd_chunk_scan_fwd_kernel::main | b1-c2-L128-h4-p128-n32-bf16 | 36.19 | 35.57 | final |
| tileops/kernels/mamba/ssd_chunk_scan/ssd_chunk_scan_kernel/perf_opt/_ssd_chunk_scan_fwd_kernel.py::_ssd_chunk_scan_fwd_kernel::main | b2-c2-L64-h4-p64-n32-bf16 | 32.06 | 32.96 | final |
| tileops/kernels/mamba/ssd_chunk_scan/ssd_chunk_scan_kernel/perf_opt/_ssd_chunk_scan_fwd_kernel.py::_ssd_chunk_scan_fwd_kernel::main | latency-130m-4k | 123.23 | 113.20 | final |
| tileops/kernels/mamba/ssd_chunk_scan/ssd_chunk_scan_kernel/perf_opt/_ssd_chunk_scan_fwd_kernel.py::_ssd_chunk_scan_fwd_kernel::main | serving-130m-4k | 870.96 | 731.05 | final |
| tileops/kernels/mamba/ssd_chunk_scan/ssd_chunk_scan_kernel/perf_opt/_ssd_chunk_scan_fwd_kernel.py::_ssd_chunk_scan_fwd_kernel::main | longctx-130m-32k | 3493.22 | 2961.78 | final |
| tileops/kernels/mamba/ssd_chunk_scan/ssd_chunk_scan_kernel/perf_opt/_ssd_chunk_scan_fwd_kernel.py::_ssd_chunk_scan_fwd_kernel::main | latency-2p7b-4k | 330.59 | 309.26 | final |
| tileops/kernels/mamba/ssd_chunk_scan/ssd_chunk_scan_kernel/perf_opt/_ssd_chunk_scan_fwd_kernel.py::_ssd_chunk_scan_fwd_kernel::main | serving-2p7b-4k | 1275.40 | 1211.05 | final |
| tileops/kernels/mamba/ssd_chunk_scan/ssd_chunk_scan_kernel/perf_opt/_ssd_chunk_scan_fwd_kernel.py::_ssd_chunk_scan_fwd_kernel::main | longctx-2p7b-32k | 4987.27 | 4724.39 | final |
| tileops/kernels/mamba/ssd_chunk_scan/ssd_chunk_scan_kernel/perf_opt/_ssd_chunk_scan_fwd_kernel.py::_ssd_chunk_scan_fwd_kernel::main | throughput-2p7b-2k | 638.30 | 606.54 | final |

**逐 workload 提升**（正 = 提速）：b1c2L64fp16 **+1.0%** | b2c4L64fp16 **+3.0%** | b1c2L128bf16 **+1.7%** | b2c2L64bf16 **−2.8%**（欠载域 <3% 噪声带，三测稳定 −1.9/−1.8/−2.8%，见 Round 4 注）| latency-130m **+8.1%** | serving-130m **+16.1%** | longctx-130m **+15.2%** | latency-2p7b **+6.5%** | serving-2p7b **+5.1%** | longctx-2p7b **+5.3%** | throughput-2p7b **+5.0%**。
- **model-scale（7 workload）几何平均提速 +9.7%**；全 11 workload 几何平均 +6.4%。
- probe 锚点：w2-780m-s4k 217.58 → 200.18µs（−8.0%，vs 旧二轮 final 204.33µs 复现并略超）。
- smoke/skip 排除：smoke 集空（11 个 benchmark 案例均无 smoke 标记/字样，profile-collection §1 分类规则）；skip 集空。
- 精度：最终文件 L0 全绿 + probe_bn64 4/4 + `--level all` 全量回归（L0 7 + contract / L1 6 / L2 4 / Boundary 4，见 logs/final_level_all.log）。
- **总体结论**：best_effort 调优于候选穷尽点收束（stop_reason=blocked，见 Round 4 分析）；全部 11 tune workload winner_merged；无 perf_feedback（[DESIGN_LIMIT] 双门槛之②不满足，见 Round 4-D）。

## 6. Skill Retrospective

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Iteration-diagnosis | **跨引擎单点验证的累积序调制**：vbrchoist 单独叠加在无 prevhoist 的 base 上时 H=80 族回退 −4.2~−6.7%（双引擎 mte2 传输 +22~28%，L2 局部性劣化），曾据此考虑 trace-time H 条件路径；prevhoist 合入后复测回退消失（+2.2~+10.2%）——MixCV 中 AIV 侧候选的单点效应受 Cube 侧 base 状态调制，单点"族分裂/方向翻转"与先验矛盾时须先检验累积序假设 | 本 log R1 机制分析 + R2 复测；perf_records round1_vbrchoist vs round2_vbrchoist（同形态不同 base）；profiles/candidate/round1 vs round2 PipeUtilization diff | iteration-diagnosis.md 增补「跨引擎候选的累积序检验」步骤：与先验矛盾的族分裂 → 先合入对侧已验证胜出再复测，再谈 per-shape 条件路径 | P |
| SKILL / E-5 工具 | **kb_stale_check 对 docs-only commit 产生全库假阳性**：启动时 stale_count=125，实际 `git diff 15ad002..96f287e` 仅 7 个 .agents/skills 文档文件、tilelang 源码零改动——五项结构否决的 carry-over 论证本可直接成立 | 本 log §0 E-5 段 + git diff --stat（7 文件全 .agents/） | kb_stale_check.py 的 stamp diff 过滤非源码路径（tilelang/ 源码树），区分"编译器变更"与"知识库变更"两类 stale | R |
| logging / 工件持久化 | **工作树未提交的 perf_opt 无恢复途径**：上一会话 14 个实验分支 + 64 行 records 随目录删除永久丢失；本次重建依赖 repro 骨架自包含性（成功，见 D 类价值点 #2） | 本 log 任务背景 + 重建全程 | conductor/statectl 在 DONE 前做 perf_opt 产物快照（或分支文件入 git stash）；SKILL.md Phase 4 增产物快照提醒 | R |
| Iteration-diagnosis / 分支验证 | **配置路径覆盖缺口**：tuned bn=128 钳位使 kernel 自带 L0/L1 测试永不触达 N_TILES_MULTI，而 tileops wrapper 默认 bn=64 走该路径——分支验证只跑 `--level`（TUNED 配置）不覆盖全部合法配置 | 本 log probe_bn64.py 4/4（双 dtype × 尾块 × 双 pp）；wrapper default_config 分析 | tilelang-op-optimize SKILL.md / tilelang-op-develop 分支验证清单增「配置路径维度」检查（默认/tuned 双配置 × 关键分支形态） | R |
| profile-collection / 工具 | **后台 runner 被持久 shell 超时连带杀死**：`cd X && nohup ... &` 中 `&` 作用于整个链，工具超时 SIGKILL 进程组；setsid 脱离后稳定（首次 baseline 采集即事故，产生 3 行重复记录） | 本 log §2 注 + logs/phase1_baseline_all.log 首次启动事故 | run_experiments_template.py 的使用说明内置 setsid 后台运行形态 | R |
| stop_condition | stop_reason=blocked 有充分支撑（候选穷尽清单 §Round 4 A/B/C + 无漂移锚点 + 工具链 docs-only delta 论证），符合"逐项留痕"要求 | 本 log Round 4 全节 | none | - |

### Value Point Proposals（含 BP_xxx）

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **跨引擎单点验证的累积序调制**（BP_mixcv_cumulative_order）：MixCV kernel 的 Vector 侧候选单点验证出现与先验矛盾的 workload 族分裂（方向翻转）时，强制先做累积序假设检验（合入已验证的 Cube 侧胜出后复测），再评估 per-shape 条件路径——本例避免了无必要的核内 H 分派并挽留一个胜出结构 | P | opt_log.md#round-1 / #round-2；perf_records round1/round2（已回写 attention.md PL-1.18 重建会话 update #2） | repro/PL-1.18-floor2-wins.py（结构形态）+ 本会话 perf_records（效应数据） | tilelang 0.1.2+96f287eeaa / CANN 8.5.0 / Ascend910B2C | iteration-diagnosis.md（BP proposal）+ attention.md（已回写） |
| **三胜出可仅凭 repro 骨架重推导**（工件全失场景实证）：perf_opt 全失后从 PL-1.18-floor2-wins.py 骨架重实现 prevhoist/l0c2x/vbrchoist，w2 锚点 217.58→200.18µs（−8.0%）vs 旧二轮 −8.6% 同分布；新 workload 集（11 个，H=24/80 双族）model-scale −5.0~−16.0%——ED-A/ED-B 自包含性经住检验 | D | opt_log.md#round-1..#final-summary + profiles/phase1/probe_w2_{anchor,final}（已回写 attention.md PL-1.18 重建会话 update #1） | repro/PL-1.18-floor2-wins.py（被复证的骨架本体） | 同上 | attention.md PL-1.18（已回写） |
| **配置路径覆盖缺口**：tuned bn 钳位使自带测试永不触达 N_TILES_MULTI 而 wrapper 默认 bn=64 会走——分支验证覆盖 = shape 维 × 配置路径维笛卡尔积（BP_config_path_coverage） | R | opt_log.md 复盘表 #4；probe_bn64.py 4/4 | perf_opt/probe_bn64.py（任务工件，provenance 允许失效；结论自包含于回写条目） | 同上 | attention.md PL-1.18（update #3，已回写）+ iteration-diagnosis.md（BP proposal） |
| **kb_stale_check docs-only 假阳性**：125 stale 全因知识库 commit（源码零改动）——stamp diff 应过滤非源码路径，区分编译器变更/知识库变更 | R | opt_log.md 复盘表 #2；`git diff --stat 15ad002..96f287e` | none（git 命令即复现） | 同上 | .agents/tools/kb_stale_check.py（proposal，evolver 路由） |
| **perf_opt 未提交产物不可恢复**：上会话 14 分支 + 64 行 records 永久丢失；建议 DONE 前产物快照 | R | opt_log.md 任务背景 + 重建全程 | none | 同上 | conductor/statectl（proposal） |
| **本会话案例索引**（重建型 Stage 4：断裂修复 + 增量调优 + 三胜出重推导，model-scale 几何 +9.7%） | C | opt_log.md 全文 + perf_records.jsonl（102 行） | — | 同上 | pattern-library/cases.md（CASE-ssd-chunkscan-migration 追记候选，evolver 蒸馏时定） |
