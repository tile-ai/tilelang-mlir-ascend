# Attention / Expert persistent 模式与硬上限（attention 族实测）

> pattern-library 主题文件（入口/预算/front-matter schema 见 [INDEX.md](INDEX.md)；`repro: repro-missing` = 待回填）。

---
id: PL-1.7-expert-persistent-boundary
kind: pattern
family: [attention, expert]
apis: [T.gemm, sync_block_set, sync_block_wait]
dtype: [fp16, bf16]
device: 910B2C
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z
toolchain: expert 侧 tilelang dev build 21586b5（2026-09-07）+ CANN 8.5.0；developer 基线侧 tilelang 67db6f3（2026-09-04）+ CANN 26.0.rc1
repro: repro-missing
---

### 1.7 Expert 双 Scope persistent 与简单 tiling 的收益分界

- **模式**：同一算子（GQA prefill FA 前向）两代同门对比——Expert persistent 24 核双 Scope（跨引擎块级流水 + GM ws 多槽 + per-slot flag）vs Developer 简单 tiling（fragment + cv_split + T.Pipelined）；同 wrapper config (64,64,1)、同 msprof op kernel-only 口径。
- **实测收益分界**（10 workload bench；跨工具链代对比，倍率含版本间差异成分——版本戳见 front-matter）：长 KV（S=2048）**1.57–1.63× 提速**（8b-long 13271 vs 20883 µs fp16）；短 KV（S=512）**~1.31× 回退**（3764 vs 2872 µs）；smoke ~1.27× 回退（282 vs 222 µs）。吞吐随 S 反转（short 2.70 vs 3.53 / long 5.58 vs 3.55 TOps/s）——工作假设（未单独验证）：persistent + 跨引擎流水的固定入口成本只在长 KV 循环摊销；集成期 bench 见短 workload 回退属该形态的预期画像（Stage 5 只记录不修复）。
- 溯源：`examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/integration_log.md` §Bench + `examples/TileOPs/profile_run_msprof_20260907_173440.log`（expert）vs `profile_run_msprof_20260904_110930.log`（developer）；复现：wrapper baseline 激活态下 TileOPs 根 `python -m pytest benchmarks/ops/bench_multi_head_attention.py -v -s`。

---
id: PL-1.8-bf16-cube-direct
kind: pattern
family: [attention, expert, cube]
apis: [T.copy, T.gemm, T.vcast]
dtype: [bf16, fp32]
device: 910B2C
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z
toolchain: tilelang dev root build 2026-09-07（21586b5）+ CANN 8.5.0 + Ascend910B2C
repro: repro-missing
---

### 1.8 bf16 Cube 直连链

- **形态**：`T.copy` bf16 GM→L1（slice 形态，见 traps-runtime.md TRAP-load-nd2nz-strided 行）+ Scope("Cube") `T.gemm` bf16×bf16→fp32（L0C 累加）+ `T.vcast` f32→bf16 回写。文档依据：`docs/Tilelang.language/线性代数操作/T.gemm.md` §2.2.1（bf16 √）+ §2.3（bf16×bf16 dst fp32 实测可用）；`testing/npuir/bf16_support_ops/test_gemm_bf16.py`（Developer 形态先例）。
- **实测（Expert 端到端，L0+契约/L1/L2/Boundary 29 用例全过）**：bf16 out tier-2 max_diff 1.56e-2 = 2 bf16 ulp、lse 1.9e-6、tier-1 翻转率 77–96/262144 ≈ 0.03%（与上轮 developer 谱系 D8 校准一致）；集成 bench 10 workload bf16/fp16 差全部 **≤0.2%**（如 8b-long 13274.85 vs 13270.57 µs）——上一 developer 轮载体链 bf16 普遍慢 2–4%，直连形态无延迟税。
- **证伪更正**：两轮迁移设计曾断言「bf16 Cube 输入 ×」并据此驱动 E5 载体链全链设计——实为负向断言未亲核文档表格（引用了正确路径、转述了相反内容）；本轮文档+测试复核推翻该断言（举证规则见 `_shared/standards/negative-claim-evidence.md`，提案见 queue VP-2026-0025）。
- 溯源：`examples/multi_head_attention/_gqa_prefill_fwd_kernel/debug_log.md` Final results + 同目录 `RETROSPECTIVE.md` Stage 3/5；复现：`python examples/multi_head_attention/_gqa_prefill_fwd_kernel/_gqa_prefill_fwd_kernel.py --level L0`。

---
id: PL-1.9-blockwidth
kind: pattern
family: [attention, expert]
apis: [T.gemm, sync_block_set, sync_block_wait, T.vcast, store_fixpipe]
dtype: [fp16, bf16, fp32]
device: 910B2C
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260908T005751Z
toolchain: tilelang 0.1.2+3a214cde7aa4f54fc4a103f0f324a43341122d68（三轮同 commit）/ CANN 8.5.0 / Ascend910B2C
repro: repro-missing
---

### 1.9 Expert 跨引擎块级流水的「块宽摊减」调优序与硬上限

- **块宽摊减律**：persistent 双 Scope（per-slot flag 块级流水）下每块跨引擎串行链 V1(softmax)→FLAG_P→C2(gemm2)→FLAG_O→V2(accumulate) 是周期地板（busy 核 aic_scalar 38–47% = flag 等待自旋，cube gemm 9–12% 近峰值）。**加大 bn 同时削减向量 op 数/块、flag 往返数/列与 gemm 碎片化**，比加深流水更根本——bn 64→256→512 使 ns=1≈ns=2（宽块上双槽 ws 流水被单槽 L1 的 v-load↔gemm2 WAR 串行抵消）；G=2 批量重基准（批末共享 max + L0C initC 对累加 + 每对 1 次 fixpipe/FLAG_O/V2）bn=256 上 -5%~-10% 但被 bn=512 反超（273 vs 327µs @fa4096；pair@bn=512 需双 V tile 驻留 L1 不可行）。
- **cbuf(L1)=512KB 硬上限**（报错文本反推；常数与双槽 bn 上限公式见 constants.md CONST-capacity-910B2C——bm=44 时 bn≤427）。**wide 单槽模式**：L1 单槽（KS=1，C2-first issue 保序消 WAR）+ GM ws/flag 双槽——L1 减半换块宽，bn=512 可行（k+v+p+q=312KB）；UB clean trace 特化（剔 mask/softcap/guard 死缓冲 ~90KB/AIV@bn=256）后 bm=44/bn=512 ≈97KB ✓。
- **S-f16 跨引擎传输可行**（store_fixpipe f32→f16 数值转换正确性见 CONST-store-fixpipe-gm-only）：字节减半，宽块上 -3%（fa2048/4096）；单块 M7 链 vcast +1 op 抵消收益（平区不采纳）。**探针教训**：探针 kernel 必须带 Cube→Vector flag 同步，缺同步的数值崩坏是探针自身竞态而非 API 缺陷。
- **工厂级多 @T.prim_func 变体分派**：trace 级特化（full/clean/G2/...）用工厂闭包层 Python if 选择不同 builder（每个 builder 独立 @tilelang.jit），绕开「parser 不折叠 if」限制；full 变体逐字节保留使非目标 dispatch 零回归（causal 实测 +0.12%/-0.26%）。
- **工厂内 tuned 分派表（config 级，S4-5 模式；VP-2026-0035 两证：attention + ada_layer_norm）**：caller 传 wrapper 默认 config 且 shape 命中 TUNED_DEFAULT_CONFIGS/TUNED_DEFAULT_BLOCK_M 时替换为调优 config，显式非默认 config 一律尊重原值（不静默覆盖用户意图）；tuned 目标 case 纳入内嵌测试套件作 blocking 精度层（fa-tuned 层：dispatch 路径 tier-1 翻转数对照基线）防分派表与全量套件漂移。第二证要点（norm 族 TUNED_DEFAULT_BLOCK_M）：① bm 甜点非单调且逐 shape——按 shape 实测显式记录，不从单点泛化「UB 松弛律」类规则（bm=7 甜点数值例见 cases.md CASE-norm-adalayern-stage4）；② 表条目做邻域闭合验证（bm±1 未测点全档劣化即确认）；③ TileOPs bench/pytest 走 wrapper `default_config`（GPU 结构性值如 block_m=1）而非工厂 UB 表默认——集成 bench 低 Ratio 常是 wrapper 默认未覆盖，Stage 4 第一杠杆即 wrapper config 覆盖（ada：bm=1→调优表贡献全组 2.14x 的最大份额）。
- **H=1 任务平衡**：公式与数值例见 constants.md CONST-aicore-910B2C（logical tasks = ceildiv(S,bm)·H·B；wall-rows = ceil(ceildiv(S,bm)/24)·bm——bm=44 使 S=1024/2048/4096 得 44/88/176 行，bm=22 使 S=512 得 24 任务 1:1）。
- **终局数据（dispatch 路径，msprof op Task Duration，median of 20）**：fa512 18.20 / fa1024 29.58 / fa2048 77.53 / fa4096 263.77 µs（vs 基线 56.62/101.81/375.22/1101.88 = -67.9%~-76.1%；吞吐 7.4/18.2/27.7/32.6 TFLOPS）。跨谱系对照（developer 简单 tiling 调优终值 16.1/33.6/105.2/300.3µs，2026-09-07，provenance 任务工作区允许失效）：fa1024/2048/4096 超 12%/26%/12%，fa512 落后 13%——单块链固定入口开销地板（17.9–18.2µs 平区，M7-clean〔clean 覆盖 nk==1 退化链〕-5.6% 后仍贴地板）。clean 特化使 non-causal 整除 dispatch 的 vcmp 标量化警告贡献归零（causal/full trace 仍在，第五轮根治见 PL-1.11）。

---
id: PL-1.9-hardlimits
kind: constant
family: [attention, expert]
apis: [T.vexp, T.vsub, T.vmul, T.reduce, T.vcast, sync_block_wait]
dtype: [fp16, fp32]
device: 910B2C
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260909T033622Z
toolchain: tilelang 0.1.2+3a214cde7aa4f54fc4a103f0f324a43341122d68 / CANN 8.5.0 / Ascend910B2C / 2026-09-09
repro: repro-missing
---

### 第二轮硬目标续调实测补充（2026-09-09，同 origin_task 谱系续）

- **f16 softmax 链（有效，四 case -3.5%~-14.1%）**：clean trace [half,bn] 大 pass 全 f16（vexp/vsub/vmul/reduce 四 op 文档支持 fp16），vcast-in/out 消失；**running max 以 f16 表示**（f16 值上 max 精确、f16→f32 vcast 无损）经 f32 无损镜像供 alpha/ell/lse 算术——直接 f32→f16 镜像 m 引入系统性 lse 平移（经 ell 的 e^-δ 缩放，1.4e-3 量级）；f16 ell 求和间距是残余 lse 误差主导项（ell~200 时 ~1e-3，有效容差 atol 1e-3+rtol 1e-3×|lse| 内 10x 裕量）。终值 17.55/27.24/69.13/226.72µs。
- **四硬件常数（唯一事实源 constants.md，此处只留 attention 应用口径）**：① L0C(cc)=128KB（bn=512 时 bm≤51 的上限公式见 CONST-capacity-910B2C——与 L1 512KB、UB 192KB 并列三硬上限，共同封死「大 bm 减 KV 重读」再平衡）；② L1 端口 r+w ≈148–154 GB/s/核（双槽**不能**解除 MTE2/MTE1 端口读写串行——Cube 操作数流吞吐地板；KV 重读流量〔tasks/核 × 2MB × 2〕÷此率 = 算子级下界，fa4096 ≈125–131µs；CONST-L1-port-bw）；③ fabric 聚合 ≥1.73TB/s（fa4096 现聚合 1.41TB/s 未触顶——地板因子排序 L1 端口/向量发射率先于 fabric；CONST-fabric-bw）；④ 向量发射 ~0.5µs/op、f16≈f32（同元素总量下宽块 op 恒优于窄块翻倍：深流水重构〔defer-2 + KV 预取先于 P-wait 入队〕虽解锁 Cube〔aic_scalar 50.7→40.5%〕但 bn=256 使 vec-active 90→180µs 净回退 +12%，bn=512 深流水需 K/V 双槽 567KB>L1 封死——「块宽摊减」律的机制解释；CONST-vector-launch-overhead）。
- **`sync_block_wait` 阻塞后续发射流**（ns=1≈ns=2、wide≈serial、深流水槽受限时无效的综合解释）：flag wait 非窄管道门控——流水化只有在 wait 到达时必然已满足（深 deferral）才有效，而深 deferral 与 L1 槽容量冲突；online 递推的 alpha/ellcur 跨 deferral 需 ping-pong（v-op codegen 拒绝多维 UB 切片操作数，平铺多 buffer + 运行时三分支合法）。
- **[DESIGN_LIMIT] 档案**：`perf_opt/perf_feedback.md` + opt_log round 8–10（硬上限测绘→f16 链→深流水回退证据链）；H=1 大 S 复合地板 = store_fixpipe 仅 GM（S/P 往返强制）+ L1 端口 + 向量发射率，非 tiling 可解。**〔第三轮修正〕**该地板系冻结设计族（per-block 链 + bm=44）局部地板——两相位重构四目标全达成（见 PL-1.9-twophase），perf_feedback.md 修正附录已回填。

---
id: PL-1.9-twophase
kind: pattern
family: [attention, expert]
apis: [T.gemm, T.reduce, T.copy, T.transpose]
dtype: [fp16, f16]
device: 910B2C
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260909T071018Z
toolchain: tilelang 0.1.2+3a214cde7aa4f54fc4a103f0f324a43341122d68 / CANN 8.5.0 / Ascend910B2C / 2026-09-09
repro: repro/PATT-twophase-restructure.py
---

### 第三轮（2026-09-09）：两相位重构达标终值

- 参考结构（`examples/flash_attention/flash_attn_npuir.py`）迁移：非持久 grid ceildiv(S,bm=96)、Cube pass-1(全 S)/pass-2(全 O_partial) 独立串行大循环（flag=n-block 下标，nk≤16）、单 L0C [96,256] gemm1/gemm2 顺序复用、K/V/P/Q L1 生命周期复用（112KB）、Vector f32 softmax 链 + UB `scales[]` 延迟 rescale 回放、O_partial f16、**Q hoist**（免每 n-block 重读）、**ell 正确递推**（`T.reduce` 必带 `clear=True`——参考无 clear 的 sum 系文档化静默错误形态，docs/Tilelang.language/规约操作/T.reduce.md §2.3）、**transpose-free lse**（毒化条目绕法，[B,H,S,1] 视图 + wrapped reshape）。
- 终值 **12.94/18.11/30.88/98.05µs**（四硬目标 14/19/33/100 全达成，fa1024/2048/4096 优于参考 18.20/32.41/99.09）；每核 L1 r+w 18.70→11.17MB（-40%）、cube 管道 busy 171→66µs；busy/wall 0.66x 与参考（0.68x）等价——加速来自流量削减而非重叠度。
- **morph 阶梯**（参考实现上逐级叠加本方契约增量的 14 点二分）是本轮定位上下文级差异（transpose 毒化）的决定性方法。bm 上探死点：(128,256) 105.16 / (112,256) 101.84 > (96,256) 98.38（双波装载失衡 + L0C 满配）。
- **causal 多头域画像与宽块解锁反转（2026-09-15 v3 重生成 + 第五轮调优，VP-2026-0063 两证合入）**：causal 多头域此前从未结构化优化（Sep-07 单遍基线 full 路径 3764/13265µs ≈ 2.25/5.17 TFLOPS vs fa 域两相位 87.6 TFLOPS——headroom >10×）；两相位 v3 窄 config（bn_eff∈{64,144}）vs 单遍 expert 同门对照（msprof kernel-only）：短 KV 1.30–1.32× 提速、长 KV 0.78× 回退——ws 物化流量（S+P+O ≈1.65–2.78GB fabric）只在宽块摊销；第五轮 UB diet（−40.9KB 手工预算，N/D staging 合并 + rowmat 删除）解锁 (64,256) 后反转——长 KV 11.1× 领先（8b-long 1191.63µs）、全域几何 8.53×（完整档案见 PL-1.11），bf16/fp16 平价维持（≤1.7%）。
- 溯源：`examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/`（opt_log rounds 1–11 + perf_records.jsonl round 8–11 + perf_feedback.md [DESIGN_LIMIT] 五段档案）；三轮同 commit 3a214cde（2026-09-08～09），origin_task 分属 20260908T005751Z / 20260909T033622Z（hardlimits 小节）/ 20260909T071018Z（本小节 + traps-compiler.md 毒化行）；结构关键更改与机制归因见 `repro/PATT-twophase-restructure.py`；端到端复现命令与 raw profile 目录对账见 opt_log 各轮（provenance 允许失效）。

---
id: PL-1.11-causal-mask-scalartrap
kind: pattern
family: [attention, expert]
apis: [T.vcmp, T.vselect, T.vmax, T.vmin, T.vsub, T.vadd, T.vmul, T.arange]
dtype: [fp16, bf16]
device: 910B2C
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260915T080600Z（第五轮 Stage 4）
toolchain: tilelang 0.1.2+6797758（2026-09-15）/ CANN 8.5.0 / Ascend910B2C；2026-09-16 a13585dc stale 重验维持（probe_ub.py 4 点复校：(64,256)/(80,256) 编译通过、(88,256)/(96,256) UB 溢出——config 空间封闭性结论跨工具链存活，opt_log 第六轮 Phase 0）；2026-09-20 4515de8 重验存活（task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z 重跑：算术惩罚主选 + band 域 vselect 全量门禁绿）
repro: repro-missing
---

### 第五轮（2026-09-15）：两相位 causal 多头域调优——vcmp 标量化陷阱与算术惩罚掩码（14.2x）

- **瓶颈画像**：两相位 v3 baseline 在 causal 多头域（manifest 5 组）aiv_scalar 94–97%、引擎 <6% 忙碌——**vcmp int16 全形态标量化**（236µs/链 @[32,256]，对角块聚合 ≈ 壁钟 45%）；因子分离实测：单头 causal 比单头非causal **慢 28x/块**（同 bm/bn 同块工作量），多头/strided-KV 因子无罪。
- **算术惩罚掩码**（band-free trace，`S_kv % bn_eff == 0`）：`scr = clamp(diff - K_blk, 0, 1)`（diff = j-i 经负步长 arange `[-1,1]` 一次性物化，f32 精确整数）→ `S += scr·(-1e38)`——全向量化、masked 位精确 -1e38（f32 大数吸收小 S）与 vselect 假支逐位一致。**宽块红利在掩码修复后才显形**：bn=256 vs 144 再 −40%（修复前宽块反升 27%——「宽块摊销」假设须先除标量化税）。
- **NaN 免疫边界**：加法掩码无法清除 NaN（NaN+x=NaN；vmax/vmin 钳位只治 ±Inf，P8 探针）——band-carrying trace（ragged/fractal-min，`S_kv % bn_eff != 0`）必须保留 vselect 选择语义（值替换）；**gemm2 K 维 fractal band 的 0×NaN=NaN 整列污染**（P_band=0 × V_band=未初始化 l1_b NaN 位）用 **zbuf l1_b 每核一次零初始化**根治（[bn_eff,dim] 零张量内部参数，成本 +0.06%）——「stale V rows contribute exactly 0」类假设对 NaN 位不成立（baseline 潜伏同缺陷；上下文依赖触发：完整门禁上下文 3/3 复现、孤立重跑 0/12——前置 case 的 L1 残留决定触发）。
- **−1e38 哨兵与 softcap 顺序（VP-2026-0014 两证合入补遗，首轮 expert 设计实证 + 第五轮复用）**：masked 哨兵用有限 −1e38 替 −inf——e^{−1e38−m} 对有限 m 下溢为精确 0 与 e^{−inf} 同效，从根上规避 −inf−(−inf) 的 NaN 路径；mask 应用放 softcap 之后——mask→softcap 顺序下 softcap(−1e38)≈−softcap 非精确 0，破坏 P(OOB)=0 不变式（softcap 先行、掩码后置可同时吃掉 NaN/Inf 垃圾分数）。
- **trace 分派形态**：宽度切换缓冲（`fast_w/legacy_w = bn_eff if cond else 16`）+ 纯名常量 if 预组合（parser 折叠边界详见 traps-compiler.md TRAP-tvm-parser-rules：else 分支破坏折叠、表达式条件生成 runtime if）；softcap trace 的 vtanh 内部 UB 工作区 ∝ bn（>192 溢出，帽 192）。
- **符号约定变体（2026-09-17 ssd_chunk_scan 第二族实证补充）**：`pen = vmin(i−j, 0)·PEN`（符号约定与 clamp(diff−K_blk,0,1)·(−1e38) 相反、同为有限大数哨兵）——j>i 位 `exp(x + (−1e30))` 下溢为**精确 +0.0**、j≤i 位惩罚恒 0 指数不变（torch 逐位验证 `torch.equal=True`）；PEN 取**有限**大数（1e30/1e38）而非 −inf 是规避 `−inf−(−inf)=NaN` 的关键（与上条哨兵 bullet 同源）。`T.vmin(idx,0,pen)`→`T.vmul(pen,PEN,pen)`→`T.vadd(diff,pen,diff)` 三式均为文档化形（vmin/vadd 标量形有生产先例：engram_fwd.py L76）；分派落地 = 工厂 trace-time `Q % bs` 判定 + 常量 buffer 按域分配不混占 UB。
- **终值**（msprof op Task Duration，median of 20，(64,256)）：smoke 49.6 / 8b-short 395 / 8b-long 1191.6 / 70b-short 399.8 / 70b-long 1207 µs（fp16）——**vs 两相位 baseline 4.4–14.2x（几何 8.53x）**，长 KV 从上一代单遍 0.78x 回退反转为 11.1x 领先。config 空间封闭性：flag 预算（bn ≥ ceil16(ceildiv(S_kv,15))）使 band-free 宽度只剩 256；UB（×1.10–1.12 开销）封 bm≥88；bm=80 长域 −4.9% 但 smoke +16.7% 回退被域检查拒。
- 溯源：`examples/multi_head_attention/_gqa_prefill_fwd_kernel/perf_opt/`（opt_log Phase 1 判别链 + Rounds 1–4 + perf_records.jsonl + profiles/phase1|round2|round3|final/）；探针 probe_vcmp_forms/forms/nan_clamp/rt_slice2.py；复现：`python perf_opt/bench.py perf_opt/_gqa_prefill_fwd_kernel.py --case 8b-long --use-default-config --msprof-loop 25` 外部 msprof op 同 opt_log 命令模板。

---
id: PL-1.12-task-pipeline-depth2
kind: pattern
family: [attention, expert]
apis: [T.sync_block_set, T.sync_block_wait, T.copy, T.gemm, T.alloc_L1]
dtype: [fp16, bf16]
device: 910B2C
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260916T041000Z（第六轮 Stage 4）
toolchain: tilelang 0.1.2+a13585dc / CANN 8.5.0 / Ascend910B2C；2026-09-20 4515de8 重验存活（task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z 重跑：深度 2 流水 + 消费侧前导 set + wait 任务头前置结构复刻，首编即过、L0–Boundary 全绿）
repro: repro/PL-1.12-bn-clamp-bm-guard.py
---

### 第六轮（2026-09-16）：task 级流水深化——屏障冗余审计 + Cube 深度 2 + f32 S 载体（roofline Ratio 口径）

- **TASKDONE 屏障冗余审计（最大单点，全域 −7~−21%）**：两相位 persistent 结构的 task 边界全 drain 屏障（Cube 等TASKDONE 才 hoist 下一 task）在数据竞争角度**冗余**——ws_s/ws_p/ws_o 的所有跨 task WAR 由「Cube 单指令流程序顺序 + P-ready 消费链 + AIV 串行」闭合（逐对象论证：ws_s[T+1] 写者晚于 p2[T] 的 wait(i)（蕴含 Vec p1[T] 已读）；ws_p[Vec T+1 写] 晚于 S-ready(T+1)（Cube T+1 set，晚于 p2[T] 读）；ws_o[Cube T+1 写] 晚于 P-ready(T+1)（Vec T+1 set，晚于同 AIV 的 p2[T] 读））。删除后 task T+1 的 Cube pass-1 与 task T 的 Vector pass-2/epilogue 重叠。**方法**：Phase 0 列出每个 barrier 保护的对象集，逐对象找更细粒度的已有信号覆盖——零实验成本定位。
- **Cube 深度 2 任务流水**（短域 −7.4%）：指令流重排 prologue[hoist(0)+p1(0)] + 迭代 [hoist(T+1)+p1(T+1,slot_n); p2(T,slot)]——p2(T) 的 P-ready 等待窗口被下一 task 的 S 生产填充。**前提**：Q 迁独立 l1_q（否则 Q-hoist(T+1) 与 p2(T) 的 P staging 在 l1_a WAR，附赠长域 −4.3%——l1_q 解除 WAR 提升调度自由度）；ws 加任务槽维 [24,2,...]；flag id = i + slot×nk_total，**可行性判据 2×nk_total ≤ 15**（nk=8 长域超限，保持单槽 r6a 形态）。flag 计数时间线：同 id 的 set/wait 按依赖链严格交替（Cube 的 p2 wait 保证不超前 Vec p1 的 set）。
- **Vec 侧同构改造 blocked（反向教训）**：p1(T+1) 提前后，其 S load（MTE2 写 ub_f16_ND）与 p1(T) 尾块 P store（MTE3 读同 buffer）背靠背——**跨引擎同 buffer WAR 无同步原语**（r7f 中被 p2/epi 时间距离侥幸掩盖）；修复需第二份 ND staging（UB 贴限）或自等自身 flag（双 AIV gather 死锁风险）。**引擎独占 buffer 是流水深度 >1 的隐含前提**。另：legacy 单 flag-id 空间下 advance 会消费 task T 的 O-ready 计数读未写数据（L1 全挂实证）。
- **f32 S 载体迁移（同型 dtype 对照法）**：bf16（f32 ws_s）比 fp16（f16 ws_s）short 快 6.7% → fp16 域 ws_s 改 f32（省 vcast up、S 无损传输），short −2.3~−5.0%；长域 +1.3% 真实回退（fix/GM 流量 ×2，A/B 交错复测 977→989 两轮方向一致）——**短域赚长域亏，per-shape 决策按噪声阈值与域检查基准判定**。
- **终值**（msprof Task Duration / Ratio=Perf/359.33 TOps/s，(64,256) + per-shape bm80（S≥256∧D=128））：8b-long 989.26µs/23.63% / 70b-long 1003.19/23.14% / 8b-short 292.25/14.66% / 70b-short 288.99/14.66% / smoke 44.07/2.68%（fp16）——vs 第五轮终值全域 −11~−28%。**Ratio 指标结构**：Perf = counted FLOPs/duration（Computility 按 48 核计而实机 24），奖励利用率不奖励 padding 缩减——bm=80 短域「时长+2.6% + counted flops+17% → Ratio+1.6pt」在该口径下合法但须如实披露。
- 溯源：`examples/multi_head_attention/_gqa_prefill_fwd_kernel/perf_opt/`（opt_log 第六轮 + perf_records round 5–9 + profiles/r6_phase1|r6_round6|r6_final/）；ratio_extract.py（bin roofline 提取）。

- **update（2026-09-17 ssd_chunk_scan Stage 4，同族轻量形态）**：**消费侧前导 set** 变体——ws 加槽维 ×2 + 4 flag（ready/cons × slot），Cube 循环前 prologue `set(cons0); set(cons1)`（表达"初始空闲"），Vector 侧 T=0/1 的 slot-free wait 立即通过、**免除运行时分支**；每 flag 的 set/wait 严格交替（prologue-set → V-wait(T) → C-set(T) → V-wait(T+2)…），无死锁。与 PL-1.12 的指令流重排形态互补（本变体不动循环结构，只改握手拓扑）。实测 w2 −39.6%（608→367µs，ping-pong 解除）。**配套铁律**：跨引擎 ws 必须**块连续布局**——band 化（行 stride 512B）使 Vector MTE3 时间 3×（106→220µs），吞掉全部 Cube 收益（PL-1.14）。
- **update（2026-09-16 r9 precision_fix，同任务第七会话）——bn 钳位域 × bm=80 的 UB 耦合与分派守卫**：E6 flag 预算守卫在 S_kv>3840 时把 bn_eff 钳到 >256（如 S_kv=4096→288，nk=15，band_carry 使 legacy 掩码链全宽 [half,288]），叠加 r7e 的 bm=80（half=40）使 BishengIR UB 需求 210080B > 196608B/AIV（"ub overflow"，`--enable-auto-multi-buffer=false` 下实测；该域 {Sq≥256, D=128, S_kv≥3841} 此前无测试/manifest 覆盖）。**修复 = 分派守卫收紧**：bm=80 域追加 `bn_min ≤ TUNED_DEFAULT_BN`（bn 钳位域回退 bm=64）——全部 UB buffer 首维 = half，需求随 half 线性缩放（210080×0.8≈168KB，裕量 ~28KB），性能域（S_kv≤2048）分派输入不变零影响（实测 8b-long fp16 +0.08% / 8b-short fp16 +0.09%，run 噪声）。**两个可迁移判据**：① per-shape bm 放宽必须与 bn 钳位/E6 类守卫做**联合 UB 核算**（两守卫各自安全、组合超限——单点证明纪律同样适用于 dispatch 规则组合）；② 本结构手工核算 214080B vs 实际 210080B（ratio 0.981，actual **低于**手工）——×1.10–1.12 通胀系数（PL-1.11/CG-2026-0008）在 auto-multi-buffer=false 的显式 alloc 结构不必然成立，核算时留 12% 余量即可、不必按 1.7x 悲观。守卫 delta 骨架 + 镜像核算公式（含 pre-fix 失败复现断言）：`repro/PL-1.12-bn-clamp-bm-guard.py`；编译失败全档案 `perf_opt/logs/r9_precision_fix/repro_full_fwd_bf16_r8h.log`（provenance，允许失效）。

---
id: PL-1.13-aiv-dup-subid-split
kind: pattern
family: [expert, mixcv, persistent]
apis: [T.Kernel, T.Scope, sync_block_set, sync_block_wait, T.serial]
dtype: [fp16, bf16]
device: 910B2C
status: verified
origin_task: ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z（Stage 4；2026-09-17 蒸馏 D2 溯源归位——原回写误标 20260917T0855Z，实际 task_id 以 .stage_state.json 为准）
toolchain: tilelang 0.1.2+1990aa9fe4 / CANN 8.5.0 / Ascend910B2C / 2026-09-17；2026-09-20 4515de8 重验存活（task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z 重跑复刻蛇形分片，全量门禁绿）
repro: repro/PL-1.13-aiv-dup-subid-split.py
---

### Mix kernel 双 AIV 重复执行与 subid 分片（−28~−36%）

- **现象与判据**：`T.Kernel(n, is_npu=True)` + `T.Scope("Vector")` 的程序默认在每 block 的**两个 AIV 上重复执行**（Mix Block Dim = 2×Block Dim）。profile 判据：PipeUtilization 中 vector0/vector1 子块的 vec/mte ratio **完全相同**而非各半（ssd_chunk_scan w2: 两 AIV vec_ratio 均 0.66）。未分片时每 AIV 承担 100% 向量工作（GQA "dual-producer" 同 flag 语义的成因）。
- **分片形态（GQA v11 边界表达式推广）**：`for i in T.serial(lt_count): lt = i + subid + (i%2)*(L_tiles − 2i − 2*subid)` + 双向 clamp——蛇形均衡（L=4 → {0,3}/{1,2}，s-block 权重 5/5）；退化 L 折叠为重复 lt=0（bit-identical 良性）。分片维度须与 ws 写区域正交（按 lt 分片 × ws 按 lt 索引）；双 AIV 仍执行相同 set/wait 序列（one-set-multi-wait 已证）。**禁用 if 守卫**（TRAP-tvm-parser-rules 运行时分支风险）。
- **实测**（msprof op Task Duration，median of 20）：w2 323.01→211.85µs（−34.4%）、w3 −35.9%、w4 −27.9%；蛇形 vs 交错：w2 ab_test **tie**（run-state 双态），w3/w4 −1.8~−2.4%——按「打平→负载均衡优先」采纳蛇形。配套深度 2 任务流水 + ws 块连续布局（见 PL-1.12 update / PL-1.14），baseline→final 全域 −64~−68%（2.79–3.10×）。
- 溯源：`examples/ssd_chunk_scan/_ssd_chunk_scan_fwd_kernel/perf_opt/`（opt_log R4/R5 + perf_records round 4/5）；结构骨架与机制归因见 `repro/PL-1.13-aiv-dup-subid-split.py`。

---
id: PL-1.16-expert-dualscope-bypass
kind: pattern
family: [attention, expert, mixcv, persistent]
apis: [T.Scope, T.alloc_L1, T.alloc_L0C, T.alloc_ub, T.sync_block_set, T.sync_block_wait, T.gemm]
dtype: [fp16, bf16]
device: 910B2C
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z（首证，VP-2026-0013）/ ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z（第二证：persistent+gemm 模式级不兼容触发类扩展，mamba 族）
toolchain: 首证 tilelang dev root build 21586b5（2026-09-07）；第二证 tilelang 0.1.2+1990aa9fe4 / CANN 8.5.0 / Ascend910B2C（2026-09-17）；2026-09-20 4515de8 重验存活（task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260920T122332Z 重跑：双关闭编译 + L0–Boundary 全量门禁绿——且比旧 perf_opt 终版的单关闭更强，UB 记账更贴手工核算）
repro: repro/TRAP-DEVMODE-PERSIST-GEMM.py
---

### Expert 双 Scope 流水形态（Developer 阻塞的结构级绕法）✅ 两任务实证

- **判据**：Developer kernel 出现 ① aiv_scalar >50%（谓词 mask 预填标量化）② CG-2026-0001 崩溃类（Pipelined 体内条件构造 / 跨块计算重叠 / 标量谓词写）③ **persistent 分核 + gemm 混排运行时崩溃**（traps-runtime.md TRAP-DEVMODE-PERSIST-GEMM——ssd 第二证扩展的触发类；user_requirement 指定 Developer 时以该实测为仲裁依据切换）之一时，评估 Expert 双 Scope 形态再定编程模式，而非在 Developer 内回退。
- **结构形态**：双 Scope（Cube：T.alloc_L1/L0C + load_nd2nz/T.copy + T.gemm + T.store_fixpipe；Vector：T.alloc_ub + v 前缀链）；跨引擎数据经 GM workspace 多槽 + per-slot flag（T.sync_block_set/wait 握手）；staggered stream；运行时 if 在 Expert T.serial 流内合法（Developer 的 CG-2026-0001 禁忌不带入 Expert）。
- **Expert 硬边界**：kernel 内无 fragment 抽象（T.alloc_fragment/T.Pipelined/T.Parallel 不可用）；`pass_configs` 关闭 `TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION` 与 `NPUIR_ENABLE_AUTO_MULTI_BUFFER`（highperf/GQA 先例；ssd 第二证：Expert 手动 CV split + 多 UB buffer 时不关闭该 pass 会重排/重作用域 buffer → codegen "cannot find variable" 崩溃）；wrapper 契约兼容（工厂内层闭包 + workspace 显式参数——workspace 需求不构成改 wrapper 的理由）。
- **实测收益/代价**：attention 族长 KV 1.57–1.63×、短 KV ~1.31× 回退（PL-1.7 同门对照——短 workload 为主的算子慎选）；ssd（mamba 族 MixCV persistent）：Developer 崩溃 → Expert 切换后全量门禁绿 + Stage 4 调优几何 2.91×。
- **未文档化假设**：T.sync_block_set/wait 的 id 预算无文档（仅 T.set_flag.md §2.1 标 event_id 0–15）——flag 族 id 数量以先例推断（constants.md CONST-flag-id-budget ≤15/核）。

---
id: PL-1.18-ssd-steady-structure-floor
kind: pattern
family: [expert, mixcv, persistent, mamba]
apis: [T.gemm, T.copy, T.alloc_L1, T.sync_block_set, T.sync_block_wait, T.Kernel]
dtype: [fp16, bf16]
device: 910B2C
status: verified
origin_task: ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z（Stage 4 第二轮 R7/R8，w4/w3 取景框）/ ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T003531Z（二轮调优 update：段数更正 + 三胜出 + 五否决 + repro 登记）
toolchain: tilelang 0.1.2+1990aa9fe4 / CANN 8.5.0 / Ascend910B2C / 2026-09-17；2026-09-21 二轮调优 tilelang 0.1.2+15ad002b3d（与 4515de8 同源，git diff 零改动）/ CANN 8.5.0 / Ascend910B2C
repro: repro/PL-1.18-floor2-wins.py
---

### SSD chunk scan 稳态结构地板：Cube mte2 段数墙的五方向否决（R7/R8）

- **稳态画像（w4：B2·C128·Q256·H64，16384 任务 / 24 核 = 683 任务/核串行）**：Cube mte2 **83.5% 忙比**（3217µs，12978 条 nd2nz，~256ns/条 = 64 段 × ~4ns/128B，段传输主导）+ cube_wait 0.919 / mte1_wait 0.908（数据供应饥饿）；AIV vec 62% **非关键路径**。**短任务串 workload（w2：32 任务/核）的 mte2 72% 是流水爬坡瞬态、低估引擎占比——persistent 任务流水 kernel 的瓶颈诊断须以最长任务串 workload 画像为取景框**。
- **段数墙构成**（每任务 Cube mte2）：ws_lcb band 重组 640 段（Σ(lt+1)=10 块）+ x 256 + ws_c 128 + prev 128 ≈ 1216 段 × ~4ns —— 这是「Vector 产因子 → GM ws 中继 → Cube 消费」Expert 结构的物理流量（GQA 读放大 H/G 由 L2 吸收，cube read_hit 91% / AIV 99%，非带宽墙）。**拆分口径注（2026-09-20 ssd 重跑任务 Stage 2 检视登记）**：与 constants.md CONST-mte2-degradation 指令维度口径的段数拆分互斥（彼处记 ws_lcb 640 + ws_c 256 + x 256 + prev 64）——总和一致（≈1216）而分项矛盾，源出两任务不同估算/标定路径；引用以总段数为准，分项拆分待下次 Stage 4 段数墙 profile 复核厘清。〔2026-09-21 复核解决：两套拆分与总数均漏算 prev_states 的 lt 循环 ×4 重读——真实 240KB/1920 段（指令数互证见下方 update①），引用以指令数互证口径为准〕
- **五个候选方向的实测否决**（全部 msprof op 同 session 同口径）：
  1. **band 增量组装**（嵌套包含 → L1 跨 lt 累积，640→256 段）：**数学不可行**——band 块 (lt,s_blk) 内容 = lcb[l0+i, s0+j]，**行内容随 lt 变化**（dA_l 依赖 l），band(lt) 与 band(lt−1) 列前缀无公共可复用内容；L0 实测 L_tiles=1 全过、L_tiles≥2 全挂（max_diff 5.9e-3/7.9e-3）。
  2. **深度 3 任务流水**（ws 三槽 + 6 flag ≤15）：w2 +0.1% / w3 −2.7% / w4 +1.4% 平区——**3 任务 in-flight 的 ws 工作集 9.2→13.8MB 劣化 L2 局部性，mte2 每条 256→346ns（busy 85%→93% 但更慢）**；任务流水深度存在 L2 甜点（本结构=2）。
  3. **AIV 减负**（vbrc hoist 等）：无墙钟收益（AIV 非关键）且 hoist 的 vsub alias 形态 +17~20% 税（layout.md PL-1.15 形态二）。〔2026-09-21 修正：税源 = vsub dst=src2 alias 而非 hoist 本身——干净形态（fresh dst + 死后复用）实测 −0.3~−3.5% 胜出（update③）；「AIV 非关键」结论随对侧优化漂移（w2 转 AIV-bound，见 update③ 获益分布）〕
  4. **x 预取**（x copy 提到 factors-ready wait 前）：稳态 wait ≈ 0（AIV 快于 Cube），ab_test 交错协议 tie（+0.48%, p=0.25）——无间隙可填。
  5. **L1 双缓冲软件流水**（_a/_b 槽 + lt 编译期展开 + prefetch 先行）：w2 +11.9% / w3 +17.9% / w4 +18.8%——mte2 纯传输 3217→3056µs（idle 确被填）但 **mte2_wait 0.864→0.976：prefetch 的 MTE2 写（L1 _b 槽）与 gemm 操作数装载的 MTE1 读（L1 _a 槽）在 L1 端口层互拖，+700µs 代价 > ~160µs 收益**——910B2C 此 BiSheng 调度下 L1 双缓冲 family blocked（与 PL-1.9-hardlimits 的 L1 端口常数互证）。
- **〔2026-09-21 二轮调优 update，origin_task ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260921T003531Z（4515de8 重跑谱系 Stage 4），tilelang 0.1.2+15ad002b3d + CANN 8.5.0 + 910B2C〕地板修正与三项新胜出（7 轮 10 分支，几何 1.078×，w2 223.53→204.33 / w3 638.85→601.61 / w4 3902.26→3616.80µs）**：
  1. **段数墙记账修正（D，更正上文 prev 128 段）**：prev_states 拷贝位于 lt 循环内且对 lt 无依赖——Q=256 时每任务 **4× 重读（512 段/64KB，非 128 段/16KB）**；实测 Cube mte2 指令数 19.0/任务（= 1(x)+Σ[1(ws_c)+1(prev)+(lt+1)(band)]）与代码结构精确互证。两套旧拆分（1216 段）均漏算 ×4，真实 240KB/任务（1920 段）。**互证方法（P）**：aic_mte2_instructions ÷ 任务数 对照 kernel 拷贝语句清单——逐维核对循环体内张量索引的循环不变性是发现冗余装载的机械化路径。
  2. **三项胜出结构（P，全 msprof op Task Duration median-of-20 + ab_test）**：①prevhoist——prev 载装载提升出 lt 循环（mte2 字节 −20%、指令 19→16/任务）→ −2.0~−3.4%；②l0c2x——**L0C acc 乒乓**（配对循环 lt=2j/2j+1 交替 l0_acc_a/b，相邻 lt 的 hist→band→out gemm 链 2 深重叠，cube_wait 0.897→0.836）→ −1.7~−4.6%；③vbrchoist——**vbrc hoist 干净形态**（lt 不变行广播提升到 lt 层 + vsub 写 fresh dst〔无 alias〕+ diff_mat 死后复用为 dt 广播目标 = 净零 UB buffer；旧 v11 失败归因 vsub dst=src2 alias 税而非 hoist 本身，本条修正）→ −0.3~−3.5%（AIV-bound 的 w2 与 AIV 共临界的 w4 获益，Cube-bound 的 w3 不响应——逐 workload 约束判定与响应幅度互证）。
  3. **新否决（补入五方向后的清单）**：⑥L1 双缓冲在 prevhoist+l0c2x 新状态下二次否决（+2.9~+6.3%——端口竞争状态无关，mte2 idle ~18-22% 为端口轮转而非可填空闲）；⑦**运行时 if 进 Cube 热循环 = 调度毒**（once-true 守卫 `if jp == 0:` 包 32KB 拷贝，w4 +15.65%——跨迭代调度被控制流打断，远超字节模型预期）；⑧x 装载在 mte2 流头是局部最优（无守卫静态后移也 +3.1~+10.5%：32KB 大块在流头给长启动窗口）；⑨lt 内发射序变体（band 先行/先行+跟进）两形态均回退或无增益（最早 gemm 的操作数必须最先到达）；⑩acc 深度 4 无增益（距离-2 WAR 已被 ping-pong 相位隐藏）。
  4. **字节节省→墙钟换算率 ~20% 现象（D）**：prevhoist 削 20% mte2 字节仅兑现 −2~−3.4% 墙钟（节省的传输时间大部分转化为 mte2 idle 增长：w4 idle 613→910µs）——mte2 忙比不是线性可兑换资源，端口轮转地板下需同时解除 gemm 链串行（②）才能部分兑现。
- **方法论**：稳态画像暴露的「新瓶颈」不必然存在「可打空间」——段数是中继结构的物理流量；结构候选受数学事实（band 行绑定）/ L2 局部性（工作集）/ L1 端口（读写竞争）三重约束夹击，逐一实测是唯一裁决方式。单 buffer 消 idle 的正路是减少段数本身（数学/结构层），而非更深流水/更早预取。
- 溯源：`examples/ssd_chunk_scan/_ssd_chunk_scan_fwd_kernel/perf_opt/`（opt_log R7/R8 + §5 清单第二轮五行；perf_records round 7/8；分支文件 _opt_v10/v11/v12/v13/v14_*.py）。
