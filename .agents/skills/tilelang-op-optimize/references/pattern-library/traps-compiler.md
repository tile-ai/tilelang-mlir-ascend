# 已知编译器/解析器陷阱（工具链版本绑定 ⚠️）

> 本文件是 pattern-library 主题文件之一（入口与预算见 [INDEX.md](INDEX.md)）。条目带 front-matter（schema 见 INDEX.md §3）；`repro: repro-missing` 表示待回填最小复现代码。运行时/数值/语义类陷阱见 [traps-runtime.md](traps-runtime.md)。
>
> **证伪协议（强制，canonical 见 INDEX.md §2，两者同步演进）**：
>
> 1. **否定任何 API/模式前，必须用文档合法形态测试**——先查 `docs/Tilelang.language/` 确认 API 的合法参数/形式，穷举代表性写法后再下结论。实测教训：曾以非法 3-cycle permutation（`[1,2,0]`）与 Parallel 循环累加形式测出"编译失败"，误判"transpose 链/C 轴累加不可用"，掩盖了 2–13x 收益。
> 2. **一切编译器约束结论必须盖工具链版本戳**（tilelang commit/build 时间 + 来源任务），工具链变更（源码修改/重编译）后**自动视为待重验**，不得直接引用旧结论。
> 3. 证伪更正时须在 opt_log 写明"误判根因 + 合法形态 + 新数据"。

---
id: TRAP-C1-old-compiler-transpose
kind: trap
family: [general]
apis: [T.transpose]
dtype: []
device: 910B2C
status: overturned
origin_task: AvgPool2dFwdOp-optimize-2026-08
toolchain: 2026-08-27 前旧 build
repro: repro-missing
---

### C1 旧编译器 transpose/C-slice 形态不可用

**已失效**（2026-08-27 tilelang 重编译后推翻；合法形态见 [layout.md](layout.md) PL-1.1/PL-1.2）。

---
id: TRAP-C10-strided-gather
kind: trap
family: [general]
apis: []
dtype: []
device: 910B2C
status: verified
origin_task: AvgPool2dFwdOp-optimize-2026-08
toolchain: 截至 2026-08-28 build
repro: repro-missing
---

### C10 跨步 gather 向量指令缺失

非 C 轴连续的跨步访问小心——跨步 gather 向量指令缺失（截至 2026-08-28 build 有效）。

---
id: TRAP-C11-parser-vartable
kind: trap
family: [general]
apis: []
dtype: []
device: 910B2C
status: verified
origin_task: mixed（2026-08 溯源）
toolchain: 截至 2026-08-28 build
repro: repro-missing
---

### C11 parser var-table 作用域问题

有效（截至 2026-08-28 build）。

---
id: TRAP-C12-copy-dtype-cast
kind: trap
family: [elementwise]
apis: [T.copy]
dtype: [fp16, bf16, fp32]
device: 910B2C
status: verified
origin_task: lerp_tensor-_make_lerp_tensor_kernel-20260907T010433Z
toolchain: 截至 2026-08-28 build；lerp 证伪 2026-09-07
repro: repro-missing
---

### C12 `T.copy` 静默跨 dtype 转换

有效（截至 2026-08-28 build）；Developer GM→UB lowering 为隐藏 hivm::VCastOp（`docs/Tilelang.language/内存操作/T.copy.md` §3），跨 dtype copy 不减少 vector pass 且舍入不可控——勿用作 cast 融合（lerp_tensor 2026-09-07 文档证伪）。

---
id: TRAP-UB-multibuffer-inflation
kind: trap
family: [elementwise, general, reduction]
apis: [T.Pipelined, T.serial]
dtype: []
device: 910B2C
status: verified
origin_task: mish-optimize-2026-08（首证）/ lerp_tensor-_make_lerp_tensor_kernel-20260907T010433Z（第二证）/ ada_layer_norm-_ada_layer_norm_kernel-20260910T145715Z（第三证：结构依赖精测；2026-09-10 蒸馏溯源归位——原回写误标 145312Z，实际 optimize task_id 以 stage_state.json 为准）/ multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z（第四证首源：基础分配差发现）/ multi_head_attention-_gqa_prefill_fwd_kernel-20260915T080600Z（第四证重校：×1.10–1.12 4 点标定 + UB diet 绕法）
toolchain: 2026-08-24 mish 首证；2026-09-07 lerp_tensor 第二次独立证据；2026-09-10 ada_layer_norm 精测（tilelang 0.1.2+a83118285a + CANN 8.5.0）；2026-09-15 基础分配差（首源 tilelang 0.1.2+28783f45 / 重校 0.1.2+6797758，均 + CANN 8.5.0）
repro: repro-missing
---

### auto-multi-buffer UB 膨胀：footprint 与 staging 数量解耦（~20/26 B/elem 平台）+ num_local/静态 extents 依赖

大 block_size 探索须按膨胀后预算评估；削 buffer 不必然解锁（lerp fp32 3-buffer 12288/16384 仍 blocked）。有效（2026-08-24 mish 首证；2026-09-07 lerp_tensor 第二次独立证据：`ub overflow, requires 1966080 bits while 1572864 bits available`）。

**第三证精测（ada_layer_norm，2026-09-10，编译溢出报文逐配置读数）**——三条可操作规律：

1. **膨胀后 footprint 是"结构类平台值"，不随 staging 数量线性增长**：persistent serial（num_local_tasks≥2）激活时，fp16/bf16 中转循环（vcast 链）无论 resident 10 B/elem（3 缓冲）还是 14 B/elem（5 缓冲），multi-buffered 需求同为**精确 20.01 B/elem**；fp32 直连无论 8 B 还是 16 B resident，同为**精确 26.00 B/elem**。⇒ UB 预算 guard 从"resident × 膨胀系数"改为"按结构类查平台值"（transit≈20/21、fp32≈26/27 B/elem）。
2. **需求随 num_local_tasks 变化**：同配置（bm=8×N=1152 transit）nl=3 可编译（20.01 B/elem=184.3KB）、nl=4 溢出（22.00 B/elem=202,784B）——分核参数（num_kernels）会改变 UB 可行域，调 num_kernels 必须连带复验编译。
3. **静态 slice extents 也改变需求**：`M % block_m == 0` 特化（real_m 常量化、静态 extents）使 transit 需求 20.01→22.00 B/elem——静态化优化在 UB 贴限时可能直接翻车。

**第四证/姊妹现象（2026-09-15 两任务实证，VP-2026-0059 两证合入）——基础分配差（multi-buffer 关闭仍在）**：`--enable-auto-multi-buffer=false` 下 BishengIR per-AIV UB 实际需求 ≈ 真实手工预算 × **1.10–1.12**（首证 28783f45：178.2KB 预算实测 ~204.1KB +15%；第二证 6797758 4 点标定：(64,256) +9.6% / (80,256) +12.2% / (96,256) +12.0% / (128,256) +12.1%）——手工逐 buffer 预算表**不能当可编译证明**，设计期按系数放大且勿漏计小 buffer（第五轮实证：DESIGN §4.5 漏计 ub_cond2 8.2KB）；宽 config 上靶前以编译探针校准（`ub overflow` 报文 requires bits 即精确实测需求）。系数已录入 constants.md CONST-capacity-910B2C；与 CG-2026-0008（能力缺口本体，recurring）互链。绕法实证：UB diet 削减手工预算 ≥12.4KB（N/D staging 合并 −24.6KB + rowmat 删除 −16.3KB）解锁 (64,256) 宽块。**反例边界（2026-09-16 第六轮 r9，a13585dc）**：显式 alloc 结构（全 UB buffer 首维 = half）实测 0.981×manual（actual 低于手工）——系数结构依赖、非单向上偏，该结构 ~12% 余量即足（constants.md CONST-capacity-910B2C / PL-1.12 r9 update）。

机制推测（未实证）：pass 按 buffer 角色/循环形态决定复制哪些实例，非全量 ×2。溯源：`examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/perf_opt/opt_log.md`（Iteration 2/3 探针记录 + probe_ub_v2.py）；第四证溯源：首证 `examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/integration_log.md` Round 2 + 第二证 `examples/multi_head_attention/_gqa_prefill_fwd_kernel/perf_opt/opt_log.md` Phase 1 编译探针（probe_ub.py，origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260915T080600Z）。

---
id: TRAP-vbrc-scalar-shared
kind: trap
family: [expert]
apis: [T.vbrc, T.if_then_else]
dtype: []
device: 910B2C
status: verified
origin_task: mixed（2026-08 溯源）
toolchain: 截至 2026-08-28 build
repro: repro-missing
---

### vbrc 标量→shared 不可用；serial 变量条件 segfault

vbrc 标量→shared 不可用；serial 变量条件 if_then_else select segfault（有效，截至 2026-08-28 build）。

---
id: TRAP-threads-kwarg-noop
kind: trap
family: [general, migration]
apis: [T.Kernel]
dtype: []
device: 910B2C
status: verified
origin_task: lerp_tensor-_make_lerp_tensor_kernel-20260907T010433Z
toolchain: tilelang-mlir-dev dev root build 2026-09-07 源码通读
repro: repro/TRAP-threads-kwarg-noop.py
---

### `T.Kernel(..., threads=)` 在 npuir 无效果

`src/ir.cc` L175–203 KernelLaunch NPU 分支仅消费 grid_size、block_size 不生成任何 threadIdx 绑定（Python 签名仍接受并透传该 kwarg）；GPU 源码迁移遇 `threads=` 直接移除即可，threads/npt 折叠为单一 block_size 由 wrapper 参数传入（复现：`python repro/TRAP-threads-kwarg-noop.py`——机械核对 src/ir.cc NPU 分支无 threadIdx 绑定）。有效（tilelang-mlir-dev dev root build 2026-09-07 源码通读；origin provenance：lerp_tensor 任务 REVIEW.md 附#28——任务工作区，未上库，允许失效）。

---
id: TRAP-tvm-parser-rules
kind: trap
family: [expert]
apis: [T.prim_func, T.rs, T.alloc]
dtype: []
device: 910B2C
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z
toolchain: tilelang dev root build 2026-09-07（HEAD 21586b5）+ CANN 8.5.0 + Ascend910B2C；developer 谱系 v3_deadbuf 跨工具链二次一致（tilelang 67db6f3 + CANN 26.0.rc1）
repro: repro/TRAP-tvm-parser-rules.py
---

### TVM script 解析器四条硬规则

违反报 Undefined variable / trace error：`if` 语句一律生成 TIR runtime if（不折叠、赋值作用域困在 if 体内）→ trace-time 选择用标量三元式或 Python 层预计算；三元式**两臂都会求值**（buffer 切片臂不能用哨兵 tensor → 无条件全尺寸 alloc）；条件 `T.alloc` 非法（哨兵只可用于 shape 三元式且该 buffer 不得在运行 if 体内引用）；`with T.rs()` 块内局部量块外不可见（标量常量提到 Scope 顶层）。条件 alloc 规则与 developer 谱系 v3_deadbuf 教训跨工具链二次一致。复现：`python /tmp/opencode/probe_forms.py on`（四形态逐一验证，session-local；镜像于 `examples/multi_head_attention/_gqa_prefill_fwd_kernel/debug_log.md` D4）。

**〔第六轮补充实证，2026-09-16，tilelang a13585dc + CANN 8.5.0，origin_task multi_head_attention-_gqa_prefill_fwd_kernel-20260916T041000Z〕**精确折叠边界修正（上文「一律 runtime if」过粗）：**纯名 Python bool 的无-else `if X:` 在 trace 时折叠**（第五/六轮 `if band_free:` / `if use_pipe2:` 均产出单 trace，实证两次）；**else 分支的存在破坏折叠或作用域**——`if use_pipe2: slot = task_id % 2 else: slot = 0` 报 Undefined variable（即使条件是纯名常量），绕法 = factory 层 Python int 常量 + 单算术表达式 `slot = (task_id % 2) * pipe2_on`（r7f Vec slot 报错→算术消解通过；r8g 生成脚本同型失败三证）。runtime 条件 `if task_id + 1 < N:` 内的 T.serial/rs/变量定义合法（r7f Cube 段编译通过）。

---
id: TRAP-expert-v-operands
kind: trap
family: [expert, attention]
apis: [T.vcmp, T.vbrc, T.vtanh]
dtype: [fp32, int16]
device: 910B2C
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z
toolchain: tilelang dev root build 2026-09-07（21586b5）+ CANN 8.5.0 + Ascend910B2C
repro: repro-missing
---

### Expert v 算子操作数规则

`T.vcmp` 标量操作数拒绝 `tir.Cast`（codegen `NpuirOperand::FromExpr cannot handle the expr with type of "tir.Cast"`）；**〔2026-09-15 第五轮实测补充，工具链 6797758〕vcmp 在 int16 上对所有操作数形态（运行时标量 PrimExpr / 文档合法的同 shape int16 张量）一律降级为标量指令执行**（编译警告 `will execute by scalar instruction with low efficiency`，[32,256] 实测 **236µs/链**）——在 causal mask 链上占整 kernel 壁钟 ~45%（aiv_scalar 94–97% 的主成分，与 flag 自旋同形，须查编译警告区分）。**绕法 = 算术惩罚掩码**：`scr = clamp(tpl - thr, 0, 1)`（vmax/vmin 标量广播，文档合法）→ `S += scr·(-1e38)`（f32 大数吸收，masked 位精确变 -1e38，与 vselect 假支逐位一致；NaN 无法被算术掩埋——stale 列/band 垃圾须 vselect 选择语义或源头零初始化）。标量 PrimExpr B 操作数合法有先例（`examples/deepseek_v4/example_sparse_attn_kernel_highperf.py` L353）；`T.vbrc` 标量源须 let 绑定局部量（字面量触发 rank 断言）；`T.vtanh` 需 fp32 同 dtype 操作数（int16 scratch 不可复用，MLIR verify 报错）。**v-op 切片仅接受静态边界**（运行时 Var 触发 parser `int(Var)` 拒绝，P10 探针）；**脚本解析器只折叠纯名常量 if**（`not X`/`X and Y` 表达式生成 TIR runtime if，须预组合纯名）；**条件 T.alloc 永不折叠**（含常量条件，v3_deadbuf 的强化实证——用宽度切换 `w = X if cond else 16` 替代）。证据：debug_log.md D4/D5 + 本轮任务工作区 `examples/multi_head_attention/_gqa_prefill_fwd_kernel/perf_opt/`（溯源：probe_vcmp_forms.py P6 / probe_forms.py P1-P5 / probe_rt_slice2.py P10 / opt_log.md Round 2）；origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260915（第五轮 Stage 4，工具链 6797758）。

**〔第六轮补充实证，2026-09-16，tilelang a13585dc，origin_task 同上〕**v-op 操作数拒绝**带 runtime 首维索引的多维 buffer 切片**：`T.vlog2(ub_ell_snap[st, 0:half, 0:1], ...)`（snap 为 [2,half,1]，st 是 runtime Var）在 codegen `getBroadcastDim` 报 `buffer_shape0.size() == buffer_shape1.size()` 检查失败（3 维 vs 2 维不折叠）；`T.vmul(ub_m_snap[st, ...], ...)` 同型。绕法：**先 `T.copy` 拷回单槽 buffer 再喂 v-op**（plain `T.copy` 的源/目标接受 runtime 索引切片——`ws_s[kernel_id, slot, ...]` 先例，r8g epi 修复实证）。与既有「v-op 切片仅静态边界」（P10）同族：静态边界 + runtime 首维索引亦不可用。
---
id: TRAP-transpose-epilogue-poison
kind: trap
family: [attention, expert, general]
apis: [T.transpose]
dtype: [fp16, f16]
device: 910B2C
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260909T071018Z
toolchain: tilelang 0.1.2+3a214cde + CANN 8.5.0 + Ascend910B2C，2026-09-09
repro: repro-missing
---

### 活跃源 `T.transpose([N,1]→[1,N])` epilogue 毒化整 kernel（2.6x）

epilogue 位置的 transpose 若其源为活跃生产者（被前序 v-op 写过且结果被消费），整个 kernel 管道重叠崩塌（实测 fa4096 98→255µs 级）；单 op 无罪（vlog2/vmul/vadd/死源 transpose 各自 97µs 级）、4 个平凡 vadd 无罪——**组合阈值效应，pass 层根因未定位（未文档化假设，依据 = 14 点 morph 实测矩阵）**。绕法：目标 GM 张量增维视图（如 [B,H,S,1]，与 [B,H,S] 同内存）+ UB→GM 自然 2D 区域拷贝（`ub[0:real_m,0:1] → dst[·,·,bx:bx+real_m,0]`）+ host 侧 reshape 还原契约形状——零 transpose、零拷贝。**全谱系存量税**：v9 级 per-block kernel 同样中招（226.72→181.46µs，-20%，perf-only 探针）。证据：`perf_opt/opt_log.md` round 11 morph 矩阵（14 点二分）+ perf_feedback.md 修正附录 + `profiles/round11/`、`profiles/final3/` raw；复现：morph 阶梯脚本形态与 14 点实测矩阵镜像于 opt_log round 11 Diagnosis（scratch `/tmp/opencode/ref_morph.py`，session-local），终值对账 `msprof op --kernel-name=_gqa_prefill_fwd_main_mix_aic --launch-count=20 --warm-up=5`（median of 20）跑 bench `--use-default-config`。
