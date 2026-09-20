# 已知运行时/数值/语义陷阱（工具链版本绑定 ⚠️）

> 本文件是 pattern-library 主题文件之一（入口与预算见 [INDEX.md](INDEX.md)）。条目带 front-matter（schema 见 INDEX.md §3）；`repro: repro-missing` 表示待回填最小复现代码。编译器/解析器类陷阱见 [traps-compiler.md](traps-compiler.md)。
>
> **证伪协议（强制，canonical 见 INDEX.md §2，两者同步演进）**：
>
> 1. **否定任何 API/模式前，必须用文档合法形态测试**——先查 `docs/Tilelang.language/` 确认 API 的合法参数/形式，穷举代表性写法后再下结论。
> 2. **一切运行时/数值结论必须盖工具链版本戳**（tilelang commit/build 时间 + 来源任务），工具链变更后**自动视为待重验**，不得直接引用旧结论。
> 3. 证伪更正时须在 opt_log 写明"误判根因 + 合法形态 + 新数据"。

---
id: TRAP-C9-taskqueue-async
kind: trap
family: [general]
apis: []
dtype: [fp32]
device: 910B2C
status: verified
origin_task: mixed（2026-08 溯源）
toolchain: 截至 2026-08-28 build
repro: repro-missing
---

### C9 `TILELANG_ENABLE_TASKQUEUE=false` 异步 launch 损坏 fp32 数据

有效（截至 2026-08-28 build）。

---
id: TRAP-UB-dst-align
kind: trap
family: [general]
apis: [T.copy]
dtype: []
device: 910B2C
status: verified
origin_task: mixed（2026-08 溯源）
toolchain: 截至 2026-08-28 build
repro: repro-missing
---

### UB dst 非零起点切片 + 32B 倍宽度触发 VEC 对齐错误

有效（截至 2026-08-28 build；host pad 或 0 起点拷贝绕开）。

---
id: TRAP-fp16-opmath-golden
kind: trap
family: [elementwise, migration]
apis: [vcast, vadd, vsub, vmul]
dtype: [fp16, bf16]
device: 910B2C
status: verified
origin_task: lerp_tensor-_make_lerp_tensor_kernel-20260907T010433Z
toolchain: torch 2.9.0+cpu + tilelang-mlir-dev dev root build 2026-09-07
repro: repro-missing
---

### torch CPU golden 的 fp16 opmath 域分歧

torch.lerp（torch 2.9.0+cpu）对 fp16 输入经**fp32 opmath + 单次舍回**计算，NPU fp16 原生域三步链（逐步 fp16 舍入）与之差 ~2–3 ulp fp16，N=2^24、atol=rtol=1e-3 下违反率 ~0.125% 且与 shape/block_size/核数无关（纯舍入路径统计性质，非 tiling/同步缺陷）；bf16 golden 与 fp32 中转 kernel 逐位一致。对齐通解：`vcast(rint)` 升 fp32 → fp32 域 v-prefix 链 → `vcast(rint)` 单次舍回，差 ≤1 ulp fp16（rtol≥5e-4 即覆盖）；NaN/Inf 角点 IEEE 传播不受中转影响。复现（provenance，允许失效）：`python examples/lerp_tensor/_make_lerp_tensor_kernel/history_version/_make_lerp_tensor_kernel_impl_s3_attempt1.py --level L0`（fp16 违反）vs `python examples/lerp_tensor/_make_lerp_tensor_kernel/_make_lerp_tensor_kernel.py --level all`（全过）。

---
id: TRAP-load-nd2nz-strided
kind: trap
family: [attention, expert, cube]
apis: [T.load_nd2nz, T.copy]
dtype: [fp16, bf16, fp32]
device: 910B2C
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z
toolchain: tilelang dev root build 2026-09-07（HEAD 21586b5）+ CANN 8.5.0 + Ascend910B2C
repro: repro/TRAP-load-nd2nz-strided.py
---

### `T.load_nd2nz` 对跨步（非尾二维连续）src 区域静默平坦误读

BSHD [B,S,H,D] 按固定头取 (S,D) tile（区域 [1,real_m,1,dim]，dim1 步长 H·D）被按「基址起平坦连续内存」读取——无告警无报错，数值表现为整块乱值（ws_s 与任何 head 的 QK^T 均不匹配；flat-read 假设逐点复现 got[i,j]==flat_q[i]·flat_k[j]）；`T.copy` **base+size 形态**对相同跨步区域同样错误（diff 3.4），**slice 形态** `T.copy(q[bz, s_lo:s_lo+real_m, by, 0:dim], l1[0:real_m, 0:dim])` bit-exact 正确且同为 PIPE_MTE2 单次搬运——跨步 GM 区域装载的实测正确形态为 slice 形态 T.copy（文档「src 支持 2-4D tensor」未标注连续性约束；能力缺口登记 CG-2026-0004）。复现：`repro/TRAP-load-nd2nz-strided.py`（知识域——slice 绕法 bit-exact 断言）；原 session 探针（provenance，session-local 允许失效）：probe_stridecopy.py（slice vs base+size 对照）、probe_l1slot.py——关键发现镜像于任务 debug_log D1/D2。

---
id: TRAP-T-copy-region-semantics
kind: trap
family: [attention, expert, general]
apis: [T.copy, T.transpose]
dtype: [fp16, f16, fp32]
device: 910B2C
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z
toolchain: tilelang dev root build 2026-09-07（HEAD 21586b5）+ CANN 8.5.0 + Ascend910B2C
repro: repro/TRAP-T-copy-region-semantics.py
---

### `T.copy` 区域语义三规则

① 标量基址 + size 的 extents **前向补 1**（size=[1,N] 落末维、[N,1] 落倒数第二维；`tilelang/language/copy.py` L42-95 源码语义）——lse 写出方向由它决定；② **src/dst 区域元素数不匹配时引擎按 src 搬运并越界写坏相邻 GM**（`[0:half]` 源写 `[0:real_m_half]` 目的区砸坏相邻 tensor，症状酷似跨 pipe 竞态——仅部分核数据坏、set_flag/wait_flag 无效；判据 = 溢出值恰为源 buffer 内容；只在尾块位形 Sq%bm≠0 出现，整除 shape 全量测试发现不了；正确形态 = src 切片 `[0:real_rows]`）；③ [N,1] UB 源 + size=[1,N] 按步长越界读（MTE DDR fault）——行向量写 GM 尾维连续区先 `T.transpose` 到 [1,N]（Developer shared 源不需要，UB 源必须）。**推荐通用形态：src/dst 双显式 slice**。复现：`repro/TRAP-T-copy-region-semantics.py`（知识域——合法 slice 形态断言）；原 session 探针（provenance，session-local 允许失效）：probe_lsemulti2.py、probe_copyforms.py——关键发现镜像于任务 debug_log D2/D3。**〔2026-09-10 重验注记〕**repro 转正重跑（repro/TRAP-T-copy-region-semantics.py）：slice 形态 PASS 存活；历史 base+size=[HALF,D] 标量基址形态（原会话通过）在当前工具链复现 MTE DDR fault——按版本戳规则该形态**待重验**，条目结论以 slice 形态为准。

---
id: TRAP-zero-input-crash
kind: trap
family: [expert, general]
apis: [T.prim_func]
dtype: [bf16, fp16, fp32]
device: 910B2C
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z
toolchain: tilelang dev root build 2026-09-07（21586b5）+ CANN 8.5.0 + Ascend910B2C
repro: repro/TRAP-zero-input-crash.py
---

### 零输入 kernel（out_idx-only）运行期必崩 "MTE DDR address out of range"

启动参数错乱——崩溃形态与真实越界写完全同款，曾误诊为拷贝形态 bug 消耗 3 轮探针；探针 kernel 保留 ≥1 个输入 tensor 后同形态全过。诊断「部分核/部分行数据坏」先做元素级 fp64 对照判定哪侧偏离真值（本例 golden 才是偏离方：|kernel−true|=4.3e-4 vs |golden−true|=5.8e-3，bf16 P 量化噪声放大），避免在错误层面（跨 pipe 竞态假设）空转。复现：`repro/TRAP-zero-input-crash.py`（知识域——dummy input 绕法断言）；原 session 探针（provenance，session-local 允许失效）：probe_copymin.py 加 dummy input 前后对照——关键发现镜像于任务 debug_log D2/D6。

---
id: TRAP-vrsqrt-plain-precision
kind: trap
family: [norm, reduction, elementwise]
apis: [T.vrsqrt, T.vsqrt, T.vdiv]
dtype: [fp32, fp16, bf16]
device: 910B2C
status: verified
origin_task: ada_layer_norm-_ada_layer_norm_kernel-20260910T132324Z
toolchain: tilelang 0.1.2+a83118285a + Ascend910B2C + CANN 8.5.0 + torch 2.9.0+cpu / 2026-09-10
repro: repro/TRAP-vrsqrt-plain-precision.py
---

### `T.vrsqrt` plain 模式为近似指令（max rel err ~2.9e-3）

npuir 上 `T.vrsqrt` lower 为 plain 近似 Vector 指令：输入几何扫描 [1e-6, 1e6] 实测 max rel err 2.87e-3（返回值量化到 ~10–11 位有效数字，如 1010/1024）。传播效应（ada_layer_norm Stage 3 实测）：fp32 1e-5 门禁下 87% 元素违反、fp16 1e-3 门禁下 6%、bf16 1.6e-2 宽容差掩盖——「fp32 大面积违反而 bf16 全过」是常数相对误差（近似指令）的指纹，属舍入路径性质而非数据依赖/同步缺陷。`examples/norm/layer_norm.py` 通过的 1e-2 容差会掩盖该问题，不构成 vrsqrt 精度佐证。

**绕法**（ada_layer_norm 交付形态）：`T.vsqrt` + `T.vdiv` 组合（sqrt(v)/v ≡ 1/sqrt(v) 实数恒等，两 op 全精度）实测 max rel err 1.07e-7；修正形式 Newton×2 迭代亦可达 6.3e-8（代价 12 个微型 op）。含 rsqrt 且容差 <1e-2 的算子（layernorm/rmsnorm/softmax 归一化族）在本工具链上以组合形态达标。定位手法：中间量分层导出（`out_idx` 多输出落 GM + fp64 精确值对照）单次运行把误差定位到具体指令（ada 案例：mean/d 精确 1.1e-7 而 rstd 偏 1.57e-3 ⇒ 误差独占于 vrsqrt 段）。配套事实：文档文件名与 API 导出名存在偏差是常态（T.rsqrt.md → `T.vrsqrt`；T.vLn.md → `T.vln`），`examples/` 实调代码是 API 名核对入口。

复现：`python repro/TRAP-vrsqrt-plain-precision.py`（断言量级：raw > 1e-3 现象存在 + bypass < 1e-6 绕法通过；工具链修复后首断言翻转即条目推翻信号，届时刷新版本戳）。溯源（provenance，允许失效）：`examples/ada_layer_norm/_ada_layer_norm_kernel/`（Stage 3 分层探针 + Implementation Notes attempt-1 precision fix 段；repro 原件 `examples/ada_layer_norm/_ada_layer_norm_kernel/repro/TRAP-vrsqrt-plain-precision.py`）。

---
id: TRAP-L1-band-dst-tail-overrun
kind: trap
family: [runtime, l1, mixcv]
apis: [T.copy, T.alloc_L1]
dtype: [fp16, bf16]
device: 910B2C
status: verified
origin_task: ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z（Stage 4；2026-09-17 蒸馏 D2 溯源归位——原回写误标 20260917T0855Z，实际 task_id 以 .stage_state.json 为准）
toolchain: tilelang 0.1.2+1990aa9fe4 / CANN 8.5.0 / Ascend910B2C / 2026-09-17
repro: repro/PL-1.13-aiv-dup-subid-split.py
---

### L1 band 组装 dst 列区间未按尾块裁剪 → 越界写污染相邻 L1（布局敏感潜伏缺陷）

- **形态**：向 `[bl, Q]` L1 band buffer 逐块搬运时写 `l1[0:bl, s0:s0+bs]`——末 s-block 的 `s0+bs > Q`（Q%bs≠0）越出缓冲，越界写破坏相邻 L1 分配。**触发依 L1 布局而变**：fp16 同 shape 靠布局运气通过（L0 PASS 假象），bf16 布局命中关键 operand（max_diff 7.8e-3）——单 dtype 门禁放行双 dtype 契约的潜伏缺陷。
- **绕法**：src/dst 列宽同步裁剪 `ts = T.min(bs, Q−s0)`（`ws[..., 0:bl, 0:ts] → l1[0:bl, s0:s0+ts]`）；行维可保留 [0:bl]（gemm size 的 tmc 排除 stale 行，输出裁剪丢弃）。
- **教训**：分支精度回归须含 **bf16 × 非整除 shape** 组合（本缺陷潜伏 4 轮，仅 Boundary 用例拦截）；另注：tilelang 磁盘缓存会在源变更后命中旧二进制（同 max_diff 复现"修复无效"假象）——kernel 源变更后清 `~/.tilelang/cache`。
- 溯源：`examples/ssd_chunk_scan/_ssd_chunk_scan_fwd_kernel/perf_opt/`（opt_log §6.1；logs/final_level_all.log 修复前后）。

---
id: TRAP-DEVMODE-PERSIST-GEMM
kind: trap
family: [mixcv, expert, persistent]
apis: [T.Kernel, T.gemm, T.Scope, T.vmul]
dtype: [fp16, bf16]
device: 910B2C
status: verified
origin_task: ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z（Stage 3）
toolchain: tilelang 0.1.2+1990aa9fe4 / CANN 8.5.0 / Ascend910B2C（npu-smi 26.0.rc1）/ 2026-09-17
repro: repro/TRAP-DEVMODE-PERSIST-GEMM.py
---

### Developer 模式 + persistent 分核 + gemm 混排运行时崩溃（"unaligned UUB addresses"）

- **现象**：persistent `T.Kernel(24, is_npu=True)` 核内 `T.serial` 多任务 + `T.gemm` + v-prefix 向量 op 混排的 kernel 在 Developer 模式（auto CV-split）**运行时崩溃**——"Illegal instruction, which is usually caused by unaligned UUB addresses"（vector core exception，retCode=0x31）；Expert 模式（显式 `T.Scope("Cube")`/`T.Scope("Vector")` + alloc_L1/alloc_L0C/alloc_ub + sync_block_set/wait）同结构正常。
- **判定依据（仓库实态扫描）**：全仓 24 处 `is_npu=True` 中 Developer 模式仅用于非 persistent 网格（`examples/flash_attention/flash_attn_npuir_dev.py`、`examples/deepseek_v32/fp8_lighting_indexer.py`），persistent 混合算子（先例：`examples/multi_head_attention/_gqa_prefill_fwd_kernel/`〔任务工作区，溯源〕、`examples/deepseek_v32/sparse_mla_fwd_exp.py`）全部 Expert——「persistent 分核 + gemm」落在 Developer 支持域之外（docs 无 Developer/Expert 适用范围条款；未文档化假设，依据本任务双模式 repro + 全仓用例扫描）。
- **模式切换实证**：本任务 user_requirement 指定 Developer 模式，Stage 3 首实现即崩；切换 Expert（GQA/sparse_mla 先例 + `pass_configs={TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION: False}`，不关闭时多 UB buffer 被 pass 重排/重作用域 → codegen "cannot find variable"）后 L0/L1/L2/Boundary 全过——Expert 双 Scope 是该结构类的可用形态（结构级绕法模式见 attention.md PL-1.16；能力缺口登记 CG-2026-0010）。
- 复现：`python repro/TRAP-DEVMODE-PERSIST-GEMM.py`（Expert 路径数值断言通过 + Developer 路径崩溃观察——设备故障 out-of-band 即现象本体，工具链修复后 Developer 路径存活即条目推翻信号，届时刷新版本戳）。溯源（provenance，允许失效）：`examples/ssd_chunk_scan/_ssd_chunk_scan_fwd_kernel/repro/DEVMODE_PERSIST_CRASH.py`（双模式对照原始探针）+ 同目录 RETROSPECTIVE.md Stage 3 章节。

---
id: TRAP-BENCH-CONFIG-CALIBRATION
kind: trap
family: [benchmark, harness, stage4]
apis: []
dtype: [fp16, bf16]
device: 910B2C
status: verified
origin_task: ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z（Stage 4 第二轮 R7——续跑口径断裂）
toolchain: tilelang 0.1.2+1990aa9fe4 / CANN 8.5.0 / Ascend910B2C / 2026-09-17
repro: repro-missing
---

### 采数 harness 默认配置与交付配置断裂（续跑/翻转场景的 +19~21% 假回退）

- **现象**：调优产物的 kernel 内嵌 TUNED_DEFAULT_CONFIG（如 block_n=128），但采数 harness（bench/runner）的 argparse 默认回落到自己的 fallback（如 `min(64,N)`）——第一轮靠**显式传参**保持一致（而 opt_log 采集命令模板漏记该参数）；第二轮续跑者按模板走 runner 默认 → 所有分支数据系统性偏慢 +19~21%（w2 258.86 vs 217.65µs），初判为「设备漂移 ±20%」，实际是配置口径断裂。**双配置同 session A/B 探针（bn=64/128：261.55 vs 226.26µs，+15.6%）先于「设备漂移」假设**。
- **绕法**：① bench harness 的默认配置从**被测 kernel 模块的 TUNED 常量**解析（`getattr(mod, "TUNED_DEFAULT_CONFIG", {}).get(...)`），baseline 无常量时 fallback 兼容——使「交付配置=测量配置」成为机械保证；② 续跑/翻转轮次第一步 = current best 的同 session 重测校准（同口径基准行）；③ 调优日志的采集命令模板必须与实际执行完全一致（含全部显式参数）。
- **教训**：跨 session 的 perf_records 对比，「设备状态漂移」与「配置口径断裂」症状相同（系统性偏移）——配置探针成本 ~2 分钟，应优先排除；跨 run 双态（BP_run_state_bimodality ±3~5%）解释不了 ±20% 量级的偏移。
- 溯源：`examples/ssd_chunk_scan/_ssd_chunk_scan_fwd_kernel/perf_opt/`（opt_log R7 口径断裂段；profiles/round7_bn_check/；bench.py R7 修复）。
