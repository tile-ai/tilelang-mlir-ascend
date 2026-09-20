# 硬件常数表（设计期 roofline 与调优定量口径）

> 本文件是 pattern-library 主题文件之一（入口与预算见 [INDEX.md](INDEX.md)）。**本表是已实测硬件常数的唯一事实源**（D-2：evolver 蒸馏新 D 类常数时追加进本表，Tier 0）；设计期 roofline 估算协议（如何用本表算下界）见 `_shared/standards/hardware-cost-model.md`。条目带 front-matter（schema 见 INDEX.md §3），`source` 指向首次实测条目 ID（provenance，允许失效——常数数值与适用条件已自包含于本表）。
>
> **使用纪律**：① 引用常数须带版本戳（工具链变更后自动待重验，`kb_stale_check.py` 检测）；② 常数绑定设备（当前全部为 Ascend910B2C）；③ 外推前核对适用条件（如 MTE2 曲线绑定 4-set 轮换访问模式）。

---
id: CONST-capacity-910B2C
kind: constant
family: [general]
apis: []
dtype: []
device: 910B2C
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260908T005751Z / mish-optimize-2026-08 / multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z（第六轮 r9 反例数据点）/ ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z（第三数据点：Expert 显式 alloc 结构依赖确认 + bl=128 容量否决）
toolchain: tilelang 0.1.2+3a214cde + CANN 8.5.0（L1/L0C）；2026-08-24 build（UB 膨胀）；2026-09-15 重校 6797758；2026-09-16 反例 a13585dc；2026-09-17 ssd 1990aa9fe4
source: PL-1.9-blockwidth / PL-1.9-hardlimits / TRAP-UB-multibuffer-inflation / PL-1.12-task-pipeline-depth2
repro: repro/PL-1.12-bn-clamp-bm-guard.py
---

### 片上容量三硬上限（Ascend910B2C）

- **UB = 192KB/AIV**；auto-multi-buffer 会使 UB 占用**膨胀 ~1.7x**——大 block_size 探索须按膨胀后预算评估（BishengIR `ub overflow, requires N bits while 1572864 bits available` 报错文本反推）。**〔2026-09-15 第五轮重校，工具链 6797758〕auto-multi-buffer=false 下 BishengIR 实际需求仍 ≈ 真实手工预算 × 1.10–1.12**（4 点标定：(64,256) +9.6% / (80,256) +12.2% / (96,256) +12.0% / (128,256) +12.1%——CG-2026-0008；设计期 UB 表须按此系数放大，且勿漏计小 buffer）。**〔2026-09-16 第六轮 r9 反例数据点，工具链 a13585dc〕系数结构依赖**：显式 alloc 结构（全部 UB buffer 首维 = half 的逐 buffer 显式分配，`--enable-auto-multi-buffer=false`）实测手工核算 214080B vs BishengIR 实际 210080B（ratio **0.981，actual 低于手工**）——×1.10–1.12 上偏系数在该结构不成立，该结构实测 ~12% 余量即足（不必按 1.7x 悲观；正负两向偏差均无文档，逐 buffer 分配可见性诉求见 CG-2026-0008）。档案：attention.md PL-1.12 r9 update；核算镜像 + pre-fix 溢出断言：`repro/PL-1.12-bn-clamp-bm-guard.py`。**〔2026-09-17 ssd_chunk_scan 第三数据点，工具链 1990aa9fe4，Expert 显式 alloc 结构——不计 CG-0008 occurrences，属结构依赖确认〕**：Expert 直差分因子链手工峰值 ~179KB ≈ 上限 93%；v3 实测 +16KB 手工 → 195.1KB 实测报错（偏差 <2%，显式 alloc ≈ 手工核算再证）；bl=128 变体编译实测需 582KB（bs=128）/ 357KB（bs=64）vs 192KB——容量否决（DESIGN §1.6.3 变体 B 以实测常数定格）。
- **L1 (cbuf) = 512KB/核**（`cbuf overflow, requires N bits while 4194304 bits available` 反推，2026-09-08）。双槽 k/v/p 的 bn 上限公式：`2·slots·bn·dim·2 + slots·bm·bn·2 + bm·dim·2 ≤ 512KB`（bm=44 时 bn≤427；wide 单槽模式 bn=512 可行，k+v+p+q=312KB）。
- **L0C (cc) = 128KB**（`cc overflow, requires 1310720 bits while 1048576 bits available` 反推）：`l0c_s[bm,bn]f32 + l0c_o[bm,dim]f32 ≤ 128KB` ⇒ bn=512 时 bm≤51。

---
id: CONST-L1-port-bw
kind: constant
family: [attention, expert, cube]
apis: [T.gemm]
dtype: []
device: 910B2C
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260909T033622Z
toolchain: tilelang 0.1.2+3a214cde + CANN 8.5.0 / 2026-09-09
source: PL-1.9-hardlimits
repro: repro-missing
---

### L1 端口读写带宽 ≈ 148–154 GB/s/核

探针复刻 KV+gemm1 数据流 + 真实 kernel 反推双证。ping-pong 双槽**不能**解除 MTE2（GM→L1 写）与 MTE1（L1→L0 读）的端口读写串行——Cube 操作数流的吞吐地板。用法：persistent 结构的 KV 重读流量（tasks/核 × 2MB × 2）÷ 此率 = 该结构的算子级下界（fa4096 ≈ 125–131µs 实测例）。

---
id: CONST-vector-launch-overhead
kind: constant
family: [attention, expert, elementwise]
apis: [v-prefix ops]
dtype: [fp16, f16, fp32]
device: 910B2C
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260909T033622Z
toolchain: tilelang 0.1.2+3a214cde + CANN 8.5.0 / 2026-09-09
source: PL-1.9-hardlimits
repro: repro-missing
---

### 向量算子发射开销 ~0.5µs/op；逐元素速率 f16≈f32

无 f16 打包加速；每 op ~0.5µs 固定发射成本主导——同元素总量下宽块 op（bn=512）恒优于窄块 op 翻倍（bn=256）。用法：向量链时延下界 ≈ op 数 × 0.5µs。

---
id: CONST-fabric-bw
kind: constant
family: [attention, expert]
apis: []
dtype: []
device: 910B2C
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260909T033622Z
toolchain: tilelang 0.1.2+3a214cde + CANN 8.5.0 / 2026-09-09
source: PL-1.9-hardlimits
repro: repro-missing
---

### fabric 聚合带宽 ≥1.73 TB/s（24 核 × 72GB/s，pp 探针）

shared=distinct=pp 无争用惩罚。fa4096 级 workload 现聚合 1.41TB/s 未触顶——地板因子排序通常为 L1 端口 / 向量发射率先于 fabric。

---
id: CONST-mte2-degradation
kind: constant
family: [elementwise]
apis: [T.copy]
dtype: [fp16]
device: 910B2C
status: verified
origin_task: lerp_tensor-_make_lerp_tensor_kernel-20260907T025419Z / ada_layer_norm-_ada_layer_norm_kernel-20260910T145715Z（L2 驻留口径补充）/ ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z（指令维度口径补充）
toolchain: tilelang 0.1.2+ed787bb（2026-09-07 build）/ CANN 8.5.0；L2 口径 tilelang 0.1.2+a83118285a / 2026-09-10；指令维度口径 tilelang 0.1.2+1990aa9fe4 / 2026-09-17
source: PL-1.6-copy-floor / CASE-norm-adalayern-stage4
repro: repro-missing
---

### MTE2 有效带宽随总流量退化（4-set 轮换，fp16 bs4096，48 vector core）

1M 33.9 → 16M 30.2 → 64M 20.9 → 256M 18.4 GB/s/core；256M 的 3:1 R/W 混合全流量 **~1.26 TB/s 为该设备地板**——"~1.8 TB/s 峰值参考"对混合读写流量不可达（勿据其估算大 N headroom）；1M 档读带宽含 L2 成分。grid-stride（48 核聚集移动窗口）实测优于每核连续分块 +3.2%（256M）。

**L2 驻留口径（ada_layer_norm 2026-09-10 补充）**：sets=1 访问模式（同输入张量逐 launch 复用，Stage 5 bench harness 同款）下 MTE2 有效读带宽 43.1 GB/s/core（聚合 ~2.07 TB/s，2048×4096 fp16，43.1×48 核）——远高于上表 HBM 侧曲线同量级值；带宽数字须注明数据驻留状态。设计期 roofline 按上表 HBM 口径估 L2 驻留 workload 会高估时延（ada 实证：DESIGN 估算 53–67µs vs 实测 90.5µs 基线〔失准主项为发射/重叠〕，调优后 40.5µs 优于估算下界——sets=1 下带宽不是地板项，瓶颈转为 Vector/UB 流量 vec_ratio 0.91）。

- **update（2026-09-17 ssd_chunk_scan Stage 4 追加指令维度口径）**：GM→L1 nd2nz 指令代价 ≈ **300ns/指令**、≈ **4ns/128B 段**（w2 标定：16 指令/任务×32 任务 = 156µs mte2 busy；段数 = ws_lcb 640 + ws_c 256 + x 256 + prev 64/任务）——小块搬运是**指令数/段数受限**而非带宽受限（41GB/s ≪ 148GB/s L1 端口）；减少指令数（合并装载）或增大行宽（128B→256B 段）是仅有的两个削减方向。AIV 重复执行时两 AIV 子块指标镜像（判据见 PL-1.13）。

---
id: CONST-copy-floor-method
kind: pattern
family: [elementwise]
apis: [T.copy, T.vadd]
dtype: [fp16, bf16, fp32]
device: 910B2C
status: verified
origin_task: lerp_tensor-_make_lerp_tensor_kernel-20260907T025419Z
toolchain: tilelang 0.1.2+ed787bb（2026-09-07 build）/ CANN 8.5.0
source: PL-1.6-copy-floor
repro: repro/CONST-copy-floor-method.py
---

### copy-floor 标定法（方法学常数）

与目标 kernel 同结构（同 persistent/分块/映射）的 3 载入 + 2 vadd 保活（DCE-proof）+ 1 写出探针，一步判定计算链暴露量。实测例：lerp_tensor 7-pass fp32-transit 链 ≥16M 全 dtype delta ≤1.1µs（16M 仅 0.34µs）——auto multi-buffer 将全部向量 pass 隐藏于 MTE2 窗口内。标量削减判定式：mte2_ratio ≥0.95 时 per-tile 标量/分支开销已被隐藏。

---
id: CONST-flag-id-budget
kind: constant
family: [attention, expert]
apis: [sync_block_set, sync_block_wait]
dtype: []
device: 910B2C
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260909T071018Z
toolchain: tilelang 0.1.2+3a214cde + CANN 8.5.0 / 2026-09-09
source: PL-1.9-twophase（VP-2026-0042 证据链）
repro: repro-missing
---

### flag id 预算 ≤ 15/核

跨引擎 per-slot flag 的 id 数量预算上限（第二轮 round-10 否决两遍式时的推理输入之一——该推理本身被第三轮"全局 [S,S] ws + n-block 下标 flag（nk≤16）"推翻，但预算上限本身实测存活）。两相位结构用 n-block 下标作 flag（nk≤16）贴上限可行。

---
id: CONST-store-fixpipe-gm-only
kind: constant
family: [attention, expert, cube]
apis: [store_fixpipe]
dtype: [fp16, f16, fp32]
device: 910B2C
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260909T071018Z
toolchain: tilelang 0.1.2+3a214cde + CANN 8.5.0 / 2026-09-09
source: PL-1.9-hardlimits（[DESIGN_LIMIT] 修正附录存活清单）
repro: repro-missing
---

### store_fixpipe 仅支持 L0C→GM（存活硬上限）

S/P 类中间矩阵无法经 fixpipe 直写 L1/UB——跨引擎传输强制 GM 往返（S-f16 传输是字节减半的缓解，非消除）。store_fixpipe f32(L0C)→f16(GM) 为正确数值转换（max_rel 4.7e-4 = 1 f16 ulp）。

---
id: CONST-aicore-910B2C
kind: constant
family: [general]
apis: []
dtype: []
device: 910B2C
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z
toolchain: 2026-09 build
source: PL-1.7-expert-persistent-boundary
repro: none
---

### 物理核数 = 24 AI Core（Ascend910B2C）

persistent 结构按 24 核设计（logical tasks = ceildiv(S,bm)·H·B；wall-rows = ceil(ceildiv(S,bm)/24)·bm——bm=44 使 S=1024/2048/4096 得 44/88/176 行）。实查命令：`NPUUtils.get().get_aicore_num()`（设计期分核三要素之一，禁止假设值）。
