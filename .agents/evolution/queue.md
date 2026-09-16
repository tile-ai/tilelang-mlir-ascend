# 进化提案队列（Evolution Queue）

> TileLang-Op-Conductor 自进化机制的提案队列。由 `tilelang-skill-evolver` 维护（唯一写入者）。
>
> - **用途**：暂存未达合入阈值的 P 类（模式方法）提案与待人工审批的 R 类（流程规则）diff 提案；D/C 类（实测数据/案例索引）不进本队列，由 evolver 直接合入 pattern-library。
> - **字段规范**：见 `.agents/skills/tilelang-skill-evolution/references/queue-schema.md`（本文件头部仅列速查）。
> - **状态**：`pending`（待确认/待审批）→ `verified`（Tier 1 证据达阈值待合入）→ `merged`；或 `rejected` / `expired`（90 天未决）/ `conflict`（与现有条目矛盾且无法自动裁决，留人工）。
> - **审阅方式**：Tier 2（R 类）提案经用户批准后由 conductor 调度 evolver（`mode=apply`）执行写入；Tier 1 达阈值后由 evolver 在下次蒸馏时合入。

## 字段速查

```yaml
proposal_id: VP-{YYYY}-{NNNN}     # 唯一 ID，由 evolver 递增分配
type: D / P / R / C               # D 实测数据 / P 模式方法 / R 流程规则 / C 案例索引
title: <一句话标题>
evidence:                         # provenance（ED-A 语义，2026-09-10 起）：来源 task_id + 出处描述——允许失效，不做路径存在性核验
  - <task_id + 出处（如 该任务 opt_log round 3 / 路径#定位——路径允许失效）>
repro: <知识域自包含脚本路径（pattern-library/repro/<id>.py，须存在且可执行）或 repro-missing 或 none>
toolchain_stamp: <tilelang build/commit + 设备 + CANN 版本，或 none>
target_doc: <目标文件仓库相对路径（pattern-library 相关指向 pattern-library/ 主题文件）>
delta: <五种合法动作之一：add / update / consolidate / negate / deprecate + 条目正文>
status: pending / verified / merged / rejected / expired / conflict
confirmations: {n}/{2}            # Tier 1 合入阈值：2 次独立证据（不同任务）；E-3 快速通道：另一上下文 repro_runner/ab_test 复现成功计 2/2
created_by: <task_id + 日期>
decided_by: evolver / human / -   # 裁决者
```

> **历史条目路径映射（2026-09-10 pattern-library 拆分）**：本文件中既有条目的 `target_doc: .../references/pattern-library.md` 与 delta 中的「§1/§2/§4」章节引用，合入时按 pattern-library/INDEX.md 的历史引用对照映射到主题文件（§1→layout/elementwise/attention.md、§2→traps-compiler/traps-runtime.md、§4→cases.md、§1.6 常数→constants.md）——历史提案文本不改写，合入锚点按映射解析。
>
> 完整字段规范：`.agents/skills/tilelang-skill-evolution/references/queue-schema.md`（本文件头部仅列速查）。

---

## Pending

### Tier 1（P 类，2 次独立证据后合入）

## VP-2026-0002

- type: P
- title: sub-fp32 逐元素算子 fp32 中转模式：vcast(rint) 升 fp32 → fp32 域 v-prefix 链 → vcast(rint) 单次舍回（bf16 dtype 支持 + fp16 golden 对齐双触发）
- evidence:
  - examples/lerp_tensor/_make_lerp_tensor_kernel/RETROSPECTIVE.md（Stage 1 bf16 触发 + Stage 3 fp16 修复模式提案）
  - examples/lerp_tensor/_make_lerp_tensor_kernel/_make_lerp_tensor_kernel.py（模块「Implementation Notes (attempt-2 precision fix)」等价性论证 + attempt-2 全量 62 PASS）
  - docs/Tilelang.language/数学操作/T.vadd.md §2.2.1（v-prefix 算术 dtype 矩阵不含 bf16）
  - docs/Tilelang.language/数据类型转换操作/T.vcast.md §2.2.1（f16→f32 仅 rint；f32→f16/bf16 含 rint；bf16↔f32 仅 rint）
- repro: repro-missing（知识域最小 repro 待同族任务回填；原任务内复现命令见 evidence 末行 provenance 项）
  - 〔provenance，允许失效〕任务内复现命令：python examples/lerp_tensor/_make_lerp_tensor_kernel/_make_lerp_tensor_kernel.py --level all
- toolchain_stamp: tilelang-mlir-dev dev root build（2026-09-07）；本机 Ascend aicore=24；golden torch 2.9.0+cpu
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library.md
- delta: |
    add §1 新小节「sub-fp32 逐元素 fp32 中转模式」：
    双触发条件：① 契约 dtype 含 bf16——v-prefix 算术（vadd/vsub/vmul）dtype 矩阵不含 bf16，vcast 升 fp32 是唯一计算路径；② fp16 需对齐 torch CPU golden——golden 经 fp32 opmath + 单次舍回，原生域逐步舍入差 2–3 ulp（见 pattern-library §2 opmath 行）。
    链结构：GM→UB(同 dtype) → vcast(rint)→fp32 → fp32 域 v-prefix 原地链 → vcast(rint)→原 dtype → UB→GM。
    实测：max_diff ≤1 ulp fp16（抽样 binade 4.883e-04～1.953e-03）、全量 0 violations（fp16 的 1e-3 rtol 恒覆盖 1 ulp）；NaN/Inf 角点 IEEE 传播与舍入路径无关（中转后 inf-corner 逐位不变）。
    UB 预算：中转路径按 Σ(elem_bytes×buffer_count) 复算——bf16 24B/elem（上界 196608/24=8192）、fp16 20B/elem（上界 9830，保守 guard 8192）。
- status: pending
- confirmations: 1/2
- created_by: task lerp_tensor-_make_lerp_tensor_kernel-20260907T010433Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0003

- type: P
- title: 纯 Vector 一维逐元素 persistent 模板：num_kernels=min(ceil(N/block), aicore×2) + T.serial 静态边界 grid-stride + 动态尾块零起点切片
- evidence:
  - examples/lerp_tensor/_make_lerp_tensor_kernel/DESIGN.md#5.5（分核三要素）
  - examples/lerp_tensor/_make_lerp_tensor_kernel/RETROSPECTIVE.md（Stage 1 提案 + Transferable Lessons）
  - examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/opt_log.md（persistent nc=48 全档贴 copy 地板；grid-stride 实测最优访问模式）
  - examples/elementwise/vec_add_1d.py L31–L38（尾块处理官方先例）
- repro: repro-missing（知识域最小 repro 待同族任务回填；原任务内复现命令见 evidence 末行 provenance 项）
  - 〔provenance，允许失效〕任务内复现命令：python examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/_make_lerp_tensor_kernel.py --level all
- toolchain_stamp: tilelang-mlir-dev dev root build（2026-09-07）；本机 Ascend aicore=24（纯 Vector 翻倍 48 vector cores）
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library.md
- delta: |
    add §1 新小节「纯 Vector 一维逐元素 persistent 模板」（核数实查/纯 Vector 翻倍规则已有 ascend-constraints.md「物理核数限制」行，仅交叉引用不重复）：
    模板：num_kernels = min(ceil(N/block_size), aicore×2)；核内 T.serial(ceil(num_logical/num_kernels)) 静态边界（N 工厂特化保证编译期常量，可直接用 T.ceildiv+min 求值，见 §1.5 host 层常量折叠行）+ block_id = i×num_kernels + cid + if block_id < num_logical 屏蔽越界。
    尾块：tail = T.min(block_size, N−t0) + UB 零起点切片 T.copy(x[t0:t0+tail], x_ub[0:tail])——v-prefix 整 buffer 操作对 UB 陈旧段一并计算、只拷出 [0:tail]，正确性无碍。
    提取期注意：TileOPs wrapper 的 E3/E4 适配注释（如 threads= 取代）是提取期笼统预估，核数/grid 取值以 DESIGN.md 分核三要素为准，冲突时设计文档优先。
- status: pending
- confirmations: 1/2
- created_by: task lerp_tensor-_make_lerp_tensor_kernel-20260907T010433Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0004

- type: P
- title: BP_grid_stride_vs_contig_chunk：带宽饱和区连续分块映射回退 +3.2%（256M），grid-stride 聚集窗口对 HBM 混合流更优
- evidence:
  - examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/opt_log.md#iteration-3（v3_op2：256M fp16 1765.01 vs 1710.07，+3.2%；1M/16M tie）
  - examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/profiles/round3/v3op2_256m_fp16_bs4096
- repro: repro-missing（知识域最小 repro 待同族任务回填；原任务内复现命令见 evidence 末行 provenance 项）
  - 〔provenance，允许失效〕任务内复现命令：msprof op --kernel-name=main --launch-count=15 ... python bench.py --impl ./_make_lerp_tensor_kernel_opt_v3_op2.py --dtype float16 --N 268435456 --block-size 4096（对照 grid-stride 基线）
- toolchain_stamp: tilelang 0.1.2+ed787bb（2026-09-07 build）/ Ascend910B2C / CANN 8.5.0
- target_doc: .agents/skills/tilelang-op-optimize/references/bottleneck-patterns.md
- delta: |
    add 新 BP 条目 BP_grid_stride_vs_contig_chunk：
    触发信号：带宽饱和区（mte2_ratio ≥0.95）考虑把 grid-stride 任务映射改为每核连续分块以改善「DRAM 局部性」。
    反证/判定：lerp_tensor 256M 实测连续分块 +3.2% 回退、小 N tie——grid-stride 的 N 核聚集移动窗口（fewer open pages / 行缓冲命中）对 HBM 混合读写流更优，「连续流局部性」直觉在混合流下反向；数据侧见 pattern-library §1.6 访问模式条。
    动作：饱和区维持 grid-stride；连续分块仅在纯读/写单一流向或 L2 友好 workload 复测后考虑。
    验证：Task Duration 非回退 + mte2_ratio 不恶化。
- status: pending
- confirmations: 1/2
- created_by: task lerp_tensor-_make_lerp_tensor_kernel-20260907T025419Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0005

- type: P
- title: 标量/分支削减优化在 MTE2 饱和区无效的判定式：mte2_ratio ≥0.95 时标量已被搬运隐藏，先查饱和再立项
- evidence:
  - examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/opt_log.md#iteration-2（v2_op2 静态尾块全档 tie/-0.6%；guard-free 机制证据 defer <0.3%）
  - 同文件 Baseline 表（≥16M mte2_ratio 0.95–0.99；1M 档 0.70–0.84 流水浅区）
- repro: repro-missing（知识域最小 repro 待同族任务回填；原任务内复现命令见 evidence 末行 provenance 项）
  - 〔provenance，允许失效〕任务内复现命令：msprof op ... python bench.py --impl ./_make_lerp_tensor_kernel_opt_v2_op2.py --dtype float16 --N 1048576 --block-size 4096
- toolchain_stamp: tilelang 0.1.2+ed787bb（2026-09-07 build）/ Ascend910B2C / CANN 8.5.0
- target_doc: .agents/skills/tilelang-op-optimize/references/bottleneck-patterns.md
- delta: |
    update BP_ub_traffic_floor 触发信号，追加：「per-tile 标量/分支开销（动态尾块 T.min、if-guard、地址计算）立项削减前，先查 mte2_ratio——≥0.95（搬运饱和）时标量已被隐藏，实测静态尾块/去 guard 全档 tie/-0.6%（lerp_tensor 2026-09-07）；仅 mte2_ratio <0.9 的流水浅区（小 N 档 0.70–0.84）可能有收益」。数据侧见 pattern-library §1.6 标量削减判定式条。
- status: pending
- confirmations: 1/2
- created_by: task lerp_tensor-_make_lerp_tensor_kernel-20260907T025419Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0006
- type: P
- title: 精度失败定征顺序：dtype 隔离 → 违反率与 shape/block/核数无关性 → golden opmath 域小实验（避免盲目排查分核/内存层级）
- evidence:
  - examples/lerp_tensor/_make_lerp_tensor_kernel/RETROSPECTIVE.md Stage 3 Transferable Lessons（定征顺序实证有效）
  - examples/lerp_tensor/_make_lerp_tensor_kernel/history_version/_make_lerp_tensor_kernel_impl_s3_attempt1.py（定征输入：fp16 ~0.125% 违反、bf16 逐位一致、fp32 全过、违反率与 shape 无关）
- repro: 对 attempt-1 基线分 dtype 跑 --level L0，观察违反的 dtype 隔离性与违反率 shape 无关性
- toolchain_stamp: tilelang-mlir-dev dev root build（2026-09-07）；本机 Ascend aicore=24；golden torch 2.9.0+cpu
- target_doc: .agents/skills/tilelang-error-fixer/references/precision-patterns.md
- delta: |
    add 新文件首条目（合入时创建 references/precision-patterns.md）「精度失败定征顺序」：
    ① dtype 隔离：哪个 dtype 挂 / 哪些全过（全 dtype 挂 → 结构/同步缺陷；单 dtype 挂 → 舍入路径统计性质）。
    ② 无关性检查：违反率与 shape/block_size/核数无关 → 纯舍入路径性质，排除 tiling/分核/同步排查方向。
    ③ golden opmath 域小实验：数行脚本核实 golden（torch CPU op）在目标 dtype 的实际计算域（fp16 常见 fp32 opmath + 单次舍回），再决定 kernel 原生域 vs fp32 中转（模式见 pattern-library §2 opmath 行 + VP-2026-0002）。
- status: pending
- confirmations: 1/2
- created_by: task lerp_tensor-_make_lerp_tensor_kernel-20260907T010433Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0007
- type: D
- title: T.Kernel(..., threads=) 在 npuir 无效果（KernelLaunch NPU 分支仅消费 grid_size）——设计侧约束表路由
- evidence:
  - src/ir.cc L175–203（NPU 分支无 block_size→threadIdx 消费路径）
  - examples/lerp_tensor/_make_lerp_tensor_kernel/REVIEW.md 附#28
  - tilelang/language/kernel.py L218–223（Python 签名仍接受并透传）
- repro: grep -n -A 30 "is_npu_kernel_frame" src/ir.cc
- toolchain_stamp: tilelang-mlir-dev dev root build（2026-09-07，源码通读）
- target_doc: .agents/skills/tilelang-op-design/references/ascend-constraints.md
- delta: |
    add §1「本项目已知限制」表新行：
    | GPU `threads=` kwarg 无效果 | `T.Kernel(..., threads=)` 的 block_size 在 npuir KernelLaunch 不生成 threadIdx 绑定（Python 签名仍透传） | GPU 源码迁移残留 `threads=` 无声无效，threads/npt 语义需显式落到 block_size | 迁移时直接移除 `threads=`，折叠为单一 block_size 由 wrapper 参数传入（实证见 pattern-library §2 threads= 行；lerp_tensor 2026-09-07） |
- status: pending
- confirmations: 1/2
- created_by: task lerp_tensor-_make_lerp_tensor_kernel-20260907T010433Z 2026-09-07
- decided_by: -
- decided_note: 三件套齐备；因 target_doc 为 op-design references（Tier 1 写域）而入队——设计期（GPU→NPU 迁移）是该事实的主要消费场景，事实本体已 Tier 0 合入 pattern-library §2。

## VP-2026-0013
- type: P
- title: attention 族 Developer 性能阻塞（aiv_scalar>50% / CG-2026-0001 崩溃类）时先评估 Expert 双 Scope 形态再定编程模式——结构级绕法含硬边界清单
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/DESIGN.md#§0（动因：developer 基线 ~3.5 TOps/s = msprof 算力上限 359 TOps/s 的 ~1%，落后 torch-SDPA 10–25×）+ #§0.6 E2/E3
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/opt_log.md#§0/§4/Round-7（developer 谱系阻塞项来源——**谱系注明**：tilelang 67db6f3 + CANN 26.0.rc1，2026-09-07 晨 developer 轮 Stage 4 产物，非本 expert 任务产物）
  - examples/deepseek_v4/example_sparse_attn_kernel_highperf.py（Expert 结构全同构先例）
  - pattern-library §1.7（expert vs developer 同门 bench：长 KV 1.57–1.63× / 短 KV ~1.31× 回退——Expert 形态的收益分界数据）
- repro: 触发条件——任一 attention 族 Developer kernel 出现 aiv_scalar >50%（谓词 mask 预填标量化）或 CG-2026-0001 崩溃类触发时，对照 highperf 结构评估 Expert 重写（结构形态见 delta）
- toolchain_stamp: expert 侧 tilelang dev build 21586b5（2026-09-07）+ CANN 8.5.0 + Ascend910B2C；developer 阻塞证据侧 tilelang 67db6f3 + CANN 26.0.rc1（跨代对照已注明）
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library.md
- delta: |
    add §1 新小节「Expert 双 Scope 流水形态（Developer 阻塞的结构级绕法）」：
    判据：attention 族 Developer kernel 出现 aiv_scalar >50% 或 CG-2026-0001 崩溃类（Pipelined 体内条件构造 / 跨块计算重叠 / 标量谓词写）时，先评估 Expert 形态再定模式，而非在 Developer 内回退。
    结构形态：双 Scope（Cube：T.alloc_L1/L0C + load_nd2nz/T.copy + T.gemm + T.store_fixpipe；Vector：T.alloc_ub + v 前缀链）；跨引擎数据经 GM workspace 多槽 + per-slot flag（T.sync_block_set/wait）；staggered stream；运行时 if 在 Expert T.serial 流内合法（Developer CG-2026-0001 崩溃面不适用——Developer 的禁忌不带入 Expert 设计）。
    Expert 硬边界：kernel 内无 fragment 抽象（T.alloc_fragment/T.Pipelined/T.Parallel 不可用）；pass_configs 关闭 TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION 与 NPUIR_ENABLE_AUTO_MULTI_BUFFER（highperf 先例）；wrapper 契约兼容（工厂内层闭包 + workspace 显式参数，highperf sparse_attn() 形态——workspace 需求不构成改 wrapper 的理由）。
    实测收益/代价：见 §1.7（长 KV 1.57–1.63×、短 KV ~1.31× 回退——短 workload 为主的算子慎选）；vcmp 标量形态先例与低效警告见 §2 v 算子操作数行。
    未文档化假设：T.sync_block_set/wait 的 id 预算无文档（仅 T.set_flag.md §2.1 标 event_id 0–15）——flag 族 id 数量只能以先例推断（本轮 7 族在安全域）。
- status: pending
- confirmations: 1/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0015
- type: P
- title: Expert 跨引擎 workspace 的跨任务槽位复用握手：per-task 槽位重启 + FLAG_TASKDONE 边界（避免 causal 变长 NK 的全局块前缀索引）
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/DESIGN.md#§6.3（flag 协议安全性传递性论证）+ #§7.2（四族 flag + FLAG_V 写侧 staging 补全）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/REVIEW.md（维度 5 附核对记录：三族 + 单槽 + 跨任务五条论证独立推演全部成立）
  - 〔2026-09-16 第六轮限定，task multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z〕TASKDONE 屏障在两相位 v3 结构经 WAR 全链审计判定**冗余**并删除（全域 −7~−21%）：ws_s/ws_p/ws_o 的跨 task WAR 由「Cube 单指令流程序顺序 + P-ready 消费链 + AIV 串行」闭合（attention.md PL-1.12）——握手模式是保守安全默认而非必要条件，**本条合入时须附该限定**（结构特定的程序序闭合可免除屏障；另见 VP-2026-0080 屏障冗余审计 BP 提案，两案互链）
- repro: 任一 per-task 变长内循环（causal NK 变长）的 Expert staggered 流设计
- toolchain_stamp: tilelang dev root build 2026-09-07（HEAD 21586b5）+ CANN 8.5.0 + Ascend910B2C
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library.md
- delta: |
    add §1 附注（Expert 流水模式条目〔VP-2026-0013〕的握手子模式，合入时紧随其后）：
    模式：per-task 槽位重启（slot = i % slots 任务内重计数、跨任务从 0 重启）+ FLAG_TASKDONE 跨任务握手（Vector epilogue 完成后 set、Cube 下一任务首个 S 存储前 wait）——跨任务槽位复用安全，避免 causal 变长 NK 下的全局块前缀索引（数据依赖、不可静态化）。
    配套：flag 协议安全性按「写 i+slots 前读 i 已完成」逐族传递性推演（set 的程序序前驱 → wait 的后继），跨任务握手族单列；写侧 staging（FLAG_V 族，highperf V0S→C1L 同构）补全读侧无序窗口。
    实测：L2-gap100 四例 + kv32-guard（M7 单块退化共存位形）全过（DESIGN §8.2 / debug_log.md Final results）。
- status: pending
- confirmations: 1/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0016

- type: P
- title: slots≥2 分块装载的尾块守卫从不变式推导（nk_total ≤ slots）而非 shape 枚举——枚举法漏浅块数域；判据同源反推测试域
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/DESIGN.md#§0.6-E7 + #§8.2（缺口域 4 例：S_kv=100 ∈ (bn,2bn)、ns=2，条件与测试同源）+ #§7.2（Vector 零填充预备段）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/REVIEW.md（第 1 轮问题 3：v0 守卫仅覆盖 nk_total==1，slots=2/nk_total=2 时部分块为槽位首次使用——L2 套件 kv32/tail520 均不在缺口域，缺口靠检视推导而非测试暴露）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/debug_log.md（Final results：L2-gap100 四例 causal/non-causal × fp16/bf16 全过）
- repro: repro-missing（知识域最小 repro 待同族任务回填；原任务内复现命令见 evidence 末行 provenance 项）
  - 〔provenance，允许失效〕任务内复现命令：python examples/multi_head_attention/_gqa_prefill_fwd_kernel/_gqa_prefill_fwd_kernel.py --level all（L2 段 gap 四例即判据同源测试域）
- toolchain_stamp: tilelang dev root build 2026-09-07（HEAD 21586b5）+ CANN 8.5.0 + Ascend910B2C
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library.md
- delta: |
    add §1 新小节「多槽分块装载的尾块守卫不变式」：
    判据推导：部分块 p=nk_total−1 的残留安全性 ⟺「该槽位 p%slots 此前已被整块装载」⟺ nk_total > slots；唯一破坏点 = 核启动后槽位首用即部分块（残留为未初始化 L1/GM〔torch.empty〕，0×NaN=NaN 污染有效行）——正确守卫条件 = nk_total ≤ slots（含跨任务归纳：槽位尾行状态恒 ∈ {零, 有限真实数据}）。
    枚举法反例：从 shape 枚举（seq_len_kv < bn ⟺ nk_total==1）出发漏浅块数域（nk_total=2、slots=2）且测试域与条件不同源；判据同源反推测试域 = 按判据取边界值 S_kv ∈ (bn,2bn) + 必须触发 slots=2。
    配套：Vector 零填充预备段 + FLAG_V 握手（无它则 staging 写与 Cube 读无序）；守卫面随路线结构性简化（删载体族后 GM 未初始化域缺口自动消失——路线裁决的「正确性面」第四维）。
- status: pending
- confirmations: 1/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0017
- type: P
- title: bf16 输出精度门禁设计须评估 golden 自身 P 量化噪声的放大位形——tier-2 绝对下限 max(5e-3, 2·flip_rel·vmax) + tier-3 窄域分类机制
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/debug_log.md#D6（L1-70b-long-bf16：1/16777216 元素超 tier-2 1.1%；fp64 逐元素验证 |kernel−true|=4.3e-4 vs |golden−true|=5.8e-3——golden 是偏离方；P 1-ulp 翻转经 causal 早行 p≈0.33×|v|≈2.06 放大 → 5.4e-3 > tier-2）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md（Stage 3 Precision-gate 行）
  - 交叉引用：queue VP-2026-0006（精度失败定征顺序——同文件合入时互链）、pattern-library §2 torch golden opmath 行（golden 侧域差异同类现象）
- repro: bf16 randn 域 attention：causal 早行（有效位少 × P 量化翻转 × 大 |v|）位形对照 tier-2 门限；元素级 fp64 对照定位偏离侧
- toolchain_stamp: tilelang dev root build 2026-09-07（HEAD 21586b5）+ CANN 8.5.0 + Ascend910B2C；golden = torch SDPA NPU
- target_doc: .agents/skills/tilelang-error-fixer/references/precision-patterns.md
- delta: |
    add 新条目「bf16 输出门禁的 golden 噪声放大位形与 tier-3 分类」（合入时与 VP-2026-0006 定征顺序条目同文件、互相交叉引用）：
    机理：golden 自身 P 量化到输入 dtype 的 1-ulp 翻转，在 causal 早行（有效位置少、p 大、|v| 大）放大为输出偏差 5.4e-3 > tier-2（5e-3）——kernel 比 golden 更接近 fp64 真值（4.3e-4 vs 5.8e-3）仍可能超差；该机理是源算法 P-quantization-to-input-dtype 的固有性质，非 kernel 缺陷。
    门禁设计：tier-2 绝对下限应取 max(5e-3, 2·flip_rel·vmax)（bf16 randn 域 ≈ 4e-2）；未重校门禁前的过渡机制 = tier-3 窄域分类：发生率 ≤0.01% + 放大包络（2·flip_rel·|v|max）+ lse 门不受影响 + 逐例上报——真实缺陷（如丢 mask：79.8% mismatch）仍为响亮失败。
    定位手法：元素级 fp64 对照判定偏离侧（golden vs kernel 谁离真值远）再定机制，避免把 golden 噪声当 kernel 缺陷排查（亦见 pattern-library §2 零输入探针行的 fp64 对照法）。
- status: pending
- confirmations: 1/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0018
- type: P
- title: AIV/lane 半分类设计的参数化标量阈值按最劣 lane 全局坐标独立重推 + 双 lane 端点数值例（单 AIV 思维自检必然逃逸）
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/REVIEW.md（第 1 轮问题 1：v0 L220/L536 K_blk = base − row0，正确为 base + row0；smoke bx=4/块 4/vid=1 行 288 数值例复现——正确 j≤32 vs 错误 j≤−32）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/DESIGN.md#§0.6-E3（v1 修复：行映射式 q_pos = s_lo + row0 + i 先行 + 阈值 PrimExpr 标量）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/history_version/design_v0.md → DESIGN.md 修订链
- repro: 任一含 vid/row0 半分 + 块级 mask/阈值构造的 Expert 设计：对每个 per-lane 标量公式代入 vid=0 / vid=1 两个端点数值例
- toolchain_stamp: 设计/检视期证据（无运行时依赖）；审查环境 tilelang 0.1.2 + 910B2C（2026-09-07）
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library.md
- delta: |
    add §1 附注（并入 VP-2026-0014 向量 mask 构造链小节的「分界/阈值推导模板」子条，合入时紧随其后）：
    修复模板：① 先写行映射式（lane 的全局坐标，如 q_pos = s_lo + row0 + i）再代源条件推导阈值（K_blk = base + row0）；② 阈值走 PrimExpr 标量而非动态 offset（T.arange.md offset 参数文档类型为 int，运行时 PrimExpr 支持无文档）；③ 共用分界（K_A 类）对非推导 lane 标注保守性（仅性能不损正确性——取整方向反例见 §4 expert attention 案例行：ceildiv 版对 vid=0 是激进上界、floordiv 版精确）。
    逃逸机理：以 vid=0（row0=0）代入的阈值在 vid=1 恰好相差 2·row0 且符号方向隐蔽（该 +row0 写成 −row0 时 vid=0 完全正确、vid=1 全错）——单 AIV 思维的自检必然逃逸，双端点代入是最低成本拦截。
- status: pending
- confirmations: 1/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0034
- type: P
- title: BP 跨引擎 flag 串行链：dual-Scope persistent 结构下块宽摊减优先于流水深度（bn↑ 同削 flag 往返/向量 op/gemm 碎片，ns/slots 在宽块上失效）
- evidence:
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/opt_log.md#Iteration-1（busy 核分解：aic_scalar 41%/aiv_scalar 33% = flag 等待自旋，aic_cube 仅 9% 近峰值——ns=1→2 在 bn=64 上零变化）+ #Iteration-3（bn=512 单槽 -15~-16% vs G2@bn=256；bn=512 双槽 L1 溢出 invalid_compile）+ #Final-Summary（跨轮 busy 核画像主线：flag 等待 25–47% 恒存，块宽摊减是唯一有效方向）
  - pattern-library §1.9（数据侧：块宽摊减律 + cbuf=512KB 上限公式 + wide 单槽模式 + H=1 平衡公式；origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260908T005751Z）
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/opt_log.md#Round-10（2026-09-09 同谱系第二轮续调补充〔task 20260909T033622Z，按 VP-2026-0035 同 op 谱系先例不计独立确认〕：v10_dp bn=256 深流水全 case 回退 +11~12%——向量发射开销定律〔每 op ~0.5µs 固定成本〕给出块宽摊减律的机制解释；另实测第三硬上限 L0C=128KB）
- repro: 同结构 kernel（Expert dual-Scope persistent + per-slot flag）扫 bn 64→256→512 × ns∈{1,2}，fa2048/fa4096 non-causal fp16 对照 msprof op Task Duration
- toolchain_stamp: tilelang 0.1.2+3a214cde / CANN 8.5.0 / Ascend910B2C / 2026-09-08
- target_doc: .agents/skills/tilelang-op-optimize/references/bottleneck-patterns.md
- delta: |
    add 新 BP 条目 BP_cross_engine_serial_chain（目录行置于 BP_pipeline_overlap 之后，正文紧随 BP_pipeline_overlap 节，交叉引用之）：
    触发信号：Expert dual-Scope persistent kernel（Cube/Vector + GM workspace 多槽 + per-slot flag 跨引擎流水）busy 核 aic_scalar 或 aiv_scalar 占比 30%+ 且归因为 flag 等待自旋；cube/mte 已近峰值而总时延远超计算分量；加深流水（slots/ns↑）零变化或被抵消。
    常见反证/不确定点：非跨引擎结构的 MTE/compute 串行归 BP_pipeline_overlap；bn 增大触发 L1(cbuf)=512KB / UB=192KB 上限、bm 增大触发 L0C(cc)=128KB（bm≤51@bn=512，2026-09-09 第二轮实测；上限公式见 pattern-library §1.9）——上限不否定方向，转 wide 单槽（L1 单槽 + GM ws/flag 双槽）继续换块宽。
    推荐动作：先块宽摊减（bn↑ 同时削减向量 op 数/块、flag 往返数/列、gemm 碎片化）再评估流水深度——宽块上 ns=1≈ns=2（双槽收益被单槽 L1 v-load↔gemm2 WAR 串行抵消）；配 bm 任务平衡（wall-rows = ceil(ceildiv(S,bm)/24)·bm）。机制基础（2026-09-09 第二轮实测）：向量算子按发射开销计费——每 op ~0.5µs 固定成本、f16≈f32 逐元素速率，同元素总量下窄块 op 翻倍使 vec-active 翻倍（v10_dp bn=256 实测 +12% 回退）。
    验证指标：Task Duration 下降 + busy 核 scalar（flag 自旋）占比下降 + causal/full 等非目标 dispatch 非回退（full 变体逐字节保留）。
- status: pending
- confirmations: 1/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260908T005751Z 2026-09-08
- decided_by: -
- decided_note: -

## VP-2026-0038
- type: P
- title: 跨引擎/在线递推 kernel 精度失败的误差结构定征：全块均匀且可复现 → 排除竞态、指向跨迭代状态生命周期（rescale 因子覆盖）
- evidence:
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/opt_log.md#Round-10-OP2（v10_dp 首版精度 FAIL：全块均匀误差 → alpha/ellcur 跨 defer-2 生命周期被 V1(j) 覆盖；3 路 ping-pong〔平铺 buffer + 运行时三分支〕修复后 L0 全过）
  - 同文件#Skill-Retrospective-第二轮（「先看误差结构再定位」纪律：无它会把递推因子覆盖误判为竞态而放弃深流水方向）
  - pattern-library §1.9 sync_block_wait 行（修复形态交叉引用：v-op codegen 拒绝多维 UB 切片操作数，平铺多 buffer + 运行时三分支为合法形态）
- repro: 任一含 online 递推状态（running max/sum + rescale 因子）跨流水 deferral 生命周期的 Expert kernel：首版精度 FAIL 时先记录误差的空间分布（全块均匀 vs 部分核/非确定）再定排查方向
- toolchain_stamp: tilelang 0.1.2+3a214cde7aa4f54fc4a103f0f324a43341122d68 / CANN 8.5.0 / Ascend910B2C / 2026-09-09
- target_doc: .agents/skills/tilelang-error-fixer/references/precision-patterns.md
- delta: |
    add 新条目「误差结构定征（均匀 vs 部分核）」（合入时与 VP-2026-0006 定征顺序、VP-2026-0017 golden 噪声放大位形同文件并互相交叉引用）：
    判据：精度 FAIL 的误差空间分布是竞态/同步缺陷与系统性状态缺陷的判别器——全块均匀且可复现（各块/各核一致）→ 排除竞态假设，指向跨迭代状态覆盖（online 递推的 rescale 因子〔running max/sum 的 α、ell〕在流水 deferral 生命周期内被后继迭代写入）；部分核/非确定分布 → 竞态/同步方向（flag 协议、握手窗口）。
    修复形态：递推状态跨 deferral 需按流水深度 ping-pong 多路；v-op codegen 拒绝多维 UB 切片操作数——平铺多 buffer + 运行时索引三分支是合法形态（pattern-library §1.9）。
    定位顺序与 §2 零输入探针行的元素级 fp64 对照法互补：先判分布（均匀/部分）定缺陷类别，再下钻元素级对照定偏离侧。
- status: pending
- confirmations: 1/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260909T033622Z 2026-09-09
- decided_by: -
- decided_note: -

## VP-2026-0043
- type: D
- title: layernorm 矩式方差（var=E[x²]−E[x]²）fp32 计算域失败定量：mean≈3000/σ≈0.5 时 fp64err 1.4e3（两遍式 6.3e-4，差 6 个量级）、违反率 25%，fp16 粗栅格 83%——「须评估 catastrophic cancellation」从定性升定量
- evidence:
  - examples/ada_layer_norm/_ada_layer_norm_kernel/verify_equiv.py（moment_form + run_rejection_evidence 段——torch CPU 设计期脚本，无 NPU 依赖）
  - examples/ada_layer_norm/_ada_layer_norm_kernel/RETROSPECTIVE.md（Stage 1 Value Point Proposals 首行）
- repro: repro-missing（知识域最小 repro 待同族任务回填；矩式 vs 两遍式差分最小化仅需数行 torch CPU 脚本）
  - 〔provenance，允许失效〕任务内复现命令：python3 examples/ada_layer_norm/_ada_layer_norm_kernel/verify_equiv.py（输出「C2 rejection evidence」段）
- toolchain_stamp: torch CPU（设计期脚本）+ tilelang 0.1.2+a83118285a / 2026-09-10
- target_doc: .agents/skills/tilelang-op-design/references/algorithm-candidates.md
- delta: |
    update ALG-layernorm R1 行（R1 已有指向本条的 pending 指针——VP-2026-0044 三件套合入时预写；本条闭环时把定量句写入 R1 并把指针转为条目引用）：
    R1 单遍式矩方差的 catastrophic cancellation 定量边界（ada_layer_norm 2026-09-10 verify_equiv 实测，torch CPU）：mean≈3000/σ≈0.5（pad 校正小 N 大均值位形）时矩式 fp64err 1.4e3 vs 两遍式 6.3e-4（差 6 个量级）、容差违反率 25%；fp16 粗栅格（6e4 行）违反率 83%——两遍式为默认候选。配套：GPU 源的 256-pad 方差校正在 NPU 精确 N 式下可整体舍弃且数值更优（恒等式机器验证 0 违反，pad 列贡献 m² 显式减去）。
- status: pending
- confirmations: 1/2
- created_by: task ada_layer_norm-_ada_layer_norm_kernel-20260910T132324Z 2026-09-10
- decided_by: -
- decided_note: 因 target 为 Tier 1 写域（algorithm-candidates.md）+ 缺知识域 repro 而入队（VP-2026-0007 先例形态）；R1 行的 pending 指针已随 VP-2026-0044 合入预写。

## VP-2026-0052
- type: P
- title: 分层中间量导出定位法（系统性 vs 随机误差判别）：out_idx 多输出落 GM 对照 fp64 + violation 率 dtype 指纹（bf16 全过/fp16 ~6%/fp32 ~87% ⇒ 常数相对误差 ~1e-3 = 近似指令特征）单次运行定位到具体指令
- evidence:
  - examples/ada_layer_norm/_ada_layer_norm_kernel/RETROSPECTIVE.md（Stage 3 Value Point Proposals 第三行：探针 stage A/B 对照——S1/mean/d 精确 1.1e-7 而 rstd 偏 1.566e-3，误差独占定位在 vrsqrt 指令）
  - 同文件 Stage 3 首跑三 dtype violation 率分布（12 失败用例的 violations/N 统计）
  - pattern-library TRAP-vrsqrt-plain-precision（2026-09-10 Tier 0 合入；「定位手法」段即本方法的首次实证）
- repro: 复现条件——任一含统计链中间量（mean/var/rstd 等）的精度失败定位：中间量经 out_idx=[1,2] 多输出落 GM 对照 fp64 精确值逐级 dump；无固定脚本（方法形态）
- toolchain_stamp: tilelang 0.1.2+a83118285a + Ascend910B2C + CANN 8.5.0 + torch 2.9.0+cpu / 2026-09-10
- target_doc: .agents/skills/tilelang-error-fixer/references/precision-patterns.md（合入时与 VP-2026-0006 定征顺序、VP-2026-0017 golden 噪声、VP-2026-0038 误差结构定征同文件互链）
- delta: |
    add 新条目「中间量分层导出定位」（合入时置于 VP-2026-0006 定征顺序条目之后）：
    手法：kernel 统计链中间量（S1/mean/d/S2/var/rstd）经 out_idx 多输出落 GM，对照 fp64 精确值逐级比较——单次运行把误差定位到具体指令段（ada 实证：mean/d 精确 1.1e-7 而 rstd 偏 1.57e-3 ⇒ 独占于 vrsqrt）。
    指纹判据（先于探针预判根因类别）：「violation 率 vs 容差」的 dtype 分布——bf16 全过 / fp16 ~6% / fp32 ~87% ⇒ 常数相对误差 ~1e-3 量级 = 近似指令特征（非数据依赖、非同步 bug）；与 §2 零输入探针行的元素级 fp64 对照法互补（先判分布定类别，再下钻元素级定偏离侧）。
- status: pending
- confirmations: 1/2
- created_by: task ada_layer_norm-_ada_layer_norm_kernel-20260910T132324Z 2026-09-10
- decided_by: -
- decided_note: 原提案 target 为 tilelang-debug-helper SKILL.md（Tier 2 写域）；知识本体为精度定位手法（数据/模式类），改路由至 error-fixer precision-patterns.md（P 类 Tier 1，与 VP-2026-0006/0017/0038 同域互链）——保守序内降档处理，如需保持原 target 请在审批时说明。

## VP-2026-0057
- type: P
- title: 两相位结构的 causal 多头域适配机制集（persistent 变长任务循环 / 尾块钳位 + 装载不满宽掩码门控 / K_A 对角链承接 / flag 预算-bn 耦合守卫 / per-n-block flag 跨任务复用条件）——fa 域母本的 causal 化五机制
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/multi_head_attention/_gqa_prefill_fwd_kernel/DESIGN.md v3 §0.6 E1/E2/E3/E6/E7 + §5.2/§7.2（机制设计）+ §6.4（flag 协议传递性论证）
  - Stage 3 实证：同目录 RETROSPECTIVE.md Stage 3（2026-09-15）章节——`--level all` 29/29 全绿（L0 4 + L1 8 + L2 10 + Boundary 7；E7 缺口域四例 + M7 单块退化 + GQA 位形覆盖）
  - 母本对照：examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/_gqa_prefill_fwd_kernel.py `_builder_2phase`（fa 域 B=1/H=1/整除/非 persistent 特化——五机制均为其不具备的 causal 多头增量）
  - 〔2026-09-16 第六轮机制集存活验证，task multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z——同 op 谱系不计独立确认（VP-2026-0035 先例）〕跨工具链迁移 6797758→a13585dc 后 stale 重验（probe_ub.py 直跑 4 点复校，config 空间封闭性结论全部维持）+ 第六轮全部门禁绿；机制⑤ 预言的双槽 id-offset 变体（flag id = i + slot×nk_total，2×nk_total ≤ 15）由 r7f 实现并实测（短域 −7.4%）；机制④ E6 钳位域在 S_kv=4096 首次触发并经 r9 守卫收窄修复（联合 UB 核算判据）。**合入注意**：pipe2 slot-offset 与 E6 钳位分析已写入 attention.md PL-1.12，本条合入时以五机制清单为主体、机制③⑤细节指向 PL-1.12 防重复
- repro: repro-missing（机制效应只在完整 kernel 规模显现；ED-B delta 骨架待同族任务回填）
- toolchain_stamp: tilelang 0.1.2+28783f454705cadab047805c1e0f5e054ba4b967 + CANN 8.5.0 + Ascend910B2C / 2026-09-15
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/attention.md
- delta: |
    add 条目 PL-1.9-twophase-causal（置于 PL-1.9-twophase 之后；attention.md 已近 16KB 预算，合入时触发 consolidate）：
    **两相位结构 causal 多头域适配五机制**（fa 域母本 → causal 多头域的必要增量，Stage 3 29/29 实证）：
    ① per-task 变长 NK 的 persistent 任务循环 + 轮转均衡（母本为非持久定长）；② 尾块分形下限钳位（M/N≥16、K≥32 的 tmc/tnc 公式）+ **尾带 OOB 掩码门控按「装载不满宽」口径**——Vector 装载宽是 tn_real 而非分形宽，统一门控 `has_kv_tail = (S_kv % bn_eff != 0)` + 运行时 `i >= K_A`（non-causal 时 K_A=kv_full；按「分形带存在」判定会漏 rem≥32 且 16|rem 的不满宽残留）；③ causal 对角 mask 链平移进 pass-1（K_A floordiv 两段式 + per-AIV K_blk = base + row0 阈值）；④ flag 预算-bn 耦合守卫：per-n-block 下标 flag 方案要求 NK_max + TASKDONE ≤ 16 → 工厂期强制 `bn_eff = max(bn_caller, ceil16(ceildiv(S_kv, 15)))`——守卫恒生效、设计默认仅替换 wrapper 默认值（config 契约分层：正确性守卫不可被用户 config 绕过）；⑤ per-n-block 同 id 三次握手的跨任务复用安全条件（单槽）：Cube 到达任务 t+1 的 set_S(i) 前程序序必经 pass-2(t) 的 wait_P(i)（同 id 前一生命周期消费点）——id 事件在任务边界前排空；双槽变体需 id-offset（2·NK_max+2 ≤ 16 才合规）。
- status: pending
- confirmations: 1/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0058
- type: D
- title: T.ceildiv 负数域 lower 静默错误（divsi 截断 + 后续 pass 复活恒正改写）——persistent 空核越界任务 MTE 崩溃；绕法 = cid = T.min(task_id·N+kernel_id, num_logical-1) 幂等钳制
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md Stage 3（2026-09-15）Skill Flow Issues 首行 + Value Point Proposals 首行
  - 复现链：one-task (1,1,1,64,64) 崩溃（num_logical < 24 时空核执行越界任务 cid ≥ num_logical → s_lo ≥ S_q → MTE write-out-of-range）+ dump_ir2.log L130-134/L371-374（多 pass 快照对照：lower 成 `divsi(x,N)+1`，x∈(-N,0] 时返回 1 而非 0；恒正参数的 `floordiv(n-k+N-1,N)` 改写被后续 pass 重新规范化回错误形态——纯表达式改写不可靠）
  - 幂等性验证：two-batch (2,1,1,64,64) 钳制下 PASS（幽灵任务重算最后一个合法任务，同输入同指令序位级一致；各核独立 ws 槽 + 核内 flag 通道保证安全；饱和域 num_logical ≥ 24 零开销）
- repro: repro-missing（NPU 编译探针 + dump IR 对照待同族 persistent 任务回填；任务内复现条件 = 任一 num_logical < 24 的 persistent case 如 (1,1,1,64,64)）
- toolchain_stamp: tilelang 0.1.2+28783f454705cadab047805c1e0f5e054ba4b967 + CANN 8.5.0 + Ascend910B2C / 2026-09-15
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-compiler.md
- delta: |
    add 条目 TRAP-ceildiv-negative-lower：
    **`T.ceildiv` 对非正参数的 lower 是静默错误**：`T.ceildiv(n-k, N)` lower 成截断除法 `divsi(x,N)+1`，x∈(-N,0] 时返回 1 而非 0——persistent kernel 空核（kernel_id ≥ num_logical，仅 num_logical < 物理核数 24 时存在）执行越界任务直接 MTE write-out-of-range 崩溃；更隐蔽的是后续 BishengIR pass 会把恒正参数的 `floordiv(n-k+N-1, N)` 改写重新规范化回 divsi+1 错误形态（纯表达式改写不可靠）。**绕法 = cid 钳制幂等方案**：`cid = T.min(task_id*N + kernel_id, num_logical-1)` 把幽灵任务钉到最后一个合法任务——幂等安全（同输入同指令序位级一致重算 + 各核独立 ws 槽）、饱和域零开销、比 if 守卫（parser var-table 风险）与表达式改写（pass 复活）都稳。官方 persistent 模板（docs/开发指南.md §3.3）在 num_logical ≥ 核数域从未触发该分支。
- status: pending
- confirmations: 1/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0060
- type: D
- title: lse [B,H,S,1] 视图直写形态实测可用——`T.copy(ub[0:rm,0:1], lse[bz,by,s:s+rm,0])`（2D src → 1D dst 切片）bit-exact；[N,1] 源 MTE DDR fault 限定于特定 size= 组合而非该直写形态的普遍性质
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md Stage 3（2026-09-15）Skill Flow Issues 末行（probe_lse_store.py 三形态 A/B/C 全过 bit-exact + 全量 29/29 lse 门通过；E6 transpose-free 方案落地无障碍）
  - 被澄清条目：pattern-library traps-runtime.md TRAP-T-copy-region-semantics 规则③（"[N,1] UB 源 + size=[1,N] 按步长 N 越界读——lse 需先 T.transpose 到 [1,N]"，源自 Sep-07 任务 E1-E7 impl note 2）；traps-compiler.md TRAP-transpose-epilogue-poison 的绕法（[B,H,S,1] 视图 + 2D 区域拷贝）即本形态，第三轮已生产使用
- repro: repro-missing（NPU 探针；任务内复现条件 = probe_lse_store.py 三形态对照——session-local provenance，关键发现镜像于 RETROSPECTIVE Stage 3）
- toolchain_stamp: tilelang 0.1.2+28783f454705cadab047805c1e0f5e054ba4b967 + CANN 8.5.0 + Ascend910B2C / 2026-09-15
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-runtime.md
- delta: |
    update 条目 TRAP-T-copy-region-semantics 规则③（边界澄清，原规则保留）：
    ③ [N,1] UB 源 + **size=[1,N]** 组合按步长越界读（MTE DDR fault）——fault 限定于该 size= 组合；**2D src → 1D dst 切片直写 `T.copy(ub[0:rm, 0:1], gm[b, h, s:s+rm, 0])` 实测 bit-exact 可用**（2026-09-15 探针三形态全过 + 29 用例 lse 门；Sep-07 记载的「[N,1] UB 源 MTE DDR fault」不构成对该直写形态的否定）——行向量写 GM 尾维连续区首选 [B,H,S,1] 增维视图 + 2D 区域直写（transpose-free lse，与毒化条目绕法一致），T.transpose 到 [1,N] 仅在必须 base+size 形态时需要。
- status: pending
- confirmations: 1/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0061
- type: P
- title: 错误指纹速查——lse 恒等于 log2(S_kv) ⇒ S 通道读到未初始化零页（伴生指纹：前几行 NaN = 越界更远脏页），秒级定位上游 workspace 生产者/槽位错误
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md Stage 3（2026-09-15）Value Point Proposals 第 2 行 + Skill Flow Issues 第 2 行（two-task (1,1,1,128,128)：lse=7.0=log2(128) 恒值直接锁定 ws_s 通道；bx=1 任务全错〔lse 恒值 + 4 行 NaN〕而 bx=0 完美 = per-core 槽 row0 索引 bug——全局/本地行基巧合掩盖）
- repro: 复现条件——two-task (1,1,1,128,128) 类位形（S 通道读到 torch.empty 零页时指纹必现）；无固定脚本（指纹判据形态）
- toolchain_stamp: tilelang 0.1.2+28783f454705cadab047805c1e0f5e054ba4b967 + CANN 8.5.0 + Ascend910B2C / 2026-09-15
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-runtime.md
- delta: |
    add 诊断指纹条目（置于 TRAP-zero-input-crash 之后，同「部分行数据错」诊断族）：
    **lse 恒值 log2(S_kv) 指纹**：lse = log2(ell) + m·LOG2E——输入全零分数（S 通道读到未初始化零页 workspace）时 ell=S_kv、m=0 → lse 精确恒值 log2(S_kv)；「前几行 NaN = 越界更远读到脏页」为伴生指纹。判据指向「上游生产者未写 / 读错槽位」（workspace 槽位索引、flag 时序），先查 Cube 是否写/Vector 是否读对槽位再查计算链，无需逐指令分析（2026-09-15 per-core ws 槽全局行基误用案例：bx=0 任务「全局=本地」巧合掩盖、bx≥1 全错，指纹秒级锁定）。
- status: pending
- confirmations: 1/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0062
- type: P
- title: MTE 越界类三层崩溃定位法（崩溃边界矩阵 → 错误值指纹 → TILELANG_DUMP_IR + 新 TILELANG_CACHE_DIR）——tilelang 缓存命中跳过编译也跳过 IR 打印，dump 必须换 cache 目录
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md Stage 3（2026-09-15）Value Point Proposals 第 3 行 + Transferable Lessons（one-task 崩溃 40 分钟定位：矩阵 5 分钟锁定「nl<24 崩 / bx≥1 错 / NK 无关」→ IR divsi 对照 → pass 复活确认；probe_nd2nz_offset.py 排除法）
  - 工具行为实证：TILELANG_CACHE_DIR 命中缓存时跳过编译也跳过 IR 打印——TILELANG_DUMP_IR=TRUE 对已缓存 kernel 无输出
- repro: 复现条件——任一 MTE 越界崩溃的 persistent kernel（任务数 × NK × bx 三维隔离矩阵 + dump IR 对照）；cache 行为 = 对已缓存 kernel 设 TILELANG_DUMP_IR=TRUE 观察无输出、换新 TILELANG_CACHE_DIR 后有输出
- toolchain_stamp: tilelang 0.1.2+28783f454705cadab047805c1e0f5e054ba4b967 + CANN 8.5.0 + Ascend910B2C / 2026-09-15
- target_doc: .agents/skills/tilelang-debug-helper/references/mlir-dump-guide.md
- delta: |
    add 增补两节（置于环境变量段之后）：
    **① dump IR 必须配新 TILELANG_CACHE_DIR**：tilelang 缓存命中时跳过编译也跳过 IR 打印——`TILELANG_DUMP_IR=TRUE` 对已缓存 kernel 无输出，须指向新 cache 目录强制重编译。
    **② 崩溃边界矩阵法（MTE 越界类三层定位）**：一层 崩溃边界矩阵（任务数 × NK × bx 三维隔离小用例各跑一次，快速锁定崩溃维度——如「nl<24 崩 / bx≥1 错 / NK 无关」）→ 二层 错误值指纹（恒值/NaN/部分行分布——如 lse 恒 log2(S_kv) ⇒ S 通道零页，见 traps-runtime 指纹条目）→ 三层 TILELANG_DUMP_IR（新 cache 目录）看 TIR/MLIR 双层指令形态（lower 结果 vs pass 后形态对照，确认被哪层改写）。
- status: pending
- confirmations: 1/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0080
- type: P
- title: BP_sync_barrier_redundancy：persistent/两相位结构的同步屏障冗余审计前置（列出每个 barrier 保护对象集，逐对象找更细粒度已有信号覆盖）——TASKDONE 全 drain 屏障删除实测全域 −7~−21%，纯程序顺序论证零实验成本
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z（第六轮 Stage 4）：opt_log Phase 1 诊断归因（单槽 ws + TASKDONE 全 drain 屏障使每 task 边界完全串行；WAR 全链审计结论：屏障在数据竞争角度冗余——ws_s/ws_p/ws_o 的所有跨 task WAR 由「Cube 单指令流程序顺序 + P-ready 消费链 + AIV 串行」闭合）+ Round 6 r6a（删除后全域 −7~−21%：8b-long 1189.18→1021.58 / 8b-short 397.79→315.58 / smoke 49.97→46.26µs；全引擎利用率等比提升 cube 20.4→23.7% / vec 49.9→57.4% / mte2 52.1→62.1% 而 GM 流量不变——节约纯等待）
  - 知识域锚点：attention.md PL-1.12「TASKDONE 屏障冗余审计」段（方法 + 逐对象论证 + 数字自包含）
  - 交叉引用：queue VP-2026-0015（FLAG_TASKDONE 跨任务握手模式——保守安全默认，本条是其「可审计免除」限定，两案合入时互链）；VP-2026-0034（BP_cross_engine_serial_chain——块宽/深度之外的第三杠杆）
- repro: repro-missing（delta 最小形态 = 删除 task 边界的 TASKDONE set/wait 同步对〔2 个 sync op〕，效应依赖完整 kernel 的跨 task 指令流衔接；机制归因与逐对象 WAR 论证自包含于 PL-1.12——知识域骨架待同族 persistent 任务回填）
- toolchain_stamp: tilelang 0.1.2+a13585dc / CANN 8.5.0 / Ascend910B2C / 2026-09-16
- target_doc: .agents/skills/tilelang-op-optimize/references/bottleneck-patterns.md
- delta: |
    add 新 BP 条目 BP_sync_barrier_redundancy（目录行 + 正文，追加于 BP_run_state_bimodality 之后）：
    触发信号：persistent / 两相位（跨引擎 task 循环 + GM workspace）结构存在 task 边界全 drain 屏障（TASKDONE 类：下一 task 的生产等待上一 task 全部消费完成）；时长因子分解显示 per-task 固定成本 F 主导（引擎利用率低而 GM 流量已定、busy 核 scalar 占比高但非 flag 自旋类）。
    审计方法（零实验成本）：列出每个 barrier 保护的对象集（ws buffer × 读写者对），逐对象找更细粒度的已有信号覆盖——同引擎单指令流程序顺序（天然保序）、消费链 flag（P-ready/S-ready 类既有握手）、AIV 串行流。全部对象被覆盖 ⇒ 屏障在数据竞争角度冗余、可删。
    实测（attention 第六轮 r6a，a13585dc）：TASKDONE 删除全域 −7~−21%，引擎利用率等比提升而 GM 流量不变（归因成立的判据：节约的是纯等待）。
    边界：删除前提是 WAR 链逐对象闭合论证（数据竞争角度），非盲目删同步；跨引擎同 buffer WAR 无同步原语的形态不可删（CG-2026-0009 / PL-1.12「Vec 侧同构改造 blocked」教训——引擎独占 buffer 是流水深度 >1 的隐含前提）。
    验证：全域 dispatch 非回退（--level all 全绿）+ 利用率等比提升 + GM 流量不变。
- status: pending
- confirmations: 1/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z 2026-09-16
- decided_by: -
- decided_note: -

## VP-2026-0081
- type: P
- title: msprof roofline Ratio 采数管线规则——--aic-metrics=Roofline 与 --kernel-name 过滤缺一不可（metrics 集不含 Roofline 不产出 visualize_data.bin；无过滤全捕获把 ZerosLike 等辅助 op 当目标，Ratio 0.15–0.65% 伪影实录）
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z（第六轮 Stage 4）：opt_log 第六轮头部「度量管线」段（TileOPs bench 框架 2026-09-16 修复：kernel-name 经 JitKernel_NPU 对象图解析 + 多 op 检测 + duration 匹配；禁止回退无过滤全捕获——profile_run_msprof_20260915_105716.log 的 0.15–0.65% 即该伪影，描述的是 ZerosLike 等辅助 op）+ Phase 1（修复后管线验证：--aic-metrics=Roofline + --kernel-name 过滤 → 单 op 捕获 mix_aic 无 ZerosLike 污染、bin roofline 条目可提取——第五轮 metrics 集不产出 bin 的问题确认并规避）+ DESIGN.md §11.2 指标结构锚定（Ratio = Perf/359.33 TOps/s，Computility 按 48 核计而实机 24 核——口径基数须实测锚定核对）
  - 现行 profile-collection.md 已有 target_kernel_name 命令模板与 captured_op 追溯要求（基础规则存在）；本条补 Roofline-metrics↔bin 耦合、全捕获伪影实录与口径基数核对（2026-09-15 会话产出过伪影 log——规则存在下的执行复发）
- repro: repro-missing（需 msprof + 目标算子环境；伪影对照 = 同 workload 无过滤 vs --kernel-name 过滤双跑，Ratio 条目差异即伪影量级）
- toolchain_stamp: CANN 8.5.0 msprof / Ascend910B2C / 2026-09-16（采数侧 tilelang 0.1.2+a13585dc）
- target_doc: .agents/skills/tilelang-op-optimize/references/profile-collection.md
- delta: |
    动作: update（captured_op 追溯要求行扩展）
    定位锚: "- `captured_op_name` 能通过命令、输出目录和运行日志追溯到本次 `target_kernel_name` 或目标 TileLang kernel。"
    old 文本: （即上述定位锚原文）
    new 文本: |
  - `captured_op_name` 能通过命令、输出目录和运行日志追溯到本次 `target_kernel_name` 或目标 TileLang kernel。**roofline Ratio 口径采数（2026-09-16 attention 第六轮实证）**：① `--aic-metrics=Roofline` 单命令同时产出 Task Duration + 诊断 CSV + `visualize_data.bin` 预计算 roofline 条目（Ratio =「GM/L2 · GM Read + Write」条目 ratio×100）——metrics 集不含 Roofline 时不产出 bin（第五轮实证）；② `--kernel-name` 过滤缺一不可：无过滤全捕获会把 ZerosLike 等辅助 op 当目标 kernel（实录：Ratio 0.15–0.65% 伪影，描述的是辅助 op 而非目标）；③ Ratio = Perf/Computility，Perf = msprof 计数 FLOPs / Task Duration——指标奖励执行利用率、不奖励 padding 缩减（flops 与时长同降时 Ratio 不动），Computility 归一基数以实测锚定核对（如 359.33 TOps/s 按 48 核计而实机 24 核——attention.md PL-1.12 指标结构段）。
    动机: Ratio 类硬目标的采数链路三处坑（bin 不产出 / 辅助 op 污染 / 口径基数错）任一都会使目标判定失真——2026-09-15 会话已产出伪影 log，2026-09-16 修复管线后才可靠；规则化使下一次 Ratio 目标任务第一轮即采对。
- status: pending
- confirmations: 1/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z 2026-09-16
- decided_by: -
- decided_note: -

### Tier 2（R 类，结构化 diff 提案，待人工批准后 mode=apply 执行）

## VP-2026-0008
- type: R
- title: op-design Phase M0 第一问增加 golden 计算域核实检查项（fp16 golden 常见 fp32 opmath，勿假设与源算子舍入路径一致）
- evidence:
  - examples/lerp_tensor/_make_lerp_tensor_kernel/RETROSPECTIVE.md Stage 3 Skill Flow Issues（设计/检视期隐含假设未被拦截，L0 实测才暴露）
  - examples/lerp_tensor/_make_lerp_tensor_kernel/.task_timeline.jsonl（Stage 3 attempt-1 precision fail，2485s）
  - examples/lerp_tensor/_make_lerp_tensor_kernel/history_version/_make_lerp_tensor_kernel_impl_s3_attempt1.py
- repro: torch.lerp CPU fp16（fp32 opmath+单次舍回）vs 逐步 fp16 舍入三步链：N=2^24 差 2–3 ulp、违反率 ~0.125%（pattern-library §2 opmath 行）
- toolchain_stamp: torch 2.9.0+cpu；tilelang-mlir-dev dev root build（2026-09-07）
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（扩展一行）
    定位锚: "2. **第一问（语义）**：解读数学语义、I/O 契约、规约语义（累加顺序）、dtype 语义、边界语义 → 产出 §0.1 / §0.2。完成标准：能写出与源算子数值一致的 golden 函数。"
    old 文本: （即上述定位锚原文）
    new 文本: |
      2. **第一问（语义）**：解读数学语义、I/O 契约、规约语义（累加顺序）、dtype 语义、边界语义 → 产出 §0.1 / §0.2。完成标准：能写出与源算子数值一致的 golden 函数。**golden 计算域核实**：golden 若为 `torch.*` CPU 实现，须用数行最小实验核实其在目标 dtype 下的实际 opmath 域——fp16 常见 fp32 opmath + 单次舍回（torch.lerp 实证：NPU 原生域逐步舍入差 2–3 ulp、容差违反率 ~0.125%，pattern-library §2）；「忠实源算子原生 dtype 舍入」与「过 golden 容差」冲突时以 golden 为准（源仓测试 ref 即 golden），kernel 侧用 fp32 中转对齐。
    动机: lerp_tensor Stage 3 attempt-1 precision fail（2485s）源于设计期「golden 舍入 = 源算子原生舍入」隐含假设；REVIEW 维度 8 预警了 v-prefix 侧舍入语义缺口但未覆盖 golden 侧 opmath 差异——设计期一道 3 行实验即可消除。
- status: pending
- confirmations: -/-
- created_by: task lerp_tensor-_make_lerp_tensor_kernel-20260907T010433Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0009
- type: R
- title: design-review 维度 2 UB 预算检查项补多 dtype 混合字节复算口径（dtype 中转路径禁沿用单一 dtype N-buffer 口径）
- evidence:
  - examples/lerp_tensor/_make_lerp_tensor_kernel/REVIEW.md 维度 2 勘误（DESIGN.md §4.5 注与表自相矛盾：bf16 误写 fp16 同款 24576，正确 196608/24=8192）
  - examples/lerp_tensor/_make_lerp_tensor_kernel/RETROSPECTIVE.md Stage 2 Value Point Proposals
- repro: 含 vcast 升精度中转路径的 block_size 上界 = UB容量 / Σ(bytes_i × count_i)（bf16 中转 24B/elem→8192；fp16 中转 20B/elem→9830 保守 8192）
- toolchain_stamp: tilelang-mlir-dev dev root build（2026-09-07）；A2/A3 UB 192KB
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（改一行）
    定位锚: "| UB 预算 | §4.5 中间缓冲区总和不超过目标平台 UB 容量（A2/A3 = 192KB） |"
    old 文本: （即上述定位锚原文）
    new 文本: |
      | UB 预算 | §4.5 中间缓冲区总和不超过目标平台 UB 容量（A2/A3 = 192KB）；含 dtype 中转（vcast 升精度）路径须按 Σ(elem_bytes × buffer_count) 混合字节宽复算，不得沿用单一 dtype 的 N-buffer 口径（lerp_tensor 实证：bf16 中转 24B/elem 上界 8192，误沿 fp16 口径 24576 需 480KB UB） |
    动机: lerp_tensor DESIGN.md §4.5 出现口径混用且注表自相矛盾，检视维度 2 原口径只查总和未查混合字节宽——勘误后才发现；若未拦截，Stage 3 assert 与 Stage 4 block_size 扫描范围都会错。
- status: pending
- confirmations: -/-
- created_by: task lerp_tensor-_make_lerp_tensor_kernel-20260907T010433Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0011
- type: R
- title: conductor optimize 场景预检 baseline 字段补测量口径标注：Stage 5 events（host 侧 wall）数据禁作 headroom 依据
- evidence:
  - examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/integration_log.md（Stage 5 bench events 口径 1M fp16 182.83us「慢于 torch」vs torch msprof 15.22us）
  - examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/.stage_state.json user_requirement（任务框定被 events 基线误导：「中小 N 慢于 torch，派发开销主导」）
  - examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/opt_log.md Baseline P1（msprof kernel-only 1M 7.30us 快于 torch 2.0–2.3x，全档快于 torch 1.65–2.3x）
- repro: 同 workload 双口径对照：events（host wall，含派发）vs msprof op Task Duration（kernel-only）
- toolchain_stamp: torch 2.9.0+cpu / torch_npu 2.9.0.post2；msprof cann-8.5.0；tilelang 0.1.2+ed787bb（2026-09-07 build）
- target_doc: .opencode/agents/conductor-scenarios/optimize.md
- delta: |
    动作: update（扩展一行）
    定位锚: "- 性能目标类型 / 数值 / baseline（字段与默认值同 `conductor-scenarios/new-op.md` §5 调优必要信息收集表，缺省 `best_effort`）；"
    old 文本: （即上述定位锚原文）
    new 文本: |
  - 性能目标类型 / 数值 / baseline（字段与默认值同 `conductor-scenarios/new-op.md` §5 调优必要信息收集表，缺省 `best_effort`）。**baseline 口径标注**：来自 Stage 5 bench（`prof_mode=events`，host 侧 wall、含端到端派发开销）的数值仅趋势参考、禁止作为 headroom 依据——kernel 时延唯一口径为 `msprof op Task Duration`（pattern-library §2 已弃用 event 口径；两口径方向可相反：lerp_tensor 2026-09-07 events 判 1M 慢于 torch、msprof kernel-only 判快 2.0–2.3x）；透传给 optimizer 的 events 基线须附此标注；
    动机: lerp_tensor optimize 任务的 user_requirement 即被 Stage 5 events 基线误导框定（「中小 N 慢于 torch」实为端到端派发开销失真）；靠 optimize skill 强制 msprof 唯一口径才免于方向错误——调度层透传时即标注可根除。
- status: pending
- confirmations: -/-
- created_by: task lerp_tensor-_make_lerp_tensor_kernel-20260907T025419Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0012

- type: R
- title: integrator 第一步后补 lint/format 自检步骤（TileOPs rule set B 对集成包生效，B023 类闭包绑定违反须在 pytest 前修复并留档）
- evidence:
  - examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/integration_log.md History mod 1（B023 "Function definition does not bind loop variable" + format.sh ruff-format auto-join；幂等重跑 integrate_kernel.py 会覆盖修复）
  - examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/history_version/_make_lerp_tensor_kernel_s5_attempt1.py（pre-fix 备份）
  - 同款再现（2026-09-10 蒸馏追加，不同任务）：examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/integration_log.md Issues 第 1 条 + op 级 RETROSPECTIVE.md Transferable 末条（ALIGNMENT/_align_up 胶水补丁被幂等重跑整文件覆盖——「重跑前先备份 history_version/，重跑后按 Debug history 复放补丁」）
- repro: repro-missing（知识域最小 repro 待同族任务回填；原任务内复现命令见 evidence 末行 provenance 项）
  - 〔provenance，允许失效〕任务内复现命令：cd examples/TileOPs && bash format.sh --files tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/_make_lerp_tensor_kernel.py（pre-fix 报 B023）
- toolchain_stamp: tilelang-mlir-dev dev root build（2026-09-07）；TileOPs pyproject rule set B；torch 2.9.0+cpu / torch_npu 2.9.0.post2
- target_doc: .opencode/agents/tilelang-op-integrator.md
- delta: |
    动作: update（第一步 bullet 列表末尾追加一条）
    定位锚: "- 若脚本报 `[warn] no DESIGN.md ...`（某函数产物目录无 DESIGN.md）：harness 流程不应出现（Stage 1 门禁保证存在）；出现时记录到 `integration_log.md` 的 issues 并继续，不手工补拷贝。"
    old 文本: （即上述定位锚原文）
    new 文本: |
  - 若脚本报 `[warn] no DESIGN.md ...`（某函数产物目录无 DESIGN.md）：harness 流程不应出现（Stage 1 门禁保证存在）；出现时记录到 `integration_log.md` 的 issues 并继续，不手工补拷贝。
  - 脚本完成后、第二步 pytest 前，对集成的 kernel 文件执行 `bash format.sh --files <integrated files>`（TileOPs rule set B 对集成包生效，conductor 产物目录不受其约束）并修复 lint 违反（如 B023 循环变量闭包绑定用 default-arg 绑定：`def _to_npu(t, _dt=torch_dtype): ...`）；format/lint 修改记入 `integration_log.md` History（mod 条目）并在 `history_version/` 留 pre-fix 备份——幂等重跑 integrate_kernel.py 会整文件覆盖这些修复，重跑前须先备份；若修改触及 pytest 不覆盖的内嵌 L0 测试代码，须单独复跑内嵌 L0。
    动机: lerp_tensor 集成后 format.sh 才暴露 1 处 B023（内嵌测试闭包），且 format 自动改动使集成副本与 conductor 产物出现双向分歧；pytest 不覆盖内嵌 L0 代码，须单独复跑。前置自检把该类修改收敛为标准流程步骤。
- status: pending
- confirmations: -/-
- created_by: task lerp_tensor-_make_lerp_tensor_kernel-20260907T010433Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0019
- type: R
- title: gate 1 S1-PLACEHOLDER 模板变量正则对中文数学集合记法误判——含分隔符的 {零, 有限真实数据} 类花括号集合被判「疑似模板变量未替换」造成 Stage 1 无效重试
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/.task_timeline.jsonl（Stage 1 两次 runtime fail：2026-09-07T12:44:47Z〔2960s〕与 13:47:06Z〔1518s〕，stage_retry_count 1→2）
  - .agents/tools/gate_lint.py#L36（TEMPLATE_BRACE_RE 定义）+ #L208-215（S1-PLACEHOLDER 判定）
  - 正则行为机械复现（2026-09-07 蒸馏会话）：`TEMPLATE_BRACE_RE.finditer("槽位尾行状态恒 ∈ {零, 有限真实数据}")` 命中 `{零, 有限真实数据}` → fail；误判文本出现在被门禁拒绝的中间版本上（归档 design_v0/v1/DESIGN.md 已无中文花括号——修订后通过）；发生账户来自 conductor 终态钩子输入（复盘工件未覆盖该两次 fail 的 gate 侧原因，同时构成复盘缺口）
  - 〔2026-09-15 复发，task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z〕新形态两处误报（Stage 1 gate 修复重试的 2×S1-PLACEHOLDER）：`e^{块内分数 − m_cur}`（CJK 数学上标，无列表分隔符）与 `_{\text{Cube 部和，f16 物化回传}}`（LaTeX \text{中文} 下标）——**前者不受本提案正则排除**（负向前瞻仅排除 [,、;；∈≤≥]，无分隔符的 `^{中文 − var}` 形态仍命中），审批时建议扩大排除集（brace 组前缀为 `^`/`_` 或含 `\text{` 时跳过）；writer 侧排版规则另立 VP-2026-0066 互补（改写消除模式命中：括号形式 / 注解移出公式，语义零变化）；纯 ASCII 花括号（`e^{m_prev − m_cur}`）不触发
- repro: python3 -c "import re; print([m.group(0) for m in re.finditer(r'\{[^{}\n]*[\u4e00-\u9fff][^{}\n]*\}', '恒 ∈ {零, 有限真实数据}')])"
- toolchain_stamp: gate_lint.py 现行版本（statectl gate 1 机械核对层）；误判发生于 2026-09-07 expert 迁移任务
- target_doc: .agents/tools/gate_lint.py
- delta: |
    动作: update（替换一行正则定义）
    定位锚: 'TEMPLATE_BRACE_RE = re.compile(r"\{[^{}\n]*[\u4e00-\u9fff][^{}\n]*\}")'
    old 文本: |
      TEMPLATE_BRACE_RE = re.compile(r"\{[^{}\n]*[\u4e00-\u9fff][^{}\n]*\}")
    new 文本: |
      # CJK-brace groups that look like unfilled template slots. Set/math
      # notation with CJK members and list separators (e.g. "{零, 有限真实数据}")
      # is legitimate prose, not a template variable: skip such groups.
      TEMPLATE_BRACE_RE = re.compile(r"\{(?![^{}\n]*[,、;；∈≤≥])[^{}\n]*[\u4e00-\u9fff][^{}\n]*\}")
    动机: 负向前瞻排除含列表分隔符/数学关系符的花括号组——已验证 `{零, 有限真实数据}`/`{保留、舍弃}` 不再命中，`{算子名称}`/`{如: xxx}` 类真模板变量仍命中；本任务 Stage 1 两次 runtime 重试中 gate 误判贡献了无效重试（设计文档的集合记法是合法内容）。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0020
- type: R
- title: gate 1 S1-REF-PATHS 路径存在性核对未剥离 `#`/`::` 锚后缀与省略号——`DESIGN.md::修订记录`/`.py#L345` 类合法引用被判「引用的仓库路径不存在」
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/.task_timeline.jsonl（同 VP-2026-0019 的两次 Stage 1 runtime fail）
  - .agents/tools/gate_lint.py#L64-84（_ref_path_failures：REF_PATH_RE 的排除字符类不含 ASCII `:`/`#`，token 连同锚后缀一起做 os.path.exists 判定）+ #L42-44（REF_PATH_RE 定义）
  - 正则行为机械复现（2026-09-07 蒸馏会话）：`examples/.../DESIGN.md::修订记录` 与 `examples/deepseek_v4/example_sparse_attn_kernel_highperf.py#L345` 两个 token 均含锚后缀导致 exists 判 false；发生账户同 VP-2026-0019（conductor 终态钩子输入 + 复盘缺口）
- repro: python3 -c "import re; print([m.group(0) for m in re.finditer(r'(?:docs|examples|testing|\.agents)/[^\s\`\x27\"<>()\[\]（）【】：，。；、！？]+', '见 examples/multi_head_attention/_gqa_prefill_fwd_kernel/DESIGN.md::修订记录')])"
- toolchain_stamp: gate_lint.py 现行版本；误判发生于 2026-09-07 expert 迁移任务
- target_doc: .agents/tools/gate_lint.py
- delta: |
    动作: update（替换 token 清理两行）
    定位锚: |
      (于 _ref_path_failures 函数体内)
          token = m.group(0).rstrip(".,;:!?、。】")
    old 文本: |
          token = m.group(0).rstrip(".,;:!?、。】")
    new 文本: |
          # Strip anchor suffixes (path#heading / path::section) before the
          # existence check; elided paths (examples/foo/...) are not concrete.
          token = re.split(r"#|::", m.group(0), maxsplit=1)[0].rstrip(".,;:!?、。】")
          if token.endswith("...") or "/..." in token:
              continue
    动机: 设计文档以 `path#heading`/`path::section` 引用仓库工件是合法引证形态（本任务 DESIGN/REVIEW 多处使用）；现行正则把锚后缀并入 token 判不存在——已验证修复后 `DESIGN.md::修订记录` → `DESIGN.md`、`.py#L345` → `.py` 正确判定，尾部省略号路径 rstrip 后落到父目录存在性判定。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0023
- type: R
- title: DESIGN 伪代码单一权威源规则——伪代码与实现注/设计步骤矛盾时必须先消歧再冻结（单边权威声明不够）
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-1-rev1（VP 行 4：v0 §3.3 伪代码 α 行 vsub(ub_m, ub_mcur) 与实现注①「快照式 m_prev」矛盾——修复 = 伪代码改写为注的形态而非加「以注为准」标记）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-3（Skill Flow Issues Design-internal 行：§3.3 伪代码 mask→softcap 与 E3/§3.1 softcap→mask 矛盾——照抄伪代码会破坏 P(OOB)=0 不变式，实现被迫自行消歧）
- repro: 复现条件——含伪代码 + 实现注双源且顺序/操作数不一致的设计文档（本任务 v0 两处实证）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2 + 910B2C（2026-09-07）
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（Phase 4 章节列表后、Phase 5 前插入附注）
    定位锚: |
      11. 交付清单

  ```
  ### Phase 5：质量自检
  ```
  old 文本: |
      11. 交付清单

  ```
### Phase 5：质量自检
```
  new 文本: |
      11. 交付清单

      > **伪代码单一权威源**：§3 伪代码即权威实现形态，实现注只解释不纠偏——两者矛盾时修正伪代码本身（改写为注的形态），而非加「以注为准」标记；伪代码与其他章节（设计步骤/论证）多源矛盾时必须先消歧再冻结——单边权威声明不够（2026-09-07 attention expert 任务实证：mask/softcap 顺序矛盾若照抄伪代码会破坏 P(OOB)=0 不变式，softcap(−1e38)≈−softcap 非精确 0）。

```
### Phase 5：质量自检
```
  动机: 同一设计文档两处伪代码-注/步骤矛盾（α 行操作数顺序、mask/softcap 顺序），一处靠检视拦截、一处漏到 Stage 3 由实现者自行消歧——冻结前的消歧义务是缺口。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0024
- type: R
- title: tilelang-op-design Phase 6 修订纪律三条——多路线裁决三步法+正确性面第四维 / 公式修复「公式-论证-自述」三同步 / 概念删除与计数预算标注 grep 清零判据
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/history_version/design_v0.md → design_v1.md → DESIGN.md（v0→v1→v2 修订链）+ DESIGN.md 修订记录（v1/v2 两节「为何不会再犯」）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-1-rev1（路线裁决三步法 + 第四维 + grep 清零 + 三同步各 VP 行）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-1-rev2（三同步模板 + 计数/预算标注同步义务：「8 参」「6 输入」需带边界 pattern 防子串误命中）
- repro: 复现条件——任一 Stage 2 不通过后的针对性修订（本任务 2 轮修订、牵连面 10+ 章节实证）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境同上
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（Phase 6 小节扩展）
    定位锚: |
      ### Phase 6：针对性修订

  ```
  仅修正未通过自检的项目。信息确实不足的标注为「待确认」并说明原因。
  ```
  old 文本: |
      ### Phase 6：针对性修订

  ```
仅修正未通过自检的项目。信息确实不足的标注为「待确认」并说明原因。
  ```
  new 文本: |
      ### Phase 6：针对性修订

  ```
  仅修正未通过自检的项目。信息确实不足的标注为「待确认」并说明原因。

  检视不通过后的针对性修订（revision mode）附加纪律（2026-09-07 attention expert 任务 v0→v1→v2 修订链实证）：

1. **多路线修复必须写明路线裁决**：检视给出多条修复路线时，修订版写明裁决依据与回退设计，不默认选改动面小的路线。裁决步骤：① 证据分级（文档规格 > 前序设计断言 > 形态缺口——缺口降级为「未实证假设」而非「不支持」）；② 结构收益量化（如直连使 dtype 同构、删 flag/ws 族）；③ 回退完整性（被否决路线降级为 R-x 回退而非删除）；④ 正确性面作为独立第四维（flag 族数、未初始化域、握手竞态数随路线的削减）。
2. **公式修复的「公式-论证-自述」三同步**：修正分界/阈值/取整公式（ceildiv↔floordiv 类）时同步恢复依赖该公式的整条下游论证链（保守性论证 / 收益前提 / 量化自述占比）——只替换公式 token 而不同步论证会保留旧矛盾或制造新矛盾；修复前执行代数重推 + 整除/不整除双数值例。
3. **概念删除与计数/预算标注的完成判据 = token grep 清零**：删除某概念（API/buffer/flag/参数）或切换预算基准后，以旧 token（含「N 参」「N 输入」「NKB」类标注）全文 grep 清零为完成判据（子串误命中用带边界符号的精确 pattern；合法语境白名单 = 回退方案/修订记录/历史版引用）。
```
  动机: 本任务两轮修订中：路线裁决无模板（自行三步法完成）、v1 删参未同步计数（第 2 轮问题 2/3）、K_A 修复若无三同步会留下 §5.4 占比与 M5 前提的内部矛盾——三条纪律均已被 3 轮检视逐项验证有效。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0025
- type: R
- title: 弃选论证证据规则补第 6 条「前序实证的证据边界」——归档工件「N/M 实证通过」只证明实测路径可行，不得外延为替代路径不可行
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-2（问题 2：上一轮 23/23 载体链实证被用作「直连 bf16 不可行」的证据；负向断言「T.gemm bf16 ×」两轮原样继承且引用同一文档——两轮均未打开文档核对表格）
  - docs/Tilelang.language/线性代数操作/T.gemm.md#§2.2.1/§2.3（实际 bf16 √ + 实测可用——断言被推翻的对照证据）
  - examples/multi_head_attention/_prev_task_20260904_developer/_gqa_prefill_fwd_kernel/debug_log.md（前序实证的原始出处——provenance 交叉核对）
- repro: pytest testing/npuir/bf16_support_ops/test_gemm_bf16.py（被误读为「bf16 不可行」的反例证据链复核）
- toolchain_stamp: 证据环境 tilelang 0.1.2 + 910B2C（2026-09-07 两轮设计 + 本轮复核）
- target_doc: .agents/skills/_shared/standards/negative-claim-evidence.md
- delta: |
    动作: update（§2 规则要点追加第 6 条）
    定位锚: "，与实测矛盾即 fail。"
    old 文本: |
      ，与实测矛盾即 fail。
    new 文本: |
      ，与实测矛盾即 fail。
      6. **前序实证的证据边界**：归档工件/前序任务的「N/M 实证通过」只证明其实测过的实现路径可行，不得外延为替代路径不可行——引用前序结论做负向断言时降级为「该路径已验证」并为替代路径另行举证（文档表格/测试/先例）；从前序任务继承的负向断言同样必须亲自打开所引文档复核到具体表格行/限制条款（「引用了正确路径、转述了相反内容」= 两轮设计同款失效形态，2026-09-07 attention expert 任务 bf16 断言实证：T.gemm.md §2.2.1 实为 bf16 √，前序断言驱动了整条载体链重设计后被推翻）。
    动机: bf16 负向断言跨两轮任务存活并驱动 E5 全链设计（载体链 +32KB UB、2 个 flag 族、Vector 载体负载），本轮文档+测试亲核推翻后直连形态 bf16/fp16 延迟差 ≤0.2%（pattern-library §1.8）——继承断言不亲核的代价是整条设计路线。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0026
- type: R
- title: design-review 维度 8 补「公式与参数化标量验证」强制块——⟺ 分界声明代数重推+双数值例 / 量化自述对账 / per-lane 双端点代入 / 继承负向断言亲核
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/REVIEW.md（第 2 轮问题 1：K_A ceildiv 假等价在 v0/v1 两轮存活——`块 n 全有效 ⟺ n < ceildiv(a, bn)` 的 +1/−1 补偿直觉使公式「看起来对」；8 个 bx 数值枚举 design−precise=1、免 mask 域违例 2016 位置/任务）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-2-r2（VP 行：自述即测试向量——§5.4「每任务 1–2 块走 mask」与公式联立 NK−K_A∈[1,2] 直接暴露公式错误）+ #Stage-2-r1（per-lane 双端点行）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/history_version/design_v0.md#L221（同款公式两轮存活证据）
- repro: 复现条件——任一含「块级分界 + 条件跳过」的 DESIGN 检视（causal K_A / sliding window / 守卫域判定）
- toolchain_stamp: 证据环境 tilelang 0.1.2 + 910B2C（2026-09-07，3 轮检视）
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（维度 8 引言 blockquote 追加一段）
    定位锚: "不得只复算数字而继承设计的前提。"
    old 文本: |
      **弃选/否决类论断（API 不支持 / 代价高 / 无先例）的前提同样必须亲自查证**——打开其引用的 API 文档核对代价机制（如 repack 是核内 UB 级还是 GM 级），不得只复算数字而继承设计的前提。
    new 文本: |
      **弃选/否决类论断（API 不支持 / 代价高 / 无先例）的前提同样必须亲自查证**——打开其引用的 API 文档核对代价机制（如 repack 是核内 UB 级还是 GM 级），不得只复算数字而继承设计的前提。**公式与参数化标量验证（强制）**：① 一切「⟺」分界/阈值/取整类声明（causal 分界、sliding window 边界、守卫域判定）执行代数重推 + 整除/不整除双数值例（a=65/bn=64/n=1 类反例）——ceildiv↔floordiv 差 1 的补偿直觉使公式「看起来对」，文字论证无法发现（K_A 反例两轮存活）；② 设计文档内的量化自述（块数/占比/频次）是与公式对账的不变式，自述-公式联立矛盾即线索；③ AIV/lane 半分类的参数化标量阈值代入两个 lane 端点数值例（vid=0/vid=1——单 AIV 思维推导在另一 lane 符号方向隐蔽出错）；④ 从前序任务/归档工件继承的负向断言必须亲自打开所引文档复核到表格单元格——前序实证只对其实测过的路径有效。
    动机: 本任务 3 轮检视中 K_A 假等价公式在第 1 轮维度 0/8 下 pass（文字论证自洽）、per-lane 符号错误靠 smoke 数值例才发现——四项核对均为「文字论证全对、公式错」形态的最低成本拦截器。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0027
- type: R
- title: 检视报告修改建议中的替代公式/参数写入前必须先经独立数值验证——检视建议是 Stage 1 revision 的逐字输入，建议错误的修复成本高于漏检
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/REVIEW.md（第 1 轮问题 1 修改建议③把 ceildiv 版 K_A 误背书为「保守下界」；第 2 轮问题 1 代数重推 + 反例 c=64/bn=64/n=1 + 8 个 bx 数值枚举确证为假等价——ceildiv 对 vid=0 是激进上界）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/DESIGN.md#v2-仲裁说明（v1 曾把建议③逐字固化进「K_A 保守性说明（防 Stage 3 误改）」——错误由「待核公式」升级为「已论证事实」）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-2-r2（Skill Flow Issues 首行 + Transferable「检视者对自己前轮输出的建议负有验证义务」）
- repro: 复现条件——检视报告给出具体替代公式/取整方向且被修订版逐字采纳（本任务 v1 实证）
- toolchain_stamp: 证据环境 tilelang 0.1.2 + 910B2C（2026-09-07）
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（Phase 4 追加一段）
    定位锚: "按 §5 模板写入 `examples/{project}/{op}/REVIEW.md`。"
    old 文本: |
      按 §5 模板写入 `examples/{project}/{op}/REVIEW.md`。
    new 文本: |
      按 §5 模板写入 `examples/{project}/{op}/REVIEW.md`。

  ```
  修改建议中给出的具体替代公式/参数/取整方向，写入前先经独立数值验证（代数重推或数值例）——检视建议是 Stage 1 revision 的逐字输入，建议自身携带错误会被修订版固化升级为「已论证事实」（2026-09-07 attention expert 任务实证：第 1 轮建议③把 ceildiv 版 K_A 误背书为「保守下界」，v1 逐字固化，第 2 轮代数重推 + 8 个 bx 数值枚举才确证为假等价）；检视者对自己前轮输出的建议负有与设计同等的验证义务——重审时必须重推建议本身的正确性，「建议被执行了」与「建议的目标达成了」是两回事。
```
  动机: 建议③误背书使错误公式获得检视权威加持、跨 v0/v1 两版存活且被「防误改」说明保护——该形态比漏检更难被发现，写入前验证是唯一低成本拦截位。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0028
- type: R
- title: design-review 修订版重审工作流——复用白名单/公式重推黑名单 + diff 锁定深查范围 + 新引入 API 亲验 + 建议原文逐句验收清账 + 三态判定
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/REVIEW.md（第 3 轮：v1→v2 diff 仅 8 处与修订记录声明一致；「复用 R2」标注 + K_A 全链重推策略；问题 4 三态判定「核心诉求成立 + 残留建议级」实践）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-2-r2/r3（复用白名单/重推黑名单行、diff 全量比对法、不变量对账清单、API 亲验行、原建议文本即 requirements 行）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/history_version/design_v1.md（diff 比对基准）
- repro: 复现条件——Stage 2 第 N 轮（N≥2）修订版重审（本任务 R2/R3 两轮实证）
- toolchain_stamp: 证据环境 tilelang 0.1.2 + 910B2C（2026-09-07）
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（Phase 2 追加重审规则）
    定位锚: "按 §3 的维度逐项检查（迁移任务 0–8，非迁移任务 1–8），每项标记 `pass / warn / fail` 并记录证据（DESIGN.md 章节号 + 源码证据 + 引用文件）。"
    old 文本: |
      按 §3 的维度逐项检查（迁移任务 0–8，非迁移任务 1–8），每项标记 `pass / warn / fail` 并记录证据（DESIGN.md 章节号 + 源码证据 + 引用文件）。
    new 文本: |
      按 §3 的维度逐项检查（迁移任务 0–8，非迁移任务 1–8），每项标记 `pass / warn / fail` 并记录证据（DESIGN.md 章节号 + 源码证据 + 引用文件）。

  ```
  修订版重审（第 N 轮，N ≥ 2）附加规则（2026-09-07 attention expert 任务 3 轮检视实证）：
- **复用白名单 / 重推黑名单**：前轮已验证的事实核对类结论（文档原文/源码行号/路径存在性/数字复算）可复用（标注「复用 R{N-1}」）；一切分界/阈值/取整公式与保守/激进的定性判断无论修订是否触碰均须重推——修订未触碰的段落不等于正确的段落（K_A 公式 v0/v1 两轮存活，唯一新阻塞恰在「复用区」）。
- **diff 锁定深查范围**：对「其余内容逐字保持上一版」类声明，以 `diff design_v{N-1}.md DESIGN.md` 全量比对验证——既验证声明属实（无未声明改动的隐性风险面），又自动锁定本轮深查范围（diff 命中处 = 必查）。
- **修订版新引入的 API 一律亲验存在性**（伪代码/映射表中首次出现者）——设计自述的「实查/亲验」记录不构成佐证，重跑同款 3 行验证才是检视证据。
- **修复验证对照建议原文逐句清账**：原建议文本是修复验收的 requirements 文档（每个验收子句独立判定达成/残留）；多目标建议部分达成时判「核心诉求成立 + 残留建议级」三态（记维度 warn + 修改方向），不整体放行也不整体阻塞。
```
  动机: 本任务 R2 漏检根因 = 复用策略未区分结论类型（事实核对 vs 推演判断）；R3 以 diff 锁定 + 公式全量重推 + 三态判定完成收口——规则均经实战验证。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0029
- type: R
- title: design-review 维度 5 新增「flag 协议安全性」检查行——per-slot flag 逐族传递性推演法（写 i+slots 前读 i 已完成）
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/DESIGN.md#§6.3（三族 + 单槽 + 跨任务五条传递性论证）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/REVIEW.md（维度 5 附核对记录：本轮对五条论证独立推演全部成立）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-2（VP 行 5：推演法可直接复用）
- repro: 任一 per-slot flag + 多槽 workspace 的 Expert 设计检视
- toolchain_stamp: 证据环境 tilelang 0.1.2 + 910B2C（2026-09-07）
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（维度 5 表追加一行）
    定位锚: "| 尾块处理 | §6.4 说明 shape 不整除时的尾块逻辑 |"
    old 文本: |
      | 尾块处理 | §6.4 说明 shape 不整除时的尾块逻辑 |
    new 文本: |
      | 尾块处理 | §6.4 说明 shape 不整除时的尾块逻辑 |
      | flag 协议安全性 | per-slot flag + 多槽 workspace 设计按「写 i+slots 前读 i 已完成」逐族传递性推演（set 的程序序前驱 → wait 的后继），跨任务握手（TASKDONE 类）单列核对（2026-09-07 attention expert 任务 §6.3 五条论证推演法） |
    动机: Expert 跨引擎流水的正确性核心在 flag 协议而非循环结构——维度 5 原三行（循环/同步模式/尾块）不覆盖 flag 族传递性，本轮靠检视者自发推演完成，方法已验证可复用。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0030
- type: R
- title: design-review 维度 7 新增「统计行对账」检查行——四态/多态处置表的统计行与表体处置列做分项-标签逐行比对（不只对总数）
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/REVIEW.md（第 3 轮维度 7 warn：统计行分项归类与表体处置列标签 2 处错位——表体字面分布 6/7/6/4 vs 统计 5/8/5/5，跨 v0/v1/v2 三版存活）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-2-r3（Skill Flow Issues 首行：前三轮检视只数总数 23=23，未做分项-标签逐行比对）
- repro: 复现条件——含四态/多态分类统计行的设计文档（本任务 §0.5 处置表三版存活实证）
- toolchain_stamp: 证据环境 tilelang 0.1.2 + 910B2C（2026-09-07）
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（维度 7 表追加一行）
    定位锚: "| 内部一致 | API 映射、内存规划、Tiling 三处描述相互一致，无矛盾 |"
    old 文本: |
      | 内部一致 | API 映射、内存规划、Tiling 三处描述相互一致，无矛盾 |
    new 文本: |
      | 内部一致 | API 映射、内存规划、Tiling 三处描述相互一致，无矛盾 |
      | 统计行对账 | 四态/多态处置表的统计行与表体处置列做分项-标签逐行比对（不只对总数）——总数对齐但分项归类错位（6/7/6/4 vs 5/8/5/5）在 v0/v1/v2 三版存活，仅靠总数对账无法发现（2026-09-07 attention expert 任务实证） |
    动机: 总数对账只能发现双计类错误；分项归类与标签的错位需逐行比对——低成本高收益动作，防止错误统计被后续章节继承。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0031
- type: R
- title: conductor 修订调度 prompt 附加仲裁规则——检视建议跨轮冲突时显式标注「以最新轮为准 + 禁止恢复项」；design_error_summary 位置冲突以 REVIEW.md 为准
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/DESIGN.md#v2-仲裁说明（「以第 2 轮报告的 floordiv 公式为准，严禁按第一轮检视建议③的字面恢复」——本轮 conductor 调度输入显式指令防止了修订者回退到已推翻建议）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-1-rev2（Skill Flow Issues 两行：仲裁输入行 + design_error_summary 位置引用笔误行——「M5 收益前提（§1.6.2 #5）」实位于 §1.6.1）
- repro: 复现条件——Stage 2 多轮检视且后轮推翻前轮建议（本任务 v0→v1→v2 实证）
- toolchain_stamp: 流程规则（conductor 调度层）；证据环境 2026-09-07
- target_doc: .opencode/agents/tilelang-op-conductor.md
- delta: |
    动作: update（设计修订循环步骤 3 追加仲裁规则）
    定位锚: "3. 预算未超限 → `statectl start 1 --dir ...`（一次性修订通行证重入）→ 重新调度 designer（`mode=revision`），prompt 传入 `last_design_path` / `design_error_summary` / `revision_index` / `previous_revisions`（迁移任务另传 `source_op_path`，见 migration.md §5）→ 新 `DESIGN.md` 后按正常流程进入 Stage 2 重新检视（迁移任务 9 维度；所有任务含维度 8）。"
    old 文本: |
      3. 预算未超限 → `statectl start 1 --dir ...`（一次性修订通行证重入）→ 重新调度 designer（`mode=revision`），prompt 传入 `last_design_path` / `design_error_summary` / `revision_index` / `previous_revisions`（迁移任务另传 `source_op_path`，见 migration.md §5）→ 新 `DESIGN.md` 后按正常流程进入 Stage 2 重新检视（迁移任务 9 维度；所有任务含维度 8）。
    new 文本: |
      3. 预算未超限 → `statectl start 1 --dir ...`（一次性修订通行证重入）→ 重新调度 designer（`mode=revision`），prompt 传入 `last_design_path` / `design_error_summary` / `revision_index` / `previous_revisions`（迁移任务另传 `source_op_path`，见 migration.md §5）→ 新 `DESIGN.md` 后按正常流程进入 Stage 2 重新检视（迁移任务 9 维度；所有任务含维度 8）。修订调度 prompt 附加仲裁规则：① 前轮检视建议被后续轮次推翻时，显式标注「以最新轮为准 + 禁止恢复项」（如「严禁按第 1 轮建议③字面恢复」）——无此指令时修订者可能因「上一轮建议已被执行过」而复用错误公式（2026-09-07 attention expert 任务实证：误背书的 ceildiv 建议曾逐字固化进 v1）；② `design_error_summary` 的位置引用与 REVIEW.md 位置列表冲突时以 REVIEW.md 为准（或摘要只引问题编号不重复位置）——摘要位置笔误会误导修订定位。
    动机: 检视建议跨轮冲突时修订者面对两个「检视权威」；显式仲裁输入是被实证有效的防回退机制（本任务 v2 修订记录明示该指令起了作用）。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0032

- type: R
- title: integrate_kernel.py 补 stale perf_opt lineage 告警——换代重集成不清理上一任务 perf_opt/，占位符会静默解析到旧谱系调优 kernel（工厂签名兼容、测试不拦截）
- evidence:
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/integration_log.md（§Bench observations 末条 + Files 清单：perf_opt/ 保留的 developer 谱系产物〔mtime 09-07 04:51，43KB〕早于 expert 基线 kernel〔09-07 17:22，52KB〕）
  - examples/multi_head_attention/RETROSPECTIVE.md#Stage-5（Skill Flow Issues 行 + Transferable 首条）
  - 〔2026-09-15 预警场景实际发生，task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z〕换代重集成（单遍 expert→两相位 v3）再次命中且以另一形态：integrate_kernel.py 幂等检查误判 already integrated（正则匹配到 r1-r3 调优轮注释掉的 baseline 行），wrapper 未重写、Sep-07 谱系 perf_opt import 静默保持激活（`w._gqa_prefill_fwd_kernel.__module__` 修复前解析到 perf_opt 旧谱系 kernel）；证据：integration_log.md Round 2 Integration steps Step 1（`wrapper_rewritten: false`）+ Step 2 人工胶水修复记录 + op 级 RETROSPECTIVE.md Stage 5 round-2 Skill Flow Issues 首行。互补缺陷另立 VP-2026-0064（幂等检查剥离注释行——修「不重写」侧；本条修「重写后不告警」侧），同 VP-2026-0055/0056 集群建议同批审批
- repro: repro-missing（知识域最小 repro 待同族任务回填；原任务内复现命令见 evidence 末行 provenance 项）
  - 〔provenance，允许失效〕任务内复现命令：换代重集成后 `ls -la tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/` 对比基线源 mtime——本任务实测旧谱系早 12.5 小时
- toolchain_stamp: tilelang dev build 21586b5（2026-09-07）+ CANN 8.5.0；TileOPs pyproject 环境
- target_doc: examples/TileOPs/.agents/skills/add-npu-op/scripts/integrate_kernel.py
- delta: |
    动作: update（main 复制循环内追加谱系检查）
    定位锚: |
      (main() 的 sources 复制循环内)
          integrated[src.stem] = dst
          print(f"[copy] {src} -> {dst}")
    old 文本: |
          integrated[src.stem] = dst
          print(f"[copy] {src} -> {dst}")
    new 文本: |
          integrated[src.stem] = dst
          print(f"[copy] {src} -> {dst}")
          perf_dst = target_dir / "perf_opt" / src.name
          if perf_dst.exists() and perf_dst.stat().st_mtime < src.stat().st_mtime:
              print(
                  f"[warn] stale perf_opt lineage: {perf_dst} predates the "
                  f"baseline source {src}; it likely derives from a previous "
                  f"task lineage -- verify lineage (mtime / origin task) "
                  f"before activating the perf_opt import"
              )
    动机: 换代重集成（developer→expert）整文件覆盖基线但保留旧 perf_opt/；wrapper 的 perf_opt import 占位符切换后会解析到旧谱系 kernel——工厂签名兼容、smoke/full/bench 均不拦截，只有谱系核对能发现（本任务实测两谱系相差 12.5 小时与 9KB/640↔949 行）。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0033
- type: R
- title: op_template.py 的 _run_case 加长签名传参规范注释——位序敏感 case 参数（≥4 个或多个同类整型）用 kwargs 或 case 元组，防 heads↔seq 写反
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-3（Test-harness 行：6 参位序 (batch, heads, heads_kv, Sq, Skv, dim) 在 13 处手写调用中写反 2 类〔heads↔seq〕，1 次 SDPA GQA 崩溃 + 1 次 ValueError；全 MHA 位形静默通过、GQA 位形才炸）
  - /tmp/opencode/full_run2.log#L48（RuntimeError: The size of tensor a (1024) must match the size of tensor b (16)——位序写反的崩溃实录，session-local）
- repro: 复现条件——长位序 case 签名手写调用 ≥10 处（本任务 13 处写反 2 类实证）
- toolchain_stamp: 测试工程规则（无运行时依赖）；环境 torch_npu 2.x + 910B2C（2026-09-07）
- target_doc: .agents/skills/tilelang-op-develop/templates/op_template.py
- delta: |
    动作: update（_run_case 定义行前追加规范注释）
    定位锚: "def _run_case(M, N, dtype, tag):"
    old 文本: |
      def _run_case(M, N, dtype, tag):
    new 文本: |
      # Case-signature rule: when a kernel's case parameters grow beyond ~4
      # positional args, or contain several same-type integers (shape / dims /
      # head counts), pass them as kwargs (or wrap a case tuple) at every call
      # site -- transposed positional args (heads<->seq) pass silently on
      # all-MHA shapes and only crash on GQA/dispatch variants (2026-09-07
      # attention expert task: 2 of 13 hand-written calls transposed).
      def _run_case(M, N, dtype, tag):
    动机: 位序写反在 GQA 位形（heads≠heads_kv）才暴露，全 MHA 测试全绿掩盖错误——模板级规范使新算子测试套件从第一处调用就免于此类浪费。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: -
- decided_note: -

## VP-2026-0036
- type: R
- title: optimize SKILL.md Phase 2 步骤 7 补 blocked/invalid 分支归宿——perf_records.jsonl 仅收录完成验证与测量的分支（duration_us 恒为实测数、不得 null），结构性不可行证据归 opt_log invalid 行
- evidence:
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/opt_log.md#Iteration-3-注记（gate 4 窄修复记录：4 个 blocked-compile 分支〔bn=512 双槽 L1 溢出〕曾以 duration_us: null 写入 perf_records.jsonl，按契约移除、证据归 opt_log invalid_compile 行）
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/.task_timeline.jsonl（Stage 4 attempt 1 fail 7546s〔gate S4-PERF-RECORDS-SCHEMA〕→ attempt 2 窄修复 complete 314s）
  - .agents/skills/_shared/standards/signal-registry.md#§5（字段契约：duration_us 类型 number；「每轮每个实验分支验证后追加一行」）
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/opt_log.md#终局验证-注记（2026-09-09 第二轮同主题实证〔task 20260909T033622Z，并入不新开 id〕：final causal 首行 3779.0 为手抄转录笔误，append-only 纪律下以更正行 3773.39 + opt_log 注记收尾——手抄数字是 null 之外第二类结构化记录失真源）
- repro: 复现条件——调优轮出现 blocked-compile/invalid 分支时的记录行为（本任务 Round 3：bn=512 双槽 4 分支全部编译失败无测量值）
- toolchain_stamp: gate_lint.py 现行 S4-PERF-RECORDS-SCHEMA 规则；发生环境 tilelang 0.1.2+3a214cde / CANN 8.5.0 / Ascend910B2C / 2026-09-08
- target_doc: .agents/skills/tilelang-op-optimize/SKILL.md
- delta: |
    动作: update（步骤 7 扩展）
    定位锚: "7. 对精度通过的分支，用 `msprof op` 采集目标 kernel 性能；**验证完成后向 `perf_records.jsonl` 追加一行**（含 `parent_id` = 派生来源候选；append-only，禁止改写/删除既有行）。"
    old 文本: |
      7. 对精度通过的分支，用 `msprof op` 采集目标 kernel 性能；**验证完成后向 `perf_records.jsonl` 追加一行**（含 `parent_id` = 派生来源候选；append-only，禁止改写/删除既有行）。
    new 文本: |
      7. 对精度通过的分支，用 `msprof op` 采集目标 kernel 性能；**验证完成后向 `perf_records.jsonl` 追加一行**（含 `parent_id` = 派生来源候选；append-only，禁止改写/删除既有行）。**blocked-compile / invalid 分支不写入 perf_records.jsonl**——`duration_us` 恒为实测数值、不得为 null（signal-registry §5 契约，gate 4 `S4-PERF-RECORDS-SCHEMA` 强校验）；其结构性不可行证据（报错文本、L1/UB 溢出公式）记入 opt_log 当轮实验分支表的 invalid/blocked 行，证据不丢失。**行内数值一律从 msprof 提取输出直接复制（或管道）生成，禁止手抄**（2026-09-09 第二轮实证：final causal 首行 3779.0 转录笔误，靠 append-only 更正行 + opt_log 注记收尾）。
    动机: 契约的「验证后追加」语义未在写入指令处显式表达负面清单，optimizer 把 blocked 分支也「忠实记录」进结构化文件——2026-09-08 expert Stage 4 实证：4 个 bn=512 双槽 L1 溢出分支误记 null 致 gate 失败一次 attempt（7546s），窄修复又耗 314s；显式归宿规则可省整次重试。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260908T005751Z 2026-09-08
- decided_by: -
- decided_note: -

## VP-2026-0037
- type: R
- title: profile-collection.md msprof 输出目录权限规则从叶子目录扩展到父目录——group-writable 父目录下仅 chmod 700 输出目录采集仍被拒（8/8 baseline 首采失败实证）
- evidence:
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/opt_log.md#Skill-Retrospective（BP-R1 proposal：本轮 8 个 baseline 采集全部因 group-writable 父目录失败一次后才修正，前置检查可省一轮）
  - .agents/skills/tilelang-op-optimize/references/profile-collection.md#L96（现行规则仅 chmod 输出目录自身，未覆盖父目录）
- repro: 复现条件——在 group-writable 父目录（如默认 umask 002 的共享 checkout）下新建 msprof 输出目录、仅对叶子目录 chmod 700 后执行 msprof op 采集
- toolchain_stamp: CANN 8.5.0 msprof / Ascend910B2C / 2026-09-08
- target_doc: .agents/skills/tilelang-op-optimize/references/profile-collection.md
- delta: |
    动作: update（权限规则行扩展）
    定位锚: "新建 `msprof op` 输出目录后，先确保 group/other 不可写，例如 `chmod 700 {output_dir}`；否则某些环境会拒绝采集或写入失败。"
    old 文本: |
      新建 `msprof op` 输出目录后，先确保 group/other 不可写，例如 `chmod 700 {output_dir}`；否则某些环境会拒绝采集或写入失败。
    new 文本: |
      新建 `msprof op` 输出目录后，先确保该目录**及其父目录**（至少直接父目录）group/other 不可写，例如 `chmod 700 {output_dir}` 并同步检查父目录；msprof 的权限校验覆盖父目录——仅叶子目录 700 而父目录 group-writable 时采集仍被拒（2026-09-08 expert Stage 4 实证：8/8 baseline 首采失败，父目录收紧后一次通过）。
    动机: 现行规则只覆盖输出目录自身；本任务 8 个 baseline 采集全部因 group-writable **父目录**失败一次后才修正——预检规则覆盖父目录可直接省一轮重采。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260908T005751Z 2026-09-08
- decided_by: -
- decided_note: BP-R1 整合去重：optimizer 自提的「profile-collection.md 补权限预检」经查现行文件已有叶子目录规则（67db6f3 期引入），本任务证据表明其不充分——按 update 形态提案而非 add。

## VP-2026-0039
- type: R
- title: profile-collection.md 补「探针 kernel 测硬件上限」标准形态——复刻目标数据流多模式对照 + 编译报错反推容量 + 真实 kernel 反推双证
- evidence:
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/opt_log.md#Round-8（probe_kvbw 四模式〔single/shared/distinct/pp〕同为 ~110.8µs kernel-only → L1 端口 r+w 148–154GB/s/核；bm=64@bn=512 `cc overflow` 报错反推 L0C=128KB；fabric ≥1.73TB/s 聚合）
  - 同文件#Skill-Retrospective-第二轮 有效手法①（硬上限测绘先行——一轮内把设计空间不确定性收敛为可算术的复合地板，直接决定 DESIGN_LIMIT 判定的证据质量）
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/perf_feedback.md（三硬上限构成 [DESIGN_LIMIT] 2.09x/2.27x 缺口归因的实测基础）
  - pattern-library §1.9（数据侧落点：L0C/L1 端口/发射定律/fabric 条目，origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260909T033622Z）
- repro: 复刻目标 kernel 数据流（同结构装载/gemm 模式）的最小探针 + 编译期容量压限（`overflow, requires N bits while M bits available` 报错文本）+ 真实 kernel msprof 计数器反推（流量÷时间）三方对照
- toolchain_stamp: tilelang 0.1.2+3a214cde7aa4f54fc4a103f0f324a43341122d68 / CANN 8.5.0 (msprof) / Ascend910B2C / 2026-09-09
- target_doc: .agents/skills/tilelang-op-optimize/references/profile-collection.md
- delta: |
    动作: update（文件末尾追加一节）
    定位锚: |
      返回：

  ```
  ```text
  PERF_DATA_COLLECTED
  ```
  ```
  old 文本: |
      返回：

  ```
  ```text
  PERF_DATA_COLLECTED
```
  ```
    new 文本: |
      返回：

  ```
```text
  PERF_DATA_COLLECTED
```

---

## 6. 探针 kernel 测硬件上限（DESIGN_LIMIT 证据 / 设计空间测绘）

  适用：调优目标涉及硬天花板判定（设计层归因、[DESIGN_LIMIT] 反馈、结构方向投入前的可行性测绘）时，在迭代候选之外用探针 kernel 把硬件/编译器上限测成可算术的数字。标准形态（2026-09-09 attention expert 第二轮实证：一轮测绘直接支撑 DESIGN_LIMIT 判定）：

  1. **复刻目标数据流的最小探针**：与目标 kernel 同结构的装载/gemm/访问模式，多模式对照（single / shared / distinct / ping-pong 双槽）区分 fabric、端口、容量因子——四模式同耗时 ⇒ 端口读写串行（双槽不能解除），非 fabric 争用。
  2. **编译失败报错反推容量上限**：刻意压限触发 BishengIR `overflow, requires N bits while M bits available` 报错（L0C cc / L1 cbuf / UB），从报错文本反推硬容量与 bm/bn 联合约束式。
  3. **真实 kernel 计数器反推双证**：探针值须与真实 kernel 的 msprof 计数器反推值互证（如 L1 r+w 流量 ÷ (mte2+mte1) 时间 = 端口带宽）；单源探针结论降级为待证。
  4. **探针纪律**：跨引擎探针必须带 Cube→Vector flag 同步（缺同步的数值崩坏是探针自身竞态而非 API 缺陷，pattern-library §1.9）；探针 kernel 保留 ≥1 个输入 tensor（§2 零输入行）。
  5. **归宿**：探针结论回写 pattern-library §1（数据侧）；触及编译器能力缺口时登记 `.agents/evolution/capability-gaps.md`；探针 kernel 脚本与日志入 `perf_opt/logs/`（jsonl 不收录非本算子分支的探针行，证据存日志与 opt_log 当轮表）。
```
  动机: DESIGN_LIMIT 判定与结构方向裁决的证据质量取决于硬上限数字的可信度；本轮 round 8 以该形态一轮完成 L0C/L1 端口/fabric 三项测绘 + 复合地板算术，直接封死无效调参方向（bm 上调/深流水/两遍 softmax）并支撑 2.09x/2.27x 缺口归因——无此形态需多轮试错反推。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260909T033622Z 2026-09-09
- decided_by: -
- decided_note: -

## VP-2026-0040
- type: R
- title: iteration-diagnosis.md「已知迭代结论」补续调任务前置基线复核——上轮交付驱动完整性（手改漂移检测）+ 基线重采对账（偏差 ≤ 噪声阈值）
- evidence:
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/opt_log.md#第二轮-前置修复（上轮终版测试驱动被用户手改：run_fa_tuned 仅余 fa4096、main 绕过 --level 分派；恢复后 `--level all` 全过〔logs/round8/driver_restore_level_all.log〕——手改漂移重新验证 ✓）
  - 同文件#Round-8-Diagnosis-1（基线复核：4 case 重采 18.19/29.69/77.62/263.93µs，与上轮终值记录偏差 ≤0.4%——谱系/环境一致性验证后才开调）
- repro: 复现条件——任一以「上轮终值为基线」的续调 optimize 任务（用户在手改过的 perf_opt 交付物上发起第二轮）
- toolchain_stamp: tilelang 0.1.2+3a214cde7aa4f54fc4a103f0f324a43341122d68 / CANN 8.5.0 (msprof) / Ascend910B2C / 2026-09-09
- target_doc: .agents/skills/tilelang-op-optimize/references/iteration-diagnosis.md
- delta: |
    动作: update（「已知迭代结论」小节扩展）
    定位锚: |
      ### 已知迭代结论

  ```
每轮必须读 `opt_log.md` 中已有结论：

  ```text
  哪些优化点已验证有效
  哪些优化点已被当前 workload 证明无收益
  哪些修改导致精度失败 / 编译失败 / profile invalid
  current best 是哪个版本
  本轮 profile 相比上一轮 profile 哪些指标发生变化
  ```
  ```
  old 文本: |
      ### 已知迭代结论

  ```
  每轮必须读 `opt_log.md` 中已有结论：

  ```text
  哪些优化点已验证有效
  哪些优化点已被当前 workload 证明无收益
  哪些修改导致精度失败 / 编译失败 / profile invalid
  current best 是哪个版本
  本轮 profile 相比上一轮 profile 哪些指标发生变化
```
  ```
  new 文本: |
      ### 已知迭代结论

  ```
  每轮必须读 `opt_log.md` 中已有结论：

  ```text
  哪些优化点已验证有效
哪些优化点已被当前 workload 证明无收益
  哪些修改导致精度失败 / 编译失败 / profile invalid
  current best 是哪个版本
本轮 profile 相比上一轮 profile 哪些指标发生变化
```

**续调任务（基线 = 上轮终值）的额外前置**（2026-09-09 attention expert 第二轮实证）：开调前完成两项复核——① 上轮交付测试驱动完整性：确认 run/case 分派逻辑与上轮终版一致（用户可能手改交付物——实证：run_fa_tuned 仅余 1 case、main 绕过 --level 分派；恢复后 `--level all` 全过再开调）；② 基线重采对账：重采上轮终值 workload 与上轮记录对账，偏差 ≤ 噪声阈值（实证 4 case ≤0.4%）才算谱系/环境一致，超阈值先查谱系与环境再开调。
```
  动机: 续调任务若直接信任被手改的测试驱动或未复核的基线，后续所有测量与归因都建立在失真地基上；本轮前置修复 + 重采对账各耗数分钟，拦截的是整轮方向的偏移风险。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260909T033622Z 2026-09-09
- decided_by: -
- decided_note: -

## VP-2026-0041
- type: R
- title: iteration-diagnosis.md 补「参考实现对照（morph ladder）」标准定位法——结构移植后性能不达预期时在参考实现上逐级叠加契约增量二分定位（op 级无罪 ≠ 组合无罪）
- evidence:
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/opt_log.md#Skill-Retrospective-第三轮（有效手法① morph 阶梯决定性 + skill 流程问题①：形态学排查消耗 4 个 no_gain 分支后才启动，应更早切换；流程问题②：第二轮 [DESIGN_LIMIT] 地板归因未做同 contract 最小增量对照——有则 -20% transpose 税第二轮即被发现）
  - 同文件#Round-11-Diagnosis（morph 14 点矩阵：MORPH=0 参考原版 95.8 → MORPH=1 4D BSHD 输入 96.1 无罪 → MORPH=2c +lse 254.5 爆炸 → write-only 97.2 无罪 → 单 op 全无罪（vlog2/vmul+vadd/死源 transpose/4 个平凡 vadd）→ 任意含「活跃源 transpose」组合全部 253–255——唯一毒化点定位）
  - pattern-library §2 活跃源 transpose 毒化行（组合阈值效应数据侧，origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260909T071018Z）
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/perf_feedback.md 修正附录（负结论推翻闭环的同款方法论缺口）
- repro: 复现条件——任一「结构移植/重写后性能不达预期（>1.3x）且仓库内存在同族参考实现」的调优轮（本轮实证：v11 初版 250.68 vs 参考 95.8–99µs，逐字转录参考 body 仍 255.53——上下文级差异形态学排查完全无效）
- toolchain_stamp: tilelang 0.1.2+3a214cde7aa4f54fc4a103f0f324a43341122d68 / CANN 8.5.0 (msprof) / Ascend910B2C / 2026-09-09
- target_doc: .agents/skills/tilelang-op-optimize/references/iteration-diagnosis.md
- delta: |
    动作: update（「### 代码结构观察」小节末条之后、「### 已知迭代结论」标题之前插入新小节）
    定位锚: |
      - 冗余访问：是否有中间结果写回 GM 后又读回，或重复读取同一批 GM 数据。

  ```
### 已知迭代结论
  ```
    old 文本: （即上述定位锚原文）
  new 文本: |
      - 冗余访问：是否有中间结果写回 GM 后又读回，或重复读取同一批 GM 数据。

  ```
### 参考实现对照（morph ladder）

  仓库内存在同族参考实现（`examples/` 同 API 结构先例）且本方 kernel 结构移植/重写后性能显著不达预期（>1.3x）时，在逐项排查 op 形态 / pass_configs / 装载写法（含逐字转录参考 body）均无差异后，**优先切换 morph 阶梯定位，不在形态学排查上继续消耗实验分支**（2026-09-09 attention expert 第三轮实证：4 个形态学分支 no_gain 后 morph 阶梯一轮定位毒化点；且 [DESIGN_LIMIT] 地板归因若做此对照，-20% transpose 毒化税在第二轮即被发现）：

1. 在参考实现上逐级叠加本方契约增量（输入布局 → 输出契约 → 逐 op），每级一个实测点，二分定位差异源——**单 op / 单开关的无罪结论不可外推为组合无罪**（编译级阈值效应：活跃源 transpose 单 op 无罪、死源无罪、4 个平凡 vadd 无罪，任意含活跃源 transpose 的组合全部 2.6x 劣化，pattern-library §2）。
2. 对照点分级：契约级（输入/输出布局）先于 op 级；write-only 探针（只写不算）区分「计算成本」与「数据流形态触发」。
3. 纪律：perf-only 探针（故意破坏输出只测性能）在记录中显式标注；morph/探针脚本与实测点矩阵记入 opt_log 当轮（perf_records.jsonl 不收录非本算子 dispatch 的探针行）；scratch 探针结论须镜像回 opt_log 才可作为证据引用。

### 已知迭代结论
```
  动机: 「逐字转录参考仍慢 2.6x」类上下文级差异只有 morph 阶梯能定位（形态学排查对该类完全无效）；组合阈值效应使 op 级排除法结构性漏检——两条均指向该方法应为标准动作而非末位手段。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260909T071018Z 2026-09-09
- decided_by: -
- decided_note: -

## VP-2026-0045
- type: R
- title: op-design Phase 2 第 5 步等价性验证补双判据——基线式自身数值退化（pad 校正消灾难/大中间量结合序）时「候选 vs 基线」单判据假性 FAIL；补「候选对 fp64 误差 ≤ 基线误差（候选不得更差）」判据
- evidence:
  - examples/ada_layer_norm/_ada_layer_norm_kernel/verify_equiv.py（compare() 双准则实现 + 首跑输出 randn-N2 fp32 案例 crit=b：基线 fp64err 1.58e-4 vs 候选 1.9e-7——候选更接近精确数学却被判 FAIL，首跑 3/8 违反）
  - examples/ada_layer_norm/_ada_layer_norm_kernel/RETROSPECTIVE.md（Stage 1 Skill Flow Issues 首行 + Value Point Proposals 第二行：补判据后 45 案例全过且不掩盖真实回归）
- repro: 复现条件——任一基线式含 pad 校正/大中间量结合序的等价性验证（双判据形态自包含于 delta）
  - 〔provenance，允许失效〕任务内复现命令：python3 examples/ada_layer_norm/_ada_layer_norm_kernel/verify_equiv.py（输出「cases passing via criterion (b)」行）
- toolchain_stamp: torch CPU（设计期脚本）+ tilelang 0.1.2+a83118285a / 2026-09-10
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（第 5 步「对照对象」bullet 扩展）
    定位锚: "   - **对照对象**：容差内等价的采纳项，对照基准 = 原式（基线）在同 dtype 舍入路径下的输出；golden opmath 域问题（如 fp16 torch CPU 经 fp32 opmath）在角点用例中显式覆盖（lerp_tensor 实证：此类分歧设计期可机器拦截，Stage 3 才暴露损失 2485s）；"
    old 文本: （即上述定位锚原文）
    new 文本: |
       - **对照对象与判定双判据**：容差内等价的采纳项，对照基准 = 原式（基线）在同 dtype 舍入路径下的输出；golden opmath 域问题（如 fp16 torch CPU 经 fp32 opmath）在角点用例中显式覆盖（lerp_tensor 实证：此类分歧设计期可机器拦截，Stage 3 才暴露损失 2485s）。**基线式自身数值退化时（GPU pad 校正的 catastrophic cancellation、大中间量乘法结合序）单判据「候选 vs 基线」会产生假性 FAIL**——补充判据 (b)：候选对 fp64 精确数学的误差 ≤ 基线误差（候选不得更差）；判定 = (a) 候选 vs 基线容差内，或 (b) 成立；(b) 路径通过须在结果表标注角点与双侧 fp64err（ada_layer_norm 实证 2026-09-10：randn-N2 fp32 基线 fp64err 1.58e-4 vs 候选 1.9e-7，补判据后 45 案例全过且不掩盖真实回归）；
    动机: D-1 验证的对照口径只写了「候选式 vs 基线式（同 dtype 舍入路径）」，基线退化场景把更优的候选判 FAIL——首跑假性 FAIL 3/8 案例浪费一轮排查；双判据以「候选不得更差」守住回归底线。
- status: pending
- confirmations: -/-
- created_by: task ada_layer_norm-_ada_layer_norm_kernel-20260910T132324Z 2026-09-10
- decided_by: -
- decided_note: -

## VP-2026-0046
- type: R
- title: op-design Phase 4 内存规划章节补 auto-multi-buffer 膨胀预算标定规则——按结构类平台值（TRAP-UB 第三证）而非「resident × 系数」，无同构先例时显式标 estimate + Stage 3 compile-probe 验证步骤
- evidence:
  - examples/ada_layer_norm/_ada_layer_norm_kernel/RETROSPECTIVE.md（Stage 3 Skill Flow Issues 首行：DESIGN §4.5 按「驻留 × 1.7」估膨胀，§8.3 L0-1 据此选 fp32 bm=3 首跑即编译失败 requires 245,760B > 192KB，一轮编译探针才修正为 20 B/elem 律并降档 bm 3→2）
  - examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/perf_opt/opt_log.md（Iteration 2/3 探针：transit 20.01 / fp32 26.00 / nl=4 22.00 B/elem——平台值精测）
  - pattern-library traps-compiler.md TRAP-UB-multibuffer-inflation（第三证，2026-09-10 已合入）
- repro: compile-probe 复现条件：persistent serial（num_local_tasks≥2）+ 20×bm×N > 196608 → bishengir 报文 `ub overflow, requires <bits> while 1572864 bits available`（报文 requires bits 即精确实测 UB 需求）
- toolchain_stamp: tilelang 0.1.2+a83118285a + Ascend910B2C + CANN 8.5.0 / 2026-09-10
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（Phase 4 章节列表第 4 项扩展）
    定位锚: "4. 数据规格与内存规划"
    old 文本: |
      4. 数据规格与内存规划
    new 文本: |
      4. 数据规格与内存规划（UB 膨胀预算按结构类平台值标定：persistent serial auto-multi-buffer 下 transit ≈20 B/elem、fp32 ≈26 B/elem、nl≥4 或静态 extents +2（pattern-library TRAP-UB-multibuffer-inflation 第三证）——不以「resident × 系数」估算；无同构先例的结构显式标 estimate 并写明 Stage 3 compile-probe 验证步骤：`ub overflow` 报文 requires bits 即该配置精确实测 UB 需求，读数反推系数后修正工厂 guard 与默认 bm 表）
    动机: ada_layer_norm DESIGN 按 ×1.7 估算（当时无同构先例），Stage 3 首跑编译失败一轮（245,760B > 192KB）才被 compile-probe 修正为 20 B/elem 律——设计期标定规则 + 探针验证步骤可把该修复前移或至少把验证步骤预置于设计中。
- status: pending
- confirmations: -/-
- created_by: task ada_layer_norm-_ada_layer_norm_kernel-20260910T132324Z 2026-09-10
- decided_by: -
- decided_note: -

## VP-2026-0047
- type: R
- title: design_calc_check.py 补两类表格形态解析——「按 N 分行的驻留预算表」（§4.5）与「workload 表格式分核表」（§5.5）当前均 skip，VP-2026-0009 混合字节口径类错误的机械拦截位在这些形态上失效
- evidence:
  - examples/ada_layer_norm/_ada_layer_norm_kernel/RETROSPECTIVE.md（Stage 2 Skill Flow Issues 首行：ub_budget=skip、core_split=skip 的机械复核 JSON + REVIEW.md 人工逐行复算补位记录）
  - .agents/tools/design_calc_check.py#L100-103（check_ub_budget 行过滤 `^[a-zA-Z_]|缓冲` 把 N 值数字行当 header 跳过）+ #L156-158（skip detail "no per-level buffer tables parsed"）+ #L231-233（core_split skip "insufficient parsed dims"）
  - 交叉引用：queue VP-2026-0009（UB 预算混合字节口径——本条是其机械拦截位的形态扩展）
- repro: 复现条件——DESIGN.md 采用「按 N 分行的驻留预算表」（§4.5）或「workload 表格式分核表」（§5.5）形态时跑 design_calc_check 两个检查项（当前均 skip）
  - 〔provenance，允许失效〕任务内复现命令：python3 .agents/tools/design_calc_check.py check --design examples/ada_layer_norm/_ada_layer_norm_kernel/DESIGN.md
- toolchain_stamp: design_calc_check.py 现行版本；失效环境 tilelang 0.1.2+a83118285a / 2026-09-10
- target_doc: .agents/tools/design_calc_check.py
- delta: |
    动作: update（两处解析扩展）
    定位锚 1: |
      (check_ub_budget 函数体内)
          if not any(re.match(r"^[a-zA-Z_]", c) or "缓冲" in c for c in cells):
              continue  # header rows
    old 文本 1: （即上述定位锚 1 原文）
    new 文本 1: |
          if not any(re.match(r"^[a-zA-Z_]", c) or "缓冲" in c for c in cells):
              # Row-N residence tables (N / block_m / 驻留字节 columns):
              # numeric first cell (N) + a per-elem-bytes or 驻留 column ->
              # parse as per-N footprint rows (bytes_per_elem * N) for the
              # UB budget check, so mixed-dtype byte widths are still caught.
              if not (cells and re.match(r"^\d", cells[0]) and any(
                  ("B/elem" in c or "字节" in c or "驻留" in c) for c in cells
              )):
                  continue  # header rows
    定位锚 2: |
      (check_core_split 函数体内，skip 分支)
              "detail": f"insufficient parsed dims {nums}; declared logical "
    old 文本 2: （即上述定位锚 2 原文所在 skip 分支）
    new 文本 2: |
      同分支前增加 workload 表格式识别：首列为 workload 名、含 "MxN"/shape 与 bm 列的表格行，从 (M, bm) 重算 num_logical = ceildiv(M, bm) 与表中声明值对照（无法解析全部维度时保留 skip，但 detail 注明已识别的 workload 行数）
    动机: ada_layer_norm 的 §4.5/§5.5 采用「N 分行驻留表 + workload 分核表」形态，两项机械检查双双 skip，混合字节口径错误只能靠人工逐行复算拦截——正是 VP-2026-0009 定义的机械拦截位，扩展解析使该防线对常见表格形态生效。
- status: pending
- confirmations: -/-
- created_by: task ada_layer_norm-_ada_layer_norm_kernel-20260910T132324Z 2026-09-10
- decided_by: -
- decided_note: -

## VP-2026-0048
- type: R
- title: core-split-strategy.md §2.2 reviewer 侧物理核数实查的替代验证条款——reviewer 无 NPU 设备连接时以「实查记录存在 + 常数表交叉 + 同款先例」三证放行，设备变更场景标注复核责任在 Stage 3/4
- evidence:
  - examples/ada_layer_norm/_ada_layer_norm_kernel/RETROSPECTIVE.md（Stage 2 Skill Flow Issues 第三行：reviewer 无法独立重放 NPUUtils 实查〔需设备连接〕，本次以设计实查记录 §5.5 + CONST-aicore-910B2C + lerp 同款查询三方交叉佐证放行）
  - .agents/skills/_shared/standards/core-split-strategy.md §1 ②（实查记录要求，无 reviewer 侧独立验证手段条款）
- repro: 只读核对（无运行时依赖）——reviewer 环境无 NPU 设备连接时核数实查不可重放的位形
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 tilelang 0.1.2+a83118285a / 2026-09-10
- target_doc: .agents/skills/_shared/standards/core-split-strategy.md
- delta: |
    动作: update（§2.2 透传文本扩展）
    定位锚: "> 维度 3（Tiling 策略）须核对分核策略三要素齐全、物理核数为 NPUUtils.get().get_aicore_num() 实查值（纯 Vector 算子翻倍）、核数与 block 取值自洽、核内串行边界静态。"
    old 文本: （即上述定位锚原文）
    new 文本: |
      > 维度 3（Tiling 策略）须核对分核策略三要素齐全、物理核数为 NPUUtils.get().get_aicore_num() 实查值（纯 Vector 算子翻倍）、核数与 block 取值自洽、核内串行边界静态。reviewer 环境无 NPU 设备连接、实查不可独立重放时，以三证放行：设计实查记录存在（查询代码 + 返回值）+ constants.md 物理核数条目交叉（CONST-aicore-910B2C）+ 同款先例（examples/ 同族查询记录）；目标设备非 910B2C 时三证链失效，核数复核责任显式移交 Stage 3/4（首跑核数断言或 profile Block Dim 核对）。
    动机: reviewer 侧无设备是常态位形（检视可在无 NPU 环境执行），现行条款只有 designer 侧实查义务、无 reviewer 侧替代验证手段——三证放行把该场景的核对义务显式化，避免形同虚设或过度阻塞。
- status: pending
- confirmations: -/-
- created_by: task ada_layer_norm-_ada_layer_norm_kernel-20260910T132324Z 2026-09-10
- decided_by: -
- decided_note: -

## VP-2026-0049
- type: R
- title: design-review Phase 1 第 7 步 verify_equiv 重跑核对补口径——「与内嵌表一致」须逐格对上确界（汇总格 = 逐案例输出的上确界），聚合口径不当的汇总值不构成不一致但须记建议级微瑕
- evidence:
  - examples/ada_layer_norm/_ada_layer_norm_kernel/RETROSPECTIVE.md（Stage 2 Value Point Proposals 首行：bf16 extreme-mod 案例逐输出 max_ulp=1 而 DESIGN §1.6.1 内嵌汇总格写 0——violation 计数与 EQUIV_PASS 判定不受影响，但「与内嵌表一致」的核对结论未逐格对上确界）
  - .agents/skills/tilelang-design-review/SKILL.md#L139（现行口径「执行结果须与 §1.6.1 内嵌结果表一致且全部 EQUIV_PASS」——「一致」未定义粒度）
- repro: 复现条件——任一含内嵌汇总格的 verify_equiv 结果表重跑核对（汇总格 vs 逐案例输出上确界）
  - 〔provenance，允许失效〕任务内复现命令：python3 examples/ada_layer_norm/_ada_layer_norm_kernel/verify_equiv.py（bf16 段 extreme-mod 行 vs 内嵌表）
- toolchain_stamp: torch CPU + tilelang 0.1.2+a83118285a / 2026-09-10
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（Phase 1 第 7 步首条 bullet 扩展）
    定位锚: "   - **重跑等价性验证**（`DESIGN.md` §1.6.1 含采纳优化项时）：`python examples/{project}/{op}/verify_equiv.py`——执行结果须与 §1.6.1 内嵌结果表一致且全部 `EQUIV_PASS`；脚本缺失、执行失败或与内嵌表不一致 → 维度 8 fail（等价性机器验证失效）；"
    old 文本: （即上述定位锚原文）
    new 文本: |
       - **重跑等价性验证**（`DESIGN.md` §1.6.1 含采纳优化项时）：`python examples/{project}/{op}/verify_equiv.py`——执行结果须与 §1.6.1 内嵌结果表一致且全部 `EQUIV_PASS`；脚本缺失、执行失败或与内嵌表不一致 → 维度 8 fail（等价性机器验证失效）。「一致」的核对口径：汇总格（如「逐位一致 max ulp=0」）须等于逐案例输出的**上确界**（ada_layer_norm 实证：bf16 extreme-mod 逐案例 max_ulp=1 而汇总格写 0——violation 计数与判定不受影响，记建议级微瑕并要求修正汇总格，不判 fail）；
    动机: 「与内嵌表一致」未定义粒度时，聚合口径差异（min vs sup）与真实漂移不可区分——逐格对上确界是零成本口径收紧。
- status: pending
- confirmations: -/-
- created_by: task ada_layer_norm-_ada_layer_norm_kernel-20260910T132324Z 2026-09-10
- decided_by: -
- decided_note: -

## VP-2026-0050
- type: R
- title: design-review 维度 1「API 存在性」补 docs 无文档 API 的 examples 佐证豁免口径——T.min 类 API 在 docs/Tilelang.language/ 全子目录无独立文档（含未映射目录已枚举），3 处 examples 实调构成合法佐证
- evidence:
  - examples/ada_layer_norm/_ada_layer_norm_kernel/RETROSPECTIVE.md（Stage 2 Skill Flow Issues 第二行：`ls docs/Tilelang.language/` 无 T.min.md；examples/flash_attention/flash_attn_npuir_dev.py L86 / lerp / logsumexp 三处实调）
  - docs/Tilelang.language/ 全子目录枚举（含创建操作/索引与元素操作/条件操作/排序操作/逻辑操作/原子操作/编译器提示操作——负向断言举证标准要求 docs 路径，但该 API 连正向使用也无文档可引）
- repro: 复现条件——语言文档全部子目录枚举无该 API 独立条目，且仓库示例代码实调检索得 ≥2 处跨算子目录的使用
  - 〔provenance，允许失效〕任务内核对命令：ls docs/Tilelang.language/数学操作/ | grep -i min（无 T.min 条目）；grep -rn "T\.min(" examples/ --include="*.py" | head（多处实调）
- toolchain_stamp: 仓库 docs 现状 2026-09-10；与 tilelang 版本无关
- target_doc: .agents/skills/tilelang-design-review/SKILL.md
- delta: |
    动作: update（维度 1 表「API 存在性」行扩展）
    定位锚: "| API 存在性 | DESIGN.md §3.2 列出的每条 TileLang DSL API 能在 `examples/` 或 `tilelang/language/` 中找到使用佐证 | 设计用了 `T.flash_attention` 但项目无此 API |"
    old 文本: （即上述定位锚原文）
    new 文本: |
      | API 存在性 | DESIGN.md §3.2 列出的每条 TileLang DSL API 能在 `examples/` 或 `tilelang/language/` 中找到使用佐证 | 设计用了 `T.flash_attention` 但项目无此 API |
      | docs 缺文档 API 的佐证口径 | API 在 docs/Tilelang.language/ 全子目录（含未映射目录）无独立文档时（实证：T.min，2026-09-10 枚举），≥2 处 `examples/` 实调（跨算子目录）构成存在性佐证，检视记录注明「docs 无条目 + N 处实调」——负向断言举证标准不因文档缺口而弃用，但正向存在性不要求 docs 条目 | 仅有 docs 条目而无任何实调（文档与实现漂移风险） |
    动机: 负向断言举证标准（须引 docs 路径）遇到「API 无文档但实际存在」时举证责任无法满足——T.min 实证靠 3 处 examples 佐证放行；口径明示化避免同类场景每次重新论证。配套（超出本提案写域，需人工/contributor 处理）：docs/Tilelang.language/数学操作/ 补 T.min 条目（标量 min，含尾块 real_mode 用法）。
- status: pending
- confirmations: -/-
- created_by: task ada_layer_norm-_ada_layer_norm_kernel-20260910T132324Z 2026-09-10
- decided_by: -
- decided_note: -

## VP-2026-0051
- type: R
- title: develop SKILL.md Phase 4 结果收集补失败分类要求——compile/runtime-error 与 precision-fail 分类上报（混排同列会使 conductor 的 precision_fix 路由把编译错误判为精度错）
- evidence:
  - examples/ada_layer_norm/_ada_layer_norm_kernel/RETROSPECTIVE.md（Stage 3 Skill Flow Issues 第二行：首跑 12 个 L0 失败混有两类根因〔1 个编译期 UB 溢出 RUNTIME-ERROR + 11 个 vrsqrt 精度 FAIL〕，汇总输出未分类）
  - 同任务首跑输出（`[collected] L0-1 RUNTIME-ERROR` 与 `[L0-5] FAIL: max_diff=...` 混排同列）
  - .agents/skills/tilelang-op-develop/SKILL.md#L48（现行「3. 收集结果：max_diff、失败用例 shape、层级」——无失败分类要求）
- repro: 复现条件——任一 L0 首跑同时含编译失败与精度失败的混合位形（本任务实证）
- toolchain_stamp: tilelang 0.1.2+a83118285a + Ascend910B2C + CANN 8.5.0 / 2026-09-10
- target_doc: .agents/skills/tilelang-op-develop/SKILL.md
- delta: |
    动作: update（Phase 4 收集步骤扩展）
    定位锚: "3. 收集结果：max_diff、失败用例 shape、层级。"
    old 文本: |
      3. 收集结果：max_diff、失败用例 shape、层级。
    new 文本: |
      3. 收集结果：max_diff、失败用例 shape、层级；失败须按 **compile/runtime-error vs precision-fail 分类**上报（分类计数 + 各类代表案例），不得混排同列——直接影响四出口判定与 conductor 路由精度（ada_layer_norm 实证 2026-09-10：首跑 12 失败 = 1 编译期 UB 溢出 + 11 精度 FAIL，未分类时 precision_fix 路由会把编译错误误判为精度错）。
    动机: 四出口判定与 conductor 路由按失败类型分流；混合失败位形下无分类的汇总使路由依据失真。
- status: pending
- confirmations: -/-
- created_by: task ada_layer_norm-_ada_layer_norm_kernel-20260910T132324Z 2026-09-10
- decided_by: -
- decided_note: -

## VP-2026-0053
- type: R
- title: optimize SKILL.md Phase 2 第 6 步「每个实验分支跑 L0 精度回归」补分支类型分层——参数分支（同 kernel 不同 config）可用 bench 内嵌 golden check 等价替代，结构分支必须全量套件
- evidence:
  - examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/perf_opt/opt_log.md#Skill-Retrospective（BP proposal：参数扫描分支〔同 kernel 不同 bm〕的精度门禁 = bench `--check` 精确 (M,N,dtype,bm) golden 比对；结构分支 = 全量 `--level all`——该分层未显式定义，「每个实验分支跑 L0」对参数分支语义模糊，内嵌 L0 表不覆盖被扫的 bm）
  - 同文件 Iteration 1/2/3（参数分支 18+16+8 行记录全走 --check、结构分支走 --level all 的实际执行形态）
- repro: 复现条件——任一含参数扫描分支（同 kernel 不同 bm/config）的调优轮（本任务 round 1/3 实证形态）
- toolchain_stamp: tilelang 0.1.2+a83118285a + Ascend910B2C + CANN 8.5.0 / 2026-09-10
- target_doc: .agents/skills/tilelang-op-optimize/SKILL.md
- delta: |
    动作: update（Phase 2 第 6 步扩展）
    定位锚: "6. 每个实验分支跑 L0 精度回归。"
    old 文本: |
      6. 每个实验分支跑 L0 精度回归。
    new 文本: |
      6. 每个实验分支跑 L0 精度回归，按分支类型分层：**参数分支**（同 kernel 仅改 config，如 bm/num_kernels 扫描）可用 bench 内嵌 golden check（精确 (shape, dtype, config) 比对，如 bench.py `--check`）等价替代完整 L0 套件——内嵌 L0 表不覆盖被扫的 config 值；**结构分支**（kernel 代码改动）必须全量 `--level all`（ada_layer_norm 2026-09-10 实证形态：round 1 参数扫描 18 行全 --check、round 2 结构改动全量 + 分支 `--level all`）。
    动机: 「每个实验分支跑 L0」对参数分支语义模糊（L0 表按基准 config 参数化，不覆盖扫描值）——显式分层消除歧义，避免参数分支误跑不覆盖的全量（浪费）或结构分支误用单点 check（漏检）。
- status: pending
- confirmations: -/-
- created_by: task ada_layer_norm-_ada_layer_norm_kernel-20260910T145715Z 2026-09-10
- decided_by: -
- decided_note: -

## VP-2026-0054
- type: R
- title: ab_test.py 输出目录创建后内置 chmod 700——msprof 权限校验拒绝 group-writable 目录，工具自建目录不 chmod 时采集必失败（需 umask 077 前置调用才可避开）
- evidence:
  - examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/perf_opt/opt_log.md#Skill-Retrospective-工具/环境问题（ab_test.py 建目录未 chmod 700，msprof 拒写 "writable by any other users or group users"——需 umask 077 前置调用）
  - .agents/tools/ab_test.py#L50-51（os.makedirs 两处，无 chmod）
  - 交叉引用：queue VP-2026-0037（msprof 父目录权限规则——profile-collection.md 规则侧，本条为工具侧互补修复）
- repro: 复现条件——默认 umask 002 的共享 checkout 下以默认权限建 msprof 输出目录后执行采集（本任务实证形态；VP-2026-0037 的 8/8 首采失败为同机制父目录侧实证）
- toolchain_stamp: CANN 8.5.0 msprof / Ascend910B2C / 2026-09-10
- target_doc: .agents/tools/ab_test.py
- delta: |
    动作: update（两处 makedirs 后追加 chmod）
    定位锚: |
      (run_pair 函数体内)
          os.makedirs(out_dir, exist_ok=True)
          os.makedirs(os.path.dirname(log_path), exist_ok=True)
    old 文本: |
          os.makedirs(out_dir, exist_ok=True)
          os.makedirs(os.path.dirname(log_path), exist_ok=True)
    new 文本: |
          os.makedirs(out_dir, exist_ok=True)
          os.makedirs(os.path.dirname(log_path), exist_ok=True)
          # msprof refuses to write into group/world-writable dirs (same
          # check as VP-2026-0037's parent-dir rule); self-created dirs
          # must be tightened explicitly regardless of umask.
          os.chmod(out_dir, 0o700)
    动机: msprof 权限校验覆盖输出目录自身；工具自建目录依赖调用方 umask 077 是隐式契约——内置 chmod 消除对调用环境的依赖（规则侧 VP-2026-0037 已覆盖「须 700」，本条使工具自身满足该规则）。
- status: pending
- confirmations: -/-
- created_by: task ada_layer_norm-_ada_layer_norm_kernel-20260910T145715Z 2026-09-10
- decided_by: -
- decided_note: -

## VP-2026-0055
- type: R
- title: integrate_kernel.py 产物侧校验两缺陷——① wrapper 契约名在产物中无顶层定义时 gen_init_py 静默降级 [warn]（__init__.py 仍声称 __all__ 含该名，smoke 才 ImportError 且无修复指引；幂等重跑覆盖集成侧补丁）；② gen_kernel_source_block 的 stem fallback 把非函数名（常量/helper）当模块路径生成无意义 perf_opt 占位行
- evidence:
  - examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/integration_log.md（Debug history attempt-1：首跑 `[warn] ALIGNMENT: not found in integrated modules` ×2 → `[error] import smoke test FAILED`；Issues 第 1/2 条）+ ada_layer_norm.py L61/L63（`# from .ada_layer_norm_kernel.perf_opt.ALIGNMENT import ALIGNMENT` 类占位行）
  - examples/TileOPs/.agents/skills/add-npu-op/scripts/integrate_kernel.py#L158-162（defining_module==None → [warn] 注释行）+ #L211（`stem = defining_module(name, integrated_files) or name`）
  - 交叉引用：queue VP-2026-0032（同文件 stale lineage 告警——缺陷集群）、VP-2026-0012（幂等重跑覆盖集成侧修复——本任务 Stage 5 同款再现，证据链互链）
- repro: 重跑 python .agents/skills/add-npu-op/scripts/integrate_kernel.py --meta tileops/kernels/norm/ada_layer_norm/.migration_meta.json（覆盖集成副本后 smoke 必现 ImportError——provenance，允许失效）
- toolchain_stamp: tilelang 0.1.2+a83118285a + Ascend910B2C + CANN 8.5.0 + torch 2.9.0+cpu / 2026-09-10
- target_doc: examples/TileOPs/.agents/skills/add-npu-op/scripts/integrate_kernel.py
- delta: |
    动作: update（两处）
    定位锚 1: |
      (gen_init_py 函数体内)
              stem = defining_module(name, integrated_files)
    （其后 [warn] 分支，L158-162）
    old 文本 1: |
              stem = defining_module(name, integrated_files)
    new 文本 1: |
              stem = defining_module(name, integrated_files)
    （[warn] 分支升级为可操作 [error]：parse_wrapper_imports 得到的 wrapper 契约名在产物顶层定义缺失时——提示两条修复路径：集成包内逐字 re-export GPU extracted 的同名定义（Stage 5 胶水先例，见 ada_layer_norm integration_log attempt-1）/ 调整 scaffolder-integrator 契约透传 helper 名；并保留已修补的非产物段不再整文件覆盖〔幂等重跑保护〕）
    定位锚 2: |
      (gen_kernel_source_block 函数体内)
              stem = defining_module(name, integrated_files) or name
    old 文本 2: |
              stem = defining_module(name, integrated_files) or name
    new 文本 2: |
              stem = defining_module(name, integrated_files)
              # Non-function names (constants/helpers) have no per-func
              # perf_opt module: fall back to the single integrated module
              # stem and import by name, or emit no placeholder line --
              # never use the bare name as a module path.
    动机: 契约名缺失从「静默 warn + smoke 期 ImportError 无指引」升级为前置可操作 error（修复手法已在 ada 案例验证）；占位行 stem fallback 对常量/helper 生成语义错误的模块路径（注释态无害但误导后续 wrapper 切换）。两缺陷与 VP-2026-0032/0012 同属 integrate_kernel.py 幂等与校验缺口集群，建议同批审批。
- status: pending
- confirmations: -/-
- created_by: task ada_layer_norm-_ada_layer_norm_kernel-20260910T132324Z 2026-09-10
- decided_by: -
- decided_note: -

## VP-2026-0056
- type: R
- title: integrator 已知修复目录「import / 路径错误」行补 extracted-import 契约缺口形态——wrapper 从 GPU extracted 模块导入的全部名字（含常量/helper）构成集成 re-export 契约；NPU 重设计丢弃对应机制时最小修复 = 集成副本逐字 re-export 同名定义
- evidence:
  - examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/integration_log.md（Debug history attempt-1 完整修复链：ALIGNMENT/_align_up 逐字 re-export + __init__ 修正 re-export；安全判别——名字只影响被 NPU 工厂接受但忽略的 flag 或无消费者的 config 选择器路径）
  - examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm.py L13-21（逐字源：GPU extracted _ada_layer_norm_fwd_kernels.py）
  - examples/TileOPs/tileops/kernels/norm/ada_layer_norm/RETROSPECTIVE.md（Stage 5 Transferable Lessons：wrapper 政策函数与 pytest 政策测试仍校验 GPU 语义，属 wrapper 胶水契约——集成期不顺手 NPU 化）
- repro: 复现条件——任何 wrapper extracted-import 含非 kernel 名字（常量/helper）且 NPU 产物不定义它的迁移（重跑 integrate_kernel.py 覆盖集成副本 → smoke 必现 ImportError）
- toolchain_stamp: tilelang 0.1.2+a83118285a + Ascend910B2C + CANN 8.5.0 + torch 2.9.0+cpu / 2026-09-10
- target_doc: .opencode/agents/tilelang-op-integrator.md
- delta: |
    动作: update（已知修复目录表「import / 路径错误」行扩展）
    定位锚: "| import / 路径错误 | 检查聚合 `__init__.py` 与相对导入 |"
    old 文本: |
      | import / 路径错误 | 检查聚合 `__init__.py` 与相对导入 |
    new 文本: |
      | import / 路径错误 | 检查聚合 `__init__.py` 与相对导入。**extracted-import 契约缺口**（ImportError 指向常量/helper 名而非 kernel 名时）：wrapper 从 GPU extracted 模块导入的全部名字构成集成包 re-export 契约；NPU 重设计丢弃对应机制（padding/cp.async 类）致产物不顶层定义这些名字时，最小修复 = 集成副本逐字 re-export GPU extracted 的同名定义（语义零漂移，wrapper 政策函数与 pytest 政策测试继续校验 GPU 语义——集成期不顺手 NPU 化）；安全判别：名字只影响被 NPU 工厂接受但忽略的 flag 或无消费者的 config 选择器路径。修复后幂等重跑会覆盖补丁（同 VP-2026-0012/0055 caveat） |
    动机: 该形态在本任务以 1 attempt 闭环（修复手法验证有效），但检索无门——已知修复目录只有泛化的「检查 __init__」；契约语义（名字清单 = 契约）与最小修复形态入目录后，同类迁移任务可一次定位。
- status: pending
- confirmations: -/-
- created_by: task ada_layer_norm-_ada_layer_norm_kernel-20260910T132324Z 2026-09-10
- decided_by: -
- decided_note: -

## VP-2026-0064
- type: R
- title: integrate_kernel.py 幂等检查只认未注释的激活 import——rewrite_wrapper_import 的 "already integrated" 探测正则匹配到被注释的 baseline 行（调优轮 perf_opt 激活态遗留），wrapper 不重写、旧谱系 perf_opt kernel 静默保持生效
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/integration_log.md Round 2 Integration steps Step 1（`[wrapper] ... already integrated`〔wrapper_rewritten: false〕时 wrapper L65 baseline 被注释、L67 perf_opt import 生效；`[smoke]` 仅 import 冒烟不区分激活源）+ Step 2 人工胶水修复（`w._gqa_prefill_fwd_kernel.__module__` 修复前解析到 perf_opt 旧谱系）
  - examples/TileOPs/.agents/skills/add-npu-op/scripts/integrate_kernel.py#L224-226（text 全文 re.search，不剥离注释行）
  - 交叉引用：queue VP-2026-0032（stale perf_opt lineage 告警——预警场景本轮实际发生的另一形态，修「重写后不告警」侧）、VP-2026-0055/0056（同文件幂等与校验缺口集群，建议同批审批）
- repro: 复现条件——wrapper 处于 perf_opt 激活态（调优轮遗留：baseline import 被注释）时重跑 integrate_kernel.py → 输出 already integrated 且 wrapper_rewritten=false，旧谱系 perf_opt import 保持激活
- toolchain_stamp: tilelang 0.1.2+28783f45 + CANN 8.5.0 + Ascend910B2C / 2026-09-15；脚本现行版本
- target_doc: examples/TileOPs/.agents/skills/add-npu-op/scripts/integrate_kernel.py
- delta: |
    动作: update（幂等检查剥离注释行后匹配）
    定位锚: |
      (rewrite_wrapper_import 函数体内)
          text = wrapper_path.read_text(encoding="utf-8")
          if re.search(rf"from \.{op_slug}_kernel(?:\.\w+)?\s+import", text):
              return False
    old 文本: |
          text = wrapper_path.read_text(encoding="utf-8")
          if re.search(rf"from \.{op_slug}_kernel(?:\.\w+)?\s+import", text):
              return False
    new 文本: |
          text = wrapper_path.read_text(encoding="utf-8")
          # Idempotency must consider only ACTIVE (uncommented) imports: a
          # commented-out baseline line (left by a perf_opt-activation tuning
          # round) must not satisfy this probe -- otherwise the wrapper is
          # never rewritten and the stale perf_opt import stays active
          # (factory-compatible signature; import smoke does not detect it).
          active_lines = [
              line for line in text.splitlines()
              if not line.lstrip().startswith("#")
          ]
          if re.search(rf"from \.{op_slug}_kernel(?:\.\w+)?\s+import", "\n".join(active_lines)):
              return False
    动机: 换代重集成遇 perf_opt 激活态 wrapper 时被注释的 baseline 行骗过幂等检查——旧谱系调优 kernel 静默保持生效且 smoke 不拦截（2026-09-15 实际发生，靠人工 __module__ 核对才发现）；与 VP-2026-0032 成对互补。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0065
- type: R
- title: develop SKILL.md Phase 4 L1 门禁补「wrapper-default 派发路径」——带 config 替换语义的 kernel，出厂 dispatch 的 traced 变体必须 ∈ 被门禁编译验证过的变体集
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/multi_head_attention/RETROSPECTIVE.md Stage 5 round-2 Skill Flow Issues 第 2 行（Stage 3 L1 门禁用逐字 config `fn(64,64,ns=2)` → bn_eff∈{64,144} 验证 manifest 域，wrapper 实际派发走 E6 替换路径 (64,64,1)→(64,256)；L0-3/L0-4 仅 dim=64 smoke 验证替换路径）
  - integration_log.md Round 2 §Design-layer finding（E6 替换路径在 dim=128 causal 域的首次编译发生在 Stage 5 bench 且直接 UB 溢出 8/10；pytest 4 参数全 causal=False 三重掩盖）
  - 〔2026-09-16 第二证（不同任务），task multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z r9 precision_fix〕同类失效再发生：r7e per-shape bm=80 分派 × E6 bn 钳位（S_kv≥3841 → bn_eff=288）的组合变体从未被任何门禁编译（standalone 29 例 S_kv≤2048 / manifest S≤2048 / 此前 pytest 域未覆盖该 S），首次编译发生在 TileOPs pytest full-fwd-bf16 且直接 UB 溢出 13472B（bishengir-compile exit 1）——「出厂 dispatch 的 traced 变体 ∉ 门禁验证集」的同一失效类，且两个守卫各自安全、组合超限（联合核算判据 + repro：attention.md PL-1.12 r9 update / repro/PL-1.12-bn-clamp-bm-guard.py）。Tier 2 人工门不因证据数豁免，供审批参考；与 VP-2026-0074（修复侧 config-契约范式）互补
- repro: 复现条件——任一 kernel 工厂带 config 替换/分派语义的迁移任务（wrapper default_config 派发的 traced 变体未入 L1 变体集时，出厂首编后移到 Stage 5 bench 暴露）
- toolchain_stamp: tilelang 0.1.2+28783f45 + CANN 8.5.0 + Ascend910B2C / 2026-09-15
- target_doc: .agents/skills/tilelang-op-develop/SKILL.md
- delta: |
    动作: update（Phase 4 第 2 步扩展）
    定位锚: "2. L0 通过后扩展 L1/L2/Boundary 并跑全量 `--level all`。"
    old 文本: |
      2. L0 通过后扩展 L1/L2/Boundary 并跑全量 `--level all`。
    new 文本: |
      2. L0 通过后扩展 L1/L2/Boundary 并跑全量 `--level all`。**门禁变体集覆盖 wrapper-default 派发路径**：kernel 工厂带 config 替换/分派语义时，wrapper `default_config` 实际派发的 traced 变体（含替换路径与惰性旋钮组合）必须被 L1 在目标域（dtype/dim/causal 维度）编译验证——至少一组与 wrapper `default_config` 全同的调用 + 替换路径的目标域案例（2026-09-15 反例：E6 替换路径 bn_eff=256 在 dim=128 causal 域首次编译发生在 Stage 5 bench 且 UB 硬溢出 8/10，L1 逐字 config + L0 dim=64 smoke + pytest 全 non-causal 三重掩盖）。
    动机: 出厂 dispatch 变体不在门禁变体集内时，唯一编译门控后移到 Stage 5 bench——集成期编译失败按功能缺陷查根因的成本远高于 L1 期发现。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0066
- type: R
- title: design SKILL.md Phase 4 补 CJK 排版规则——数学记号中 CJK 只出现在圆括号或正文、不进 {...}（含 LaTeX \text{中文} 下标与 e^{中文 − var} 上标），规避 S1-PLACEHOLDER 误报；命中合法记号时改写消除模式命中而非申诉豁免
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md Stage 1 gate-修复重试章节 Skill Flow Issues 首行（DESIGN.md L41/L446 两处误报 `e^{块内分数 − m_cur}`、`_{\text{Cube 部和，f16 物化回传}}`；修复 = 括号形式改写 + 注解移出公式，`grep -P '\{[^}]*[\x{4e00}-\x{9fff}][^}]*\}'` 复核清零，语义零变化）
  - 交叉引用：queue VP-2026-0019（gate_lint 正则侧修复——本条为 writer 侧互补；2026-09-15 新形态 `e^{中文 − var}` 无列表分隔符，VP-2026-0019 现提案正则不覆盖，见其证据链追加）
- repro: 复现条件——DESIGN.md 数学记号含 CJK-in-braces 形态（`grep -P '\{[^}]*[\x{4e00}-\x{9fff}][^}]*\}' DESIGN.md` 有命中）时跑 gate 1
- toolchain_stamp: gate_lint.py 现行版本；误报发生于 2026-09-15 任务
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（Phase 4 首行后追加一条 blockquote）
    定位锚: "基于 [templates/design-template.md](templates/design-template.md) 模板，填充所有章节："
    old 文本: |
      基于 [templates/design-template.md](templates/design-template.md) 模板，填充所有章节：
    new 文本: |
      基于 [templates/design-template.md](templates/design-template.md) 模板，填充所有章节：

      > **CJK 排版规则（S1-PLACEHOLDER 误报规避）**：DESIGN.md 数学记号中的 CJK 只出现在圆括号或正文，不进 `{...}`（含 LaTeX `\text{中文}` 下标与 `e^{中文 − var}` 形态上标）——gate 1 的 CJK-in-braces 模式会命中此类记号判「疑似模板变量未替换」。命中合法记号时的修复姿势 = 改写消除模式命中（括号形式 `e^(...)` / 注解移出公式为正文括注），语义零变化、门禁可机械复核，不申诉豁免（2026-09-15 实证：两处误报改写后 `grep -P` 复核清零；纯 ASCII 花括号不触发）。
    动机: gate 1 误报消耗整段修复重试（本任务 Stage 1 重试的 2/4 项）；writer 侧预防与 gate 侧正则修复（VP-2026-0019）双管齐下，正则修复未落地前排版规则是唯一防线。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0067
- type: R
- title: design SKILL.md「超长文档分段落盘」从条件建议升级为阈值无条件触发——VP-2026-0022 于 2026-09-11 apply 后 2026-09-15 仍复现首写截断（「预计超过」的自行判断是失效点），heredoc 分段追加为可靠替代
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md Stage 1 v3 章节 Skill Flow Issues 首行（1116 行 / ~95KB DESIGN.md 首次 write 即截断——上版建议已合入仍复现且更早失败；bash heredoc 6 段追加成功）
  - .agents/skills/tilelang-op-design/SKILL.md Phase 4 blockquote（VP-2026-0022 2026-09-11 apply 文本——条件式「预计超过 ~60KB」）
  - 旁证（同任务 Stage 3 侧 sibling）：.task_timeline.jsonl Stage 3 attempt 1 fail 809s（空返回——单次 LLM 响应生成整个大文件被 provider 截断；VP-2026-0021 同样 2026-09-11 apply 后仍首发失败，conductor 增量写入流程指令后才成功）——分段纪律的条件式表述在两个 agent 上同日失效
- repro: 复现条件——单次 Write 输出 >60KB 的 DESIGN.md（2026-09-15 本任务首设计 attempt，规则已合入状态下复现）
- toolchain_stamp: 会话层行为（无运行时依赖）；复现环境 2026-09-15
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（替换 Phase 4「超长文档分段落盘」blockquote 措辞）
    定位锚: "> **超长文档分段落盘**：DESIGN.md 预计超过 ~60KB / ~1200 行时，分段落盘后合并（每段先 Write 到 /tmp 再 cat 合并，或分节增量追加），不以单次整文件 Write 交付——单次巨型 Write 会因 JSON 体积截断失败（2026-09-07 attention expert 任务：1249 行 DESIGN 首写即截断，5 段拼接才成功，白耗一次 attempt 2960s）。"
    old 文本: |
      > **超长文档分段落盘**：DESIGN.md 预计超过 ~60KB / ~1200 行时，分段落盘后合并（每段先 Write 到 /tmp 再 cat 合并，或分节增量追加），不以单次整文件 Write 交付——单次巨型 Write 会因 JSON 体积截断失败（2026-09-07 attention expert 任务：1249 行 DESIGN 首写即截断，5 段拼接才成功，白耗一次 attempt 2960s）。
    new 文本: |
      > **超长文档分段落盘（>60KB 一律执行）**：DESIGN.md 预计超过 ~60KB / ~1200 行时一律分段落盘（bash heredoc 分段追加为可靠替代——单次 Write 截断后 heredoc 不受工具 JSON 体积限制），不以单次整文件 Write 交付，不以「预计不会超」的自行判断跳过——单次巨型 Write 会因 JSON 体积截断失败（2026-09-07 首证：1249 行首写即截断；**2026-09-15 复发**：本规则 2026-09-11 合入后 designer 仍先单次 Write 且首写即截断〔~95KB〕，heredoc 6 段恢复——条件式「预计超过」判断是失效点）。
    动机: 条件式建议在两个 agent（design/develop）上同日失效（Stage 3 侧 VP-2026-0021 同款复发）——阈值触发改为无条件 + 明示 heredoc 替代，把「是否分段」从 agent 自由裁量变为机械规则。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0068
- type: R
- title: design SKILL.md Phase 2 等价性机器验证补「判定口径」bullet 三条——ULP 阈值按恒等式域界第一性推导（勿拍小常数）/ 逐项隔离显式关闭其他量化开关 / 探针行与判定行分离
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md Stage 1 gate-修复重试章节 Value Point Proposals 后三行（M1 首跑 15 ulp 对拍定 8/16 阈值假 FAIL → 域界推导 |t|≤87 → 上界 ~120 ulp，阈值 128 后 viol 双口径 0；M14 沿用默认管线被 M12 的 ~2.6e-3 相对误差污染 viol 87% 假 FAIL → 隔离后 0；基线式失败演示只进 info 不参与判定）+ verify_equiv.py（13/13 EQUIV_PASS，双跑 diff 一致）
- repro: 复现条件——任一含 softcap/深尾域或多项量化开关的 verify_equiv 编写（任务内复现命令见 evidence provenance 项，torch CPU）
- toolchain_stamp: torch 2.7.1+cpu / 2026-09-15
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（Phase 2 第 5 项「对照对象」bullet 后追加一条）
    定位锚: "   - **对照对象**：容差内等价的采纳项，对照基准 = 原式（基线）在同 dtype 舍入路径下的输出；golden opmath 域问题（如 fp16 torch CPU 经 fp32 opmath）在角点用例中显式覆盖（lerp_tensor 实证：此类分歧设计期可机器拦截，Stage 3 才暴露损失 2485s）；"
    old 文本: |
      （即上述定位锚原文，保持不变）
    new 文本: |
      （定位锚原文保留）其后追加：
         - **判定口径**：① ULP 阈值按恒等式结构第一性推导，不拍小常数——exp 类恒等式的 ulp 差与指数入参域界成正比（|δt| ≤ c·u·|t|max → P 域 ulp ≤ c·|t|max·ln2；softcap 域 |t|≤87 → 上界 ~120 ulp，实测 15——拍 8/16 在深尾假 FAIL）；② 逐项隔离对照显式关闭其他量化/物化开关（被验证项之外全 off，否则他项误差污染本项参照门——M14 实测 viol 87% 假 FAIL → 隔离后 0）；③ 基线式失败模式演示（溢出/NaN/发散）只进 info 不进 PASS/FAIL（探针行与判定行分离）（2026-09-15 attention 两相位 verify_equiv 实证，13/13）。
    动机: 首跑两类假 FAIL（阈值拍脑袋 + 隔离缺失）都发生在「脚本已产出、口径不当」阶段——判定口径入 skill 可在首次编写时避开，省一轮 gate 往返（本任务 gate 重试 4 项中 2 项属 EQUIV-EXEC 类）。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0069
- type: R
- title: algorithm-research.md §6 信息源层「本仓 examples 同类实现」补全目录清单完成标准——结构先例检索以 grep -rl 族关键词显式列举命中清单并逐一归类为完成，不以命中第一个目录为完成
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md Stage 1 v3 章节（Research-flow 行：本轮以 `grep -rl "softmax\|attention" examples/` 显式列举 12+ 文件并逐一归类〔结构原型 / tuned 母本 / 机制先例 / 证据档案〕才闭环；仅按 AGENTS.md 关键词路由会漏 deepseek/mixcv/torch_tl_ops 侧先例）+ Transferable 首条
  - 前序根因（provenance）：examples/TileOPs/.../perf_opt/perf_feedback.md 修正附录（上版漏检两遍式先例 → 三轮调优 + [DESIGN_LIMIT] 才纠正）
- repro: 复现条件——任一迁移/重设计任务的结构先例检索（对照「已调研」断言 vs grep -rl 清单）
- toolchain_stamp: 流程规则（无运行时依赖）；证据环境 2026-09-15
- target_doc: .agents/skills/tilelang-op-design/references/algorithm-research.md
- delta: |
    动作: update（§6 第 3 层扩展）
    定位锚: "3. 本仓 `examples/` 同类实现所用算法；"
    old 文本: |
      3. 本仓 `examples/` 同类实现所用算法；
    new 文本: |
      3. 本仓 `examples/` 同类实现所用算法——**完成标准 = 全目录清单**：结构先例检索以 `grep -rl "<族关键词>" examples/` 显式列举命中文件并逐一归类（结构原型 / tuned 母本 / 机制先例 / 证据档案），不以命中第一个目录（如关键词路由的 flash_attention/）为完成（2026-09-15 实证：上版漏检 deepseek/mixcv 侧两遍式先例致 [DESIGN_LIMIT] 才纠正，本轮清单法 12+ 文件闭环；检索动作可复现、清单可检视，优于「已调研」断言）；
    动机: 调研漏检的代价是整条结构选型错误（上版三轮调优 + DESIGN_LIMIT 才暴露）；清单化完成标准使检视可机械化对账。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0070
- type: R
- title: algorithm-research.md §3 R4「流水可融合性」补行为维度注记——flag wait 是否阻塞发射流属结构属性须引实测档案佐证；候选间流量近似（<10%）不构成结构裁决依据，裁决权移交行为维度
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md Stage 1 v3 章节（Info-source 行 + Transferable 第 2 条：R3 流量口径下单遍链与两相位仅差 ~10%〔2.51 vs 2.78GB〕，实测性能差 2.31×，决定项是 R4 的发射流行为〔wait 与依赖对齐方式〕——R3/R4 分工「流量筛选 vs 行为定序」应显式提示）
- repro: 复现条件——任一候选间 fabric 流量差 <10% 的结构裁决（纸面流量对比无法区分的位形）
- toolchain_stamp: 流程规则；证据环境 2026-09-15（2.31× 锚点 = fa 域 perf_records 实测，tilelang 0.1.2+3a214cde 2026-09-09）
- target_doc: .agents/skills/tilelang-op-design/references/algorithm-research.md
- delta: |
    动作: update（§3 R4 检查清单第 5 项扩展）
    定位锚: "  5. **流水可融合性**：多阶段可否 `T.Pipelined` 双缓冲；Cube 段与 Vector 段可否 CV 融合并行；"
    old 文本: |
      5. **流水可融合性**：多阶段可否 `T.Pipelined` 双缓冲；Cube 段与 Vector 段可否 CV 融合并行；
    new 文本: |
      5. **流水可融合性**：多阶段可否 `T.Pipelined` 双缓冲；Cube 段与 Vector 段可否 CV 融合并行——flag wait 是否阻塞发射流属**结构属性**（非 tiling 可解），结论须引用 examples/ 实测档案（msprof pipe 利用率 / aic_scalar 占比）而非纸面推断；R3 类流量近似（候选间差 <10%）不构成结构裁决依据，此时裁决权移交本维度行为侧（2026-09-15 实证：单遍链 vs 两相位 fabric 流量差 ~10% 而实测差 2.31×，决定项是发射流行为）；
    动机: 上版误选单遍结构的根因之一即「流量相近时按流量拍板」——R3（流量筛选）与 R4（行为定序）分工显式化后，流量平局候选自动进入行为维度举证。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0071
- type: R
- title: T.sync_block_set.md 补「同一 event_id 多次 set 的聚合语义」条款——2-AIV 同 id 双生产（各 set 一次、Cube 单次 wait）的放行条件（双 AIV 均完成 vs 任一完成）无文档规定，设计只能以先例背书
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md Stage 2（2026-09-15）Skill Flow Issues 首行（T.sync_block_set.md §1 仅「同一 block 中的其他执行单元」一句、无多生产者语义条款；reviewer 只能以双生产先例〔E1-E7 29/29 + v11nt fa-tuned〕背书；设计侧列为 R-1 风险项 + 拆双 id 回退）
  - docs/Tilelang.language/同步管道操作/T.sync_block_set.md §1（现行文本无该条款）
- repro: 只读核对（无运行时依赖）——docs/Tilelang.language/同步管道操作/T.sync_block_set.md §1 无聚合语义条款
- toolchain_stamp: 仓库 docs 现状 2026-09-15；与 tilelang 版本无关（文档条款缺失）
- target_doc: docs/Tilelang.language/同步管道操作/T.sync_block_set.md
- delta: |
    动作: update（§1 简介段后追加语义说明段）
    定位锚: "简介：`tilelang.language.sync_block_set` 用于在当前 block 内设置一个同步标志（flag），通知同一 block 中的其他执行单元可以继续执行。"
    old 文本: |
      简介：`tilelang.language.sync_block_set` 用于在当前 block 内设置一个同步标志（flag），通知同一 block 中的其他执行单元可以继续执行。
    new 文本: |
      简介：`tilelang.language.sync_block_set` 用于在当前 block 内设置一个同步标志（flag），通知同一 block 中的其他执行单元可以继续执行。

      **同一 event_id 的多次 set 语义**：同一 block 内两个执行单元（如 2-AIV 半分类）对同一 event_id 各 set 一次、消费侧单次 wait 的聚合/覆盖语义（双生产者均完成才放行，还是任一完成即放行）本文档未规定——依赖该形态的设计应以仓内可运行先例佐证（examples/flash_attention/flash_attn_npuir.py 两相位、examples/multi_head_attention/ 双生产形态），并在设计文档中单列风险项与拆双 id 回退方案。
    动机: 2-AIV 半分类是 attention 族 Expert 设计的常见形态，聚合语义无文档时 reviewer 无法独立核验 flag 协议安全性（只能先例背书）——条款化（哪怕先标注「未规定 + 推荐单生产者」）使该风险项可在 docs 层闭环。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0072
- type: R
- title: T.gemm.md §1 签名模式标注与 Expert 生产用法口径不一致——L1/L0C 形态在 Scope("Cube") 内生产使用 T.gemm 名称（flash_attn_npuir / fp8_lighting_indexer 先例），与 [Expert mode] T.npuir_dot 标注的关系未澄清（两任务连续检视提出）
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md Stage 2（2026-09-15）Skill Flow Issues 第 2 行（T.gemm.md L8-9 两行签名标注 vs examples/flash_attention/flash_attn_npuir.py L102-109 Scope("Cube") 内 `T.gemm(l1, l1, l0c, initC, b_transpose, size)` 生产使用——reviewer 依先例判定可行但文档层无法自洽闭环）
  - 前序独立证据：同文件 Sep-07 Stage 2 round 2 Skill Flow Issues 第 3 行（examples/deepseek_v32/fp8_lighting_indexer.py L72-79 同款形态落差——两轮检视连续提出）
- repro: 只读核对（无运行时依赖）——T.gemm.md §1 两行签名标注与仓内 Expert 生产先例的形态落差（先例路径见 evidence）
- toolchain_stamp: 仓库 docs 与 examples 现状 2026-09-15；与工具链版本无关（文档-先例落差）
- target_doc: docs/Tilelang.language/线性代数操作/T.gemm.md
- delta: |
    动作: update（§1 签名区追加关系说明行）
    定位锚: "T.npuir_dot(src1, src2, dst, size=[], initC=False, a_transpose=False, b_transpose=False) # [Expert mode]"
    old 文本: |
      T.gemm(src1, src2, dst, size=[], initC=False, a_transpose=False, b_transpose=False) # [Developer mode]
      T.npuir_dot(src1, src2, dst, size=[], initC=False, a_transpose=False, b_transpose=False) # [Expert mode]
    new 文本: |
      T.gemm(src1, src2, dst, size=[], initC=False, a_transpose=False, b_transpose=False) # [Developer mode]
      T.npuir_dot(src1, src2, dst, size=[], initC=False, a_transpose=False, b_transpose=False) # [Expert mode]

      > 注：Expert 模式 `Scope("Cube")` 内以 L1/L0C 张量为操作数的形态（如 `T.gemm(l1, l1, l0c, initC, b_transpose, size)`）在仓内先例（examples/flash_attention/flash_attn_npuir.py、examples/deepseek_v32/fp8_lighting_indexer.py）中以 `T.gemm` 名称生产使用；上两行签名标注按 fragment（Developer）形态理解，L1/L0C 形态下 `T.gemm` 与 `T.npuir_dot` 的别名/等价关系以先例为准。
    动机: 同一形态落差在两轮独立检视中被连续提出（reviewer 每次都须以先例对抗文档标注）——一句关系说明消除文档-先例张力，使维度 1 API 存在性核对可从 docs 层闭环。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0073
- type: R
- title: develop SKILL.md Phase 2 补「母本移植逐项甄别」检查项——结构母本只提供 API 形态与程序序，布局假设/整除域特化/非 persistent 假设须按目标任务重推导；移植后首个回归须含 S_q > bm 的多头用例
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md Stage 3（2026-09-15）Value Point Proposals 第 4 行 + Transferable 第 2 条（v11_2phase _builder_2phase 为 fa 域〔B=1/H=1/整除/非 persistent〕特化实验形态：全局行 ws 布局/无 mask 链/无 TASKDONE/无钳位在 persistent+多头+causal+尾块域全部需适配——ws 槽全局行基误用即直接诱因之一：bx=0 任务「全局=本地」巧合掩盖、bx≥1 全错〔lse 恒 log2(S_kv) 指纹〕，row0 索引修复后 29/29）
- repro: 复现条件——任一从实验文件/先例 kernel 移植结构的开发任务（对照母本假设清单 vs 目标任务域）
- toolchain_stamp: tilelang 0.1.2+28783f454705cadab047805c1e0f5e054ba4b967 + CANN 8.5.0 + Ascend910B2C / 2026-09-15
- target_doc: .agents/skills/tilelang-op-develop/SKILL.md
- delta: |
    动作: update（Phase 2 清单追加第 4 项）
    定位锚: "3. 遵循项目根 AGENTS.md："不要凭记忆猜 API"、"从示例入手"——先 Glob `examples/` 同类实现参考。"
    old 文本: |
      3. 遵循项目根 AGENTS.md："不要凭记忆猜 API"、"从示例入手"——先 Glob `examples/` 同类实现参考。
    new 文本: |
      3. 遵循项目根 AGENTS.md："不要凭记忆猜 API"、"从示例入手"——先 Glob `examples/` 同类实现参考。
      4. **母本移植逐项甄别**：从实验文件/先例 kernel 移植结构时，「结构母本」只提供 API 形态与程序序——布局假设（全局行 vs per-core 本地行）、整除域特化（尾块/掩码/TASKDONE 缺失）、非 persistent 假设须按目标任务域重推导并显式列差异清单（2026-09-15 实证：v11_2phase 全局行 ws 布局直接移植致 per-core 槽 row0 索引 bug——bx=0 任务巧合掩盖、bx≥1 全错，lse 恒 log2(S_kv) 指纹锁定；移植后首个回归须含 S_q > bm 的多头用例）。
    动机: 母本的隐藏域假设（整除、全局行基、无尾块）只在目标域位形暴露——差异清单 + 定向回归用例把该类 bug 从「bx≥1 用例碰运气发现」变为「首个回归必现」。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0074
- type: R
- title: integrator 已知修复目录「config 参数（block/tile）不适配 NPU」行补 config-契约集成范式——kernel 带 config 替换语义时 wrapper default_config 是出厂契约决策：惰性旋钮路由到门禁验证过的逐字路径（1 行胶水修复恢复整个域可编译）
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/integration_log.md Round 2 §调试历史 attempt 1（分类命中该已知修复行 → 根因链 5 项 → 最小复现双路径 A fail / B ok → 修复 = wrapper default_config num_stages 1→2：(64,64,2) 在 E6 替换集外 → 逐字路径 bn_eff∈{64,144} = Stage 3 L1 门禁验证过的 traced 变体；num_stages 为结构惰性旋钮〔单槽两相位 stage 不变，DESIGN R-7〕语义零影响 → smoke 4/4 + full 6/6 + bench 10/10 复核）
  - .opencode/agents/tilelang-op-integrator.md L102（现行已知修复行——未展开带替换语义 kernel 的契约层选法）
- repro: 复现条件——任一 kernel 工厂带 config 替换/分派语义的集成期编译失败（wrapper 还原 (64,64,1) 时 `pytest "benchmarks/ops/bench_multi_head_attention.py::test_mha_fwd_bench[llama-3.1-8b-short-float16]" -v --tb=short` 复现 UB 溢出）
- toolchain_stamp: tilelang 0.1.2+28783f45 + CANN 8.5.0 + Ascend910B2C / 2026-09-15
- target_doc: .opencode/agents/tilelang-op-integrator.md
- delta: |
    动作: update（已知修复目录表「config 参数」行扩展）
    定位锚: "| config 参数（block/tile）不适配 NPU | wrapper `default_config` 调整（仅 wrapper 文件） |"
    old 文本: |
      | config 参数（block/tile）不适配 NPU | wrapper `default_config` 调整（仅 wrapper 文件） |
    new 文本: |
      | config 参数（block/tile）不适配 NPU | wrapper `default_config` 调整（仅 wrapper 文件）。**config-契约范式（kernel 带 config 替换语义时）**：wrapper `default_config` 是出厂契约决策而非 GPU 遗产直搬——替换集外的**逐字路径 + 惰性旋钮**（如 num_stages，结构 stage 不变的设计保留项）是集成层对齐「Stage 3 门禁验证过的 traced 变体」的合法工具：改惰性旋钮选中逐字路径即路由回门禁变体集（2026-09-15 实证：num_stages 1→2 使 (64,64,2) 落在 E6 替换集外 → bn_eff∈{64,144} 逐字路径，1 行胶水修复恢复 manifest 域可编译，pytest/bench 全绿复核）；修复后以工厂解析检查（`w.{func}.__module__`）确认激活源 |
    动机: 该修复形态已验证（1 attempt 闭环）但检索无门——现行行只说「调整 default_config」，未展开替换语义 kernel 的契约层选法（何时改旋钮、为何逐字路径合法）；范式入目录后同类集成失败可一次定位。Stage 3 门禁对 wrapper-default 派发路径的覆盖缺口另立 VP-2026-0065（防患侧），本条为修复侧。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0075
- type: R
- title: optimize SKILL.md 证伪协议第 4 条「标量占比触发换轴」前置编译警告排查——aiv_scalar 高占比与 flag 自旋同形，`will execute by scalar instruction` 警告是两者的区分器（vcmp int16 标量化占壁钟 45% 实证）
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T080600Z（第五轮 Stage 4）：examples/multi_head_attention/_gqa_prefill_fwd_kernel/perf_opt/opt_log.md#Phase-1（baseline aiv_scalar 94.5–96.8% 全 24 核均匀、两引擎 >90% 时间等待、aic_cube 0.7–1.4%——非引擎吞吐非带宽瓶颈）+ #判别实验（TILELANG_DUMP_IR + stderr 定位 `Op 'hivm.hir.vcmp' will execute by scalar instruction with low efficiency`，仅 causal trace 有、每变体 2 处）+ #Round-2（根因定案：vcmp int16 全形态标量化 236µs/链 @[32,256]，对角块聚合 ≈ 壁钟 45%；修复后 14.2x）
  - pattern-library/traps-compiler.md TRAP-expert-v-operands 第五轮补充段（「与 flag 自旋同形，须查编译警告区分」）+ attention.md PL-1.11（数据面档案）
  - 误诊路径对照：bottleneck-patterns BP_cross_engine_serial_chain（VP-2026-0034）触发信号即「busy 核 scalar 占比 30%+ 且归因为 flag 等待自旋」——两种根因（标量化陷阱 / flag 自旋）在 aiv_scalar 指标上同形，归因步骤缺区分器是误诊源头
- repro: `TILELANG_DUMP_IR=TRUE`（配新 `TILELANG_CACHE_DIR`，见 VP-2026-0062——缓存命中跳过编译也跳过 IR 打印）重编译目标 kernel，stderr grep `will execute by scalar instruction`；命中则热点段含被降级的 v-op（对照 msprof aiv_scalar 占比与「链宽 × 实测单链耗时」估算贡献占比）
- toolchain_stamp: tilelang 0.1.2+6797758（2026-09-15 07:45 HEAD）+ CANN 8.5.0 / Ascend910B2C
- target_doc: .agents/skills/tilelang-op-optimize/SKILL.md
- delta: |
    动作: update（证伪协议第 4 条原位扩展）
    定位锚: "4. **诊断信号强制触发换轴分析**：msprof 显示热点段标量执行占比 > 50% 时，强制评估向量化轴/布局重排候选（pattern-library/layout.md），不得只在原轴上微调参数。"
    old 文本: （即上述定位锚原文）
    new 文本: |
      4. **诊断信号强制触发换轴分析（先排查标量化，再评估换轴）**：msprof 显示热点段标量执行占比 > 50% 时，第一步先用 `TILELANG_DUMP_IR=TRUE`（配新 `TILELANG_CACHE_DIR`）重编译并 grep stderr 的 `will execute by scalar instruction` 警告——命中说明热点 v-op（如 vcmp int16）被降级为标量指令执行（traps-compiler.md TRAP-expert-v-operands：与 flag 等待自旋在 profile 上同形，第五轮实证占壁钟 ~45%），按对应 TRAP 条目绕法处理；未命中且归因为等待/自旋时再强制评估向量化轴/布局重排候选（pattern-library/layout.md），不得只在原轴上微调参数。
    动机: 第五轮 baseline aiv_scalar 94–97% 与良性 flag 自旋同形，若直接走「换轴」路径会漏掉 vcmp int16 全形态标量化这一占壁钟 45% 的根因（本轮靠 IR dump 警告 + 微探针才定案，修复即 14.2x）；编译警告 grep 是零成本区分器，前置可避免一类高代价误诊。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T080600Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0076
- type: R
- title: iteration-diagnosis.md 注意事项补微探针输出分片规范——`T.Kernel` 多 AIV 并发写同一 GM 输出缓冲存在竞态，探针结论（「链坏」类）可能是探针自身 artifact 而非被测机制缺陷
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T080600Z（第五轮 Stage 4）：examples/multi_head_attention/_gqa_prefill_fwd_kernel/perf_opt/probe_penalty_chain.py（P9 修正前后对照）+ opt_log.md#Round-2-实现要点（调试链：P9 探针因 T.Kernel(1) 双 AIV 写同一 GM 输出而竞态，产出误导性「链坏」结论，浪费一轮调试）
  - 同族先例（不同形态）：pattern-library/attention.md PL-1.9-blockwidth「探针教训」——缺 Cube→Vector flag 同步的数值崩坏是探针自身竞态而非 API 缺陷（跨引擎形态）；本条补多 AIV 写竞态形态，两形态同属「探针异常先怀疑探针自身」
- repro: 微探针 kernel 以 `T.Kernel` 启动 ≥2 AIV 且输出为单一 GM 张量时，对照按 subid 分片（`out[subid, ...]` 各写各区）或单 AIV 版本的输出稳定性（P9 实证：分片前结果不可复现地「坏」、分片后稳定）
- toolchain_stamp: tilelang 0.1.2+6797758（2026-09-15 07:45 HEAD）+ CANN 8.5.0 / Ascend910B2C
- target_doc: .agents/skills/tilelang-op-optimize/references/iteration-diagnosis.md
- delta: |
    动作: update（「## 注意事项」列表末尾追加一条）
    定位锚: "- 如果所有候选优化点都不够直接，先补充 profile 或检查 profile 口径，不要盲目扩大搜索空间。"
    old 文本: （即上述定位锚原文）
    new 文本: |
  - 如果所有候选优化点都不够直接，先补充 profile 或检查 profile 口径，不要盲目扩大搜索空间。
  - 微探针（判别实验 kernel）的输出缓冲按 AIV 分片（`out[subid, ...]` 各写各区）或显式单 AIV——`T.Kernel` 多 AIV 并发写同一 GM 输出区域存在竞态，会产出误导性的「机制坏」结论（2026-09-15 实证：双 AIV 竞态使向量链探针误报「链坏」，浪费一轮调试；同族形态见 pattern-library PL-1.9 探针教训——探针数值异常先怀疑探针自身同步/写冲突，再怀疑被测机制）。
    动机: 微探针是第五轮定案 vcmp 标量化的关键手段，但其自身的多 AIV 写竞态曾误导一轮调试——判别实验的可信性以探针正确性为前提，分片输出是零成本规范。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T080600Z 2026-09-15
- decided_by: -
- decided_note: -

## VP-2026-0077
- type: R
- title: optimize SKILL.md Phase 3 TileOPs 集成算子条目补「perf_opt 双镜像同步」规则——conductor 工作区与 tileops 集成目录为独立 inode，单侧修复时 pytest 门禁测旧拷贝（错误位级相同无法区分源与镜像）
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z：DESIGN.md §11.2 集成拓扑注意（「TileOPs wrapper import 的是 tileops/kernels/.../perf_opt/ 镜像拷贝（独立 inode），perf_opt 侧任何修改须双镜像同步，否则 pytest 门禁测的是旧拷贝」——conductor 收束补记）
  - 同任务 opt_log Round 10（r9 precision_fix：修复 conductor 工作区 perf_opt 后手动同步 TileOPs 集成镜像 `tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/_gqa_prefill_fwd_kernel.py`，wrapper 的 baseline/perf_opt 切换块不动；同步后 TileOPs pytest 6/6 passed）
  - 双 inode 实核（2026-09-16 蒸馏会话）：`examples/multi_head_attention/_gqa_prefill_fwd_kernel/perf_opt/_gqa_prefill_fwd_kernel.py` 与 `examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/_gqa_prefill_fwd_kernel.py` 两份 81708B 拷贝并存（mtime 06:22 / 06:26）
- repro: 复现条件——conductor 工作区与 TileOPs 集成目录存在两份独立 perf_opt 拷贝的 optimize/precision_fix 任务，单侧修改工作区 kernel 后直接跑 TileOPs pytest（门禁测的是未同步的集成镜像，错误位级相同无法区分源与镜像）
- toolchain_stamp: tilelang 0.1.2+a13585dc + CANN 8.5.0 + Ascend910B2C / 2026-09-16；TileOPs pyproject 环境
- target_doc: .agents/skills/tilelang-op-optimize/SKILL.md
- delta: |
    动作: update（Phase 3 第 3 条扩展）
    定位锚: "3. TileOPs 集成算子（算子目录为 `tileops/kernels/{family}/{op_slug}/{op_slug}_kernel/`）：`perf_opt/` 建在该目录下；wrapper 的 baseline/perf_opt 双 import 切换块由 conductor 在回归通过后翻转采纳（perf_opt 默认激活），本 skill 不修改 wrapper。若 tuned kernel 与基准 kernel 的默认参数不同（如 block_size），须在 `perf_opt/{op}.py` 中以模块级常量暴露 tuned 默认值，供 wrapper 切换块成对引用。"
    old 文本: （即上述定位锚原文）
    new 文本: |
      3. TileOPs 集成算子（算子目录为 `tileops/kernels/{family}/{op_slug}/{op_slug}_kernel/`）：`perf_opt/` 建在该目录下；wrapper 的 baseline/perf_opt 双 import 切换块由 conductor 在回归通过后翻转采纳（perf_opt 默认激活），本 skill 不修改 wrapper。若 tuned kernel 与基准 kernel 的默认参数不同（如 block_size），须在 `perf_opt/{op}.py` 中以模块级常量暴露 tuned 默认值，供 wrapper 切换块成对引用。**双镜像同步（conductor 工作区 + TileOPs 集成目录）**：任务工作区（`examples/{project}/{op}/perf_opt/`）与 TileOPs 集成目录（`tileops/.../perf_opt/`）是独立 inode 的两份拷贝——凡修改任务工作区 perf_opt kernel（含 precision_fix 会话），须同步拷贝到 TileOPs 集成镜像（wrapper 的 baseline/perf_opt 切换块不动）后再跑 TileOPs pytest/回归门禁；单侧修复时门禁测的是旧拷贝，且位级相同的错误使门禁无法区分源与镜像（2026-09-16 r9 precision_fix 实证：同步后 pytest 6/6）。
    动机: TileOPs wrapper import 的是集成目录镜像而非任务工作区文件；r9 precision_fix 若只修工作区，pytest 门禁会在旧拷贝上复现位级相同的 UB 溢出——既浪费修复验证又可能误判修复无效。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z 2026-09-16
- decided_by: -
- decided_note: -

## VP-2026-0078
- type: R
- title: statectl.py 状态迁移命令加并发互斥锁（flock）——双 conductor 会话并行驱动同一任务目录时 load→modify→write 交错，产出零产物 runtime fail 与交叉状态写入（4 次实证）
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z：.task_timeline.jsonl（6 次 fail 中 4 次 runtime 归因双会话竞争：03:38–06:15 期间另一窗口与本会话交错驱动——attempt 1「03:38 并行调度 #1 已 runtime 失败（819s 零产物）」记录于 .stage_state.json user_requirement；attempt 2 期间出现外来 set/snapshot 写入〔03:58:15〕；stage_retry_count 被推至 6 污染 E-2 指标）
  - .agents/tools/statectl.py 现行版本（Ctx.load→write 为无锁 read-modify-write；cmd_start 无活跃 attempt 互斥；atomic_write_json 只保证单次写原子性不保证序列化）
- repro: 复现条件——两个进程同时对同一 op 目录执行 statectl start/set/fail（本任务 2026-09-16 03:38–06:15 实录：interleaved set/snapshot + 819s 零产物 runtime fail ×1 + 竞争性 fail ×3）
- toolchain_stamp: statectl.py 现行版本；发生环境 2026-09-16（与 tilelang 工具链无关）
- target_doc: .agents/tools/statectl.py
- delta: |
    动作: update（main() 分发加互斥锁 + 新增 acquire_op_lock 辅助函数）
    定位锚: |
      def main(argv=None) -> int:
          parser = build_parser()
          args = parser.parse_args(argv)
          if getattr(args, "func", None) is None:
              parser.error("missing command")
          return args.func(args)
    old 文本: （即上述定位锚原文）
    new 文本: |
      MUTATING_ACTIONS = {"init", "start", "fail", "complete", "set", "snapshot", "repair"}


      def acquire_op_lock(op_dir: str):
          """Exclusive lock for state-mutating commands.

          Two conductor sessions driving the same op dir interleave
          load->modify->write cycles: silent state corruption, phantom
          runtime fails and cross-written snapshots (2026-09-16 gqa
          task: 4 competing runtime fails, first one 819s zero-artifact).
          flock converts that into a loud immediate E-BUSY.
          """
          import fcntl

          lock_path = os.path.join(op_dir, ".stage_state.json.lock")
          fh = open(lock_path, "w", encoding="utf-8")
          try:
              fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
          except OSError:
              try:
                  holder = open(lock_path, encoding="utf-8").read().strip() or "?"
              except OSError:
                  holder = "?"
              fh.close()
              emit(
                  {
                      "action": "busy",
                      "errors": [
                          err(
                              "E-BUSY",
                              f"另一 statectl 会话持有 {lock_path}（holder pid {holder}）"
                              "——同一任务目录禁止并行驱动；确认无活跃会话后可删除锁文件重试",
                          )
                      ],
                  },
                  1,
              )
              return None
          fh.write(f"{os.getpid()}\n")
          fh.flush()
          return fh


      def main(argv=None) -> int:
          parser = build_parser()
          args = parser.parse_args(argv)
          if getattr(args, "func", None) is None:
              parser.error("missing command")
          lock_fh = None
          if args.command in MUTATING_ACTIONS:
              lock_fh = acquire_op_lock(os.path.abspath(args.dir or os.getcwd()))
              if lock_fh is None:
                  return 1
          try:
              return args.func(args)
          finally:
              if lock_fh is not None:
                  lock_fh.close()  # releases flock
      （注：只读命令 show/verify/gate/timeline-summary 不加锁；migration 子树若同构需要可另行评估，本提案只覆盖主状态命令）
    动机: 并行双会话的每一次交错都产生真实成本（819s 零产物 attempt + 交叉写入使时间线不可信 + stage_retry_count 污染 E-2 北极星指标）；flock 是 OS 级机械守卫，比「conductor 自觉不并行」可靠——同任务目录的第二驱动者立即收到 E-BUSY 而非静默竞争。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z 2026-09-16
- decided_by: -
- decided_note: -

## VP-2026-0079
- type: R
- title: statectl.py cmd_fail 补「未闭合 start 前置校验」——重复 fail 记账（无中间 start 的第二个 fail）虚增 stage_retry_count，污染重试指标与 E-2 抽取
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z：.task_timeline.jsonl L14–15（06:15:32 fail 971s / 06:15:39 fail 978s——第二个 fail 前无任何 start，duration 均回溯到同一 start#5，stage_retry_count 5→6 虚增；conductor 归因记账侧重复调用）
  - .agents/tools/statectl.py 现行版本（cmd_fail 非 BLOCKED 路径无「本 stage 存在未闭合 attempt」校验；stage_attempt_duration 只回溯最近 start，第二个 fail 复用同一 start 的 duration）
- repro: 复现条件——对同一 stage 连续调用两次 statectl fail（无中间 start）：第二条 fail 照常追加时间线并递增 stage_retry_count（本任务 2026-09-16 06:15 实录，7 秒间隔双记账）
- toolchain_stamp: statectl.py 现行版本；发生环境 2026-09-16
- target_doc: .agents/tools/statectl.py
- delta: |
    动作: update（cmd_fail 非 BLOCKED 路径头部插入校验块）
    定位锚: |
      （cmd_fail 函数体内，E-TERMINAL 守卫块之后、design_revision 分支之前）
          if state["phase"] in TERMINAL_PHASES:
              errors.append(err("E-TERMINAL", f"phase={state['phase']} 已终态，禁止 fail"))
              return emit({"action": "fail", "stage": n, "errors": errors}, 1)
    old 文本: （即上述定位锚原文）
    new 文本: |
          if state["phase"] in TERMINAL_PHASES:
              errors.append(err("E-TERMINAL", f"phase={state['phase']} 已终态，禁止 fail"))
              return emit({"action": "fail", "stage": n, "errors": errors}, 1)

          # A fail must close an OPEN attempt: the most recent start/fail/
          # complete event for this stage has to be a start. Duplicate fail
          # invocations (2026-09-16: two fails 7s apart, no intervening
          # start) otherwise inflate stage_retry_count and pollute the
          # E-2 retry metrics.
          last_stage_action = None
          for ev in read_timeline_events(ctx.op_dir):
              if ev.get("stage") == n and ev.get("action") in (
                  "start", "fail", "complete"
              ):
                  last_stage_action = ev.get("action")
          if last_stage_action != "start":
              errors.append(
                  err(
                      "E-NO-OPEN-ATTEMPT",
                      f"Stage {n} 无未闭合的 start（最近事件为 {last_stage_action}）"
                      "——重复 fail 记账拒绝，不递增 stage_retry_count",
                  )
              )
              return emit({"action": "fail", "stage": n, "errors": errors}, 1)
    动机: fail 是「闭合一个 attempt」的语义事件；无 start 的第二个 fail 只能来自记账侧重复调用——照单全收会虚增重试计数（本任务 6→实为 5），并使 duration 回溯到已消费的 start 产出误导性 duration_s。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z 2026-09-16
- decided_by: -
- decided_note: -

## VP-2026-0082
- type: R
- title: optimize SKILL.md T-4 实验批处理 runner 补适用边界——结构重构类分支（指令流重排/流水深度/同步协议）默认手工 diff 串行，生成脚本化仅用于参数扫描类
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z：opt_log Skill Retrospective（第六轮）第 6 条（T-4 runner 未搭建的决策依据：分支数 8、依赖交互调试多，手工串行可控性更高；r8g 结构分支的生成脚本引入 3 次脚本 bug，叠加 2 次编译错 + 1 次精度失败 + 1 次 flaky 超时，消耗 ~40% 轮次时间）
- repro: 复现条件——以脚本生成结构重构类分支（指令流重排/多槽 buffer/flag 协议变更）且需多轮交互调试的调优轮（本任务 r8g 实证形态）
- toolchain_stamp: 会话层工作流（无运行时依赖）；发生环境 tilelang 0.1.2+a13585dc / 2026-09-16
- target_doc: .agents/skills/tilelang-op-optimize/SKILL.md
- delta: |
    动作: update（T-4 段末尾追加一句）
    定位锚: "脚本模板与适配说明见 [autotune.md](references/autotune.md)「实验批处理 runner」节。"
    old 文本: （即上述定位锚原文）
    new 文本: |
      脚本模板与适配说明见 [autotune.md](references/autotune.md)「实验批处理 runner」节。**适用边界**：runner 适用于参数扫描/配置类分支（同 kernel 改 config）；**结构重构类分支（指令流重排 / 流水深度 / 同步协议 / 多槽 buffer 变更）默认手工 diff 串行**——此类分支依赖完整 kernel 上下文且需交互调试，脚本生成的变体一旦携带 bug，其调试成本高于手工串行（2026-09-16 attention 第六轮 r8g 实证：生成脚本引入 3 次脚本 bug，叠加编译/精度/flaky 共消耗 ~40% 轮次时间）。
    动机: T-4 的收益模型（交互轮次下降 60%+）建立在「分支可批量执行」的前提上；结构分支不满足该前提，脚本化反而放大失败面——边界标注使 runner 用在参数类刀刃上。
- status: pending
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260916T033847Z 2026-09-16
- decided_by: -
- decided_note: -

## Decided（merged / rejected / expired / conflict 归档）

## VP-2026-0014
- type: P
- title: 块内 mask 的向量构造链模式：arrange strides → vsub → vcmp(int16 索引) → vand → vselect；−1e38 有限哨兵 + mask 后置 softcap + 两段式分界零 mask 开销块
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/DESIGN.md#§0.6-E3 / #§1.6.2-#1
  - examples/deepseek_v4/example_sparse_attn_kernel_highperf.py#L345-354（vcmp 标量 PrimExpr + vand 链）/ #L611（PIPE_V 内 T.arange strides）
  - developer 谱系对照：同函数 T.Parallel 谓词预填形态 aiv_scalar 88–89% 实测（opt_log §1，**谱系注明** tilelang 67db6f3 + CANN 26.0.rc1）
  - 第二证（不同任务，2026-09-15 蒸馏追加）：task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T080600Z 第五轮 Stage 4——opt_log.md Phase 1 判别链 + Round 2：**vcmp int16 全形态标量化实测**（236µs/链 @[32,256]，对角块聚合 ≈ 壁钟 45%，probe_vcmp_forms.py P6）——原 delta 链结构中 vcmp(int16) 主路径形态被证伪；核心方向（向量链替代谓词预填/规避标量化）二次实证：**算术惩罚掩码**（clamp+vmax/vmin 标量广播+vmul，band-free trace）修复后 14.2x；两段式分派升级为 trace 级（band-free 算术惩罚 / band-carrying 保留 vcmp/vselect 值替换语义——vcmp 标量化代价在该 trace 仅次要路径）
- repro: 任一含块内 mask 的 attention/causal/sliding-window 算子，以向量链替代谓词预填后对照 aiv_scalar 占比与 Task Duration（第五轮复现：`python perf_opt/bench.py perf_opt/_gqa_prefill_fwd_kernel.py --case 8b-long --use-default-config --msprof-loop 25` 外部 msprof op 同口径）
- toolchain_stamp: 第一证 tilelang dev root build 2026-09-07（HEAD 21586b5）+ CANN 8.5.0（developer 对照侧 67db6f3 + CANN 26.0.rc1）；第二证 tilelang 0.1.2+6797758（2026-09-15）+ CANN 8.5.0 / Ascend910B2C
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library.md（历史路径——按 queue 头部映射规则解析为 pattern-library/attention.md）
- delta: |
    （第二证修正后合入形态）原 add §1「向量 mask 构造链」小节 → 实际合入：主体（算术惩罚掩码 + trace 两段式分派 + NaN 免疫边界 + 终值）已由第五轮 optimizer 任务内回写 attention.md PL-1.11 + traps-compiler.md TRAP-expert-v-operands 补充段；evolver 补遗两点入 PL-1.11：「−1e38 哨兵与 softcap 顺序」bullet（e^{−1e38−m} 对有限 m 下溢精确 0 规避 −inf−(−inf) NaN 路径 + mask 后置 softcap 保 P(OOB)=0 不变式）。③ K_A 分界公式取整方向推导模板仍归 VP-2026-0018（pending，合入时并入）。
- status: merged
- confirmations: 2/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- confirmed_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T080600Z 2026-09-15（第五轮 Stage 4：vcmp 标量化证伪 + 算术惩罚掩码 14.2x——同主题不同任务的第二次独立实测，含链形态修正）
- decided_by: evolver
- decided_note: 两次独立证据来自不同任务（2026-09-07 expert 迁移 / 2026-09-15 第五轮调优），达 Tier 1 阈值 2/2。第二证对原提案是「方向确认 + 形态修正」关系：向量链替代谓词预填的核心主张被 14.2x 二次实证，但 vcmp(int16) 主路径形态被证伪（标量化陷阱）——合入时以修正后形态为准（PL-1.11 算术惩罚掩码 + band-carrying 保留 vselect），原 delta 的 vcmp 链仅适用于 band-carrying 次要 trace。本蒸馏周期执行合入。

---

## VP-2026-0059
- type: D
- title: 逐 buffer 手工 UB 预算 vs BiShengIR 基础分配差 ~26KB@192KB（auto-multi-buffer 关闭仍溢出）——预算表 ≠ 可编译证明，宽 config 上靶前须编译探针
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/integration_log.md Round 2 §Design-layer finding（DESIGN §4.5 预算 causal trace (bm=64,bn_eff=256,dim=128) = 178.2KB ≤ 192KB "✓"、§5.3 复述；实际 ~204.1KB 硬编译失败 `ub overflow, requires 1666048–1672192 bits while 1572864 bits available`）
  - 排除 multi-buffer 通胀：`bishengir-compile --enable-auto-multi-buffer=false` 仍溢出 → 基础分配差（疑未记载 extra/sync buffers，Stage 5 未进一步定位）；dim=64 smoke ~161.7KB 可过
  - 双路径复现（/tmp/opencode/s5_repro_mha_config.py，工厂直调 (4,512,32,128) causal fp16）：A `(64,64,1)` → E6 替换 (64,256) FAIL；B `(64,64,2)` → 逐字 bn_eff∈{64,144} OK（out/lse finite）
  - 第二证（不同任务，2026-09-15 蒸馏追加）：task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T080600Z 第五轮 Stage 4 Phase 1——probe_ub.py 编译探针 4 点标定（工具链 6797758，跨 commit 复现并精化为系数律）：(64,256) +9.6% / (80,256) +12.2% / (96,256) +12.0% / (128,256) +12.1% → **BishengIR UB 实际需求 ≈ 真实手工预算 × 1.10–1.12**；另实证 DESIGN §4.5 漏计 ub_cond2 8.2KB（「勿漏计小 buffer」）+ UB diet 绕法（N/D staging 合并 −24.6KB + rowmat 删除 −16.3KB → (64,256) 宽块解锁，解锁后掩码修复红利显形）
- repro: repro-missing（NPU 编译探针；两次任务内复现命令见各 evidence 末行——首证工厂直调双路径对照 / 第二证 probe_ub.py 变体扫描）
- toolchain_stamp: 首证 tilelang 0.1.2+28783f454705cadab047805c1e0f5e054ba4b967 / 第二证 tilelang 0.1.2+6797758（2026-09-15 07:45 HEAD），均 + CANN 8.5.0 / Ascend910B2C（跨 commit 复现）
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/traps-compiler.md
- delta: |
    update 条目 TRAP-UB-multibuffer-inflation（追加第四证/姊妹现象，与 20/26 B/elem 平台律并列）——已执行：正文追加「第四证/姊妹现象（基础分配差）」段（×1.10–1.12 4 点标定 + 预算表≠可编译证明 + 编译探针校准法 + UB diet 绕法实证 + 与 constants.md CONST-capacity-910B2C / CG-2026-0008 互链）+ front-matter origin_task/toolchain 机械更新（追加两任务）。
- status: merged
- confirmations: 2/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- confirmed_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T080600Z 2026-09-15（第五轮 Stage 4：跨 commit 复现 + 4 点标定精化为系数律 + UB diet 绕法）
- decided_by: evolver
- decided_note: 两次独立证据来自不同任务且跨工具链 commit（28783f45 / 6797758 均复现基础分配差），达 Tier 1 阈值 2/2。定量系数（×1.10–1.12）已由第五轮 optimizer 任务内回写 constants.md CONST-capacity-910B2C；本蒸馏周期合入 TRAP-UB-multibuffer-inflation 第四证段（陷阱视角与常数视角互链）。CG-2026-0008 同轮升级 recurring。

---

## VP-2026-0063
- type: D
- title: causal 多头域画像（从未结构化优化，headroom >10×）+ 两相位收益的 config 依赖性——窄 config 短 KV 1.30–1.32x 提速 / 长 KV 0.78x 回退 vs 单遍 expert；宽块解锁后反转（第五轮刷新）
- evidence:
  - task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z：examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/integration_log.md Round 2 §Bench（vs Sep-07 expert 基线同门对照列；profile_run_msprof_20260915_065835.log vs 20260907_173440.log）
  - 基线画像（provenance，Sep-09 采数）：perf_opt/perf_records.jsonl round 11 reg* 行——上一版 full 路径 (64,64,1) reg8bshort 3764.05µs（2.25 TFLOPS）/ reg8blong 13265.27µs（5.17 TFLOPS）vs fa4096 两相位 87.6 TFLOPS
  - 第二证（不同任务，2026-09-15 蒸馏追加）：task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T080600Z 第五轮 Stage 4——opt_log.md Final Summary：UB diet 解锁 (64,256) 后 config 依赖性画像刷新——长 KV 从 0.78x 回退**反转为 11.1x 领先**（8b-long 16962.44→1191.63µs = 14.23x vs baseline / 11.1x vs 单遍 13265µs）、短 KV 7.25–7.35x、全域几何 8.53x；原 delta 的预言「宽块收益需 mask 链缓冲精简（DESIGN §5.4 Stage 4 候选形态）」被完整实现（UB diet −40.9KB + 掩码修复后宽块红利显形 bn=256 vs 144 再 −40%）
- repro: repro-missing（端到端 msprof bench；任务内复现 = wrapper tuned 分派激活下 `msprof op --kernel-name=_gqa_prefill_fwd_main --launch-count=20 --warm-up=5 ... python perf_opt/bench.py ... --use-default-config`，raw 对账 profiles/{phase1,final}/）
- toolchain_stamp: 首证 tilelang 0.1.2+28783f454705cadab047805c1e0f5e054ba4b967 / 第二证 tilelang 0.1.2+6797758（2026-09-15），均 + CANN 8.5.0 / Ascend910B2C（基线侧 0.1.2+3a214cde 2026-09-09 采数——跨 commit 对比，倍率含版本差异成分）
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library/attention.md
- delta: |
    update 条目 PL-1.9-twophase（附加 causal 域数据点 bullet）——已执行：追加「causal 多头域画像与宽块解锁反转」bullet（headroom >10× 画像 + 窄 config 1.30–1.32x/0.78x config 依赖性 + 第五轮 UB diet 解锁后 11.1x 反转与几何 8.53x + bf16/fp16 平价，完整档案指向 PL-1.11）。
- status: merged
- confirmations: 2/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T025507Z 2026-09-15
- confirmed_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260915T080600Z 2026-09-15（第五轮 Stage 4：宽块解锁 + 0.78x→11.1x 反转的刷新证据）
- decided_by: evolver
- decided_note: 两次独立证据来自不同任务（v3 重生成集成期 / 第五轮调优期），达 Tier 1 阈值 2/2。第二证是证据刷新而非矛盾：0.78x 回退系窄 config 下的结论，第五轮在宽 config 下将其反转——同一 config 依赖性主张的完整闭环。本蒸馏周期合入 PL-1.9-twophase causal 域 bullet。

---

## VP-2026-0035
- type: P
- title: Stage 4 交付的工厂内 tuned 分派表模式：TUNED_DEFAULT_CONFIGS 仅替换 wrapper 默认 config（显式 config 一律尊重）+ tuned 目标 case 纳入内嵌测试套件作 blocking 精度层
- evidence:
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/_gqa_prefill_fwd_kernel.py（L100–110 TUNED_DEFAULT_CONFIGS 表 + L187–195 分派逻辑「仅 caller 传 wrapper 默认 (64,64,1) 且 shape 命中时替换」+ L1447–1490 run_fa_tuned blocking 门〔4 目标 case dispatch 路径 tier-1 全 0 flips〕）
  - 同目录 opt_log.md#Iteration-7（最终组装：工厂内 tuned 分派表〔S4-5 模式〕；caller 传 wrapper 默认时生效、显式 config 一律尊重——保 wrapper 契约）
  - 谱系先例（同 op 前任务、非独立计数证据）：examples/multi_head_attention/_prev_task_20260907_developer_optimize/perf_opt/opt_log.md（S4-5 工厂内 shape 分派调优 config）
  - 第二证（不同任务，2026-09-10 蒸馏追加）：examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/perf_opt/_ada_layer_norm_kernel.py（TUNED_DEFAULT_BLOCK_M per-shape 表 + select_row_config(M,N) 同形态返回值 + dtype 预算回退）+ 同目录 opt_log.md#Iteration-4（「TUNED_DEFAULT_BLOCK_M 显式逐 shape 记录而非规则化」——泛化假设否定后分派表为正解）+ #Iteration-6（U 曲线闭合验证表条目鲁棒性）+ #Final-Summary（wrapper 切换块引用建议）
- repro: repro-missing（config 级交付形态模式，无单一代码 diff；两次实现的分派表结构镜像于各自 perf_opt 交付文件〔provenance，允许失效〕）
- toolchain_stamp: 第一证 tilelang 0.1.2+3a214cde / CANN 8.5.0 / 2026-09-08；第二证 tilelang 0.1.2+a83118285a / CANN 8.5.0 / 2026-09-10
- target_doc: .agents/skills/tilelang-op-optimize/references/pattern-library.md（历史路径——按 queue 头部映射规则解析为 pattern-library/attention.md §1.9）
- delta: |
    add §1.9 附注 bullet（合入时紧随「工厂级多 @T.prim_func 变体分派」bullet 之后）：
    「工厂内 tuned 分派表（config 级，S4-5 模式）」：wrapper 契约保持的分派语义——caller 传 wrapper 默认 config 且 shape 命中 TUNED_DEFAULT_CONFIGS 时替换为调优 config；caller 传任何显式非默认 config 一律尊重原值（不静默覆盖用户意图）。tuned 分派覆盖的目标 case 须纳入内嵌测试套件作 blocking 精度层（fa-tuned 层：dispatch 路径 tier-1 翻转数对照基线），防止分派表与全量套件漂移。
- status: merged
- confirmations: 2/2
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260908T005751Z 2026-09-08
- confirmed_by: task ada_layer_norm-_ada_layer_norm_kernel-20260910T145715Z 2026-09-10（norm 族独立第二证：TUNED_DEFAULT_BLOCK_M + 非单调 bm 甜点 + 邻域闭合验证 + wrapper default_config 集成差异）
- decided_by: evolver
- decided_note: 两次独立证据来自不同任务不同族（attention 2026-09-08 / ada_layer_norm 2026-09-10），达 Tier 1 阈值 2/2，本蒸馏周期合入 pattern-library/attention.md §1.9（附注 bullet 追加于「工厂级多 @T.prim_func 变体分派」之后，含第二证要点：非单调 bm 逐 shape 记录 / 邻域闭合验证 / wrapper default_config 集成差异——合入文本见该文件）。

---

## VP-2026-0044
- type: P
- title: row-reduction 族 NPU 迁移基准形态（三件套）：全 N 驻留 UB + 按 M 行块分核（persistent num_kernels = min(ceil(M/bm), aicore×2)）+ 尾块 T.min + src/dst 双显式 slice + 垃圾行丢弃；N 超驻留上限才 N-tile 流式
- evidence:
  - 首证（logsumexp，2026-08）：examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_single/_logsumexp_kernel_single.py（L64-74 核实：T.Kernel(ceildiv(M, block_m)) grid 形态 + alloc_shared((block_m, N)) 全 N 驻留 + real_m = T.min(block_m, M − pid_m·block_m) 尾块 + 双显式 slice；DESIGN.md 分层测试计划）
  - 第二证（ada_layer_norm，2026-09-10，persistent 形态）：examples/ada_layer_norm/_ada_layer_norm_kernel/DESIGN.md §5.5/§0.6（persistent min(ceil(M/bm), 48) 分核 + T.serial grid-stride + 48 = 24 AICore × 2 实查）+ _ada_layer_norm_kernel.py（尾块 T.min + 双显式 slice + 垃圾行丢弃）+ Stage 4 调优 2.14x 全组几何平均（persistent 形态下 loads-first 结构收益最大化）
  - examples/ada_layer_norm/_ada_layer_norm_kernel/RETROSPECTIVE.md（Stage 1 Transferable Lessons 首行：「row-reduction 族 NPU 迁移三件套已两次验证（logsumexp + 本次 ada_layer_norm）」）
- repro: 复现条件——任一 row-reduction 族（norm/规约，M 行 × N 列归约）迁移设计；驻留上限推算依赖 TRAP-UB-multibuffer-inflation 平台值（fp16 约 N>11520 @20 B/elem）
- toolchain_stamp: 首证 2026-08（详见 logsumexp DESIGN 版本戳）；第二证 tilelang 0.1.2+a83118285a + Ascend910B2C + CANN 8.5.0 / 2026-09-10
- target_doc: .agents/skills/tilelang-op-design/references/algorithm-candidates.md
- delta: |
    update ALG-layernorm / ALG-rmsnorm / ALG-reduction 三行（R4 亲和要点 + kb_links + known_impl）：
    ALG-layernorm R4 追加 row-reduction 迁移基准形态（两任务实测：logsumexp grid 形态 / ada_layer_norm persistent 形态）——全 N 驻留 UB（流量 4MN 最小）+ 按 M 行块分核（persistent num_kernels = min(ceil(M/bm), aicore×2)，纯 Vector 核数翻倍实查）+ 尾块 T.min + src/dst 双显式 slice + 垃圾行丢弃；N 超驻留上限才 N-tile 流式（+25% 流量，ada 设计期估算）；kb_links 补 TRAP-vrsqrt-plain-precision / PL-1.10-loads-first-decoupling / CASE-norm-adalayern-stage4，known_impl 补 ada perf_opt 指针。ALG-rmsnorm kb_links 补 TRAP-vrsqrt（rsqrt 直用同样受影响）+ R4 引用 ALG-layernorm。ALG-reduction kb_links 补 CASE-reduction-logsumexp / CASE-norm-adalayern-stage4 + R4 引用。
- status: merged
- confirmations: 2/2
- created_by: task ada_layer_norm-_ada_layer_norm_kernel-20260910T132324Z 2026-09-10
- confirmed_by: task logsumexp-migration-2026-08（首证证据经工件追溯，创建时一并附上——VP-2026-0001 先例形态；grid 非 persistent 形态的差异已在条目正文中如实标注）
- decided_by: evolver
- decided_note: 创建即达 Tier 1 阈值 2/2（logsumexp 2026-08 + ada_layer_norm 2026-09-10 两个不同任务），本蒸馏周期合入 algorithm-candidates.md 三行。R4 迁移基准形态的正文中「persistent 强化」仅 ada 单证（logsumexp 为 grid 形态），已在文本中标注两形态来源。

### 2026-09-10 批量实施记录（conductor-improvement-report，用户批准直接实施）

> 依据 `/home/tilelang/l00970450/upload/tilelang-mlir-ascend/conductor-improvement-report.md` 的实施路线图，用户批准 P0+P1+部分 P2 批量直接实施（本节为治理留痕，非 VP 提案）：
> - **T-1/VP-2026-0042（扩展版）**：perf-feedback.md 参照锚定门槛 + optimize SKILL.md [DESIGN_LIMIT] 证据门槛 + plateau 参照结构 diff 检查 + cases.md 参考实现集；
> - **T-2/VP-2026-0010**：iteration-diagnosis.md 交错 A/B 多 run 协议前置 + `.agents/tools/ab_test.py` 工具化；
> - **K-1/K-2/K-5**：pattern-library 拆分为目录（INDEX + 主题文件 + repro/，条目 front-matter，字节预算）；cases.md 参考实现集与 CG 互链；
> - **ED-A/B/C/D/F**：证据三件套语义重定义（distillation-rules/queue-schema/merge-policy）+ repro 规范与首批 5 个转正（实跑 PASS）+ developer/optimizer repro 沉淀步骤 + repro_runner.py/kb_lint.py；
> - **E-1/E-5/E-6（K-3）**：kb_search.py 统一检索层 + conductor 预注入 + kb_stale_check.py；
> - **E-3/E-4**：审批简报节奏 + 强证据快速通道 + apply 预验证（merge-policy §6/§8）；
> - **D-1/D-2/D-3/D-4/D-5**：verify_equiv.py 等价性机器验证 + gate S1-EQUIV-EXEC + hardware-cost-model.md + constants.md + 设计期探针选项 + algorithm-candidates.md + design_calc_check.py；
> - **T-4/T-6/E-2(度量自动化)**：run_experiments 模板 + Final Summary 跨 dispatch 汇总 + stats 指标机械抽取。
> 明确未实施：VP-2026-0019/0020（用户决定暂不应用）；P2 的 T-3 BO/T-5 谱系导航/E-2 opbench/D-4 网络检索增强。

## VP-2026-0042
- type: R
- title: optimize SKILL.md [DESIGN_LIMIT] 判定补「结构不可达」证据门槛——仓库内同族先例检索 + 同口径实测 + morph 对照排除局部地板误判 + 被推翻时修正附录机制（append-only）
- evidence:
  - examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/perf_feedback.md 修正附录（第二轮「对 H=1 大 S 硬目标不可达、fa4096 结构性地板 ≈125–131µs > 100µs」被第三轮同 API 两相位结构推翻：四目标全达成 12.94/18.11/30.88/98.05µs；两处根因 = ① Stage 1 算法调研漏掉仓库内两遍式 S/P 物化先例 examples/flash_attention/flash_attn_npuir.py（同 API 可达结构）② round-10 否决两遍式的「S 需全任务驻留 → flag 预算超 15 上限」系推理错误——全局 [S,S] ws + n-block 下标 flag（nk≤16）即可成立，参考与本轮终版为直接反例）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/DESIGN.md#§11.3（[DESIGN_LIMIT] 结论修正段：局部地板 vs 绝对地板界定——L1 端口 ~150GB/s / L0C 128KB / store_fixpipe 仅 GM 三硬上限存活，被推翻的仅是「上限内不存在达标数据流」的推断）
  - 同目录 perf_opt/opt_log.md#Final-Summary-第三轮（[DESIGN_LIMIT] 修正回填记录）
  - 交叉引用：queue VP-2026-0025（弃选举证边界——设计期同族规则）、VP-2026-0041（morph ladder——最小增量对照的定位方法）
- repro: 复现条件——任一触发 [DESIGN_LIMIT] 双门槛（设计层归因 + >2x 结构性估计）且拟断言「当前工具链/硬件上限内不可达」的调优终局（实证：第二轮断言后第三轮被同 API 结构推翻）
- toolchain_stamp: tilelang 0.1.2+3a214cde7aa4f54fc4a103f0f324a43341122d68 / CANN 8.5.0 (msprof) / Ascend910B2C / 2026-09-09
- target_doc: .agents/skills/tilelang-op-optimize/SKILL.md
- delta: |
    动作: update（Phase 3 第 4 条末尾追加证据门槛段）
    定位锚: "4. **设计层天花板判定（[DESIGN_LIMIT]，可选产出）**：读取 [perf-feedback.md](../_shared/standards/perf-feedback.md)，逐条核对触发条件——① 性能天花板由算法/设计层决定（非 tiling/参数可解，归因到 DESIGN.md 具体假设）；② 结构性加速估计 > 2x 或实测与设计假设直接矛盾。**两项同时满足** → 按其 §2 固定 schema 产出 `perf_opt/perf_feedback.md`（实测章节必须为 msprof op 口径，反馈结论含建议路由），返回时附 `[DESIGN_LIMIT]` 信号；任一不满足 → **禁止产出**该文件（参数级不足留在迭代内，不触发逆向反馈）。"
    old 文本: （即上述定位锚原文）
    new 文本: |
      4. **设计层天花板判定（[DESIGN_LIMIT]，可选产出）**：读取 [perf-feedback.md](../_shared/standards/perf-feedback.md)，逐条核对触发条件——① 性能天花板由算法/设计层决定（非 tiling/参数可解，归因到 DESIGN.md 具体假设）；② 结构性加速估计 > 2x 或实测与设计假设直接矛盾。**两项同时满足** → 按其 §2 固定 schema 产出 `perf_opt/perf_feedback.md`（实测章节必须为 msprof op 口径，反馈结论含建议路由），返回时附 `[DESIGN_LIMIT]` 信号；任一不满足 → **禁止产出**该文件（参数级不足留在迭代内，不触发逆向反馈）。**「当前工具链/硬件上限内不可达」类断言的证据门槛**（2026-09-09 attention expert 三轮闭环实证：第二轮 [DESIGN_LIMIT] 断言 fa4096 结构性地板 ≈125–131µs > 100µs 目标，第三轮两相位重构同 API 达标 98.05µs 推翻）：① 断言前强制检索仓库内同族先例（`examples/`/`testing/` 是否已有同 API 可达结构），找到先例时同口径实测 + morph 式最小增量对照（iteration-diagnosis.md「参考实现对照」）排除「冻结设计族局部地板」误判——实测外推的地板只对实测过的结构族有效（前序实证证据边界，见 negative-claim-evidence.md 第 6 条 / queue VP-2026-0025）；② 否决备选结构的理由须为实测或文档条款，纯推理否决（如「S 需全任务驻留 → flag 预算超限」）标注为未实证假设；③ 已产出的 [DESIGN_LIMIT] 被后续实测推翻时不删改原文（append-only），以「修正附录」回填 perf_feedback.md（结论限定于冻结设计族 + 修正证据表 + 存活硬上限清单 + 根因记录——本轮已验证形态）。
    动机: 结构性负结论驱动 conductor「设计层重做/等待编译器补齐」逆向路由，误判代价是整轮设计方向（第二轮误判使达标结构延后一轮才被实施，参考实现就在仓库内而两轮算法调研均未纳入）；先例检索 + 同口径实测是最低成本拦截位。
- status: merged
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260909T071018Z 2026-09-09
- decided_by: human
- decided_note: 2026-09-10 用户批准随 conductor-improvement-report T-1 批量实施（扩展版：perf-feedback.md §1 参照锚定门槛 + §2 schema 参照锚定章节 + optimize SKILL.md Phase 3 第 4 条证据门槛 + plateau 参照结构 diff 检查 + cases.md 参考实现集；gate S4-PERF-FEEDBACK-ANCHOR 同步）

---

## VP-2026-0010
- type: R
- title: optimize iteration-diagnosis Step 6 补小 kernel 交错 A/B 多 run 测量协议前置规则（<5% 差异即启用，勿等采纳后复核）
- evidence:
  - examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/opt_log.md#iteration-2（v2_op2 首测 +4.4% 复测翻转为 tie，耗一轮）
  - 同文件 #iteration-4（fp32 采纳项单轮 -5.2% 系双态膨胀，A/B/A/B 4 run 合并中位 -3.1%）
  - examples/TileOPs/tileops/kernels/elementwise/mish/mish_kernel/perf_opt/opt_log.md（首证：run 间 ±2–3.5% 环境双态层）
- repro: 16m fp32 final(bs8192) vs baseline(bs2048) 交替 A/B/A/B 各 2 run：B {158.75, 158.59} vs A {163.65, 162.27}，配对次序稳定
- toolchain_stamp: tilelang 0.1.2+ed787bb（2026-09-07 build）/ Ascend910B2C / CANN 8.5.0
- target_doc: .agents/skills/tilelang-op-optimize/references/iteration-diagnosis.md
- delta: |
    动作: update（评估规则列表新增一条）
    定位锚: "- 只有通过必测 dispatch 非回退检查的候选 winner，才能更新为全局 current best。"
    new 文本: 交错 A/B/A/B 多 run 协议前置规则（<5% 差异即启用）。
    动机: lerp_tensor 两次踩坑（v2_op2 与 fp32 采纳项）各消耗一轮复测才校正归因；协议前置可省两轮实验。
- status: merged
- confirmations: -/-
- created_by: task lerp_tensor-_make_lerp_tensor_kernel-20260907T025419Z 2026-09-07
- decided_by: human
- decided_note: 2026-09-10 用户批准随 conductor-improvement-report T-2 批量实施（iteration-diagnosis.md Step 6 评估规则 + optimize SKILL.md Phase 2 第 10 步/核心防呆 + `.agents/tools/ab_test.py` 工具化——交错多 run + 合并中位差 + 配对符号检验）

---


## VP-2026-0001
- type: P
- title: run 级双态（同 kernel 跨 run ±3–5%）需交错 A/B 多 run 协议裁决小差异——BP_run_state_bimodality
- evidence:
  - examples/TileOPs/tileops/kernels/elementwise/mish/mish_kernel/perf_opt/opt_log.md（首证 2026-08：run 间 ±2–3.5% 环境双态层、per-process 快/慢双态、<3% 结论须多 run）
  - examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/opt_log.md#iteration-4（第二证 2026-09-07：1M fp16 跨 run 7.30–7.66；fp32 采纳项单轮 -5.2% 双态膨胀 → A/B/A/B 4 run 合并中位 -3.1%）
  - examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/profiles/final/ab_*
- repro: 16m fp32：final(bs8192) vs baseline(bs2048) 交替各 2 run，msprof op --launch-count=15 取 median 后合并比较
- toolchain_stamp: tilelang 0.1.2+ed787bb（2026-09-07 build）/ Ascend910B2C / CANN 8.5.0（lerp）；mish 见其 opt_log 版本戳
- target_doc: .agents/skills/tilelang-op-optimize/references/bottleneck-patterns.md
- delta: |
    add 新 BP 条目 BP_run_state_bimodality（目录行 + 正文节，追加于 BP_parameter_uncertain 之后）：触发信号（同配置跨 run ±3–5%、<20us 或贴地板 workload 差异落 3% 门槛附近、复测翻转）/ 反证（launch-count 消失→BP_measurement_resolution_limited；大 kernel 单 run 已稳）/ 动作（首次 <5% 差异即前置交错 A/B/A/B：≥3 独立 run、交替次序、合并中位；3% 采纳判断必须合并中位 + 配对次序稳定）/ 验证（run 次序与中位记录、配对方向跨 run 一致）。
- status: merged
- confirmations: 2/2
- created_by: task lerp_tensor-_make_lerp_tensor_kernel-20260907T025419Z 2026-09-07
- confirmed_by: task mish optimize 2026-08（证据经 mish opt_log 档案追溯，创建时一并附上）
- decided_by: evolver
- decided_note: 两次独立证据来自不同任务（mish 2026-08 / lerp_tensor 2026-09-07），创建即达 Tier 1 阈值 2/2，本蒸馏周期合入 bottleneck-patterns.md（目录行 + BP_parameter_uncertain 节后追加正文）。

---

## VP-2026-0021
- type: R
- title: Stage 3 大工件会话超限空返回的防再犯——DESIGN.md 分段读取 + kernel 分段落盘纪律写入 developer agent 定义（两次空返回各耗 1579s/930s）
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/.task_timeline.jsonl（Stage 3 attempt 1 fail 1579s / attempt 2 fail 930s，verdict=runtime；attempt 3 complete 7701s）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-3（标题行：「前两次会话超限空返回，本次完成」）
  - 第三次成功的「分段读取 + 增量写入」纪律账户来自 conductor 终态钩子输入（复盘工件未展开该纪律细节——provenance 如实标注）
- repro: 复现条件——一次性整读 175KB/1249 行 DESIGN.md + 单次巨型 Write 949 行 kernel 文件的子 Agent 会话（2026-09-07 本任务前两次 attempt）
- toolchain_stamp: opencode Subagent 会话限制（2026-09-07 两次实证）；tilelang 工具链无关
- target_doc: .opencode/agents/tilelang-op-developer.md
- delta: |
    动作: update（first_impl 模式首行 bullet 扩展）
    定位锚: "- Read `DESIGN.md` + `REVIEW.md`。"
    old 文本: |
  - Read `DESIGN.md` + `REVIEW.md`。
    new 文本: |
  - Read `DESIGN.md` + `REVIEW.md`。**大工件分段读写纪律**（会话超限空返回防再犯，2026-09-07 attention expert 任务两次空返回实证〔1579s/930s 白耗〕）：DESIGN.md 超过 ~1200 行 / 150KB 时按章节分段 Read（先目录 + §0.5/§0.6 决策 + §3 伪代码，再按需下钻），不一次性整读；`{op}.py` 生成用分段落盘（逐段写 /tmp 后 cat 合并，或先写骨架再增量追加），不用单次巨型 Write——超限空返回表现为无输出直接失败（runtime verdict），与代码错误同型、无法从 stderr 区分。
    动机: 本任务 Stage 3 前两次 attempt 以完全相同的输入空返回（会话超限），第三次仅改变读写纪律即成功（7701s 完成 949 行 kernel + 29 用例全过）——每次空返回消耗一次 attempt 预算与 ~15–26 分钟墙钟。
- status: merged
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: human
- decided_note: 用户批准（E-3 审批简报，2026-09-11，组⑤「大工件/会话纪律」）；mode=apply 落盘——合入锚 .opencode/agents/tilelang-op-developer.md `first_impl` 模式首行 bullet（原行保留 + 大工件分段读写纪律文本行内扩展），E-4 预验证 PASS（副本应用锚点唯一命中 + standards_check 控制组/编辑组 delta 零新增 failure；repro 为会话级复现条件非可执行命令，回归 SKIPPED）。

---

## VP-2026-0022
- type: R
- title: 超长 DESIGN.md（>60KB）单次 Write 因 JSON 体积截断——tilelang-op-design SKILL.md Phase 4 补分段落盘规则
- evidence:
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/RETROSPECTIVE.md#Stage-1（Skill Flow Issues 首行：~1200 行 DESIGN.md 单次 Write 截断失败，5 段 /tmp 拼接后 cat 合并才成功；DESIGN.md 1235/1249 行为证）
  - examples/multi_head_attention/_gqa_prefill_fwd_kernel/.task_timeline.jsonl（Stage 1 attempt 1 fail 2960s——含该 Write 截断的恢复成本）
- repro: 复现条件——单次 Write 输出 >60KB 的 DESIGN.md（2026-09-07 本任务首设计 attempt）
- toolchain_stamp: opencode Write 工具 JSON 体积限制（2026-09-07 实证）；tilelang 工具链无关
- target_doc: .agents/skills/tilelang-op-design/SKILL.md
- delta: |
    动作: update（Phase 4 首行后追加一段）
    定位锚: "基于 [templates/design-template.md](templates/design-template.md) 模板，填充所有章节："
    old 文本: |
      基于 [templates/design-template.md](templates/design-template.md) 模板，填充所有章节：
    new 文本: |
      基于 [templates/design-template.md](templates/design-template.md) 模板，填充所有章节：

      > **超长文档分段落盘**：DESIGN.md 预计超过 ~60KB / ~1200 行时，分段落盘后合并（每段先 Write 到 /tmp 再 cat 合并，或分节增量追加），不以单次整文件 Write 交付——单次巨型 Write 会因 JSON 体积截断失败（2026-09-07 attention expert 任务：1249 行 DESIGN 首写即截断，5 段拼接才成功，白耗一次 attempt 2960s）。
  动机: 截断失败发生在长任务收尾（写盘即交付前），恢复成本一次完整 attempt；attention/mixed 类算子的 DESIGN 普遍超此规模。
- status: merged
- confirmations: -/-
- created_by: task multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z 2026-09-07
- decided_by: human
- decided_note: 用户批准（E-3 审批简报，2026-09-11，组⑤「大工件/会话纪律」）；mode=apply 落盘——合入锚 .agents/skills/tilelang-op-design/SKILL.md Phase 4 首行「基于 templates/design-template.md 模板，填充所有章节：」后追加「超长文档分段落盘」blockquote（锚行原文保留），E-4 预验证 PASS（同 VP-2026-0021 流程）。
