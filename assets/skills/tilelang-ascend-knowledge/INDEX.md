# TileLang Ascend 知识库总路由表（INDEX）

> **使用方式**：根据下方路由表定位 **1–3 张** 最相关卡片读取，无需全量加载。新增任何卡片后必须在此登记，并运行 `scripts/validate_kb.py` 校验。

## 一、第 1 层：通用基础与铁律（`foundations/`）

| 主题 | 何时读取 | 卡片 |
| --- | --- | --- |
| 编写范式选型与存储预算 | 移植初版、写新算子、或高级接口/AutoSchedule 受阻决定是否切换显式 intrinsic 时 | [writing-paradigms.md](foundations/writing-paradigms.md) |
| Roofline 定性与仿真证伪节奏 | 立项摸底、初版性能远落后、或多方向调优排序前 | [roofline-sim-workflow.md](foundations/roofline-sim-workflow.md) |
| 物理防呆评测铁律 | 编写或审查精度、缓存失效、性能计时、工具链版本与回执解析脚本时 | [eval-invariants.md](foundations/eval-invariants.md) |

## 二、第 2 层：按瓶颈症状路由微架构模式与反例（`patterns/`）

### 正向已验证模式

| 观测症状 / 目标 | 核心机理 | 适用架构 | 卡片 |
| --- | --- | --- | --- |
| Attention/融合算子 `aiv_vec` 打满呈 Vector-bound，Cube 等 Softmax | `S_trans` 列向 FP32 Softmax + 硬件 N-split `FIXPIPE` + `L1→L0A` 转置 | Ascend950 | [softmax-col-fp32.md](patterns/softmax-col-fp32.md) |
| AutoSchedule 流水未重叠、Cube 小空泡多、MTE2/FIXP 阻塞主循环 | L1 多槽环 + 双缓冲 L0A/L0B/L0C + 拆分就绪事件 + 尾段提前写回 + 跨 head 展平 | Ascend950 | [cv-explicit-ring-pipeline.md](patterns/cv-explicit-ring-pipeline.md) |
| `QK` 早已完成但 `score FIX` 延迟上千周期（队首阻塞）或长序列 `507014` 同步超时 | 前置 `score FIX` + 优先旧 `V` 供数解除队首阻塞；屏障与 FIX 事件号独占 | Ascend950 | [cube-fixpipe-hol-reordering.md](patterns/cube-fixpipe-hol-reordering.md) |
| 向量阶段碎指令多、`SimtVF` 占空比过高、多段小计算访存浪费 | 全 256B `SimdVF` 算子融合，消除 `SimtVF` 与 64-bit 除法 | Ascend950 | [simd-vf-256b-fusion.md](patterns/simd-vf-256b-fusion.md) |
| 长序列/大工作集 L2 读命中骤降、MTE2 成为慢核长尾主因 | 超阈值自适应 L2 分段 + KV-head 优先工作排布 | Ascend950 | [l2-adaptive-sectioning.md](patterns/l2-adaptive-sectioning.md) |
| 长上下文 Decode 并行不足、两段 launch 间隙、grid 尾波/全核屏障死锁、PV DFU 偏低 | Split-KV 单核 MIX 融合屏障 + `grid` 严对物理核数 + 非 2 的幂精确打包 + `N128 PV`/双槽/UnitFlag/预取展开 | Ascend950 | [decode-split-kv-fused-barrier.md](patterns/decode-split-kv-fused-barrier.md) |
| Prefix/RoPE 外部 `cat` 流量大、滑窗 mask 冗余、内部块 UB 写带宽高、逐 tile 碎读 sink | 双 `L1` Ring 零拷贝融合、滑窗 KV 块上下界拆分、无 mask 内部块免 UB 写回、AIV 入口缓存不变量 | Ascend950 | [attention-prefix-window-fusion.md](patterns/attention-prefix-window-fusion.md) |
| 多 GEMM 反向因 `dK/dV` 常驻占满 L0C 被迫用 `64×128` 半宽 MMAD（`aic_mac ~0.94` 但 MFU 卡在 ~82%） | `dk_c/dv_c` 保持常驻 + `QK/dP/dQ` 复用 2 槽 `tmp_c[2]`（256 KiB）+ `dQ` 前置藏于 3 个满 MMAD 后 + L0 奇偶交替 | Ascend950 | [l0c-pingpong-dq-first-128x128-bwd.md](patterns/l0c-pingpong-dq-first-128x128-bwd.md) |

### 已证伪反例（避免重复踩坑）

| 看似诱人的方向 | 证伪结论与核心物理原因 | 适用架构 | 卡片 |
| --- | --- | --- | --- |
| 2D Zig-Zag 任务分配降低核间负载/缓存压力 | 共享梯度（如 `dQ`）同步原子写冲突数量级暴增（~11×） | Ascend950 | [2d-zigzag-atomic-contention.md](patterns/falsified/2d-zigzag-atomic-contention.md) |
| 高 GQA 比反向改用 Q-outer 遍历消除 `dQ` 原子写 | GM 读流量（~3×）与 `dK/dV` 原子写流量（~4×）成倍放大 | Ascend950 | [q-outer-gqa-traffic-blowup.md](patterns/falsified/q-outer-gqa-traffic-blowup.md) |
| Decode 合并多 Query（如 M96）复用 KV 减半逻辑搬运字节 | split 与 workspace 翻倍、片上压力增大与逐行 mask 依赖导致整核显著回退（+6%~+58%） | Ascend950 | [decode-m96-multi-query-kv-reuse.md](patterns/falsified/decode-m96-multi-query-kv-reuse.md) |
| 单遍 Softmax 减局部 UB 读写或仅拆首轮 K/RoPE `split_ready` | 未移动稳态关键路径（如队首阻塞），整核跨度持平甚至小幅倒退 | Ascend950 | [local-ub-or-early-sync-micro-tweaks.md](patterns/falsified/local-ub-or-early-sync-micro-tweaks.md) |
| 释放 L0C 常驻 `dK/dV`、每 Q-tile 用 `FIXPIPE` 原子写回 GM 以扩 `128×128` | `FIXPIPE -> GM` 原子写暴涨 3× 反超 Cube（仿真慢 2.04×，实卡 `FIXP` 亦超基线整步） | Ascend950 | [non-resident-dkdv-atomic-writeback.md](patterns/falsified/non-resident-dkdv-atomic-writeback.md) |

## 三、第 3 层：按算子族路由 Playbook（`families/`）

| 算子族 | 覆盖范围与当前成熟度 | Playbook |
| --- | --- | --- |
| **Attention** | Prefill Fwd / Prefill Bwd / GQA Sink & 滑窗 & Prefix 融合 / FP8 MLA Decode（`active`） | [attention/PLAYBOOK.md](families/attention/PLAYBOOK.md) |
| **GEMM** | BMM、量化 W4A8 多 Pass 解包与 FP8 Cube 链路（`active`，对照仓内 `example/gemm/`） | [gemm/PLAYBOOK.md](families/gemm/PLAYBOOK.md) |
| **Convolution** | 分组 Conv1d、Tiled DMA Unfold 与向量抽样（`active`，对照仓内 `example/convolution/`） | [conv/PLAYBOOK.md](families/conv/PLAYBOOK.md) |
| **Reduction & Norm** | 独立 Softmax / RMSNorm / LayerNorm（`draft` 骨架，待后续实战沉淀） | [reduction-norm/PLAYBOOK.md](families/reduction-norm/PLAYBOOK.md) |

## 四、新增经验模板（`templates/`）

- 新增正向微架构模式：[pattern-card.md](templates/pattern-card.md)
- 新增已证伪反例：[falsified-card.md](templates/falsified-card.md)
- 新增算子族 Playbook：[family-playbook.md](templates/family-playbook.md)
