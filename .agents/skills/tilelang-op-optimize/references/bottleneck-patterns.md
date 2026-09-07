# NPU TileLang 瓶颈模式参考

## 用途

本文件是 `tilelang-op-optimize` 的可迭代瓶颈模式参考。每轮诊断时，先按 [iteration-diagnosis.md](iteration-diagnosis.md) 建立本轮诊断上下文，再从本文件匹配可能的瓶颈模式。

这些模式不是固定分类树。一个 profile 可以同时匹配多个模式；每个匹配到的模式都可以生成一个或多个候选优化点，并在同一轮分别创建实验分支验证。

## 目录

- `BP_launch_overhead`：launch/scheduling overhead 高。
- `BP_per_block_fixed_overhead`：per-block 固定开销地板。
- `BP_task_concurrency_sweet_spot`：`num_cores` 任务并发存在 U 形甜点区。
- `BP_measurement_resolution_limited`：msprof 测量噪声内候选无法可靠排序；这是测量质量模式，不代表 kernel 本身瓶颈。
- `BP_memory_bandwidth_or_mte`：GM/MTE 搬运效率不足。
- `BP_ub_pressure`：UB/L0 资源压力限制性能。
- `BP_ub_traffic_floor`：UB 流量或 vector pass 数成为地板。
- `BP_pipeline_overlap`：搬运和计算未重叠。
- `BP_block_balance_or_tail`：block 负载不均或 tail 开销。
- `BP_compute_granularity`：计算粒度或硬件管线使用不足。
- `BP_redundant_work_or_roundtrip`：重复计算或多余 GM 往返。
- `BP_parameter_uncertain`：结构稳定但参数不确定。
- `BP_run_state_bimodality`：同配置跨 run 快/慢双态，小差异裁决需交错多 run 协议。

新增模式时保持同一结构：

```markdown
## BP_xxx：一句话描述

触发信号：

- ...

常见反证/不确定点：

- ...

推荐动作：

- ...

验证指标：

- ...
```

跨模式提醒：

- `T.serial` 不是单一策略。多块迭代主要对应 `BP_launch_overhead`，内层分块主要对应 `BP_pipeline_overlap`。
- 一种 `T.serial` 用法失败，不代表另一种无效。特别是内层分块无法带来 pipeline overlap 时，仍要独立评估多块迭代是否能降低 launch/scheduling overhead。
- 一个配置失败只记为 `config_no_gain`；只有覆盖关键配置或有明确机制证据时，才能把整类方向记为 `family_no_gain`。
- 某个模式在当前环境无收益，只能记录为当前 workload / TileLang-NPUIR / CANN / Developer 模式下的实测结论，不要泛化为永久无效。
- `num_cores` 搜索的 `Task Duration` 曲线可能出现平区；平区内不要用单次 msprof 结果排序。
- `BP_measurement_resolution_limited` 是测量质量模式，用来阻止错误排序；它不说明 kernel 的真实性能瓶颈。

---

## BP_launch_overhead：launch/scheduling overhead 高

触发信号：

- `Block Dim` 远超 AI Core Count，尤其达到物理核数的数十倍或更高。
- 单 block 工作量小，单 block 元素数 / 字节数不足以摊薄调度开销。
- `Task Duration` 明显高于理论 memory-bound 时间。
- 增大 block size 曾带来性能提升，但继续增大受 UB/L0 或 tile 合法性限制。

常见反证/不确定点：

- `Block Dim` 不高，或者单 block 工作量已经足够大。
- Task Duration 接近理论 memory-bound 时间。

推荐动作：

- 将 `T.Kernel(ceildiv(N, block_size))` 改成 `T.Kernel(num_cores)`。
- 在每个物理 block 内用 `for i in T.serial(iters_per_core)` 迭代多个 logical block。
- 将 `block_size` 与 `num_cores` 纳入搜索；`num_cores` 不能只测单点。
- 至少覆盖 AI Core Count 附近、整数倍 AI Core Count、任务并发甜点区、低 imbalance 候选和当前 flat-grid 端点。

`T.serial` 多块迭代提醒：

| 用法 | 典型结构 | 目标 |
|---|---|---|
| 多块迭代 | `T.Kernel(num_cores)` + `for i in T.serial(iters_per_core)` | 降低实际 `Block Dim`，削减 launch/scheduling overhead |

验证指标：

- `Block Dim` 降低。
- `Task Duration` 降低。
- `num_cores` 曲线被记录，不能用单个坏点否定整条路线。
- 精度不退化。

## BP_per_block_fixed_overhead：per-block 固定开销地板

触发信号：

- `Block Dim` 远超 AI Core Count，且单 block 数据量很小。
- 改变 block_size 后，单 block 墙钟或每轮执行时间近似恒定。
- flat-grid 端点性能差，但减少 `num_cores` 后可能出现中间最优点。

常见反证/不确定点：

- 单 block 工作量已经足够大，减少 `Block Dim` 反而降低并行度。

推荐动作：

- 创建 `T.serial` 多块迭代或等价 grid 聚合分支。
- 扫 `num_cores` 曲线，而不是只测 AI Core Count 一个点。
- 记录 `msprof Task Duration` 与 `num_cores` 曲线；平区内先用更大 `launch-count` 复测，复测后仍打平时用 imbalance 和资源占用决胜。
- 若大 tile 触发 UB 溢出，先减少 buffer 或 tile，再重测该结构，不要直接否定整类路线。

验证指标：

- `Task Duration` 在甜点区内降低。
- `Block Dim` 降低，`num_cores` 曲线存在合理最优区间。

## BP_task_concurrency_sweet_spot：`num_cores` 任务并发存在 U 形甜点区

触发信号：

- 使用 `T.Kernel(num_cores)` + `T.serial(iters_per_core)`，`num_logical` 固定但 `num_cores` 可调。
- 低 `num_cores` 下每个任务循环次数过多，`Task Duration` 偏高，疑似 DMA/VEC 交替空转。
- 中间 `num_cores` 区间明显更快，继续增大后 `Task Duration` 又变差。
- 甜点区内 `Task Duration` 差异很小，但整除性 / imbalance 仍能区分候选。

常见反证/不确定点：

- 当前 kernel 计算量足够大，任务并发不足不是主要瓶颈。
- `Task Duration` 复测噪声大到无法确认 U 形曲线，需要先匹配 `BP_measurement_resolution_limited`。
- 最优驻留任务数依赖硬件、CANN、TileLang-NPUIR 和 workload，不能把某个 `num_cores` 固化成通用常数。

推荐动作：

- 计算 `num_logical=ceildiv(N, block_size)`、`iters_per_task=ceildiv(num_logical, num_cores)`、`min/max iters` 和 `imbalance`。
- 扫 `num_cores` 时覆盖低并发端、`4x/6x/8x AI Core Count` 任务并发锚点、低 imbalance 候选和 flat-grid 端点；只有硬件上下文或 profile 明确给出其它调度核口径时，才使用该口径修正。
- 平区内先用更大 `launch-count` 复测 `Task Duration`；复测后仍打平时用 imbalance、整除性和资源占用决胜。
- 优先选择 `Task Duration` 更低、`imbalance` 更小且 UB/L0 合法的候选。

验证指标：

- 低并发端、中间甜点区、过多任务端的曲线形态被记录。
- winner 的 `Task Duration` 在甜点区内更优。
- `min/max iters` 或整除性解释平区内部差异。

## BP_measurement_resolution_limited：msprof 测量噪声内候选无法可靠排序

触发信号：

- `Task Duration` 曲线大面积平坦，候选差异小于噪声阈值，默认按 10% 判断。
- 同一配置跨独立运行漂移明显，或 session 早期出现系统性膨胀。
- 参考实现、历史 best 的 `Task Duration` 与当前量级明显不一致。

常见反证/不确定点：

- 多次独立运行（更大 `launch-count`）后 `Task Duration` 差异稳定超过噪声阈值。
- 用户指定的唯一主指标就是 `Task Duration`，且差异显著。

推荐动作：

- 采集前先做预热，再记录有效数据。
- 候选差异小于噪声阈值时，用更大 `launch-count` 复测，并做至少 2 次独立 `msprof op` 运行。
- 复测后仍打平的平区候选，用 imbalance、整除性、资源余量和代码复杂度决胜，不要用单次 msprof 结果排序。
- 记录每次采集的 launch 数与复测次数。

验证指标：

- `opt_log.md` 记录 launch 数、独立运行次数和复测结论。
- winner 选择说明 `Task Duration` 是决胜指标，平区内已用结构性证据决胜。
- 作废的噪声运行不进入最终排序。

## BP_memory_bandwidth_or_mte：GM/MTE 搬运效率不足

触发信号：

- GM 读写量大，计算利用率不高。
- MTE2/MTE3 利用率低或搬运形状不连续。
- 单次 copy 粒度小、对齐差、连续性差。
- 多次 GM round-trip 或中间结果落 GM 后又读回。

常见反证/不确定点：

- GM 读写已接近理论带宽，继续优化搬运收益有限。
- 主要时间差距来自 launch overhead 或 UB 资源限制。

推荐动作：

- 调整 tile shape，使 GM->UB 和 UB->GM 连续且粒度足够。
- 合并写回或融合后处理，减少 GM round-trip。
- 减少重复 GM 读，增加片上复用。

验证指标：

- `Task Duration` 降低。
- MTE/Memory 指标改善。
- GM 读写次数或字节数减少。

## BP_ub_pressure：UB/L0 资源压力限制性能

触发信号：

- tile 过大，buffer 多，接近 UB/L0 容量上限。
- 增加 pipeline stage、buffer 或 block_size 后编译失败、资源冲突上升或性能下降。
- auto-multi-buffer 或手动多缓冲导致额外 buffer 占用。

常见反证/不确定点：

- 缩小 tile 会显著增加 `Block Dim`，引入 launch overhead。
- 当前瓶颈主要不是片上资源，而是 GM 或调度开销。

推荐动作：

- 做 UB/L0 buffer 生命周期分析，标出每个 buffer 的定义点、最后一次使用点、dtype/shape 和后续用途。
- 在 last-use 后复用片上 buffer，不只检查临时中间 buffer，也要检查输入 staging、cast buffer 和输出 staging。
- 优先检查输入 staging buffer 是否能在输入已 cast/搬运到后续 buffer 后，复用为 downcast 或输出 staging buffer。
- 缩小 tile 或减少 stage。
- 删除不必要的片上 buffer。
- 区分 UB staging 复用与 GM 输入/输出 alias；默认不要假设 GM 输入可以原地写成输出。

验证指标：

- 编译通过且无 UB/L0 资源异常。
- `Task Duration` 降低。
- 精度不退化。

## BP_ub_traffic_floor：UB 流量或 vector pass 数成为地板

触发信号：

- tile 已接近 UB/L0 容量上限，继续增大 block_size 收益变小或回退。
- `Task Duration` 随 block_size 下降后进入平台期，主要受 vector pass 数和 UB 字节流量约束。
- 估算的 `pass_count * tile_bytes` 能解释大 tile 下的耗时。

常见反证/不确定点：

- 参考实现或同类实现达到明显更高带宽，当前“地板”可能只是结构低效。
- PipeUtilization 的绝对时间与墙钟矛盾时，只能作方向参考。

推荐动作：

- 优先减少 vector pass、vcast、片上 buffer 数和重复 UB 读写。
- 尝试 in-place 链、公共子表达式复用、dtype dispatch 拆分或表达式等价变形。
- 检查输入 staging -> 计算 dtype buffer -> 输出 staging 的生命周期是否能合并，例如输入 UB 在 last-use 后复用为 downcast/store buffer。
- 用模型定位 block_size 交点；超过交点后不要继续只扩大 tile。

验证指标：

- pass 数、UB 字节数或 buffer 数下降。
- `Task Duration` 改善。
- 精度不退化，必测 dispatch 不回退。

## BP_pipeline_overlap：搬运和计算未重叠

触发信号：

- 代码按 tile 串行执行 GM 搬入、片上计算、GM 写回。
- MTE 和 compute 管线交替忙，串行感明显。
- timeline 或 PipeUtilization 显示阶段性空洞。

常见反证/不确定点：

- 当前 TileLang-NPUIR / CANN / Developer 模式对该结构的 pipeline overlap 无收益。
- 增加 stage 后 UB 压力变大，反而变慢。
- `T.serial` 循环体或 pipeline 触发 auto-multi-buffer，导致 UB 预算膨胀并改变可行 tile。
- 当前 kernel 没有足够计算量覆盖下一轮搬运。

推荐动作：

- 小步尝试 `T.Pipelined`、double-buffer 或 multi-buffer。
- 若单 block 内需要分 tile，可用 `T.Kernel(ceildiv(N, block_size))` + `for t in T.serial(num_tiles)` 尝试 pipeline / multi-buffer overlap。
- 估算 auto-multi-buffer 后的 UB 占用；若溢出，先做 buffer 缩减或 tile 调整后再测该结构。
- 若无收益或编译失败，记录证据并回到其它优化点，不要反复尝试。

实验解释：

- `T.Pipelined` 或 auto-multi-buffer 组合出现编译失败或无收益时，只记录为当前 TileLang-NPUIR / CANN / Developer 模式下的实测结论。
- 不要因为内层分块无收益，就推断 `T.serial` 多块迭代也无效；两者解决的问题不同。
- 不要用一个 auto-multi-buffer 膨胀后的坏配置否定 `T.Pipelined` 或 `T.serial` 整类方向。

验证指标：

- `Task Duration` 降低。
- MTE/compute overlap 指标改善。
- UB/L0 资源未恶化。

## BP_block_balance_or_tail：block 负载不均或 tail 开销

触发信号：

- shape 不能整除 tile，尾块分支明显。
- 部分 block 空转或极小。
- 不同 dispatch/workload 的耗时差异明显。
- `Block Dim` 远小于 AI Core Count，或单 block 工作量过大。

常见反证/不确定点：

- 尾块占比很小，不足以解释主要 latency 差距。
- 增加 split 会引入过高 launch overhead。

推荐动作：

- 调整 block/tile shape。
- 优化 tail 处理。
- 增加 split 或合并 logical block，视 `Block Dim` 方向而定。

验证指标：

- `Task Duration` 降低。
- 空 block / tail block 比例下降。
- `Block Dim` 更合理。

## BP_compute_granularity：计算粒度或硬件管线使用不足

触发信号：

- Cube/Vector 利用率低。
- 单 block 计算粒度太小。
- GEMM/cube 类 kernel 的 tile 不匹配硬件。
- 大量小 vector 操作碎片化。

常见反证/不确定点：

- 利用率低是因为数据供应不足、launch overhead 或资源限制导致，而不是计算本身。

推荐动作：

- 调整 `block_M / block_N / block_K / K_L1`。
- 调整 vector tile。
- 合并小循环或增加每个 block 的有效计算量。

验证指标：

- Cube/Vector 利用率改善。
- `Task Duration` 降低。
- GM/MTE 或 UB 压力没有明显恶化。

## BP_redundant_work_or_roundtrip：重复计算或多余 GM 往返

触发信号：

- 多个小 kernel 通过 GM 传递中间结果。
- 中间结果写回 GM 后又读回。
- 多个 reduce 或多 pass 遍历同一批数据。
- 循环内重复计算地址、scale、mask、index 或公共表达式。

常见反证/不确定点：

- 重复工作占比不高，不足以解释主要耗时。
- 融合后可能增加 UB 压力，需要单独验证。

推荐动作：

- epilogue fusion。
- reduce 合并 / single-pass。
- 公共子表达式复用。
- streaming / online 递推公式。

验证指标：

- GM 读写次数减少。
- vector/scalar 操作数减少。
- `Task Duration` 降低。

## BP_parameter_uncertain：结构稳定但参数不确定

触发信号：

- 已经确定结构方向，但 `block_size / num_cores / tile / stage` 多个配置都合理。
- 不同 dtype 或 dispatch path 的最优参数可能不同。
- 手动推理无法可靠选择 winner。

常见反证/不确定点：

- 当前主要瓶颈仍是结构问题，直接 autotune 只会扩大搜索空间。
- 搜索空间里大量配置明显违反 UB/L0/Block Dim 约束。

推荐动作：

- 按 [autotune.md](autotune.md) 定义小而合法的搜索空间。
- winner 必须单独跑 `msprof op` 复测，不能只用 autotune latency。

验证指标：

- best config 在本轮主指标（`msprof op Task Duration`）下最低；平区内记录 imbalance/整除性决胜依据。
- 正确性通过。

## BP_run_state_bimodality：同配置跨 run 快/慢双态，小差异裁决需交错多 run 协议

> 溯源：mish optimize 2026-08（首证，`examples/TileOPs/tileops/kernels/elementwise/mish/mish_kernel/perf_opt/opt_log.md`）+ lerp_tensor optimize 2026-09-07（第二证，`examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/opt_log.md#iteration-4`；origin_task: lerp_tensor-_make_lerp_tensor_kernel-20260907T025419Z；tilelang 0.1.2+ed787bb / Ascend910B2C / CANN 8.5.0）。

触发信号：

- 同一 kernel、同一配置，跨独立 `msprof op` run 的 Task Duration 差 ±3–5%（lerp_tensor 1M fp16：7.30–7.66us；mish：run 间 ±2–3.5%），且差异与配置无关（疑似 per-process 快/慢态，HBM 物理页/DDR 通道交织状态）。
- 小 kernel（Task Duration <20us）或逼近带宽/搬运地板的 workload，候选间差异落在噪声阈值（3%）附近。
- 复测翻转：首轮判回退（+4.4%）复测变 tie；或单轮增益（-5.2%）复测收缩（-3.1%）。

常见反证/不确定点：

- 差异随 launch-count 增大而消失 → 采集噪声，匹配 `BP_measurement_resolution_limited`。
- 大 kernel（≫20us）且远离带宽地板时，单 run median-of-15 通常已稳定。

推荐动作：

- 首次出现 <5% 量级候选差异即前置启用交错 A/B/A/B 协议：≥3 次独立 `msprof op` run、候选与基线交替次序、合并中位数比较——勿等采纳后复核。
- 过 3% 门槛的采纳判断必须基于合并中位；配对次序稳定（同一方向跨 run 成立）才判定真实差异。

验证指标：

- opt_log 记录各 run 的次序与中位数。
- A/B 配对结论次序一致（如两次 A 均慢于两次 B）才判定差异真实。
