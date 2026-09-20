# Autotune 引用索引

## 用途

当 Phase 2 的候选优化点包含 `autotune 参数搜索` 时，读取本文件。

本文件不复制 TileLang Ascend autotune 的完整 API 和示例，避免 `tilelang-mlir-ascend` 更新后两边内容不同步。具体用法、参数、示例和实现细节，以 `tilelang-mlir-ascend` 源库为准。

## 先读这些源文件

源码根目录：

```text
tilelang-mlir-ascend/
```

优先读取官方/源码侧文档：

1. `docs/Autotune使用指南.md`
2. `docs/developer/EnvironmentVariables.md`

需要确认实现细节时读取：

1. `tilelang/autotuner/tuner.py`
2. `tilelang/autotuner/param.py`
3. `tilelang/profiler/bench.py`

需要参考可运行示例时读取：

1. `testing/autotune/example_gemm_autotune.py`
2. `testing/autotune/example_gemm_carver.py`
3. `testing/autotune/example_elementwise_add_autotune.py`
4. `testing/autotune/example_elementwise_add_carver.py`
5. `testing/autotune/gemv_autotune.py`
6. `testing/autotune/gemv_carver.py`
7. `testing/autotune/reduce_min_autotune.py`
8. `testing/autotune/reduce_min_carver.py`
9. `testing/autotune/flash_attn_npuir_autotune.py`

需要参考结构性优化模式时，优先搜索并阅读：

1. `examples/elementwise/example_elementwise_exp2.py`（`T.serial` 多块迭代模式）
2. `examples/elementwise/vec_add_2d_multi_buffer.py`（multi-buffer / 多缓冲模式）
3. `examples/elementwise/` 下其它 activation、unary、binary 示例
4. 历史优化目录或同类算子的 `perf_opt/opt_log.md`

如果历史目录存在 `Optimize.md`，只能把它当作快速摘要；完整过程仍以 `perf_opt/opt_log.md` 为准。

## 在本 skill 中的使用规则

Autotune 属于 Phase 2 的一个实验分支，通常放在结构性优化之后执行。结构性分支通过正确性且方向有效后，应先对该结构做一轮 coarse autotune 或等价手动粗搜；autotune winner 不是最终结论，之后还要检查 top-k 配置和 winner 邻域，再进入精测和 winner 收束。

执行前必须满足：

- baseline kernel 已经通过正确性测试。
- 已经通过 `msprof op` 得到目标 kernel 的 baseline Task Duration。
- kernel 代码中已经暴露显式可调参数，例如 block/tile shape、K tile、pipeline stage、buffer 数量或 vector tile shape。
- 如果结构性优化已经识别为 `T.serial` 多块迭代，应显式暴露 `block_size`、`num_cores`、`iters_per_core` 或等价参数。
- 若搜索 `num_cores`，搜索空间必须覆盖 AI Core Count 附近、整数倍 AI Core Count、任务并发甜点区、低 imbalance 候选和 flat-grid 端点，不能只测单点。
- 搜索空间小而合法，优先过滤明显超过 UB/L0/Block Dim/shape 约束的配置。

结构后 coarse search：

- `T.serial` 多块迭代分支至少粗搜 `block_size × num_cores`；`iters_per_core` 若由 `ceildiv(num_logical_blocks, num_cores)` 推导，不必单独暴露。
- `T.Pipelined` / multi-buffer 分支至少粗搜 tile size、stage 或 buffer 数中实际暴露的关键参数。
- coarse search 只服务当前结构，不混入其它结构性改动；若 coarse winner 周围存在甜点区，再扩展邻近参数精搜。
- coarse search 结果只用于筛候选，最终 winner 必须单独用 `msprof op` 复测。

Autotune 后邻域精搜：

- 记录 autotune winner 和 top-k 配置；若 autotune API 不能直接导出 top-k，则至少保留所有已测 config 的 latency / correctness / failure reason。
- 围绕 winner 做邻域搜索，例如 `block_size` 上下相邻合法档位、`num_cores` 上下相邻并发档位、低 imbalance 候选和资源余量更好的候选。
- 若 autotune latency 与 `msprof Task Duration` 排序冲突，以独立复测的 `msprof op` 决胜。
- 邻域精搜可以手动或脚本执行，但必须记录 search space、结果和为什么停止扩展。

执行时必须记录：

- search space。
- best config。
- top-k config 或所有已测 config 的摘要。
- autotune latency。
- 正确性结果。
- 编译失败或 correctness 失败的配置数量。
- 每个候选的 `Block Dim`、理论 logical block 数、UB/L0 buffer 占用估算。

执行后必须：

- 用 best config 生成当前 autotune 实验分支，例如 `perf_opt/{op}_round{iter}_{opt_id}.py`。
- 若执行了邻域精搜，用精搜 winner 生成最终候选分支，并记录它与 autotune winner 的关系。
- 跑 L0 精度回归。
- 对 winner 单独跑 `msprof op`。
- 最终性能报告使用单独复测的 `msprof op`，不直接使用 autotune latency 作为最终结果。

## 实验批处理 runner（T-4）

Phase 2 的实验分支执行（L0 回归 → msprof 采集 → perf_records.jsonl 追加 → 汇总表）应交给批处理脚本而非 agent 逐个往返驱动：

1. **首轮生成**：先从 benchmark 建立 `workload_inventory.json`，再从模板 [run_experiments_template.py](run_experiments_template.py) 拷贝到 `perf_opt/run_experiments.py`，适配 `kernel_id`、直接运行 kernel 的命令与 EXPERIMENTS；模板只读取 inventory 中该 kernel 的 tune workload，smoke 精度回归另按 skill 执行；
2. **后续轮直接调用**：逐 workload 候选实验使用 `--workload {workload_id}`；baseline、核内合并和 final 用 `--phase baseline/merged/final --all` 对当前 kernel 的全部 tune workload 串行采集。脚本逐测量追加性能记录，末尾打印「候选 vs current best」汇总表；
3. **决策留在 agent**：读表决策 winner/rollback/下一轮候选——脚本只执行不判断；`<5%` 差异与平区决胜仍按 SKILL.md Phase 2 第 9 步走 `ab_test.py` 交错多 run 协议；
4. 收益：单轮 agent 交互轮次下降 60%+；大工件会话超限类失败（VP-2026-0021：两次空返回各耗 1579s/930s）被根治——长链路工具调用收敛为一次脚本执行；
5. 脚本输出与 perf_records.jsonl 对账（append-only 纪律不变），gate 4 机械校验照常生效。

## 口径约束

不要把 autotune 理解为自动完成所有性能优化。它只是在给定搜索空间中选择候选配置；winner 需要经过 top-k / 邻域检查和独立复测。

对 elementwise / activation / 轻量 vector kernel，先判断是否存在 launch/scheduling overhead。若 `Block Dim` 远超物理核数，优先把 `T.serial` 多块迭代作为结构性改写，再 autotune `block_size` 与 `num_cores`，不要只扩大 block size 或 pipeline stage 搜索空间。`num_cores` 不只是在减少任务数，也是在给硬件提供合适的软件任务并发；太少会并发不足，太多会派发开销过高。

如果所有 config 的 `msprof Task Duration` 接近，先用更大 `launch-count` 复测。复测后仍打平时，使用 imbalance、整除性和资源余量决胜；若仍无法区分，回到 Phase 2 重新分析当前现象，而不是继续扩大搜索空间。

如果 `tilelang-mlir-ascend` 中 autotune API、参数名或示例发生变化，以源库文件为准，更新本文件中的引用路径或极少量执行规则即可。
