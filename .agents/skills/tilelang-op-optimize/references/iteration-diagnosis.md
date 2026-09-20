# NPU TileLang 当前现象分析与候选优化点指南

## 目录

- 用途
- 输入
- Step 1：确认 Profile 口径
- Step 2：建立本轮诊断上下文
- Step 3：生成候选优化点
- Step 4：匹配瓶颈模式
- Step 5：分别创建实验分支
- Step 6：评估本轮 winner
- Step 7：本轮输出模板
- 注意事项

## 用途

当 `tilelang-op-optimize` 进入 Phase 2 优化闭环的每一轮“本轮优化点分析”时，读取本文件。

本文件不把 kernel 强行归到某个固定瓶颈大类，而是把 `msprof op` 数据、kernel 代码结构和历史迭代结果转换成：

```text
本轮诊断上下文 -> 当前现象 -> 候选优化点 -> 独立实验分支 -> 本轮 winner
```

核心原则：

```text
先记录事实和上下文，再描述当前现象。
一轮可以生成多个候选优化点，并分别尝试。
每个实验分支都从同一个 current best 派生，只改一个主要优化点。
实验后先选择本轮实测最好的 valid/improved 分支作为候选 winner。
完成目标 workload 的迭代后才尝试核内合并；合并版通过本 kernel 全部 tune workload 的非回退检查与 smoke 精度检查后，才能成为新的已合并 current best。
下一轮基于新的 current best profile 重新分析当前现象。
```

---

## 输入

诊断前必须具备：

- `{op}.py`、current best 或当前实验分支 `perf_opt/{op}_round{iter}_{opt_id}.py`。
- `DESIGN.md` 中的性能目标章节。
- [hardware-context.md](hardware-context.md) 中的硬件上下文。
- 按 [profile-collection.md](profile-collection.md) 从 benchmark 的非 smoke workload 逐 kernel 采集到的 profiling 结果（`msprof op` 是唯一时延口径）。
- `perf_opt/opt_log.md` 中的 `Performance Test Data` 最小表。
- 已尝试过的优化点、实验分支结果、keep/rollback 结论和最新 current best。

诊断只允许使用 `profile_status=valid` 的记录。若某个 kernel/workload 的 profile 被标记为 `profile_invalid`，或 `captured_op_name` 无法追溯到 `target_kernel_name` / 目标 TileLang kernel，不能用该数据判断硬件瓶颈。

---

## Step 1：确认 Profile 口径

先确认：

```text
Performance Test Data 已存在
profile_status == valid
kernel_id 与 workload_id 已记录，且 inventory 分类为 tune
target_kernel_name 来自当前要优化的 TileLang kernel
captured_op_name 能追溯到 target_kernel_name 或目标 TileLang kernel
Task Duration 来自 OpBasicInfo.csv 中 captured_op_name 对应记录
不是 Cast / Mul / OnesLike / Random 等框架小算子
raw_profile_dir 存在且能追溯到本次 command
同一轮采集未出现并发 msprof op 或输出目录覆盖
```

按 `kernel_id + workload_id` 分别诊断，不把不同 workload 的 Task Duration 合成平均数后下结论。smoke 不进入性能诊断，只用于合并版精度回归。

---

## Step 2：建立本轮诊断上下文

每轮先记录事实和上下文，不要急着下结论。

### Profile 事实

```text
Task Duration(us)
Current Freq / Rated Freq
Block Dim / Mix Block Dim
AI Core Count per NPU
Block Dim / AI Core Count
Cube / Vector / MTE2 / MTE3 利用率
Memory / L2 / UB / L0 指标
PipeUtilization
ArithmeticUtilization
ResourceConflictRatio
```

### 理论估算

尽量从代码和 workload 推导：

```text
logical block 数
num_cores、iters_per_task、min/max iters、imbalance，若使用 T.serial 多块迭代
单 block 元素数 / 字节数
GM 读写总字节数
理论 memory-bound 时间
Task Duration 与理论时间差距
```

### 代码结构观察

记录当前 kernel 的结构事实，重点看这些结构是否能解释当前现象：

- dispatch 入口：是否有 dtype / axis / layout / flag 等真实分支。
- kernel 边界：一次测试是否调用多个 TileLang kernel，或夹杂框架生成的小算子。
- 并行粒度：`T.Kernel(...)` 的 block 数、每个 block 处理的数据量、是否用 `T.serial` 聚合多个 logical block。
- 调度参数：`num_cores / block_size / iters_per_core` 是否只测了单点，是否需要扫曲线；`num_cores` 是否处在任务并发不足、甜点区或派发开销过高区间。
- 搬运路径：GM->UB、UB->GM 是否连续、对齐，是否存在小粒度搬运。
- 搬运/计算组织：是否按 tile 串行执行“搬入、计算、写回”，是否具备 pipeline / multi-buffer 的空间。
- 片上 buffer：UB/L0 buffer 数量、大小、生命周期和 last-use；输入 staging、cast buffer、临时 buffer、输出 staging 是否能在 last-use 后复用。
- 尾块与分支：是否存在明显 tail block、mask、if branch 或空 block。
- 冗余访问：是否有中间结果写回 GM 后又读回，或重复读取同一批 GM 数据。

### 已知迭代结论

每轮必须读 `opt_log.md` 中已有结论：

```text
哪些优化点已验证有效
哪些优化点已被当前 workload 证明无收益
哪些修改导致精度失败 / 编译失败 / profile invalid
current best 是哪个版本
本轮 profile 相比上一轮 profile 哪些指标发生变化
```

---

## Step 3：生成候选优化点

候选优化点不是互斥分类。一个 profile 可以同时生成多个优化点。

每个候选优化点必须是可执行实验，而不是抽象判断。它必须写清：

```text
想解决哪个当前现象
为什么认为这个优化点可能有效
具体怎么改
看什么指标验证
什么情况回滚
```

候选表格式：

```markdown
| opt_id | 目标现象 | 优化点 | 判断依据 | 具体改法 | 验证指标 | 状态 |
|---|---|---|---|---|---|---|
| OP1 | Block Dim 过高且单 block 工作量小 | T.serial 多块迭代 | ... | 从 current best 派生 round{iter}_op1，只改 block 分配 | Task Duration / Block Dim / correctness | candidate |
```

状态含义：

```text
candidate：本轮候选
running：实验中
improved：实测有提升
config_no_gain：当前配置无收益，只否定该配置
family_no_gain：整类方向无收益，必须有足够配置覆盖或明确机制证据
invalid：精度失败或 profile 无效
blocked：编译、环境或合法性问题阻塞
defer：证据不足，暂缓
```

---

## Step 4：匹配瓶颈模式

如果当前现象需要模式参考，读取 [bottleneck-patterns.md](bottleneck-patterns.md)，用本轮诊断上下文辅助生成候选优化点。

匹配时注意：

- 不要把瓶颈模式当成互斥分类树。
- 一个 profile 可以同时匹配多个模式。
- 尽可能把有证据支撑、能落成代码实验的优化点放入候选表。
- 如果现有模式覆盖不了当前现象，可以自己寻找优化点，但必须写清触发信号、反证/不确定点、推荐动作和验证指标。

---

## Step 5：分别创建实验分支

本轮可以尝试多个候选优化点，但每个实验分支必须满足：

```text
base = 本轮开始时的 current best
每个分支都从 base 派生
一个分支只改一个主要优化点
不要在同一个分支里同时叠加多个候选优化点
```

推荐命名：

```text
perf_opt/{op}_round{iter}_{opt_id}.py
```

例如：

```text
perf_opt/mish_round2_op1.py
perf_opt/mish_round2_op2.py
perf_opt/mish_round2_op3.py
```

如果某个优化点本质上是参数选择，可以在该分支内做小范围参数实验；但搜索空间必须服务同一个优化点，不能混入其它结构性改动。

结构性优化分支通过正确性且方向有效后，不能只保留单个手写配置。必须围绕该结构暴露的关键参数做一轮 coarse autotune 或等价手动粗搜；例如 `T.serial` 多块迭代至少搜索 `block_size × num_cores`。autotune winner 不是最终结论，还要检查 top-k 配置和 winner 邻域，必要时手动/脚本精搜，再用最终 winner进入 `msprof op` 复测。

若优化点涉及 `num_cores / T.serial` 多块迭代，不能只测一个配置。先计算 `num_logical=ceildiv(N, block_size)`，再构造候选：

- AI Core Count 附近和若干倍 AI Core Count。
- 默认用 `整数倍 AI Core Count` 作为任务并发锚点；只有硬件上下文或 profile 明确给出其它调度核口径时，才使用该口径修正。
- `num_logical` 附近可整除或低 imbalance 的候选，记录 `min_iters_per_task / max_iters_per_task`。
- 当前 flat-grid 端点只作对照，不默认作为 winner。

每个配置都记录 `msprof Task Duration` 和 imbalance。若甜点区内 `Task Duration` 差异小于噪声阈值，先用更大 `launch-count` 复测；复测后仍打平时用 imbalance、整除性和资源占用决胜，不要用单次 msprof 结果对平区内候选排序。

---

## Step 6：评估本轮 winner

每个实验分支都要独立跑：

```text
L0 精度回归
msprof op 性能采集
profile 有效性校验
```

评估规则：

- 只比较同一 `kernel_id + workload_id` 下的 valid profile。
- 有一个或多个分支提升时，先按本轮主指标（`msprof op Task Duration(us)`）选择候选 winner；Task Duration 打平时，用 imbalance、UB/L0 余量和代码复杂度决胜。
- 每个 workload 完成迭代后，把其 winner 尝试合入同一个 kernel；先尝试无条件优化，若本 kernel 其他 tune workload 有可信回退，再尝试核内 shape/dtype/参数分支并重新编译、验证。
- 只有合并后的实际 kernel 通过本 kernel 全部 tune workload 的精度和非回退检查、以及 smoke 精度检查，才能更新已合并 current best。
- 小 kernel（Task Duration <20us）或逼近带宽/搬运地板的 workload，<5% 量级的候选差异须先经**交错 A/B/A/B 多 run 协议**裁决：≥3 次独立 `msprof op` run、候选与基线交替次序、合并中位数比较——同 kernel 跨 run 存在 ±3–5% 快/慢双态（BP_run_state_bimodality），单 run median-of-15 可能把双态膨胀误判为增益或回退（lerp_tensor 实证：单轮 -5.2% 复测收缩为 -3.1%）。协议在首次出现 <5% 差异时即前置使用，而非采纳后复核。工具化执行：`python3 .agents/tools/ab_test.py --a <baseline_kernel> --b <candidate_kernel> ...`（交错多 run + 合并中位差 + 配对方向一致性判定）。
- 如果合并版只在目标 workload 提升，但本 kernel 其他 tune workload 明显回退，且核内条件分支不能安全解决，记录 rollback 或 defer，不更新已合并 current best；不得转移到 Python factory、op 层或 wrapper 分派。
- 如果所有分支都无提升、无效或阻塞，current best 保持不变。
- 其它分支记录为 `config_no_gain / family_no_gain / invalid / blocked / defer`。
- 单个配置无收益只能记为 `config_no_gain`；不能据此否定 `T.serial / T.Pipelined / num_cores` 这类整类方向。
- 只有在关键配置已覆盖，或有明确机制证据时，才能记录为 `family_no_gain`。
- 已被证明无收益的配置，下一轮不要重复尝试；整类方向只有在 `family_no_gain` 后才停止尝试，除非新的 current best profile 形态已经变化。
- autotune 只在结构稳定或结构性分支已经通过正确性后使用，不作为逃避现象分析的默认动作。

---

## Step 7：本轮输出模板

```markdown
## Iteration {iter} Diagnosis

### Diagnostic Context

- kernel_id: {kernel}
- workload_id: {workload}
- current_best: {file}

#### Profile Facts

- Task Duration: {duration_us} us
- Block Dim / AI Core Count: {block_dim} / {ai_core_count}
- Cube / Vector / MTE2 / MTE3: {metrics}
- UB/L0/resource indicators: {metrics}

#### Theory Estimate

- estimated GM bytes: {bytes}
- estimated memory-bound time: {theory_us} us
- latency gap: {brief}
- num_cores / iters / imbalance: {if_applicable}

#### Code Structure

- existing entry routing: {仅说明 workload 如何到达目标 kernel，不作为新增性能分派}
- kernel boundary: {single TileLang kernel / multiple kernels / extra framework ops}
- parallel granularity: {T.Kernel blocks, per-block workload, T.serial usage}
- transfer path: {GM->UB / UB->GM continuity, alignment, copy granularity}
- transfer/compute organization: {tile-by-tile serial transfer/compute/store or pipeline-ready structure}
- on-chip buffers: {UB/L0 buffer count, size, lifetime}
- tail and branches: {tail block, mask, if branch, empty block}
- redundant access: {GM round-trip or repeated GM reads}

#### Known Iteration Conclusions

- previous effective/no-gain optimization points: {summary}
- current best profile changes: {summary}

### Current Phenomena

- P1: {phenomenon + metrics}
- P2: {phenomenon + metrics}

### Candidate Optimization Points

| opt_id | 目标现象 | 优化点 | 判断依据 | 具体改法 | 验证指标 | 状态 |
|---|---|---|---|---|---|---|
| OP1 | {phenomenon} | {optimization_point} | {evidence} | {change} | {metrics} | candidate |

### Experiment Branches

| branch | base | opt_id | major_change | correctness | task_duration_us | profile_status | result |
|---|---|---|---|---|---:|---|---|
| {file} | {current_best} | OP1 | {change} | pass/fail | {v} | valid/invalid | improved/config_no_gain/family_no_gain/invalid/blocked |

### 当前 Kernel 全部 Tune Workload 合并检查

| merged_candidate | kernel_id | workload_id | current_best_us | merged_us | precision | status |
|---|---|---|---:|---:|---|---|
| {file} | {kernel} | {workload} | {v} | {v} | pass/fail | pass/regressed |

smoke workload 只列精度结果，不填性能时延或参与性能门禁。

### Iteration Winner

- winner: {file_or_current_best}
- reason: {本 workload 的 Task Duration 最优；完成该 workload 后另验证核内合并版在本 kernel 全部 tune workload 上无可信回退}
- new_current_best: {file}
- rollback_branches: {list}
```

---

## 注意事项

- 一轮可以尝试多个候选优化点。
- 每个实验分支只改一个主要优化点。
- 同一轮所有实验分支都必须从同一个 current best 派生。
- 组合优化只在单点优化已经证明有效后再做。
- 每个实验分支都必须跑 L0 精度回归。
- 每个有效实验分支都必须用 `msprof op` 重新采集目标 kernel；候选差异落入噪声阈值时先用更大 `launch-count` 复测，不要用单次 msprof 结果对平区内候选排序。
- 更新已合并 current best 前，必须确认本 kernel 全部 tune workload 不发生可信性能回退，并完成 smoke 精度回归。
- 区分 `config_no_gain` 和 `family_no_gain`；一个配置失败不能否定整类优化方向。
- 如果所有候选优化点都不够直接，先补充 profile 或检查 profile 口径，不要盲目扩大搜索空间。
