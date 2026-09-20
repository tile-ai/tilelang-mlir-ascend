# 性能采集流程

本文档服务 `tilelang-op-optimize` 的 Stage 4。性能调优对象是迁移后 benchmark 实际参数化的、可运行且非 smoke 的 workload；合并对象是它们实际调用的 Stage 3 kernel。`msprof op Task Duration(us)` 是唯一 kernel 时延口径，不使用 NPU event 或端到端耗时。

---

## 1. 建立 benchmark workload 清单

TileOPs 迁移/集成场景读取迁移后的 benchmark 参数化入口，展开每个案例的 `workload_id`、shape、dtype、影响 kernel 的参数和 benchmark 来源。benchmark 可以通过 helper 从 manifest 生成参数，但**只取 benchmark 实际展开的案例**；不得额外枚举 manifest、L0/L1/L2/Boundary 测试或补造 benchmark 中不存在的 case。只提取 workload 参数，不要求先运行正式 benchmark、接入 wrapper 或注入候选实现。非 TileOPs、确无迁移 benchmark 的独立算子使用用户明确指定的性能 workload，逐 kernel/逐 workload 建立 inventory；`benchmark_source` 记为 `explicit_target`。

逐案例确认它实际调用的 Stage 3 kernel，记录可唯一定位实现文件及函数的 `kernel_id` 和用于 `msprof op --kernel-name` 的 `target_kernel_name`。一个案例若调用多个目标 kernel，分别登记 workload–kernel 对；无法确认映射时记录原因并停止该案例的性能采集。TileOPs 原有 op 层路由只用于查明映射，不是 Stage 4 调优单位，也不应修改。

分类顺序：benchmark 显式 skip → `skipped`（保留 reason）；`pytest.mark.smoke` 或案例 ID / 明确 workload label 中独立的 `smoke` 词（大小写不敏感，以 `-`、`_` 等分隔，如 `smoke-1m`）→ `smoke`；其余 → `tune`。若 `full` 标记与 `smoke` ID/label 冲突，记录冲突，仍归为 `smoke`。不得凭 shape 小、参数化顺序或 manifest 注释推断 smoke。

将完整清单写入 `perf_opt/workload_inventory.json`，契约见 `_shared/standards/signal-registry.md` §5。`tune` 项进入性能 baseline、调优与性能回退门禁，并在完成后写 `tuning_status`；`smoke` 项仅用于每次合并后及最终的精度回归，不采集性能 baseline、不选 winner、不参加性能回退门禁；`skipped` 项只记录原因。无 `tune` 项的 kernel 记录 `no_tunable_workload`，不虚构性能结果。

---

## 2. 逐 kernel 采集 baseline 并排序

一个 optimizer subagent 逐个 kernel 执行。对当前 kernel 的**所有** `tune` workload 串行采集 baseline，确认每项 profile 有效后再决定调优顺序：优先实测耗时较大、瓶颈明确且可能有结构性收益的大 shape；随后尽早处理小 shape、其他 dtype、尾块和资源临界案例；其余按实测瓶颈与收益机会排序。排序与理由写入 `opt_log.md`，不得因排序跳过 workload。每个 workload 完成迭代后才尝试核内合并，并复测当前 kernel 的全部 `tune` workload。

---

## 3. 串行运行 msprof op

对每个 `kernel_id + workload_id` 逐个运行 `msprof op`，禁止并发 profiling。使用 benchmark workload 参数直接运行 Stage 3 kernel 或 `perf_opt/` 下实际候选文件，**不运行需要 Stage 5 wrapper 的正式 benchmark**。

命令模板：

```bash
msprof op \
  --kernel-name={target_kernel_name} \
  --output={output_dir} \
  --launch-count=20 \
  --warm-up=5 \
  --dump=off \
  --aic-metrics=BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,MemoryUB,MemoryL0,L2Cache,ResourceConflictRatio \
  python {actual_kernel_file}.py {workload_args}
```

`{workload_args}` 按现有 kernel 入口适配，可使用已支持的 `--case`，但不新增候选实现注入接口。日志中的 `command` 必须包含实际候选文件与 workload，避免误测原始 Stage 3 文件。

实验迭代可用较小 `launch-count` 快速判断方向，但必须在日志中记录 launch 数；baseline、候选 winner 和 final 推荐使用更稳定的 launch 数复测。

如果当前 CANN 版本不兼容逗号形式的 `--aic-metrics`，退回：

```bash
msprof op --kernel-name={target_kernel_name} --output={output_dir} --launch-count=10 --warm-up=5 --aic-metrics=Default python {actual_kernel_file}.py {workload_args}
```

输出目录只需保证唯一，推荐：

```text
perf_opt/profiles/{profile_stage}/{kernel_id}/{workload_id}_{candidate_id}/
```

新建 `msprof op` 输出目录后，先确保 group/other 不可写，例如 `chmod 700 {output_dir}`；否则某些环境会拒绝采集或写入失败。

如果 `msprof op` 在输出目录下生成 `OPPROF_xxx` 子目录，记录实际 `raw_profile_dir`。

注意：`msprof op` 多 launch 稳定性需要靠多次独立运行，不要用 CSV 记录条数推断实际 launch 次数。

实验 stdout/stderr 不写到 `perf_opt/` 顶层，统一写入：

```text
perf_opt/logs/{profile_stage}/{run_id}.log
```

---

## 4. 校验采集结果

每次采集后检查：

- 能读取目标记录`OpBasicInfo.csv`的 `Task Duration(us)`。
- `captured_op_name` 能通过命令、输出目录和运行日志追溯到本次 `target_kernel_name` 或目标 TileLang kernel。
- 采到的不是 Cast / Mul / OnesLike / Random 等框架小算子。
- kernel launch 次数足够覆盖 `warm-up + launch-count`。

注意：`captured_op_name` 不一定机械等于 Python 函数名。只要能证明它属于本次目标 TileLang kernel，即可标记为 valid。

无效 profile 记录 reason，并重新采集；无效数据不能进入诊断。

---

## 5. 记录格式

把逐 workload 采集结果写入 `perf_opt/opt_log.md` 的 `Performance Test Data` 章节。路径中的 ID 须安全转义并保持唯一；baseline 以 `phase=baseline`、`round=0` 追加到 `perf_records.jsonl`。

最小格式：

```markdown
## Performance Test Data

| kernel_id | workload_id | candidate_id | target_kernel_name | captured_op_name | task_duration_us | profile_status | raw_profile_dir | command |
|---|---|---|---|---|---:|---|---|---|
| {kernel} | {benchmark_case} | baseline | {target_kernel} | {captured_op} | {v} | valid | {raw_dir} | {cmd} |
```

另列 smoke/skip 排除原因与 smoke 精度结果。最终同一 kernel 的全部 `tune` workload 必须来自同一个 `phase=final` 候选版本，不能拼接各自最快的实验数字。

返回：

```text
PERF_DATA_COLLECTED
```
