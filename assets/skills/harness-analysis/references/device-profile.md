# 上板 trace 与 aicMetric 分析

本方法消费前序节点的 ANALYSIS_DATA；采集仍按 [上板诊断契约](../../harness-develop/references/onboard-diagnostics.md) 执行。读取已有文件只需离线环境。先读交付清单和日志，按 case、真实 kernel、launch、设备、stream/task、构建身份定位文件，再读指标。`aicMetric`、`aic_metrics` 和 `AiCMetrics` 在不同导出中可能表示配置或实际计数；出现配置名不证明计数采集成功。

## 证据入口

| 实际输入 | 读取方式 | 能支持的判断 |
| --- | --- | --- |
| trace JSON 的事件数组或 `traceEvents` | 先定位目标 `name/category/pid/tid/args`，读取目标事件及嵌套指标 | 调用实例、事件区间、该事件实际携带的 aicMetric；核内 pipe 需要专门轨迹 |
| OpBasicInfo / kernel_details / task_time 等 CSV | 按完整 kernel 名及 launch、stream/task 等字段锁定实例 | 核对目标、调用顺序和计时边界；各版本字段可不同 |
| PipeUtilization、ArithmeticUtilization、Memory/L0/UB、L2Cache、ResourceConflictRatio | 逐文件读取列名、单位、block/sub_block/core 与 AIC/AIV 字段 | 逐核差异、计算/访存/冲突的线索；保持 pass 分离 |
| PipeTimeline、Occupancy、Roofline 导出 | 匹配版本的官方导出或查看器；有可读 CSV/JSON 时离线分析 | 流水区间、所采核间分布或模型位置，分别说明覆盖 |
| visualize_data.bin、原始 PROF、DB、私有 JSON 格式 | 保留原始文件与摘要，使用采集版本的官方解析器导出可读证据 | 当前无法解析时列具体缺口；已有表格仍可局部分析 |

字段解释以采集版本的 [msOpProf 数据说明](https://github.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_performance_data.md) 和导出元数据为准。尤其分清 ratio 的 0–1、0–100 与百分数字符串，KB 的换算约定、带宽是 active 还是整个 task 分母、计数是单核还是全芯片。字段相似不自动构成跨版本映射。

原生 trace 可能只含 kernel 事件，实际 aicMetric 在 kernel_details/task_time 等表里。此时沿 stream/task/启动身份关联表格，保留 trace 的调用映射限制；不要因为事件 args 没有指标就断言整次采集缺失。未使用的 AIC/AIV 单元可能导出 0 或 N/A，先核对实际执行类型，再解释其利用率。

## 配套离线摘要工具

`scripts/profile_summary.py` 使用 Python 标准库读取明确选中的 CSV 或 Chrome trace JSON，保留原始值、事件 JSON Pointer / CSV 数据记录序号、输入和解析器 SHA256。它不解析私有二进制、不访问设备、不判断性能达标、不修改搜索树。

将输入清单写在当前 analysis run 内；相对输入路径以清单所在目录为基准。每个清单绑定一个 case、候选和真实 kernel，来源逐个声明 pass、工具版本、筛选条件及范围依据；一个源文件只出现一次，需要分析另一 launch 时使用另一清单。引用同一采集的不同文件仍逐文件统计，不自动 join 或合并 PMU 重放。

```json
{
  "schema": "operator-harness/profile-input@1",
  "identity": {
    "project_id": "example", "run_id": "analysis-1", "search_node_id": "N0",
    "candidate": "C1", "case": "case-large", "build": "BUILD_DIGEST",
    "environment": "ENVIRONMENT_REVISION", "kernel": "actual_kernel"
  },
  "sources": [
    {
      "id": "target-event", "path": "trace_view.json", "sha256": "FILE_SHA256",
      "format": "trace", "role": "kernel", "pass_id": "pipe-pass",
      "tool": "torch_npu VERSION", "time_unit": "us",
      "scope_evidence": "ANALYSIS_DATA.md 的 case/launch 映射及单位依据",
      "select": {"/name": "actual_kernel", "/ph": "X", "/args/Task Id": 7},
      "metric_fields": ["/args/aic_metrics/aiv_vec_ratio"]
    },
    {
      "id": "pipe-table", "path": "PipeUtilization.csv", "sha256": "FILE_SHA256",
      "format": "csv", "role": "metrics", "pass_id": "pmu-group-1",
      "tool": "msopprof VERSION",
      "scope_evidence": "单 launch 目录及唯一 OpBasicInfo 行；绑定 actual_kernel/case-large；已核对无其他启动",
      "select": {},
      "metric_fields": ["/aiv_time(us)", "/aiv_vec_ratio", "/aic_time(us)"]
    }
  ]
}
```

示例中的列名、层级、task ID、版本、单位和摘要均按真实文件替换。`scope_evidence` 写可追溯的文件/记录位置；这些身份和关联由分析者核对，脚本只校验输入摘要和显式选择，不能认证身份声明。`select` 是 JSON Pointer → 原始值的精确匹配，CSV 值使用字符串，trace 数字使用数字。字段中的 `/` 写作 `~1`，`~` 写作 `~0`。不使用子串/正则选 kernel。

- `role=kernel`：trace 要求 `/name` 与 identity.kernel 相同且唯一命中一个 `X` 完整事件；CSV 默认 `/Op Name`，可用 `kernel_field` 指定其他列，要求唯一行。同名多 launch 必须增加筛选。
- `role=metrics/timeline`：用于绑定了单 launch 的逐核表、计数事件或流水。只有已核对输入文件范围时才用空 `select`；含多 kernel/launch/rank 的文件必须先限定实例。无 kernel 名的逐核表不能凭文件名认领归属。
- `metric_fields`：只对指定的数值列/嵌套字段给出 min/max/mean/median、缺失数及极值原始位置；不相加比率、不同类型的核时间或各核带宽。AIC/AIV 分别选其字段；必要时按 sub_block 类型分别分析。工具保留完整 `records` 和 `available_fields`，首次不指定指标即可检查实际结构。
- 空值、N/A、NaN/Inf、带单位的字符串不变成零；未转换的值留在原始记录中。某个请求字段全无数值时返回 partial。百分数字符串等需按工具文档另做有记录的转换，不能改写原始文件。
- CSV 兼容 UTF-8 BOM 和工具导出的尾部分隔符，仅忽略表头及数据均为空的尾列；非空无名列、重复字段或不齐的记录报错。trace 字符串时间戳按十进制求差，区间端点以字符串保存，避免大绝对时间抹掉短事件精度。
- trace 的 `X` 事件按 `(pid, tid, cat, name)` 分轨计算区间并集、跨度及跨度内未覆盖时段；`C/i/I/M` 保留原始记录，`B/E` 等未配对格式标 partial。时间单位由输入显式声明，缺省 unknown；`displayTimeUnit` 不是事件单位依据。未覆盖时段只表示本次筛选事件没覆盖，不能称为硬件空闲或同步等待。轨道之间及不同文件不自动求重叠。

```bash
python3 .agents/skills/harness-analysis/scripts/profile_summary.py \
  --input .workspace/example/runs/analysis-1/profile-input.json \
  --output .workspace/example/runs/analysis-1/profile-summary.json
```

输出目录应已存在，文件独占创建，重跑使用新文件。退出 0 表示所选数据解析成功；退出 2 表示输入/输出错误或部分解析，先读 stderr 和报告中各 source 的 error/status。一个文件失败仍保留其他文件的解析结果。`parsed` 不等于 ANALYSIS_DATA ready 或诊断完成；脚本也不核实采集覆盖、必需指标、数值正确性或 case 代表性。大 trace 优先用官方工具导出有限目标范围并保留原文件及导出命令；本工具会将输入加载到内存。

## 从指标到当前瓶颈

先按正式性能结果选慢 case、退化 case 和不同 dispatch 分支；每个 case 内先定位耗时阶段，再解释该阶段。完整算子跨 kernel 时保留依赖、并发及启动边界，不把单核最大时间或 kernel 时间总和改成正式延迟。

| 信号组合 | 需要核对的实现/反证 | 可检验方向 |
| --- | --- | --- |
| MTE2/3 时间、流量或带宽突出，计算等待输入 | 连续性、实际搬运粒度、重复读取、有效数据/物理流量、L2 与并行度；高 MTE 比例不等于带宽饱和 | 调整 tile/布局、复用输入、减少中间 GM 往返；预期字节/指令数或等待下降 |
| VEC/CUBE/SFU/SIMT 活动突出 | 指令类型、dtype、依赖链、冗余 cast/计算、padding；实际后端生成代码是否实现所提 API | 减少工作量、改变归约或 GEMM 数据流、增加独立工作；预期相关指令/时间下降 |
| scalar/指令缓存开销与短 tile 同时突出 | 动态索引/分支、循环展开、代码体积、任务粒度；与 launch 开销区分 | 外提不变量、合并任务、控制展开；关注大/小 shape 的相反效果 |
| 同一采集的 pipe 轨迹有可定位等待/串行段 | 生产消费依赖、buffer 生命周期、同步方向；确认轨迹单位/时钟域和采样范围 | 调整 pipeline/buffer/同步；预期同范围等待缩短或有效重叠增加 |
| 同类核的时间/工作量尾差明显 | 核/子核映射、尾块、输入分布、有效 tile 数与核限制；AIC/AIV 分别比较 | 更换分片轴、粒度或调度；预期长尾降低，全部输出仍覆盖 |
| UB conflict/wait 与源码访问映射相关 | bank/stride/operand、生成指令、不同资源冲突分类；该证据只覆盖所测存储层 | 布局/padding/operand 偏移或调度调整；同时检查容量、额外搬运和时间 |
| kernel 很短，调用链存在启动间隔 | host/设备时间线、同步与缓存/JIT；task 与核时间差不足以独立证明 launch bound | 在融合契约内减少启动/阶段；用原计时边界复验 |

每个瓶颈需要“观测 + 源码/IR 映射 + 其他解释 + 可证伪实验”。混合瓶颈或证据不足时保留竞争解释。新补采只索取能区分解释的 case、指标、core/阶段及额度，交给 develop collect-only 后返回 analysis。统计报告本身不输出确定的 bound 标签或预估加速比。
