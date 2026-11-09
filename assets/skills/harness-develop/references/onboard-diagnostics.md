# 上板流水与指标：实现节点交给 analysis 的数据

## 主采集入口与补充入口

完整多指标诊断以 `msprof op` / `msopprof` 为主，由工具完成事件选择、重放、插桩和驱动数据采集。`torch_npu.profiler` 按需补充框架到设备的调用关系及基础 AI Core 指标；已知真实 kernel 和启动范围时，不强制先跑原生 profiler。两者的指标名字和枚举不是同一个接口，逐个切换 torch_npu 枚举不能代替完整多指标诊断。正式精度与性能验收仍按冻结协议执行，本节为独立诊断作业，不用详细采集时的耗时重写标杆或加速比。

| 用途 | 执行方式 | 交给 analysis 的主要材料 |
| --- | --- | --- |
| 单算子详细流水与资源分析 | `msprof op` 外层启动 Python 测试，按真实 kernel 名与启动序号选择；MSTX 绑定验证通过后可用脚本选区 | 实际版本的 trace.json、visualize_data.bin、OpBasicInfo、PipeUtilization、Memory 等表及日志；Occupancy/Roofline 视图数据 |
| 按需发现真实 kernel、核对调用链及基础计数 | 设备进程内 `torch_npu.profiler.profile`；Level1、CPU+NPU、Text 导出；每次一个 AiCMetrics | 原始 PROF/FRAMEWORK、trace_view.json、kernel/算子 CSV/DB（以实际版本输出为准），调用顺序/stream 与源码阶段映射 |

官方 [torch_npu 配置实现](https://github.com/Ascend/pytorch/blob/master/torch_npu/profiler/experimental_config.py) 的 aic_metrics 是单一枚举，支持 PipeUtilization、ArithmeticUtilization、Memory、MemoryL0、MemoryUB、ResourceConflictRatio、L2Cache、MemoryAccess 等。不要传逗号字符串或把多个枚举按位或起来；不合法参数可能被回退到默认项，产生“程序成功但没有所需指标”的假象。PipeTimeline、Occupancy、Roofline、Default 不属于这组原生枚举，不能用 PipeUtilization 冒充完整流水图。

[msOpProf 用户指南](https://github.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_user_guide.md) 支持逗号分隔的多指标；[PyTorch 调用示例](https://github.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_usage.md#collecting-profile-data-of-pytorch-framework-operators) 支持外层工具启动 Python。选定环境可能提供 `msprof op` 或 `msopprof`，根据实际版本帮助选择；[CANN 工具说明](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/850/devaids/optool/atlasopdev_16_0082.html) 解释了二者关系。不能将线上 master 的功能列表当成本次安装已验证的能力。

## 多指标的实际采集链路

上游 [ParseValue / LoadPmuVec](https://github.com/Ascend/msopprof/blob/master/csrc/op_profiling/common/prof_args.cpp) 将逗号列表合并为功能配置，再生成芯片对应的 AIC/AIV 事件列表。各项并非一项对应一次执行：

| 选项 | 处理方式 |
| --- | --- |
| Default | 打开基础 PMU 指标集合，不等于启用全部专项采集功能 |
| Memory | 启用内存指标；与 Default 合用时其 PMU 需求已有覆盖 |
| Occupancy | 启用相应 PMU 需求及核间负载分析视图 |
| Roofline | 启用分析视图，并打开基础 PMU 集合供计算使用 |
| PipeTimeline | 独立流水采集，并选择 INSTR_PROF_END 动态插桩，不从 PipeUtilization 百分比推导时间线 |

按指标筛选事件时，工具去重并按预定义分组排列有关联的事件；当前源码的 kernel/range 模式或全 PMU 配置直接使用芯片完整预定义事件表。每组能放多少事件、有效组数与额外插桩轮次由芯片和工具版本决定。预算不能按“指标数”或 `--launch-count` 估算重放次数，也不能承诺少填一个指标就少跑一轮。具体分组见 [defs.h](https://github.com/Ascend/msopprof/blob/master/csrc/op_profiling/common/defs.h)，application 重启和插桩安排见 [DeviceTask](https://github.com/Ascend/msopprof/blob/master/csrc/op_profiling/profiling/device/run/device_task.cpp)。

[DeviceTask 的环境设置](https://github.com/Ascend/msopprof/blob/master/csrc/op_profiling/profiling/device/run/device_task.h) 通过 LD_PRELOAD 加载 libmsopprof_injection.so，并设置 TASK_QUEUE_ENABLE 等运行选项。注入库来自 [msopcom 子模块](https://github.com/Ascend/msopprof/blob/master/.gitmodules)，需按所选主仓实际子模块版本核对：

- [ProfDataCollect](https://github.com/Ascend/msopcom/blob/master/csrc/runtime/inject_helpers/ProfDataCollect.cpp) 负责 kernel replay、同步和内存恢复；application 模式由外层重新执行应用。
- [ProfTask](https://github.com/Ascend/msopcom/blob/master/csrc/runtime/inject_helpers/ProfTask.cpp) 按组配置 AIC/AIV PMU；PipeTimeline 进入 BIU_PERF 指令采集通道并配合插桩。
- [AscendHalOrigin](https://github.com/Ascend/msopcom/blob/master/csrc/ascend_hal/AscendHalOrigin.cpp) 调用 ascend_hal 的 prof_drv_start、prof_channel_poll、prof_channel_read、prof_stop。这条链路不通过 torch_npu.profiler。

Harness 保留整组指标交给工具处理，不在 Python 中复制 PMU 事件表、驱动采集或重放循环。各次重放及专项采集共同生成报告，不能把报告中所有计数、插桩结果和时间线声称为同一次执行的同步观测。

## 采集前固定计划

port-baseline 初版与普通性能优化 develop 默认交付分析数据，不等待下一 analysis 再从零采集。TASK 预先写 `diagnostics_required`、代表性/退化 case、kernel 阶段、所需指标、环境权限和有界额度；仅代码/精度任务可以明确不采，仍给出 ANALYSIS_DATA 中的范围说明。新分派路径和发生退化的 case 优先；数据仍适用时复用确切旧产物，不机械重跑所有 shape。

先检查选定环境中 `msprof op --help`（或明确选择的 msopprof）、torch_npu API/源码、工具版本、芯片与指标支持。能力差异记录在共享环境修订/项目 TASK，不为某一芯片设计通用排障流程。当前上游把 PipeTimeline 限在特定产品，且采样可能只覆盖部分 core；这不是“所有芯片、所有核、所有指令都支持”的承诺。unsupported 不填零，也不偷偷删参数继续报告完整采集。

核对与正式性能相同的候选、编译/launch、case、dtype、layout、设备与核心限制。为了 Source/指令级映射改变构建选项时，生成独立诊断构建身份，说明与正式产物的差异，不把它的耗时当验收值。先通过精度再采常规性能诊断；明确针对错误的诊断可以有界执行，但精度失败仍不允许节点接收。

## Python：外层 msprof op 与可选 MSTX 选区

先从实际构建/launch 元数据、已有有效采集或按需补充的原生 profiler 取得真实设备 kernel 名。Python 函数名、Op 类名和 ACLNN API 名不能自动当作 kernel 名。多阶段算子分别记录实际入口；同名重复启动还需核对 stream/task/调用序号，防止把初始化或预热当目标。以下 `call/prepare/synchronize/check_reservation` 由项目已验证入口提供，分别负责完整调用、编译/分配/重置、参与设备与 stream 的同步、资源约定检查。

在单独进程模式中复用同一测试入口，关闭原生 profiler。先使用工具原生的 kernel 过滤，在环境约定的资源保护范围内启动（占位符需替换为本次真实信息）：

```bash
msprof op --aic-metrics=Default,PipeTimeline,Occupancy,Roofline,Memory \
  --kernel-name=REAL_KERNEL --launch-count=1 \
  --launch-skip-before-match=VERIFIED_SKIP --replay-mode=kernel \
  --output=DIAGNOSTIC_OUTPUT python3 operator_worker.py --mode=msprof --case=CASE_ID
```

`VERIFIED_SKIP` 是从启动序列核对的数量，计入过滤前所有设备算子，不只是匹配名字的目标；不能直接填 Python warmup 次数。优先使用预编译缓存和最小诊断入口，核对输入初始化、预热与目标的顺序，避免每个进程内 launch 序列变化使选择失效。诊断仍要复现所选正式构建与 case；采集中的 JIT/重放耗时不用于性能验收。

若本次环境的 torch_npu 与工具 MSTX 绑定已验证，可通过下面的脚本选区省去易变的序号：

```python
from torch_npu_diagnostics import marked_call

output = marked_call(
    call, prepare=prepare, synchronize=synchronize,
    check_reservation=check_reservation,
    range_name="harness_target", warmup=warmup,
)
```

其核心为预编译/预热和同步后调用 `torch_npu.npu.mstx.range_start("harness_target")`，在 range 内执行一次完整调用并同步，最后在 finally 中执行 `range_end(range_id)`。这里用 host range 筛选提交的目标算子；range 的墙钟不用于性能计时。需要在原生 profiler 查看设备侧标记时才按官方 [MSTX 接口](https://github.com/Ascend/pytorch/blob/master/torch_npu/npu/mstx.py) 传入实际 stream，并在原生 experimental_config 启用 mstx；这是另一采集模式，不和外层 msprof op 嵌套。多个 stream 的同步与覆盖须由真实入口核验。

绑定不能只凭 Python API 存在来确认。上游 [MstxMgr 实现](https://github.com/Ascend/pytorch/blob/master/torch_npu/csrc/profiler/mstx_mgr.cpp) 在 `rangeStart` 前检查启用状态；外部 profiler 路径包括读取 `PROFILER_SAMPLECONFIG` 中的 `msproftx` 或检查 MSPTI 状态。工具支持 `--mstx=on` 不代表每个安装组合都已打通这条链路。用最小目标在外层工具下核对非零 range ID、目标捕获及范围；失败时保留日志，改用前面的显式 launch 选择，不静默丢弃范围过滤，不为绕过检查启动嵌套 profiler 或擅自注入内部配置变量。

在环境约定的资源保护范围内启动，例如（REAL_KERNEL 与测试命令须替换为本候选真实信息）：

```bash
msprof op --aic-metrics=Default,PipeTimeline,Occupancy,Roofline,Memory \
  --kernel-name=REAL_KERNEL --mstx=on --mstx-include=harness_target \
  --launch-count=1 --replay-mode=kernel --output=DIAGNOSTIC_OUTPUT \
  python3 operator_worker.py --mode=msprof --case=CASE_ID
```

`operator_worker.py` 是项目实际测试适配，需实现上述 msprof 模式；不是 harness 现成的通用 runner。`msprof_command()` 为一个目标生成包含完整指标组合的无 shell argv，默认不加 MSTX，传 `range_name` 才启用选区；不再按该组合里的指标逐一生成原生 profiler 作业。记录完整 argv、退出码和日志。helper 不提交作业或授予设备权限。其他阶段可单独采集并汇总，不把一个 kernel 的数据声称为完整调用覆盖。

重要的采集语义：

- `--kernel-name` 可能按前缀匹配；即使填了完整真实名字，也须检查实际捕获结果。`--launch-count` 默认有限，不能将“有输出”当成全部阶段已采集。MSTX 过滤后仍需检查 warmup/JIT、辅助 kernel 和目标实例。
- 普通 kernel replay 自带设备内存 Backup/Restore，ReplayOnce 在重放前恢复工具跟踪的内存；应用侧 prepare 不参与内部逐轮重放。不能因此断言原地/原子更新必然不能重放，也不能把该机制当成恢复任意 Python、外部服务或通信状态的保证。核对本版本可跟踪的分配/别名与恢复日志，MC2/LCCL 等路径有不同处理。确有未覆盖状态时才选择已支持的 application replay，由每次应用重建确定性起点，并将重新初始化/JIT 纳入预算。不能盲用 range replay，相关工具/芯片上它与 PipeTimeline、Roofline 等存在组合限制。
- 重放的缓存状态、采样开销与正式性能协议可能不同。因此这些数据用于解释瓶颈，不取代已冻结的性能分母或整算子耗时，也不把不同 core/pipe 的时间直接相加。
- tool=0、文件存在、MSTX ID 非零都不是完整采集证明。核对目标命中、所选指标、芯片支持、采集范围与退出/receipt；MSTX 接口可能捕获内部异常并返回无效 ID，helper 会拒绝无效 start ID，结束和数据完整性仍需看日志与产物。

## Python：原生 profiler 采集

仅当调用映射或基础指标确有缺口时使用本节，不作为 msprof op 的固定前置步骤。配套 [torch_npu_diagnostics.py](../scripts/torch_npu_diagnostics.py) 的 `capture_diagnostics` 可直接放入设备 worker 使用；只在实际执行 NPU kernel 的进程运行。以下 `call/prepare/synchronize/check_reservation` 由项目已验证的执行入口提供，不是预装的统一算子 API：

```python
from torch_npu_diagnostics import capture_diagnostics

report = capture_diagnostics(
    call, output=run_dir / "diagnostics" / case_id / "native",
    identity={"candidate": candidate_digest, "case": case_digest,
              "build": build_digest, "environment": environment_revision},
    metrics=("PipeUtilization", "Memory"),
    prepare=prepare, synchronize=synchronize,
    check_reservation=check_reservation, warmup=warmup,
)
```

- `prepare()` 在窗口外完成 JIT/分配与输入/输出重置；随机与原地更新每次恢复相同起点。`call()` 包括该算子全部必需阶段，返回对象保留至同步结束。`synchronize()` 覆盖实际参与的设备/stream。
- 每个指标用独立 pass，在一次完整调用外包 profile，使用 `schedule(wait=0,warmup=0,active=1,repeat=1,skip_first=0)` 和一次 `prof.step()`。`tensorboard_trace_handler(..., analyse_flag=True, async_mode=False)` 同步解析，`data_simplification=False` 保留数据。具体接口见 [profiler 实现](https://github.com/Ascend/pytorch/blob/master/torch_npu/profiler/profiler.py)。
- helper 拒绝混入 msprof 专属指标，检查安装版本暴露的枚举，保存 identity、指标、版本与各文件 SHA256。失败保存 collection.json 并传播异常；已有文件不覆盖。`pending_review` 仅表示有文件生成，需核对指标列、目标 kernel、阶段和设备覆盖，不能直接写 ready。
- 不同 pass 是不同次执行，不能把计数拼接成同一时刻的并发流水；基础 PipeUtilization 只能给出忙碌比例，不能独立证明 MTE2/V/MTE3 的重叠关系。

## 交付及 analysis 的使用

每个待交付候选形成 [ANALYSIS_DATA.md](../assets/templates/analysis-data.md)，链接原始采集、全部指标表/trace/visualize_data.bin、命令日志和 SHA256 清单。二进制保留供匹配版本的官方工具/MindStudio Insight 打开；若当前 agent 无法解析，注明限制并提供可读 CSV/JSON 及观察，不虚构私有格式解析器。指标字段含义参照本次版本的 [msOpProf 数据说明](https://github.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_performance_data.md)。

analysis 先核对身份与范围，再看流水空隙/重叠、核间负载、内存流量/带宽及 Roofline 位置，将观察映射到实际实现，提出可检验假设。Occupancy 在该工具中是核间负载分析，不应直接套用其他硬件的 warp occupancy 含义；Roofline 是模型边界，不能代替实测或自动成为验收标杆。采样 core 数不等于启动核数，PipeTimeline 不是逐指令完整轨迹。

必需数据缺失时原 port-baseline/develop 保留未验收检查点，在原预算内修复采集/适配；超出权限或能力时报告具体缺口，不能默降必需项。明确未要求的可选项可记录 partial/unsupported，但不因此抹去独立有效的精度/性能结果。诊断数据用于实现/思路归因；精度失败和实现问题仍在原实现节点内修复，不能借“交付分析数据”跳过上一轮约定。
