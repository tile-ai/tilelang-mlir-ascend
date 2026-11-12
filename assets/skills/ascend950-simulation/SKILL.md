---
name: ascend950-simulation
description: 使用 msprof op simulator 仿真 Ascend950 kernel，校验输出、分析指令周期与流水重叠，并从同次 dump 补齐 GET/RLS，导出 MindStudio Insight trace。npusim 用于备用采集和交叉验证。
---

# Ascend950 仿真

默认流程：TileLang 生成 `.asc` → Bisheng 编译 → CANN runtime launcher →
`msprof op simulator` → 校验输出与采集 → 增强合并 trace → Insight 验证。
AutoSchedule 的 latency/II 成本估计不会自动运行仿真。

## 接入 Harness

本 Skill 提供 Ascend950 仿真与 trace 工具方法。在 harness 中由
[harness-analysis](../harness-analysis/SKILL.md) 按需使用，执行前读取 TASK 固定的
simulation 环境约定、候选身份、case、资源限制与预算。缺少适用的仿真绑定时，在 RESULT 记录缺口。
已有 trace 可离线处理，并保留其原始采集身份。

脚本路径相对于本 Skill 安装目录，使用 Python 3 标准库即可处理已有采集。
本包不包含 Bisheng、CANN、simulator、npusim 或 Insight 本体；运行或安装这些工具
遵循任务已有环境约定。来源与许可见 [provenance](references/provenance.md)。

## 编译与运行环境

- 使用匹配版本的 Bisheng、CANN runtime、simulator 和 msprof，加载 `set_env.sh`，
  检查 `bisheng --version`、`msprof op simulator --help` 和实际目标 arch/SoC。
  按[官方仿真指南](https://github.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_simulator_user_guide.md)
  从安装目录 `tools/simulator` 选择 SoC；Ascend950 通过 `--soc-version` 显式指定。
- 复用现有编译入口，保存 `.asc`、编译命令和二进制哈希。原生 ACL launcher 不依赖
  torch_npu；Python/JIT 路径需单独确认解释器、模块和动态库兼容性。
- 按生成源码的实际 ABI 传参；mixed 保留 `__mix__(1, 2)`，确认 AIC、AIV0、AIV1
  均有目标工作。隔离操作时保留布局、内存空间、别名、同步、VF mask/线程和循环。
- 输入和 CPU reference 在采集外准备，launcher 启动后同步 stream 并校验输出。
  周期实验的输入应能检验结果和 buffer 版本，避免额外 NPU 初始化 kernel 混入采集。
- 初始化失败时区分 launcher 与 profiler 子进程环境，避免开发 stub 抢占真实 runtime。
  源码映射需调试信息；缺少 debug_line 时不宣称有源码调用栈。

## 采集与导出

每个 kernel/方案使用新目录。先从 simulation 环境约定设置 `SIM_SOC` 和
`SIM_FREQUENCY_GHZ`。以下按官方文档使用 `--dump=on`，具体参数以安装版本帮助为准：

```bash
msprof op simulator --soc-version="$SIM_SOC" \
  --launch-count=1 --aic-metrics=PipeUtilization,ResourceConflictRatio \
  --dump=on --output=/path/to/new-profile /absolute/path/to/acl_launcher

python /path/to/ascend950-simulation/scripts/enhance_msprof_trace.py \
  /path/to/OPPROF_SINGLE_LAUNCH /path/to/new-enhanced/trace.json \
  --frequency-ghz "$SIM_FREQUENCY_GHZ"
```

- 同步分析需要 `ResourceConflictRatio`；仅开 `PipeUtilization` 会关闭同步细节。
  记录目标身份、launch/block 数、工具版本、输出误差和日志；退出成功不证明采集完整。
- 默认覆盖全部参与核。指定 `--core-id` 或发生超时截断时，明确报告范围和完整性限制。
- 输入根目录须包含同一 launch 的 `dump/` 与顶层 `simulator/trace.json`。
  单核 trace 不能代替跨 AIC/AIV 合并版；不同采集、launch 或方案之间不拼接 tick。
- 确认 simulator 时钟域和频率，不使用 host CPU 频率或其他模型的默认值。
  原生 JSON 时间为微秒，不能根据 `displayTimeUnit` 推断 `ts/dur` 单位。

增强脚本保留全部原生事件，另存 `trace.json` 与 `trace.validation.json`：

- 按 core/subcore、PC、名称、completion tick 的**多重集合**对照 dump；B/E 算一条。
  ID 可跨指令流碰撞，重复 PC 不去重；总数相等也可能掩盖名称组成不同。
- 除 GET/RLS 外出现遗漏、额外记录、时钟不匹配或未知 dump 格式时停止，不猜补。
  原生已含 GET/RLS 时先审计版本，避免重复添加；输出文件已存在时使用新目录。
- 覆盖校验仅针对 trace 中的核，检查 `excluded_dump_cores`。这是相对于 completion
  dump 的覆盖结论；还须检查日志与 dispatch dump，不能据此证明 dump 自身没有缺失。
- GET/RLS 补入原始 completion tick、PC、ID、文件和行号；未知 dispatch 不虚构。
  按 core、目标 PIPE、buf_id、buf_mode、buf_cnt epoch 配对，缺字段、重复或倒序报告 `unlinked`。
- 保留原名 GET/RLS，语义耗时为零、显示宽度 **1 tick**；另加明确标记的 1 tick
  SET/WAIT 锚点供 Insight 连线，锚点不计入真实指令数或性能统计。
- intra-block 使用原生 B/E、等待区间、ID、目标 PIPE 和完成点连线，不增加兼容事件，
  不重新配对或映射显示轨道。WAIT 可以早于 SET 开始阻塞。

导入 Insight 时读取 [查看器兼容与验证](references/insight.md)，核对实际解析与端点，
不要把 JSON 配对成功等同于 GUI 可点击。npusim 微基准、SQLite 查询、VF IPC 或独立
交叉验证见 [备用流程](references/npusim-fallback.md)。

## 分析与交付

- dispatch→completion 可能包含排队和依赖等待，不直接当作本征执行时间。
  原生同步长区间用于定位阻塞；GET/RLS 显示宽度不计入耗时。
- kernel 区间用 `max(start + duration) - min(start)`，不累加重叠区间；保留 cycles
  和频率依据，host 墙钟时间不能当作设备延迟。
- 保留多迭代 buffer 版本和生产/消费依赖，分析填充、稳态、排空及等待。
  `num_stages=2` 不证明重叠；方案比较须同工作量/输入并分别校验输出，稳态 II 用
  足够迭代的同类指令间距估计，并改变迭代数验证，不能仅凭利用率声称加速。
- 交付 shape/dtype/布局、ABI/launch/core 范围、命令/版本/哈希、正确性、原生目录、
  增强 trace、覆盖与未连接报告、周期及测量边界。原生 `visualize_data.bin` 不包含
  脚本补入的 GET/RLS，不能作为增强版的等价交付。
- **仿真测试回复须披露 GET/RLS 的零耗时、1 tick 显示和额外 1 tick SET/WAIT 锚点，
  报告实际锚点数并说明不计入真实指令/性能。** 同时说明是否复用采集、是否重跑仿真，
  分别报告正确性、采集覆盖、Insight 解析/接口/GUI 和真机验证状态及量化限制。

TileLang 操作隔离和 II 测量的细节按需读取任务选定 TileLang checkout 中
`.agents/skills/tilelang-ascend/references/profiling.md` 的 Operation cycle measurements。
这是可选的目标版本资料，不要求安装到 harness 的宿主发现目录；文件缺失时查询
同版本的 profiling 源码或文档，不将另一 checkout 的命令直接用到当前环境。
