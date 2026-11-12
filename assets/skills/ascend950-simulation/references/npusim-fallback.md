# npusim 备用路径

用于微基准、原始 SQLite 查询、VF IPC 或与 msprof 独立交叉验证。
默认 Insight 指令详情交付按主 SKILL 使用 msprof。

录制方法参考官方 [TileLang profiling 指南](https://github.com/tile-ai/tilelang/blob/main/.agents/skills/tilelang-ascend/references/profiling.md)。
从当前 `npusim record --help` 确认模型名称并设置 `NPUSIM_SOC`，其他选项按安装版本核对。

```bash
npusim record -s "$NPUSIM_SOC" -o ./sim-new -g -n all /absolute/path/to/acl_launcher
# Python/JIT 使用已验证解释器；引号内的完整调用按当前 npusim 帮助确定。
npusim record -s "$NPUSIM_SOC" -o ./sim-python-new -g -n all \
  '/absolute/path/to/python /absolute/path/to/kernel_launcher.py'
```

`-n` 控制日志核范围，launch 核数由 launcher 决定；只有 core 0 时明确局部范围。
需要 VF IPC 或重生成报告时，按安装版本帮助选择入口，复用原 archive 并保留原始采集。

检查实际单核报告与 `trace_allchips.json`，顶层 index 警告不能代替文件检查。
采集器提供 SQLite 时，可以使用 harness 的辅助导出器；先核对下述 schema，不将它视为所有官方版本的固定格式：

```bash
mkdir -p ./new-sqlite-detail
python /path/to/ascend950-simulation/scripts/export_pipe_trace.py \
  /path/to/npusim_ARCHIVE/record/kernel.db ./new-sqlite-detail/trace.json --layout pipe
```

- 导出器要求 `ExecutedInstructions` 含 start/end tick，时钟来自
  `KernelInfo.frequency_ghz`。按 core/硬件 PIPE 共轨，将 RVEC 子类型合到
  VECTOR；`--layout slots` 只是虚拟显示分槽，`--overview` 仅按需另存区间并集。
- 检查采集器的覆盖、未配对、无效记录及写入失败统计；SQLite 导出不能恢复缺失事件。
- 仅有 completion 的记录应逐条定位。只有证明为零耗时 GET/RLS 后，才与计算/搬运缺口区分；
  保留原始配对指标，不把“数量恰好等于 GET/RLS 数”当作身份验证。
- 采集器若将同步记录归一化为 1 tick，不能由此读取真实等待区间；需要等待区间时
  使用 msprof 的 ResourceConflictRatio 采集，并核对具体事件语义。
- 旧 SQLite 导出器的 intra-block 仍为 X slice，不能据 JSON 配对数宣称旧 Insight
  可点击关联；需要完整等待区间和经验证连线时使用主流程的 msprof 合并版。
- 若出现 `instr.bin`/`chip*_instr.bin` 而非 SQLite，使用同版 decoder，先验证字段
  和单位，按同版本实现确认 `is_popped` 的 dispatch/completion 含义。
- 核对目标身份、输入输出、launch/chip/core、ID/PC 和来源，独立运行间不要对齐或
  拼接绝对 tick。正确性、采集完整性、仿真周期和真机验证状态分别报告。
