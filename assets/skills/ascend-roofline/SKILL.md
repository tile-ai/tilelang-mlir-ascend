---
name: ascend-roofline
description: 为昇腾 Cube、Vector/SIMD/SIMT 和融合算子建立分层 Roofline 模型，计算有效算力利用率 MFU 与带宽利用率 BWU，并用源码、生成代码及 msprof 数据验证性能瓶颈。适用于理论上限估计、CSV 指标解释和优化方向分析。
---

# Ascend Roofline

用同一测量范围的运算量、各层字节数和时间，解释算子离哪一个资源上限还有多远。
采用统一的Roofline、MFU和BWU定义。Cube/Vector是选择资源与峰值时的区分，不是两套分析方法。
本skill是个人扩展（由`ascend-mfu`整合而来），计算口径与脚本不属于官方TileLang原生Skill快照；
硬件数值引用官方资料，并标注来源与核对日期。

## 按问题读取

| Reference | 职责 |
|---|---|
| [peaks.md](references/peaks.md) | Cube与Vector标称计算峰值、精度/核数选择；不混入访存参数 |
| [counts.md](references/counts.md) | 运算量计数规则、常见场景公式、算子→分母映射与工具对账 |
| [roofline.md](references/roofline.md) | 统一建模与瓶颈分析框架：F/D、缓存情景、分资源下限、拐点、实测验证及脚本输入 |
| [mfu.md](references/mfu.md) | 通用有效计算利用率；Cube/Vector计数、融合范围、GEMM/Attention快捷计算 |
| [bwu.md](references/bwu.md) | 通用带宽利用率；各层/方向流量、CSV、task/active时间和cache命中率 |
| [hardware.md](references/hardware.md) | 950PR/DT容量、频率、主存规格、CANN配置与未知硬件数据 |

先从问题对应的reference进入；完整分析按硬件身份/峰值→按[counts](references/counts.md)计数F与各层D→资源下限→MFU/BWU实测证据展开。
JSON目录与构造示例位于`assets/`，计算工具位于`scripts/`；避免将数据文件当成第二套概念说明。
`calculate_mfu.py`和`calculate_roofline.py`均支持Cube/Vector，`calculate_bwu.py`按资源边界计算，与引擎类别无关。

脚本使用 Python 3.9+ 标准库。`<skill-dir>` 指当前 skill 的实际路径，不依赖作者机器环境。

## 共同口径

先确定 dtype/指令路径、物理资源范围、测量边界，以及流量是源码推导还是 PMU 实测。
用户指定 L2、warm cache 或当前实现时保留该情景；资料不足可给符号表达式与条件性结果。

- **语义工作量 `F_useful`** 用于比较完成同一任务的效率；**实际执行量 `F_executed`** 用于解释机器工作，包括 padding、mask 未跳过、重计算。二者分列，不用冗余计算抬高 MFU。
- `D_r` 是穿过资源 r 边界的字节，不是把所有存储层加总。同一 FLOP 分子可对应 HBM、L2、UB 各自不同的 AI。
- 同一芯片并不保证所有指令共用一个屋顶。峰值匹配执行单元、精度、指令类别、核数和频率；FMA 峰值不能默认用于 exp/rsqrt/sqrt。
- AI 与拐点比较给出**模型限制项**，不是实测瓶颈的充分证据。结合 BWU、计算吞吐、active/task 时间、等待和对照实验判断。
- `Performance Ratio` 只有在同口径带宽斜线段才等于该层 BWU；在水平段是相应计数口径的计算利用率，不自动等于有效 MFU。
- CSV 的 `NA` 不等于0；`KB`/`GB/s` 标签不能替代单位核验；不同 launch、总计行与核行不混加。

## 交付

按问题给出可复算的 F、D、T、峰值及来源，报告实际带宽、实际性能、分层 AI、MFU/BWU 中可确定的项。
保留脚本 `--json`/`--output` 的完整输出（含所选峰值及来源）作为可复算依据。
分别标明理论情景、当前实现预测和实测结果。瓶颈结论写出证据、尚未覆盖的资源及下一步能区分原因的实验。
未知峰值或缺少计数器不妨碍输出已有结果，但不能用猜测数值宣布“打满”。
实际设备档位、CANN/msprof 版本、生成代码与采集产物留在用户项目记录；不要写成 skill 的通用硬件默认值。
