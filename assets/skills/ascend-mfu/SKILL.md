---
name: ascend-mfu
description: 计算昇腾 GEMM、Attention 或融合算子的 Cube-only MFU，按有效运算量扣除 mask/padding，并依据官方设备、精度和参与 Cube 核数选择峰值算力。支持按分层搬运量与共享带宽计算条件性 Roofline、理论 MFU 上限；包含算力表和计算脚本。
---

# Ascend Cube MFU

由个人 `ascend-mfu` Skill 迁入，作为独立扩展维护；硬件数值引用官方资料，
计算方法与脚本不属于官方 TileLang 原生 Skill 快照。

只统计 Cube 运算。使用 [计算脚本](scripts/calculate_mfu.py)，Python 3.9+，无需第三方依赖。
先读 [官方 Cube 算力表](references/cube-peaks.md) 核实设备版本、实际 Cube 指令精度及总核数；脚本拒绝未核实的组合。

## 计算口径

`MFU = useful_cube_ops / (elapsed_seconds × selected_cube_peak_ops_per_second)`。
百分比再乘 100；TFLOPS 的 T 为十进制 `10^12`。

- 无掩码 GEMM：`2 × batch × M × N × K`；一次乘加按 2 次运算。
  M/N/K 使用语义上的真实尺寸，排除 padding、越界 tile、重复计算和重计算开销。
- 有效输出位置数为 P，且各位置完整归约 K 时：`2 × batch × P × K`。
  causal 对每个矩阵按 `j <= i + offset` 精确计数，默认 offset=0（含对角线）。
  方阵默认 P=`L(L+1)/2`，不要直接用总 FLOPs 除以 2。
  非方阵需确认掩码对齐：右下对齐通常 offset=`N-M`，脚本不替用户猜测。
- 输入掩码或沿 K 变化的掩码，应数有效 `(i,j,k)` 乘加三元组，用 `valid_macs`。
  `valid_pairs` 与 `valid_macs` 均为**单个 batch 实例**的数量，脚本再乘 batch。
  各实例有效数不同则拆成多个 GEMM 条目。普通数值恰好为零不自动当作 mask。
- 即使 kernel 实际执行了被掩码的部分，这些部分也不进入 MFU 分子。
  Vector/SIMD/SIMT 的 softmax、激活、归约、格式转换、地址计算、搬运不计入分子；
  不加 Vector 峰值，不采用 `6 × 参数量` 等粗略模型运算量替代 GEMM 清单。
- 分母用 `设备版本 Cube 峰值 × 参与 Cube 核数 / 该版本总 Cube 核数`。
  核数指给该测量任务使用的物理 AIC 核，不能用 AIV 数、逻辑 block 数或忙碌率二次缩放。
  若多个阶段使用不同核集合，按同一测量范围明确资源口径，或分段计时计算；不直接平均各阶段 MFU。
- 时间必须覆盖所列 GEMM 的同一执行范围。融合算子使用完整 kernel 延迟（包括其中的
  Vector、等待和搬运时间），不能仅累加 Cube 指令活动时间，不能跨并行核累加耗时。
  使用任务约定的设备计时并排除 warmup、编译和无关 kernel；采用 profiling 协议时从原始 kernel
  记录按约定统计，不使用 host 墙钟。模型端到端时间则对应端到端 Cube 清单。
  模拟器时间需标明 simulated 及频率/设备假设，不当成实机测量。
- `--time` 是 `--iterations` 次完整工作负载的总时间；若输入已是平均延迟，保持 iterations=1。
  超过 100% 保留数值并检查口径，不截断。INT8 输出整数运算利用率，`mfu_percent=null`，避免称为浮点 MFU。

## 调用

纯 Vector/SIMD/SIMT 算子没有可计入的 Cube 工作量，Cube MFU 标为不适用；
仍按 kernel 耗时比较性能，不用 Cube 峰值归一化 Vector 运算。

将 `<skill-dir>` 替换为本 skill 的实际绝对路径。单设备、统一 Cube 精度；多卡先按各设备分解，
混合精度不可套一个峰值，须分段测量或另行报告基于各精度理想时间求和的口径。

```bash
python <skill-dir>/scripts/calculate_mfu.py --list-devices
python <skill-dir>/scripts/calculate_mfu.py --device 950pr --device-cores 32 --cores 16 --dtype bf16 --m 4096 --n 4096 --k 4096 --time 1 --unit ms
python <skill-dir>/scripts/calculate_mfu.py --device 950dt --device-cores 36 --cores 36 --dtype fp16 --m 4096 --n 4096 --k 128 --batch 32 --mask causal --time 0.5 --unit ms --json
```

掩码参数三选一：`--mask causal [--causal-offset N]`、`--valid-pairs P`、`--valid-macs Q`。
不传则 dense。`--batch` 默认 1。掩码数量为 0 合法，负数或超出尺寸范围会报错。

多个 GEMM：保存 JSON 数组，再传 `--gemms-json /absolute/path/gemms.json`。
每项支持 `name,m,n,k,batch,mask,causal_offset,valid_pairs,valid_macs`，且共享 CLI dtype。
只把一次迭代的运算列入数组，重复测量次数交给 `--iterations`。

例如单头 causal attention，序列长度 4，Q/K 维度 2，V 维度 3；有效 attention 位置 P=10：

```json
[
  {"name": "QK", "m": 4, "n": 4, "k": 2, "mask": "causal"},
  {"name": "PV", "m": 4, "n": 3, "k": 4, "valid_macs": 30}
]
```

QK 为 `2×10×2=40`，PV 为 `2×10×3=60`，合计 100 次有效 Cube 运算。
PV 的 mask 位于输入 P 的归约维，不能给 PV 的输出矩阵简单套 causal mask。
普通注意力前向总量为 `2×B×H×P×(Dqk+Dv)`，不含 softmax；变长/GQA/反向则按实际 GEMM 展开。

报告设备和核数、指令精度、有效运算量及扣除量、时间范围与单位、Cube 峰值和来源、MFU。
用 `--json` 保留完整计算依据。硬件信息不足时先从当前设备/runtime/profiler 查询，不能仅凭“950”选最高档；
有多个可行结果且无证据确定时再向用户询问。

## Roofline 与理论 MFU 上限

用户询问带宽瓶颈、理论最高 MFU 或 roofline 时，读取 [Roofline 指南](references/roofline.md)，
使用 [roofline 计算脚本](scripts/calculate_roofline.py)。保留有效与实际执行两套 Cube FLOPs；
按存储层和共享资源分别计量，区分 GM 请求与实际主存流量。默认按流量/带宽理想重叠计算，
不把实验性 buffer 布局或同步等待作为固定约束。没有成功运行时间时只给条件性上限，不报告实测 MFU。

带宽层级优先服从用户指定情景：要求 L2 roofline 时显式建立 L2 约束，不默认将全部 GM 请求计为主存流量。L2 miss、bypass 和写回按对应主存流量单独约束；读取带宽不直接当作混合读写总带宽。
