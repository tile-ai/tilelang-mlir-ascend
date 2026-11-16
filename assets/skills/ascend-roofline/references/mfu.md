# MFU：通用有效计算利用率

MFU（Model FLOPs Utilization）的核心是有效浮点工作量除以同范围时间和匹配峰值。
本skill明确采用其算子级口径，**Cube算子和Vector算子都可计算**，不把“只算矩阵乘”写成通用定义。
模型端到端MFU与算子级MFU的区别是工作量和时间范围，不是换一个公式。

## 公式与分母

```text
P_useful = F_useful / T
MFU = F_useful / (T * P_peak)
U_executed = F_executed / (T * P_peak)
```

百分比再乘100；TFLOPS中的T为10^12。F_useful排除padding、无效mask、冗余及重计算；
F_executed包含实际执行的这部分工作。U_executed不等于有效MFU，也不等于所有硬件资源的总体利用率。
运算量由选定算法和计数约定确定，不用增加冗余F来“提高MFU”。

| 分析范围 | 分子 | 匹配峰值 |
|---|---|---|
| Cube计算 | 有效Cube FLOPs | 对应实际乘数/累加路径的Cube峰值 |
| Vector计算 | 有效Vector浮点FLOPs | 对应dtype、SIMD/SIMT与指令类别的Vector峰值 |
| Cube+Vector融合kernel | 分别列F_cube和F_vector | 分别报告两个MFU，分母时间均为完整kernel时间 |
| 模型端到端 | 预先约定的有效模型工作量 | 同设备、精度和资源范围的匹配峰值；异构工作需分项建模 |

Cube-only是选择的一种统计范围，不是MFU的定义。融合算子仅报告Cube MFU时不把Vector工作加进其分子；
若要统计Vector贡献另列一项。不能把两个MFU直接相加/平均，也不能默认用`F_total/(T*(P_cube+P_vector))`。
多精度、不同指令吞吐或串行阶段的综合指标先定义调度/重叠模型，见 [roofline.md](roofline.md)。
纯整数工作量报告TOPS与整数利用率，不称浮点MFU；纯copy没有有效浮点计算，MFU记不适用，仍可计算BWU。

峰值见 [peaks.md](peaks.md)，物理资源与频率见 [hardware.md](hardware.md)。
单芯片/分区/多卡范围保持一致；同频同构多卡可合计有效工作量和各卡匹配峰值，T取共同墙钟范围，不能累加并行卡时间。
指定参与核归一化时说明资源范围；logical block dim不等于物理核数，忙碌率不能再乘进峰值降低分母。

## 有效运算计数

数值公式、计数规则与算子→分母映射统一在 [counts.md](counts.md)；本节只规定 MFU 的分子范围。

- 只有与所选峰值同口径的运算能进分子：FMA 计 2、add/mul 各 1；比较、选择、转换、整数地址计算和搬运单列，不自动加入浮点 MFU。
- `exp/sqrt/rsqrt/div` 等特殊操作按每元素 1 次语义操作单列并报 GOP/s，这不意味着与 FMA 同吞吐。
  工具加权 OPS 须保留定义且峰值匹配，否则只报该操作吞吐，不报误导性的 MFU。
- 分子用 `F_useful`：扣除 padding、越界 tile、被跳过的 mask 和重计算；普通数值恰为零不等于掩码。
- `valid_pairs/valid_macs` 均按单个 batch 实例输入，各实例不同则拆条目；`valid_macs` 用于沿归约维变化的掩码。
- causal Attention 的 QK 可数有效输出 P，PV 的 mask 在输入 P 的归约维，不能把 PV 输出直接套 causal mask。
  前向有效 Cube 量为 `2*B*H*P*(Dqk+Dv)`，Vector softmax 另计。
- Vector 示例（`y=x+z` 为 N、`y=a*x+b` 为 2N、直接求和为 `N-1` 次加法）见 [counts.md](counts.md) 的计数规则表。

## 时间与解释

T覆盖分子对应的完整任务。算子MFU用完整kernel设备耗时，包括搬运、Vector/Cube等待和同步，
不用某条流水的active time，也不跨并行核加时间。计时协议固定预热、重复和统计方式；不要混用host墙钟与kernel计时。
端到端MFU包括其约定范围的通信、调度与空泡时间；排除与包含项随报告说明。
模拟器结果注明模拟与频率假设。超过100%保留并审计，不截断。

低MFU本身不说明代码差：在低AI情景下，带宽屋顶可使理论MFU上限远低于100%。
应同时看BWU、理论时延与实测时延；MFU和BWU对两类引擎均有意义。

## 计算脚本

Python3.9+标准库。`--time`为`--iterations`次完整工作负载的总时间；平均耗时保持iterations=1。
`--useful-ops`为每次迭代有效工作量；`--engine`默认cube。

```bash
# Vector通用计数，构造示例：23M有效FLOPs/2us，对标23TFLOPS，MFU=50%
python <skill-dir>/scripts/calculate_mfu.py --engine vector --device 950pr --device-cores 28 --cores 56 --dtype fp32 --useful-ops 23000000 --time 2 --unit us --json
# 明确匹配指令/资源的峰值；数值仅演示参数，使用时替换来源与范围
python <skill-dir>/scripts/calculate_mfu.py --engine vector --dtype fp32 --useful-ops 1000000 --peak-tops 10 --peak-source 'synthetic matched FP32 resource' --time 1 --unit us --json
# 保留GEMM有效量快捷计数与旧命令兼容
python <skill-dir>/scripts/calculate_mfu.py --device 950pr --device-cores 32 --cores 16 --dtype bf16 --m 4096 --n 4096 --k 4096 --time 1 --unit ms --json
```

目录查表参数`device_cores`始终为该配置AIC数，`cores`按engine分别指AIC/AIV。
显式`--peak-tops`与目录查表参数互斥，且要求`--peak-source`。
`--list-devices`输出Cube/Vector目录；峰值未知时不编造值，保留公式与实际吞吐。

GEMM快捷模式仅用于Cube：mask三选一 `--mask causal [--causal-offset N]`、`--valid-pairs P`、`--valid-macs Q`。
多个GEMM可用`--gemms-json`输入数组，每项支持`name,m,n,k,batch,mask,causal_offset,valid_pairs,valid_macs`；
共享同一dtype，数组只列一次迭代的工作，不与`--useful-ops`混用。
例如L=4、Dqk=2、Dv=3的causal Attention：

```json
[
  {"name":"QK","m":4,"n":4,"k":2,"mask":"causal"},
  {"name":"PV","m":4,"n":3,"k":4,"valid_macs":30}
]
```

P=10，QK=40 FLOPs、PV=60 FLOPs，合计100有效Cube FLOPs。该例不包含Vector softmax。
输出保存工作量、时间范围、操作口径、所选峰值及来源；数据不足时先给可确定的部分。
