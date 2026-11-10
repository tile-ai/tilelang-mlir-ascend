# Cube-only Roofline

适用于源码级或实测流量级的带宽/计算上限分析。默认只由流量、带宽和 Cube 运算量建立理想重叠模型。
若用户只要求流量 roofline，不加入实验性的单缓冲、布局或同步等待约束。用户要求当前实现的可达性能分析时，才另加有证据的依赖链、Vector 或启动/排空限制，并区分两种结果。

## 运算量与公式

分别准备 `F_useful`（语义有效 Cube FLOPs）和 `F_executed`（实际 dense/padded/重复执行 FLOPs）。
前者扣除 mask、padding，后者据当前实现计数。mask 值为零不表示跳过了 DMA 或 MMAD。
如果尚未编译，标记源码推导；不得将猜测的 L0 复用/拆分量当作生成代码证据。

令 P 为所选设备、精度及参与核数的 Cube 峰值，资源 r 的需求为 D_r，带宽为 B_r：

- `T_floor = max(F_executed/P, D_1/B_1, D_2/B_2, ...)`
- `MFU_ceiling = F_useful/(P*T_floor)`，百分比乘100。
- 纯计算有效 MFU 屋顶为 `F_useful/F_executed`，不一定为100%。
- 以 `AI_useful=F_useful/D` 画单层 roofline：`P_useful <= min(B*AI_useful, P*F_useful/F_executed)`。
  通用无浪费折点为 P/B；固定有效比例下的折点为 `(P*F_useful/F_executed)/B`。

Vector 算术不计 MFU 分子，但可限制真实执行速度。只算搬运/Cube 时明确这些约束未覆盖。
没有成功运行时间时，只给条件性上限，实测 MFU 留空；编译失败不妨碍源码级估算。
不从同一次运行的 `bytes/time` 反推带宽再证明上限，避免循环论证。

## 按资源边界计流量

先列路径、读/写方向、dtype、尺寸、重复次数、复用范围、核/双 AIV 范围，再按共享资源归组。

- GM→UB、GM→L1 和 L0C/UB→GM 分属不同流水，可以重叠，但访问主存时共享芯片总内存预算。
  主存模型合计真正穿过该层的读+写；多物理通道已包含在整芯片带宽中，不能每条 PIPE 各领一份整芯片带宽。
- UB→L1、L1→L0A/B、L0C→UB 不加入主存字节。相同数据经过不同层，在各层各计一次是合理的，不能将所有层相加后除以主存带宽。
- 共享资源的流量先相加；独立资源的服务时间取 max。同一请求可以同时受端点和共享下游限制。
  只有证据支持共享总预算时才合并；若资源是已核实的独立双向通路，则分开约束，不凭箭头方向猜测。
- 明确 Q/O 是否包含 padded 行，host 切片不减少 kernel 的写量；检查索引/掩码是否按 head、task、双 AIV 重读。
  gather 合并 DMA 可以减少指令数，但不必然减少 payload；按实际复用次数计算，不默认每个 tile 重读所有 B。
- 逻辑 payload 不等于总线事务量：小访问、stride、cache line、bank 冲突会改变真实流量/有效带宽。
  使用事务量或实测有效带宽时标明口径，避免重复惩罚同一效率损失。

GM 是地址空间，不能仅凭 GM 指针就断言经过主存。区分三个情景：
1. 当前请求全部到达主存：明确是 cold/no-cache 假设，不等于测得的主存流量。
2. 各层实测流量：按同次测量的主存/L2等计数和匹配的带宽建模。
3. 新实现的理想最少流量：明确是重新设计假设，不冒充当前实现。

预热、重复地址、未清缓存时尤其不能将情景1作为唯一真实上限。容量足够不证明命中；
只去掉主存项、遗漏 L2 限制得到的是宽松界，不是 warm-cache 性能预测。

## 设备与带宽证据

Cube 峰值复用 [cube-peaks.md](cube-peaks.md)，不使用 Cube+Vector 总和。
内存带宽不能机械按参与 Cube 核数缩放；片上单核速率则按实际参与资源数聚合。
区分每 AIV、每双 AIV、每 AIC 和整芯片速率，避免两路 AIV 再乘2。
`bytes/cycle × frequency_GHz × resource_count` 得到十进制 GB/s；GB/s=10^9 B/s，TB/s=10^12 B/s。

每条带宽附官方规格、独立匹配微基准或假设来源。官方峰值、可持续实测值、仓库成本参数不能混称硬件峰值。
核实产品档位、精度、频率、核数和方向/布局；某一路径的带宽不能直接移用到另一目的存储层。
未知值给符号/敏感性或明确假设，不为填满表格编造数值。

设备规格可从 [产品页](https://www.hiascend.com/hardware/processor) 和
[950架构白皮书](https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/昇腾950%20NPU架构白皮书.pdf)表3-1 查询；脚本不提供默认带宽。
[官方搬运理论表](https://asc.gitcode.com/api/appendix/cube_instruction_theoretical_perf_summary.html)可核实具体3510路径；
[共享GM带宽说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha001/opdevg/ascendcbestP/atlas_ascendc_best_practices_10_0009.html)说明合并读写需求原则，不替代950各路径的具体规格。

## 计算脚本

Python3.9+，标准库；输入均为一次完整 kernel 工作量，不跨并行核累加时间。

```bash
python <skill-dir>/scripts/calculate_roofline.py --input <skill-dir>/references/roofline-example.json --output result.json
```

JSON 字段：
- `device, device_cores, cores, dtype`：同 MFU 脚本，复用算力表；此脚本仅报告浮点 MFU。
- `useful_ops, executed_ops`：非负整数且 executed_ops>0，useful<=executed；必须分别推导。
- `resources`：约束数组，可为空以只显示计算屋顶。每项含唯一 `name`（不能为cube）、
  `payload_bytes`（流量名称→字节数的对象）、`bandwidth_gbps`（此资源整体十进制GB/s）、
  `basis`（official/measured/assumed）、`source`（来源与适用条件文本）。脚本合计每项payload，资源之间取max。
  同一数据若经过多个资源，可在多个条目出现；不要在一个资源内重复计数同一次传输。
- `assumptions`：非空字符串数组，写清缓存、流量来源、带宽和重叠范围；`label`可选。
- `measured_time_us`：可选，独立测得的完整kernel时间。输出单独报告实测 MFU；比模型下限更快时警示检查情景，不截断结果。

输入示例为构造的 BF16 GEMM：真实 M=120、N=128、K=64，执行时 M 补到128，
输出为 FP32。假设每个 padded 输入读取一次、完整 padded 输出写入一次，全部访问主存，
共享读写带宽假设为1000 GB/s。它只演示输入格式与计算口径，不是设备性能记录或默认带宽。

## 输出与画图

交付设备/核数/精度、有效与执行量、逐路径字节与复用、带宽及证据、各资源下限与瓶颈、条件性MFU屋顶。
实测结果独立标记；只有源码推导时披露编译/校验未完成。
按需用标准绘图库输出PNG/SVG，横轴说明有效还是执行FLOP/Byte，纵轴与之匹配，标出缓存条件和假设带宽。
同图比较不同设备时保留各自带宽/峰值，不把不同层流量混成一个AI；静态科学图优先双对数roofline。


## L2 驻留分析

当用户指定 L2 带宽模型时，以 L2 服务的流量作为该层需求，不继续套用全主存假设。
计算 MFU 的实测公式不变；调整的是 `T_floor` 中流量对应的带宽层级。
至少区分 L2读取、L2写入、bypass/miss/写回的主存服务。缺少写通路或混合带宽时明确未覆盖，
不得将官方纯读峰值无条件应用于读写合计。缓存全命中是可选情景，不是所有算子的默认事实。

可参考目标版本的 [官方 CANN Matmul 样例](https://gitcode.com/cann/asc-devkit/blob/master/examples/01_simd_cpp_api/05_best_practices/01_matrix_compute/matmul_basic_api_high_performance/README.md)
查找读取带宽依据；核对具体设备和频率，不能自动应用于另一型号或写入/混合访问。
首次填充和后续命中不能混为一层重复读取；
需要启动时序时另加依赖，否则流量roofline按各资源服务时间取max。

在 `resources` 中分别列出 L2 读取和主存 miss/bypass/写回需求，并保存命中与复用假设。
无padding/mask的假设若指“现有dense运算全部有效、流量不变”，可复制输入令 useful_ops=executed_ops，
保留原有有效MFU结果并标注假设。若指真正删除多余工作，必须重算执行量和搬运量。
