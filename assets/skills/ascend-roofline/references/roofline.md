# Roofline 建模与瓶颈分析

## 建模对象与三套量

建模是为选定算法、输入规模、精度、资源范围和缓存情景，建立资源需求与吞吐上限的关系。
不是先寻找一个对所有实现都成立的“最大 AI”。相同数学语义的算法、融合、复用、padding 和重计算都会改变需求。

| 层次 | 运算量 | 字节量 | 用途 |
|---|---|---|---|
| 语义/理想情景 | `F_useful`：选定算法的必要有效运算，明确 FMA 等约定 | `D_min,r`：在明确容量、输入驻留和复用条件下，该层假设的最少流量 | 探索优化空间；`F_useful/D_min,r` 是条件性 AI 上限 |
| 当前实现 | `F_executed`：含 padding、未跳过 mask、冗余/重计算 | `D_impl,r`：按实际循环、分块、跨核复用和中间结果推导 | 静态实现模型；编译前推导不等于生成代码证据 |
| 实测 | 工具支持的 executed 指令/操作计数；另保留语义 F | `D_measured,r`：该层计数器覆盖的实际事务量 | 配合同次任务时间算吞吐，审计与逻辑量的差异 |

`F_useful` 也不是跨所有算法唯一的数学常数：例如 GEMM 算法改变、近似 exp 或融合代数化简须声明基准语义与精度。
固定 F 后减少 D 才是提高复用 AI；增加冗余 F 虽使 `F_executed/D` 增大，不表示任务更快。
若重计算减少了搬运，必须同时重算 F 与 D，并比较预测/实测时间。

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

## 单层公式与 MFU/BWU

令 F、D、T 属于同次工作量，P_peak 与 F 的操作口径一致，B_peak 与 D 的资源及方向一致：

```text
P_achieved = F / T
AI_r = F / D_r
B_achieved,r = D_r / T
U_compute = F / (T * P_peak)
BWU_r = D_r / (T * B_peak,r)
P_roof,r = min(P_peak, AI_r * B_peak,r)
R_r = P_achieved / P_roof,r
```

当 `AI_r*B_peak,r < P_peak`，`R_r = BWU_r`；当计算屋顶较低，`R_r = U_compute`。
百分比最后乘100。**两段都能算 MFU/BWU，但 Roofline 性能比只等于其中当前约束对应的一项。**
相同 F、T 的多层图应有相同纵坐标；横坐标因各层 D 不同而不同。
若纵坐标不同，检查是否分开统计 float/integer、指令类别或测量范围，不能直接解释为双 AIV。

本skill采用统一的算子级MFU定义 `F_useful/(T*P_peak)`，Cube与Vector都可计算，分别选匹配资源的F和P。
融合kernel可分别报告Cube MFU与Vector MFU，共同使用完整kernel时间；不默认把两者相加或平均。
这里是Model FLOPs Utilization在算子级的明确约定，模型端到端MFU仍需对应端到端工作量与时间。
工具 executed 吞吐除峰值称为 executed 计算利用率，不能不加说明地改称 MFU。
整数操作报告 TOPS/整数利用率；copy 的 F=0，直接报 BWU，不计算 `0/0` 的 Roofline 比例。

若存在冗余且只含一种匹配计算资源，令 `q=F_useful/F_executed`：

```text
T_floor = max(F_executed/P_peak, D_r/B_peak,r)
P_useful_roof = min(q*P_peak, AI_useful,r*B_peak,r)
MFU_ceiling = F_useful / (P_peak*T_floor)
```

因此 useful 图的计算屋顶为 `q*P_peak`，不能用同一 executed 屋顶宣称有效利用率应达到100%。
源码计数和工具计数必须能映射到同一种操作定义，才能求 q。

## 多资源模型与峰值选择

对多个计算类别 c 和带宽资源 r，先构造每个真实共享资源的服务时间，再取理想重叠下限：

```text
t_compute,c = F_executed,c / P_c
t_memory,r = D_r / B_r
T_floor = max(t_compute,1, ..., t_memory,1, ...)
```

共享单元上的工作应按已知吞吐/发射限制聚合，不能为每条指令分配一份独立峰值。
只有已证实相互独立的资源才能假设重叠。脚本只建模一个理想重叠阶段：遇到已确认的串行阶段
（例如先整体搬运后计算且无重叠）时，按段分别运行脚本，把各段`time_floor_us`相加作为总下限，
并逐段保留各自匹配的峰值、带宽与假设；不用某一段的峰值套整条链，也不把单段口径的时间当作整链时间。
混合 FMA、SFU、integer 或 Cube/Vector 时，不把不同 P 简单相加，也不把各类别利用率平均成 MFU。
缺少混合指令吞吐证据时逐项报告下限，注明重叠和争用尚未覆盖。
分层流量与共享资源规则见下文“按资源边界计流量”；其适用性由路径决定，不取决于算子被称为Cube还是Vector。

计算峰值及指令口径按 [peaks.md](peaks.md) 选择，运算量计数与“算子→分母”按 [counts.md](counts.md)，
容量、频率和带宽证据按 [hardware.md](hardware.md) 核对；MFU 分子范围按 [mfu.md](mfu.md)，CSV流量与命中率按 [bwu.md](bwu.md)。
本框架不内置“所有950通用”的带宽或指令峰值；缺项给符号表达式或明确假设。

## 图形和实际瓶颈

在固定 P_peak、B_peak、操作定义、资源范围下，单层经典屋顶固定：

- 线性坐标 `P=B*AI` 的斜率为带宽 B；拐点 `AI*=P_peak/B_peak`。
- 双对数坐标 `log(P)=log(B)+log(AI)`，各带宽斜线斜率均为1，B 改变截距和拐点。
- 换精度、SIMD/SIMT 路径、指令类别或有效比例 q，计算屋顶和拐点也可能改变。

`AI < AI*` 只说明该模型中带宽项更紧；小 kernel 仍可能受启动、依赖链、同步、发射粒度、低 occupancy 限制。
`AI > AI*` 也不证明计算单元饱和。应比较实测 T 与各资源下限，以及相同范围的 BWU/计算吞吐。

| 观测 | 可支持的结论 | 下一步区分原因 |
|---|---|---|
| 匹配层 BWU 接近已核验峰值，减少该层流量使时延下降 | 该层吞吐限制有证据 | 改复用/融合；核对实际 D 是否降低 |
| 匹配指令吞吐接近上限，增加独立计算后线性增时 | 对应计算资源吞吐限制有证据 | 减工作量或换指令/精度并验证正确性 |
| task BWU低、active BWU高 | 搬运活跃时快，但等待/排空/负载分布可限制整体 | 检查时序、依赖、参与核与尾块 |
| task/active BWU均低，MTE活跃占比高 | 小事务、stride、未命中延迟或端口问题候选 | 扩大搬运粒度、改对齐和并发，比较计数 |
| 算力和带宽都低 | 不能据低AI直接判 HBM-bound | 查启动成本、同步、数据依赖、寄存器/UB容量与 occupancy |

80%/90% 是工具或工程启发阈值，不是 Roofline 定律。pipeline ratio 是忙碌时间占比，
不能单独证明流水重叠良好或执行效率高；工具给出的 latency/compute/memory 标签需连同工具版本解释。
950 的SIMD与SIMT指令粒度不同，不能把910的 repeat 参数优化原样用于950 SIMT，需核对生成代码。
优化以相同语义和测量协议下的时延改善为准：BWU或executed OPS变高本身不是优化成功。

## 输出与画图

交付设备/版本、参与核数与指令路径、有效与执行运算量、逐路径字节与复用、带宽及证据、各资源下限与瓶颈、条件性利用率上限。
分别标明理论情景、当前实现预测和实测结果；只有源码推导时披露编译/校验未完成。
保留`calculate_*.py`的`--json`或`--output`完整输出，连同所选峰值来源写入项目记录，作为可复算依据。
按需用标准绘图库输出PNG/SVG：横轴注明是有效还是执行FLOP/Byte，纵轴与之一致（ops/s或百分比），
标出缓存情景、假设带宽与拐点`AI*=P_peak/B_peak`；图注写清测量口径与时间范围。
同图比较不同设备/精度时保留各自带宽与峰值，不把不同层的流量混成一个AI；静态科学图优先双对数Roofline。

## post_mhc 静态算例

```text
out[n,mo,h] = post[n,mo]*x[n,h]
              + sum_mi comb[n,mi,mo]*residual[n,mi,h]
```

设 N 个 token，M 个分支，H 个隐藏元素。采用1次乘法加M次FMA的直接算法，FMA按2：
`F_useful=N*M*H*(1+2M)`。假设 post/comb 为FP32，x/residual/out 为FP16或BF16，
各输入在所建模边界只读一次、输出写一次，系数跨H复用且不写中间张量：

| 数据 | 元素数 | 字节 |
|---|---:|---:|
| post | NM | 4NM |
| comb | NM² | 4NM² |
| x | NH | 2NH |
| residual | NMH | 2NMH |
| out | NMH | 2NMH |

```text
D_ideal = N*[4M(M+1) + 2H(2M+1)]
AI_ideal = M*H*(1+2M) / [4M(M+1) + 2H(2M+1)]
M=4, H=192: F/token=6912, D/token=3536 B, AI=1.954751 FLOP/B
H趋于无穷且M固定: AI趋于M/2
```

这是声明假设下的理想逻辑流量，不是任意实现的实测 HBM 或 L2 流量。
跨核重读、padding、spill、中间结果会增加流量；缓存驻留可减少主存流量，但仍受L2/端口限制。
存在独立乘法，不能将总FLOPs直接除2就命名为MAC数；纯FMA数量为 `N*M*M*H`，另有 `N*M*H` 次乘法。
将 AI 与所选屋顶比较，只能判断该情景的理论约束；实测瓶颈仍需要BWU和时延证据。

## 通用计算脚本

`scripts/calculate_roofline.py`支持一个匹配的Cube或Vector计算屋顶加任意多个带宽资源。
融合kernel的不同计算类别分别建模；脚本不自动解决混合指令争用/阶段依赖，不能将总F填到单一峰值下。

```bash
python <skill-dir>/scripts/calculate_roofline.py --input <skill-dir>/assets/roofline-example.json
python <skill-dir>/scripts/calculate_roofline.py --input <skill-dir>/assets/vector-example.json
```

两份例子都是构造数据，用于演示schema，不是设备测量。JSON字段：

- `engine`：cube/vector；旧输入省略时默认为cube。`dtype`、`useful_ops`、`executed_ops`、`resources`、`assumptions`必填。
- 峰值二选一：`device,device_cores,cores`按 [peaks](peaks.md) 查表；或`peak_tops,peak_source`提供明确适用的峰值及证据。
  `device_cores`始终是SKU的AIC核数，`cores`是所选引擎实际参与物理核数（Cube用AIC，Vector用AIV）。
  `peak_tops`单位10^12 ops/s，浮点对应TFLOPS，整数对应TOPS。显式峰值与目录查表字段互斥。
- `resources`每项：唯一`name`（不能等于当前engine）、`payload_bytes`（路径名到非负整数bytes）、
  `bandwidth_gbps`（十进制GB/s）、`basis`（official/measured/assumed）、`source`（来源及适用范围）。
  每个资源内合计字节，各资源服务时间取max；独立读写才分列，共享预算必须合并。
- `assumptions`为非空文本数组，记录缓存、计数、带宽及重叠假设；`label`可选。
- `measured_time_us`可选，覆盖所列工作量的完整kernel时间。没有测量就只给条件性上限。

输出包括各资源下限、模型限制项、有效/执行AI和条件性利用率上限：`utilization_ceiling_percent`对Cube/Vector与浮点/整数都给出，浮点另有`mfu_ceiling_percent`别名；提供时间时再算实际性能、每层带宽和BWU。
纯copy允许`useful_ops=executed_ops=0`，只分析非零流量约束，MFU与利用率均为null（没有计算工作量）。
整数计算给出`utilization_ceiling_percent`/`measured_utilization_percent`，不命名浮点MFU；
`executed_ops>0`但`useful_ops=0`时利用率记0%，与“不适用”的null区分。原Cube浮点输出字段保留为兼容别名。
缺少峰值时在报告中保留符号推导；不要为了运行脚本编造输入。

## 参考

- [Roofline 原始论文技术报告](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/Archive/EECS-2008-134.pdf)：操作强度与资源上限。
- [msOpProf 用户指南](https://github.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_user_guide.md)：按所用版本核对支持范围和指标。
- [msOpProf Roofline 实现](https://github.com/Ascend/msopprof/blob/master/csrc/op_profiling/profiling/device/data_visualize/roofline.cpp)：核查该版本计数、Float/Integer屋顶与诊断规则。链接分支会变化；项目记录实际使用版本。
