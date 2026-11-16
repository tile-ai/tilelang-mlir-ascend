# BWU、CSV 与存储层级

BWU对Cube、Vector及融合算子使用同一公式。区别在于它们请求的路径和流量，不在于另定义一种利用率。
资源/带宽数据见 [hardware.md](hardware.md)，理论屋顶与瓶颈判断见 [roofline.md](roofline.md)。

## 先定义边界，再取列

`BWU = D / (T * B_peak)`。D 与 B 必须匹配层级、方向、资源数及事务/有效载荷口径。
GM是地址空间，访问GM不等于每次都访问HBM。请求可以命中L2；同一字节会经过多个接口，不能在一个主存分子里重复计数。

以下为常见 msprof op CSV 名称；具体支持范围、核/子核粒度、单位和计数覆盖需按目标版本核对。

| 字段/指标 | 含义与计算关系 | 用于什么，不可混作什么 |
|---|---|---|
| `read_main_memory_datas(KB)`、`write_main_memory_datas(KB)` | 工具计量的主存读/写事务量 | 同次launch汇总后算主存BWU；不是所有GM地址访问的逻辑量 |
| `GM_to_UB_datas(KB)` | GM地址空间到UB通路的搬运量，可由L2命中或主存服务 | 对应搬入通路；不与主存读相加 |
| `UB_to_GM_datas(KB)` | UB到GM地址空间的搬出量 | 对应搬出通路；不自动等于同时间窗的主存写回量 |
| `aiv_gm_to_ub_bw(GB/s)` | GM→UB量按工具所选时长归一化的速率 | 检查分母是核时间还是task时间；不直接相加不同时长的核速率 |
| `aiv_main_mem_read_bw(GB/s)`、`aiv_main_mem_write_bw(GB/s)` | 主存对应方向量/工具时长 | 优先从原始量和共同task时间重算任务带宽 |
| `GM_to_UB_bw_usage_rate(%)`、`UB_to_GM_bw_usage_rate(%)` | 工具通路带宽相对其配置峰值的百分比 | 峰值/时间/单位可能与整芯片不同；不能平均得芯片BWU |
| `aiv_ub_read_bw_gm(GB/s)`、`aiv_ub_write_bw_gm(GB/s)` | 从UB端看与GM搬运的读/写 | UB读通常对应搬出，UB写对应搬入；先核查版本，不能因GM字样再计一份HBM量 |
| `aiv_ub_read_bw_vector(GB/s)`、`aiv_ub_write_bw_vector(GB/s)` | Vector指令读/写UB的内部端口速率 | UB内反复计算可远大于GM流量；按UB端口峰值衡量 |
| `aiv_read_hit_rate(%)`、`aiv_write_hit_rate(%)` | 对应请求/事务的L2命中比例 | 请求数比例不必等于字节数比例；读写分开，汇总用原始计数加权 |
| `aiv_mte2_time(us)`、`aiv_mte3_time(us)` | 流水活动时间 | 计算active带宽；不是整个kernel时长 |
| `aiv_mte2_ratio`、`aiv_mte3_ratio` | 活动时间/工具采用的核或任务时间 | 是时间占比，不是BWU；百分比还是小数需核对 |
| `aiv_mte2_active_bw(GB/s)`、`aiv_mte3_active_bw(GB/s)` | 该流水计量字节/active时间 | 与task带宽分列；多核active时间不直接求和作墙钟时间 |
| `aiv_vec_wait_ratio`、`aiv_mte2_wait_ratio`、`aiv_mte3_wait_ratio` | 对应流水等待占比 | 辅助时序诊断；不是可与BWU相加的比例 |
| `aiv_total_cycles`、`Current Freq`、`Rated Freq` | 周期与当前/额定频率信息 | 仅在同一时钟域、频率稳定时以cycles/frequency换算时间；优先匹配的duration |
| `aiv_icache_miss_rate` | 指令cache未命中率 | 不是数据L2命中率 |

以上AIV字段是具体CSV示例，不把BWU限制为Vector。分析Cube时选择该版本对应AIC主存、GM→L1、
L1→L0、FIXPIPE等实际路径的计数；字段可能不同，不能只改列名前缀就假定含义相同。
Cube/Vector融合任务的主存BWU应合计同一次launch中双方穿过主存边界的去重流量，除共同task时间，
再对标共享主存预算；L1/L0/UB内部流量不再次加入主存分子。

MTE2承担哪些搬入路径取决于硬件和kernel。**仅当该测量中MTE2只搬GM→UB且计数完整时，MTE2量才等于GM_to_UB量**。
有GM→L1等路径时应覆盖这些路径，并按真实共享端口分组。UB带宽描述UB端的读写服务，既包括DMA也可包括Vector访问，
不能与HBM或L2带宽直接相加当作新的“总内存带宽”。

## 从 CSV 计算

筛选同一device、kernel/launch、采集轮次和核集合，不把所有重复运行相加再除单次时间。
先确定文件含逐核记录还是已聚合记录；不要将summary和对应明细重复相加。
不同采集pass需要同一确定性工作负载及采集协议支持，不能仅凭同名kernel拼接。

设已核验该版本 `KB` 实际按1024字节存储，则：

```text
D_HBM_read = 1024 * sum(read_main_memory_datas)
D_HBM_write = 1024 * sum(write_main_memory_datas)
B_HBM_RW = (D_HBM_read + D_HBM_write) / T_task
BWU_HBM_RW = B_HBM_RW / B_peak_HBM_RW

D_GM_UB = 1024 * sum(GM_to_UB_datas)
B_GM_UB_task = D_GM_UB / T_task
BWU_GM_UB_task = B_GM_UB_task / B_peak_GM_UB
```

`T_task`通常取同次 `OpBasicInfo.csv` 的`Task Duration(us)`再乘 `1e-6`。
单核active口径：`B_MTE2_active,i=D_MTE2,i/T_MTE2_active,i`。
相同量、范围和参考时间下，`B_task=B_active*(T_active/T_task)`；不同核/时间分母不能混用。
不能把所有核活动时间求和后当作kernel时间，也不能把每核active带宽简单相加宣称整芯片任务带宽。

KB/KiB与GB/GiB分别相差1000/1024和10^9/2^30：
`132.5 KiB / 2.88 us = 47.1111 GB/s = 43.8756 GiB/s`。
若共享读写峰值假设为十进制1600 GB/s，则BWU为2.94444%，不是以43.8756 GB/s代入的结果。
这是单位演算示例，不是硬件实测记录。数字超过100%保留并审计，不能截断或修改峰值让其“对上”。

### 计算工具

`calculate_bwu.py`要求显式选择数据行和数据量列，不自动猜哪次launch或L2计数器：

```bash
python <skill-dir>/scripts/calculate_bwu.py --csv Memory.csv --row 1 --row 2 --column 'read_main_memory_datas(KB)' --column 'write_main_memory_datas(KB)' --amount-unit KiB --resource HBM --direction read+write --time-us 10 --time-scope task --time-source 'matching OpBasicInfo Task Duration; example only' --peak-gbps 1000 --peak-source 'hypothetical shared read/write peak; replace with evidence'
```

行号从表头后第一条记录算1；选择的数据量列必须不重叠且共用`--amount-unit`。
上面时间/峰值仅演示参数，使用时替换。工具输出原值、行号、文件哈希、字节量、十进制GB/s和BWU，拒绝NA/缺失/负值/重复选择。
未知峰值时同时省略`--peak-gbps`和`--peak-source`，只给实际带宽，BWU为null。
`--time-scope active`只用于已匹配活动时间的范围；该标记不自动验证或转换时间。
脚本不判断所选列是否物理重叠，调用前按本节核验。不要将百分比或GB/s列当字节列。

## L2、差值与命中率

**L2接口总流量不等于“仅cache命中的字节”，更不等于主存量与GM→UB量的差值。**
未命中请求也可穿过L2侧接口；填充、旁路和写回则取决于具体通路与计数器。

对一个限定的只读通路，若请求和主存计数在时间、范围、事务粒度上一致，
无额外预取/重复填充/旁路，才可采用简化分解：

```text
D_requested = D_hit + D_miss_served
D_main_read = D_miss_served
D_hit = D_requested - D_main_read
```

这里GM→UB可近似该通路的 `D_requested`；差值方向是请求量减主存读量。
它最多估计该通路的命中字节，不代表完整L2读写流量。
任一假设不成立时不做差值推断；出现负数先查计数范围，不能取绝对值。

计算 **L2 BWU** 应使用该版本工具/硬件定义的L2接口读/写量与同方向峰值。
若只能拿到GM→UB、UB→GM，可报告这些端口的带宽；只有确认它们完整覆盖目标L2接口、无遗漏或重复，
才把总量用于该接口BWU。缺少完整L2计数时说明“该L2口径未能确定”，不要从Roofline点倒推数值充当CSV证据。
UB↔GM量也不一定涵盖SIMT直接load/store、Cube、L1或其他请求方。

请求命中率为 `hit_requests/(hit_requests+miss_requests)`。
只有每次请求字节数一致等条件下才能当作 `D_hit/D_requested`；读写总命中率不能代替读命中率。
写命中不能保证无主存写回，所以不能用 `总请求量*(1-读写总命中率)` 推算HBM读写量。

“MTE2由HBM和L2共同供给”说的是来源分解，**不是两个理论峰值直接相加**。
若在同一T下来源字节互斥，则 `B_requested=B_hit_contribution+B_miss_contribution`；这是实测贡献量恒等式。
若某性能估算假设命中/未命中服务时间相加且p为字节命中率，其有效带宽模型为
`B_eff=1/((1-p)/B_miss+p/B_hit)`；这是串行服务假设，不是普遍硬件定律。
若并行服务则需另外建重叠模型，并受MTE端口及共享链路约束。

## 不依赖图的四项结果

当F、D_r、T和匹配峰值都已确定时：

1. 实际带宽：`D_r/T`，分HBM/L2/端口、方向、task/active。
2. 实际性能：`F_useful/T`与可得的`F_executed/T`分列；exp等报告对应特殊操作吞吐。
3. AI：`F/D_r`，同F分子配各层D；不能与理想最少逻辑流量的AI混称。
4. MFU/计算利用率与BWU：分别为`F/(T*P_peak)`和`D_r/(T*B_peak,r)`；缺峰值项不输出百分比。

原始列、单位换算、聚合范围和公式足够复算时，Roofline图仅作展示/交叉核验。
图与CSV不符时先核对工具版本、计数/时间/单位及硬件配置，而不是反调D或峰值去拟合图。

参考：[msOpProf 用户指南及 Memory/Pipe 指标](https://github.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_user_guide.md)。
正式分析保留安装版本对应的文档/实现与采集范围；上游分支中的定义可能已改变。
