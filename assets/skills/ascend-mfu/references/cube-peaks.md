# 昇腾 Cube 峰值算力

查询日期：2026-11-05。入口：[昇腾处理器](https://www.hiascend.com/hardware/processor)。
数据来自该入口的[《昇腾 950 NPU 架构白皮书》](https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/昇腾950%20NPU架构白皮书.pdf)，
表 3-1，PDF 第 13–14 页的 **Cube 算力**行。以下均为单颗芯片对应版本的总 Cube 峰值。

| 设备 | Cube 核数 | FP16 / BF16 TFLOPS | TF32 TFLOPS | FP8 / MXFP8 / HiF8 TFLOPS | MXFP4 TFLOPS | INT8 TOPS |
|---|---:|---:|---:|---:|---:|---:|
| Ascend 950PR | 28 | 378 | 189 | 756 | 1513 | 756 |
| Ascend 950PR | 32 | 432 | 216 | 865 | 1730 | 865 |
| Ascend 950DT | 28 | 378 | 189 | 756 | 1513 | 756 |
| Ascend 950DT | 32 | 432 | 216 | 865 | 1730 | 865 |
| Ascend 950DT | 36 | 486 | 243 | 973 | 1946 | 973 |

脚本读取同目录 `cube-peaks.json`。更新时一起维护本表与 JSON，并保留来源与查询日期。
保留官网公布的舍入数值；不要由另一精度倍乘推算。FP32 Vector 峰值不能充当 TF32 Cube 峰值。

所用核数为 C，设备版本总 Cube 核数为 C_total 时，计算峰值为
`表中峰值 × C / C_total`。这是按同频同构核线性缩放的计算口径；它不是另一个硬件 SKU。
例如 32 核 950PR 使用 16 个 Cube 核，BF16 峰值为 216 TFLOPS。
仅按参与核归一化会高于按整芯片归一化，报告中必须说明所用核数。

不要把 AIV 核数、逻辑 block 数、设备数量或两个 AI Die 的数量再乘进单芯片峰值。
虚拟设备/分区要映射到匹配的物理版本；无法确认映射时先确认设备信息。
官网入口此次提供 950PR/950DT；没有核实的 910、310 或其他 SKU 不套用此表。
新设备应从该入口及其官方资源核实 Cube 独立峰值、精度和核数后再添加。
若只查到 Cube+Vector 总算力，不能直接作为 MFU 分母，也不能凭未核实的 Vector 值相减。
