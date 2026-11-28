---
id: family/gemm
layer: family
arch: [Ascend950]
kind: playbook
status: active
verified_scope: tilelang-ascend
---

# GEMM / BMM / 量化 GEMM 算子族 Playbook

覆盖标准 GEMM、Batched GEMM（BMM）以及混合精度/量化 GEMM（如 W4A8）。

## 1. 核心组织原则

1. **按物理核数规划持久任务与 L1/L0 分块**：
   - 常规BMM/GEMM可从规整候选tile及 `T.Pipelined` 起步；先按dtype、实际片上容量、同时存活版本和任务数筛选，再决定tile与stage。`128×128×128`仅是候选示例，不是默认最优配置。可通过目标版本的核心数接口核对物理Cube核数（仓内参考：`example/gemm/bmm.py`）。
   - 先用 `ascend-roofline` 核算当前 shape 是 Cube-bound 还是 GM/L2 带宽受限，再决定优化重点是 K 维流水深度、L2 swizzle 还是尾块处理。
2. **量化 GEMM（如 W4A8）：把“向量解包”与“纯 Cube 乘法”按硬件优势解耦**：
   - 仓内参考样例：`example/gemm/w4a8.py` 展示了典型的三段式组织思路——
     - **Pass 1（Vector 解包重排）**：将分组缩放的有符号 INT4 权重在 Vector 单元解包、乘以 group scale 并舍入到目标 FP8（`float8_e4m3fn`），避免在 Cube 主循环内穿插重型解包拖慢流水；
     - **Pass 2（原生 FP8 Cube GEMM）**：Cube 直接读取规整 FP8 权重与激活并做 FP32 累加，保持硬件满速 FP8 吞吐；
     - **Pass 3（Vector 后处理缩放）**：施加 token/channel scale。
   - **取舍边界**：多 Pass 会增加中间权重/激活的写回与重读流量；在权重可跨多批/多步复用或解包本身极重时收益显著。若后续面临小 batch 带宽瓶颈，需对照 Roofline 评估是否融合部分解包或 epilogue。

## 2. 常见陷阱与验证要点

- **精度与累加口径**：明确输出 dtype 与累加 dtype（FP32 累加 vs 低精度、FP32 场景是否关闭 HF32），量化算子须核对舍入模式与参考实现逐项一致。
- **非对齐尾块与小矩阵**：核对 `M/N/K` 不能被 tile 整除时的边界 mask、`NaN` 哨兵预填与零填充，防止越界写回污染相邻 batch。
- **动态/缓存键检查**：`@functools.lru_cache` 编译缓存键必须包含全部影响生成代码的 shape、dtype、转置与量化分组参数。

## 3. 分层BMM的可复用分析顺序

详细方法见 [分级分块与任务局部性](../../patterns/gemm-hierarchical-tiles-and-task-locality.md) 和 [缓冲生命周期与UnitFlag](../../patterns/cube-fixpipe-unitflag-buffer-lifetime.md)。独立仓内例子为 `example/gemm/example_batch_gemm.py`，输入BMK/BNK BF16、输出FP32，使用Ascend C；不能当作现有TileOps BMM接口的直接替换。

1. **定语义**：固定shape、布局、dtype及计时边界；若需输入转置或类型转换，明确其费用属于哪段调用。
2. **算容量**：分开L1搬运块与L0计算块，逐域计入版本、padding和累加器。先保证放得下，不把L0C占满当成利用率打满。
3. **算复用和任务**：用M/N分析输入服务多少输出，用K分组分析搬运调用频率；核对任务数、尾波和完整窗口。
4. **查生成代码**：确认槽、MMAD形状、就绪/释放、序言/排空，以及首子步初始化与最终结果交接。stage配置不等于实际重叠。
5. **分开流量口径**：源码请求量、近时唯一地址、实际L2/主存事务分别记录；swizzle只能提供复用机会。
6. **验证与测量**：先独立精度，再小仿真排错，最后全核同板同频配对测量；记录实际预热/采样，完整kernel时间配匹配峰值计算MFU。

## 4. 哪些知识可以复用，哪些仍须实测

| 问题 | 可以带到下一算子的知识 | 不能直接沿用的结论 |
| --- | --- | --- |
| tile怎么选 | 按类型、容量、版本、复用、任务数逐项筛选 | 固定256或128对所有shape最优 |
| 双缓冲是否有用 | 地址分离与就绪/释放配合，为交错供数提供条件 | 声明2个stage就一定完全隐藏搬运 |
| K块增大 | 搬运调用频率可能降低，计算子步和逻辑字节可能不变 | 调用数减少多少就加速多少 |
| UnitFlag能否减少事件 | 核对绑定后端的首末子步、输出和槽复用协议 | 不核对版本就复制参数或删除所有同步 |
| swizzle是否更快 | 对比时间邻近的地址复用与实际事务 | 逻辑唯一字节就是实测HBM流量 |
| 是否compute-bound | 完整kernel、匹配峰值、分层流量共同判断 | 单样本高MFU就证明实现最优 |

证据不足的方向保留为待验证假设，不新增“已证伪”反例。具体提交、运行路径、原始流水与测量协议留在项目记录；知识库只保留方法、适用条件和验证步骤。
