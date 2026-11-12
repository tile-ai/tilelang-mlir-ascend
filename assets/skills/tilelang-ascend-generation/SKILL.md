---
name: tilelang-ascend-generation
description: 补充昇腾 TileLang 算子生成与优化方法，指导阶段拆分、SIMD/SIMT 选择、物理存储预算和自动流水。用于 analysis 制定实现方案及 develop 编写或优化候选。
---

# TileLang Ascend 算子生成补充

本 Skill 提取自个人 [Wamdus3/tilelang](https://github.com/Wamdus3/tilelang)
仓库中 `tilelang-ascend` 的增补指导，独立于官方原生 Skill 快照维护。
API、同步和布局规则以任务绑定的 TileLang checkout 为准；先读该版本的
`.agents/skills/tilelang-ascend/SKILL.md` 及相关参考。本文中的源码路径也相对于该 checkout。

## 从算子契约到执行阶段

明确全部必选 shape、dtype、布局、精度及输出，再将计算拆成有明确生产者、消费者和存储的阶段。
按阶段选择 Cube、SIMD 或 SIMT，不把 GPU 的每个线程循环机械改写为 SIMT。
连续 UB 算术优先评估 SIMD；不规则索引可独立成 SIMT gather 或使用目标版本支持的 DMA，
矩阵乘法检查 Cube 的布局与容量约束。选择需要由可用 API、lowering 和测量验证。

编写或改造 attention 等混合算子时，读取
[阶段组织与 VF 选择](references/kernel-structure.md)，检查 stage macro、布局转换、
compact-NZ packing 和输出存储复用。参考相近 kernel 的调用循环和 buffer 生命周期，
不能只复制算术阶段、tile 常数或 VF latency。

## 重复 tile 循环的实现流程

对能够跨 tile 重叠搬运与计算的混合循环，读取
[自动流水指导](references/tiled-pipeline.md) 和目标 checkout 的同步参考。

1. 将阶段映射到 core/PIPE，区分循环携带状态、不变输入和每 tile scratch。
   预算计入 padding、转换临时区、别名与候选版本数。
2. 在调用阶段的循环上建立 `T.Pipelined` 候选，依据预算显式选择 `num_stages`。
   保留递归状态依赖，合适的 scratch 交给编译器管理版本；不为简化实现将所有 scratch 固定为 1。
3. 检查实际生成的调度、版本数和物理分配，确认每个版本完整读区域的初始化。
   保留串行循环或单版本时，说明具体依赖、所有权、容量或任务范围理由。
   缺少硬件会限制验证与调优，不等于无法表达和检查可行的流水候选。
4. 交接所选 tile/流水预算、实际存储、待验证假设和 case。
   分开记录 lowering、设备编译、实卡正确性和测得性能；lowering 成功不证明完成调优。

## 接入 harness

analysis 将适用的方法落实到阶段/VF/存储方案；develop 按该方案构造候选，
在现有 RESULT 中说明实际采用的指导、对应源码或生成代码位置，以及暂未采用的相关方法及原因。
无需新增独立的 Skill 证明文件。

优化收益使用任务约定的 kernel 耗时比较。需要 Cube MFU 或条件性 Roofline 时读取
[ascend-mfu](../ascend-mfu/SKILL.md)；纯 Vector 算子以耗时评价，不套用 Cube 峰值。
