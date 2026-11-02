# 节点技能的参考来源

本项目独立维护节点技能。下面的资料用于方法参考和按需核对，不表示已整体安装、复制或验证上游技能。检查日期：2026-10-31；首批语言为 TileLang。

## 主要参考：TileLang 上游

固定 revision：`0176ba1465b12758f41802bf838ddd68d43c9ee9`。

| 实际技能或关联文档 | 采纳位置 | 采纳的方法 |
| --- | --- | --- |
| [tilelang-ascend/SKILL.md](https://github.com/tile-ai/tilelang/blob/0176ba1465b12758f41802bf838ddd68d43c9ee9/.agents/skills/tilelang-ascend/SKILL.md) | `harness-develop` | 按目标版本检索 Ascend dialect、存储与同步规则；保留 DSL 到生成代码的关系；检查边界 case 和原调用接口 |
| [tilelang-ascend/references/profiling.md](https://github.com/tile-ai/tilelang/blob/0176ba1465b12758f41802bf838ddd68d43c9ee9/.agents/skills/tilelang-ascend/references/profiling.md) | `harness-device-test`、`harness-simulate` | 固定比较边界；区分设备耗时、仿真周期和模型参数；核对采集身份、完整性、事件语义及参与核，再提出可验证假设 |
| [tilelang-build/SKILL.md](https://github.com/tile-ai/tilelang/blob/0176ba1465b12758f41802bf838ddd68d43c9ee9/.agents/skills/tilelang-build/SKILL.md) | `harness-prepare-environment` | 核对 Python/native 来源与构建状态，避免源码版本与实际加载库错配 |

[上游 LICENSE](https://github.com/tile-ai/tilelang/blob/0176ba1465b12758f41802bf838ddd68d43c9ee9/LICENSE) 标题为 MIT，并保留历史 Microsoft 合作期说明。GitHub 许可元数据为 `NOASSERTION`，本项目据原文件记录来源。此处没有复制技能正文；未来若分发上游文件或实质片段，应保留适用的许可和声明。

需要适配的边界：

- 构建技能包含 CUDA 默认值和 GPU 测试示例，不能直接作为 Ascend 安装或验收脚本。
- 上游版本中的 API、示例、调度约束与所选 CANN/TileLang 组合需要对应，不能从文档存在推断所有芯片可用。
- 上游 `npusim` recipe 与本项目拟复用的 simulator 执行入口不是同一接口；需独立验证适配。
- 计时方法必须匹配用户约定的算子边界。多个 kernel duration 的聚合不自动等于存在重叠时的完整算子延迟。
- trace 的单位、事件配对与完成含义按实际工具版本核对；分析结果不能替代实卡精度和性能验收。

## 补充参考：测试方法

[huggingface/kernels 的 triton-kernels/SKILL.md](https://github.com/huggingface/kernels/blob/633d1faef0caea5f2b9acf447e17ad5f840a52d1/kernel-builder/skills/triton-kernels/SKILL.md)，固定 revision `633d1faef0caea5f2b9acf447e17ad5f840a52d1`，根许可为 [Apache-2.0](https://github.com/huggingface/kernels/blob/633d1faef0caea5f2b9acf447e17ad5f840a52d1/LICENSE)。

只借鉴独立参考、非整齐 shape、先验证再计时及工作量单位说明，用于 `harness-develop` 与 `harness-device-test` 的测试检查。原文面向 NVIDIA/AMD，不迁入其 CUDA/Triton 命令、调参启发式或硬件结论，也不引入 Triton 作为首批语言。连续性、累加类型和归约分块必须服从算子需求，不能把示例约束改写成通用规则。

## 已有 AscendC 资料

来源：[cann/asc-devkit](https://gitcode.com/cann/asc-devkit)，检查 revision `5dd381e37e134445ae3ef160320f3a7fb7d7986d`。技能位于 `.agent/skills/`；根许可为 CANN Open Software License Agreement Version 2.0。详细迁移评估见 [复用计划](../REUSE_PLAN.md)。

| 技能 | 可复用内容 | 适用范围 |
| --- | --- | --- |
| `asc-ar-dev` | 新 API 的文档、声明和样例核验 | 辅助理解生成 AscendC；不能替代 TileLang 语言规范 |
| `asc-npu-arch` | 结构化架构事实与能力边界 | 绑定实际软件版本、实卡产品或仿真模型 |
| `asc-search` | 文档范围、版本筛选与索引新鲜度 | 当前范围为 SIMD C API，不能当作全量 TileLang/Ascend API 检索 |
| `ascendc-api-kernel-debug` | 调试输出链路与 instrumentation 清理 | 不等于流水 profiling；迁移前修正其关联技能路由 |

## 环境与验证边界

环境准备分别收集并验证 device 和 simulation 的记录，二者可以使用不同机器或容器。上板节点读取 device 约定，仿真节点读取 simulation 约定；工具版本、路径、资源限制、允许动作和授权范围各自生效，互不继承。语言版本和 case 可以共享引用，但执行可用性与验收状态分别记录。

本轮只核对上述技能正文、相关参考和许可，没有运行其示例，没有安装全局技能，没有验证 Onboard/simulator 的端到端兼容性。选择依据是任务适配和可核查内容，安装数或星数不构成 NPU 支持证明。节点的成功判定仍需对应的编译、实卡比较或完整仿真产物。
