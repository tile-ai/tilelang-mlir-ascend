# 节点技能的参考来源

本项目独立维护节点技能。外部资料用于方法参考，不构成固定版本依赖。实际 API、目录布局和支持范围以用户选定的仓库及运行环境为准；需要引用实现时，在任务自己的私有记录中保存实际版本和证据。

## TileLang

来源：[TileLang 上游仓库](https://github.com/tile-ai/tilelang)。在所选 checkout 中按需查找以下资料；文件不存在时检索当前版本的对应文档，不回退到文档中写死的 commit。

| 资料入口 | 采纳位置 | 参考方法 |
| --- | --- | --- |
| `tilelang/ascend/README.md`、Ascend 示例及 API 定义 | `harness-develop` | 按目标版本确认 dialect、存储、同步和生成代码，检查边界及调用接口 |
| `tilelang-ascend` skill 的 profiling 资料（若提供） | `harness-device-test`、`harness-analysis` | 固定比较边界，区分设备时间、仿真周期和模型参数，核对事件及阶段覆盖 |
| `tilelang-build` skill（若提供） | `harness-prepare-environment` | 核对 Python/native 的实际来源与构建状态，避免源码与加载库错配 |

上游的 CUDA 默认值和 GPU 测试命令不能直接作为 Ascend 环境命令。npusim recipe 与其他 simulator 接口分别核对；多个 kernel duration 的求和不自动等于完整算子延迟。引用或分发外部文件时核对所选版本的许可证并保留适用声明。

## 测试方法

来源：[Hugging Face kernels](https://github.com/huggingface/kernels)。按需查找其算子测试及 triton-kernels 资料，借鉴独立参考、边界 shape、先验证再计时和工作量单位说明。

这些方法用于 `harness-develop` 与 `harness-device-test`；不迁入其他硬件的命令、调参假设或支持结论，也不据此增加首批语言范围。连续性、累加类型和分块约束由算子需求确定。

## AscendC 与架构资料

来源：[cann/asc-devkit](https://gitcode.com/cann/asc-devkit)。以用户所选 SDK/仓库中的实际资料为准，技能路径、API范围和许可证在使用或分发时核对。

| 技能或能力 | 可参考内容 | 边界 |
| --- | --- | --- |
| `asc-ar-dev` | API文档、声明与样例核验 | 辅助理解生成AscendC，不替代TileLang规范 |
| `asc-npu-arch` | 架构与资源能力 | 绑定实际SDK和产品/仿真模型 |
| `asc-search` | 文档范围、版本筛选与索引新鲜度 | 先核对查询范围，不当作全量语言API检索 |
| `ascendc-api-kernel-debug` | 调试输出和instrumentation清理 | 不等同流水profiling，外部技能缺失时直接查源码/文档 |

详细适配方向见 [复用计划](../REUSE_PLAN.md)。这些外部技能不是用户必须预装的隐含依赖。

## 环境边界

开发、device和simulation分别配置与验证。版本、路径、设备映射、资源限制和授权按当前任务环境生效；上板成功不等于仿真可用，文档存在也不等于编译或运行通过。具体机器和运行记录不写入产品资源。
