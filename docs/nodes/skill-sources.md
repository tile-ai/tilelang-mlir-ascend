# 节点技能的参考来源

本项目独立维护节点技能。外部资料用于方法参考，不构成固定版本依赖。实际 API、目录布局和支持范围以用户选定的仓库及运行环境为准；需要引用实现时，在任务自己的私有记录中保存实际版本和证据。

## TileLang

来源：[TileLang 上游仓库](https://github.com/tile-ai/tilelang)。在所选 checkout 中按需查找以下资料；文件不存在时检索当前版本的对应文档，不回退到文档中写死的 commit。

| 资料入口 | 采纳位置 | 参考方法 |
| --- | --- | --- |
| `tilelang/ascend/README.md`、Ascend 示例及 API 定义 | `harness-port-baseline`、`harness-develop` | 按目标版本确认 dialect、存储、同步和生成代码，检查边界及调用接口 |
| `tilelang-ascend` skill 的 profiling 资料（若提供） | `harness-develop`、`harness-analysis` | 固定比较边界，区分设备时间、仿真周期和模型参数，核对事件及阶段覆盖 |
| `tilelang-build` skill（若提供） | `harness-prepare-environment` | 核对 Python/native 的实际来源与构建状态，避免源码与加载库错配 |

上游的 CUDA 默认值和 GPU 测试命令不能直接作为 Ascend 环境命令。npusim recipe 与其他 simulator 接口分别核对；多个 kernel duration 的求和不自动等于完整算子延迟。引用或分发外部文件时核对所选版本的许可证并保留适用声明。

## 测试方法

来源：[Hugging Face kernels](https://github.com/huggingface/kernels)。按需查找其算子测试及 triton-kernels 资料，借鉴独立参考、边界 shape、先验证再计时和工作量单位说明。

这些方法由 `harness-port-baseline` 与 `harness-develop` 复用；不迁入其他硬件的命令、调参假设或支持结论，也不据此增加首批语言范围。连续性、累加类型和分块约束由算子需求确定。

## AscendC 与架构资料

来源：[cann/asc-devkit](https://gitcode.com/cann/asc-devkit)。以用户所选 SDK/仓库中的实际资料为准，技能路径、API范围和许可证在使用或分发时核对。

| 技能或能力 | 可参考内容 | 边界 |
| --- | --- | --- |
| `asc-ar-dev` | API文档、声明与样例核验 | 辅助理解生成AscendC，不替代TileLang规范 |
| `asc-npu-arch` | 架构与资源能力 | 绑定实际SDK和产品/仿真模型 |
| `asc-search` | 文档范围、版本筛选与索引新鲜度 | 先核对查询范围，不当作全量语言API检索 |
| `ascendc-api-kernel-debug` | 调试输出和instrumentation清理 | 不等同流水profiling，外部技能缺失时直接查源码/文档 |

详细适配方向见 [复用计划](../REUSE_PLAN.md)。这些外部技能不是用户必须预装的隐含依赖。

## cannbot-skills 的分析方法

来源：[cann/cannbot-skills](https://gitcode.com/cann/cannbot-skills)。Harness 独立编写内部参考与解析工具，将以下方法融入 analysis；不直接打包外部脚本或整套调优流程，也不引入该 checkout 的运行依赖。外部资料的实际版本在用户任务证据中记录，使用时核对许可证与平台适用性。

| 来源技能与资料 | 内置能力 | 适配边界 |
| --- | --- | --- |
| `ops/tilelang-op-profiling/Ascend950`：instructions、csv_fields_reference、optimization_quickref | [上板分析](../../assets/skills/harness-analysis/references/device-profile.md)：实例筛选、逐核表/指标与源码对应、先统计再归因 | 不固定 CSV 数量、芯片阈值或峰值，不把 profiling 耗时替换正式性能 |
| `ops/tilelang-perf-optimization/Ascend950/references/evidence-and-diagnosis.md`、`plan-and-validation.md` | [算子诊断](../../assets/skills/harness-analysis/references/operator-diagnosis.md)：竞争解释、API/lowering 准入、case 风险与验证 | 采用项目实际 TileLang 后端；实现和复测仍在 develop |
| `ops/ops-profiling` | trace/指标来源区分、实际工作量与指标联合解释 | 保留 Harness 既有采集与测量协议，不迁入外部 warmup/统计规则或新采集循环 |
| `ops/ops-theoretical-performance-estimation` | 分层流量、计算类型、稳态重叠与尾块/填充排空边界 | 不移植固定硬件 JSON、标定常数或模型门槛；模型只辅助诊断 |
| `ops/pypto-op-perf-tune/perf-analyzer` | 核间负载差异、等待与依赖的联合分析思路 | PyPTO bubble 日志公式和星级阈值不直接用于 TileLang；仅消费实际工具提供的事件语义 |

搜索归属、mechanism 查重、search-plan、hot/cooled 和预算继续由 Harness 管理，详见 [analysis 架构](analysis.md)。外部 skill 自带的实现/测试循环不嵌套进 analysis。

## cannbot-skills 的开发方法

同一 cannbot 来源的方法已整理为 Harness 内部参考，外部 checkout 为可选查询资料；不复制其 agent 编排、collector、设备环境命令或固定交付结构。

| 来源 | 内置落点 | 适用边界 |
| --- | --- | --- |
| `tilelang-op-develop` / `tilelang-op-design` | [方案落实](../../assets/skills/harness-develop/references/tilelang-workflow.md)：索引、数据流、buffer 与局部设计，最早失败阶段定位 | 沿 TASK 的方案、目标 API/后端与 initial_design；不新增设计节点 |
| `tilelang-performance-best-practices` | 同上：按机制核对搬运、分块、流水、生成代码与实际收益 | 按架构/后端/算子族选择参考，容量和核数不写死，模板成熟度不等于候选验证 |
| `tilelang-review` 的正确性规则 | [正确性检视](../../assets/skills/harness-develop/references/correctness-review.md) | 按改动选择索引、尾块、整数宽度、初值、竞争与生命周期检查 |
| `tilelang-op-test-design/Ascend950/references/test-credibility.md` | [测试可信度](../../assets/skills/harness-develop/references/test-credibility.md) | 核对契约、独立参考、完整断言与实际路径；不擅改冻结容限/case |
| `tilelang-op-profiling` / `ops-profiling` | 复用 develop 的 onboard-diagnostics 与 analysis 数据解析 | 保持正式性能协议与诊断重放的区别 |

API/programming-model 资料按实际目标分支查询；目标分支缺失时查绑定源码，不自动采用其他架构的默认规则。具体能力路由与范围见 [开发能力设计](../develop-capabilities/design.md)。

## 环境边界

开发、device和simulation分别配置与验证。版本、路径、设备映射、资源限制和授权按当前任务环境生效；上板成功不等于仿真可用，文档存在也不等于编译或运行通过。具体机器和运行记录不写入产品资源。
