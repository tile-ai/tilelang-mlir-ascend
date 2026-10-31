# 现有资产复用与迁移计划

状态：2026-10-31 源码与文档初查，未运行编译、仿真或实卡验证。本轮只建立迁移计划，没有复制或启用旧 skills，也没有修改来源仓库。

## 1. 检查范围与证据

来源以用户提供的 checkout 为准。以下相对路径均相对于表中的来源仓库，不是本 harness 已有文件。个人机器路径只保留在本地研究记录，后续安装通过配置解析。

| 来源 | 检查版本 | 主要证据 |
| --- | --- | --- |
| asc-devkit | `5dd381e37e134445ae3ef160320f3a7fb7d7986d` | `.agent/skills/*/SKILL.md`、`LICENSE` |
| simulator | `b8bd99e5480aa45d0ba22a1a42931a16dc2d03c5` | `README.md`、`bin/simctl`、`bin/project-container`、`lib/env.sh` |
| Onboard | `039f98eb1b729de21bec04d4137ed43b81101671` | `README.md`、`deep_jit/remote/`、`deep_jit/frontends/tilelang_ascend.py`、`docs/remote-execution/README.md`、`docs/profiling.md` |
| TileLang 上游 | `0176ba1465b12758f41802bf838ddd68d43c9ee9` | [Ascend 文档](https://github.com/tile-ai/tilelang/blob/0176ba1465b12758f41802bf838ddd68d43c9ee9/tilelang/ascend/README.md) |
| GSD Core | `next` 的 `1ec5c812b79dc1cc487d0873d18b03449b11c106` | [阶段流程文档](https://github.com/open-gsd/gsd-core/blob/1ec5c812b79dc1cc487d0873d18b03449b11c106/docs/explanation/the-phase-loop.md) |

结构查询先使用代码图；simulator 的 `bin/`、Onboard 的 `docs/` 以及部分新增/变化的 skills 存在索引覆盖缺口，相关结论已回到源码读取。此处是有范围的复用评估，不是全部代码或全部平台的审计。

## 2. Skills：按职责迁移

现有目录的七个 skill 入口覆盖不同任务，其中部分服务于 devkit 自身开发。首批仅接入 TileLang，迁移以支持 TileLang 开发和生成产物调试为范围；独立 AscendC 开发技能保留为后续候选。

| Skill | 可复用内容 | 建议处理 |
| --- | --- | --- |
| `asc-ar-dev` | 需求分析、API 文档核对、host/device 分工、验证证据要求 | 首批提取通用方法；完整 AscendC 开发能力迁移留待后续语言扩展 |
| `asc-npu-arch` | 架构与产品映射、能力边界、结构化事实 | 共用架构知识来源；事实与版本绑定，使用前核对实际 SDK/目标芯片 |
| `asc-search` | 按版本和架构检索 API、索引新鲜度检查 | 保留检索思路和实现候选；先改为显式知识源路径，再验证任意项目目录中的调用 |
| `ascendc-api-kernel-debug` | 调试接口选择、输出链路定位、调试后清理 | 作为可选调试能力；修复关联 skill 路由并核对资料版本 |
| `asc-api-ut-gen` | API 参数化测试和覆盖率方法 | 暂不整体迁移；它面向 devkit API UT，不能替代算子任务的实卡测试 |
| `asc-doc-checker` | devkit API 文档规范 | 暂不迁移为 harness 通用文档规范 |
| `public-skills` | 旧仓库的公共技能安装与 GitCode 工作流 | 暂不迁移；不属于算子生成核心链路 |

直接复制前有明确的适配工作：`asc-search/SKILL.md` 通过自身路径向上定位 devkit，且假定命令从仓根执行；`asc-api-ut-gen` 也假定技能位于 devkit 内。npm 安装后的目录关系不同。调试 skill 还引用 `ascendc-docs-search`，需确认实际替代入口，不能生成悬空依赖。

迁移后的产品 skills 放入 `assets/skills/`，由安装器适配宿主发现路径。每项迁移记录来源 revision、改动、依赖资料和验证方式；复制文件时保留适用许可与原有声明。来源根目录当前提供 CANN Open Software License Agreement Version 2.0，应随迁移明确文件归属和分发材料。

TileLang 路径需要单独整理当前上游的文档、示例与语言规则。AscendC skills 可辅助理解生成代码和底层调试，不能直接作为 TileLang 前端的 API 规范。首批完成一份可实际运行的 TileLang 开发流程，独立 AscendC 路径在后续有需求时再评估。

## 3. simulator：复用执行与分析能力，增加任务接口

观察到的调用链：

```text
bin/simctl 的命令分派
  → cmd_profile / cmd_cycles / doctor 等处理
  → lib/env.sh 加载环境
  → msopprof simulator 或 tools/ 中的分析脚本

bin/project-container
  → docker create / exec
  → 容器内的 simctl 或用户命令
```

`cmd_profile` 已包含锁、外层超时、输出路径和失败返回码处理，并保留 partial profile；`project-container` 提供独立 workspace、容器创建和执行入口。这些适合先通过执行适配复用，再增加 MCP 外层接口。

需要补齐或重新验证：

- 从命令输出和目录约定形成结构化任务状态、产物清单与错误类别。
- 远程任务提交、传输、结果回收、取消及清理确认。
- host/docker 两种环境的显式配置、doctor 与兼容性测试。
- 将镜像、架构、频率、工具链等默认值转为环境能力和版本信息，不能推广固定实验环境的假设。
- 为每次分析记录来源候选、编译产物、采集参数及完整性；残缺 trace 不产生完整 kernel 的性能结论。

已检查入口体现了本地命令和 Docker 管理能力。本次没有验证远程适配，也没有重跑本地 Docker；四组合支持仍是待验收目标。README 中的历史实验不能视作本 harness 的测试结果。

## 4. Onboard：保留编译与设备执行能力，解耦任务组织

当前代码的 TileLang 远程调用链为：

```text
RemoteRuntime.compile_tilelang
  → frontends.tilelang_ascend.lower_to_ascendc
  → 本地 tilelang.lower，保存 AscendC/IR 并收集编译参数
  → compile_ascendc / _compile
  → SSHTransport 启动 deep_jit.remote.worker
  → Worker.compile 在远端构建产物
  → 设备 launch、同步、输出回传与比较
```

可复用部分包括语言入口、源码/头文件打包、工具链身份检查、编译缓存、缓冲区传输、设备执行以及 profiling 回传。`compile_tilelang` 与 `compile_tilelang_opentile` 已分开，应继续明确语言仓库和编译链路身份。

需要处理的契约差异：

| 观察依据 | 对 harness 的影响 |
| --- | --- |
| `remote/transport.py` 启动 SSH worker，配置键限定为主机、Python、设备、超时等 | 已检查入口没有显式 Docker 配置；本地/远程 Docker 需定义独立适配并实测 |
| `frontends/tilelang_ascend.py` 在客户端调用 lowering；远程指南要求本地 CANN 身份与依赖检查 | 编译位置与运行位置必须分别声明，远程执行不能默认承诺客户端零 SDK |
| `compile_tilelang` 要求显式入口，launch 参数与 blocks 仍由调用方提供 | 任务与语言适配需要可靠地提供 ABI、shape 和 launch 信息 |
| `docs/profiling.md` 明确采集不自动预热或重复 | harness 需补齐稳态采样、重置、逐 case 比较与统计规则 |
| 设备耗时与 `profile_pipeline` 使用不同采集方式 | 实卡性能和流水分析分别运行、记录，不能混用耗时 |
| SSH 失败/超时会关闭会话且不自动重放 | 恢复流程需要查询/确认清理状态，不能盲目重试设备副作用 |

建议先增加面向任务的适配层，使用小型 case 验证契约，再判断哪些 Onboard 内部模块需要重构。现有接口存在不等于与指定上游版本兼容；必须验证 TileLang → 编译 → 实卡运行的完整链路。

## 5. GSD：借鉴机制与整体依赖分别评估

首批已借鉴其新项目与已有项目接入流程，独立实现 `harness-new-project`、`harness-onboard`，作为 Codex skills 成对提供。流程记录写入 `.workspace/`，不依赖 GSD runtime、不保留 `gsd-*` 命令别名，也不自动提交。使用方式与固定版本来源见 [README](../README.md#使用方式) 和 [来源与许可](../assets/skills/harness-new-project/references/provenance.md)。这不涉及迁移 asc-devkit 的开发 skills。

当前文档强调阶段决策、计划、执行、验证和持久状态。可以映射到算子需求、环境、标杆、实现和迭代阶段，但它的通用阶段完成与 Ship 语义不能直接作为算子达标判据。

建议先用 [需求与工作流](REQUIREMENTS.md)中的阶段检查作为评估标准。无论是否采用 GSD，正确性比较、性能统计和候选证据失效都由独立程序执行。现已通过 `harness-resume` 提供基于宿主子 agent 的上下文隔离和交接约定；整体依赖、版本兼容和独立调度服务仍未决定。

## 6. 复用工作的完成标准

每项资产依次经过“来源与版本明确 → 契约适配 → 单独调用验证 → harness 端到端验证”。源文件复制、配置存在或 README 声明均不单独构成支持证明。

本轮先完成上述清单。服务于 TileLang 的迁移与工具封装按 [路线图](ROADMAP.md) 的 M1–M4 推进，持续记录执行组合及能力的验证状态。独立 AscendC 语言接入不纳入首批验收。
