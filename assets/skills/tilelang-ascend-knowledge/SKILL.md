---
name: tilelang-ascend-knowledge
description: 可扩展的昇腾 TileLang 算子三层知识库：按算子族与硬件瓶颈症状按需检索通用编写/评测铁律、实卡验证过的微架构模式卡、反例卡与算子族 Playbook，并在算子交付后提炼沉淀新经验。
---

# TileLang Ascend 算子知识库

本 Skill 是 Operator Harness 的长期经验沉淀与检索入口。设计原则是 **渐进式按需加载（Progressive Disclosure）** 与 **战后提炼（Post-Delivery Distill）**，避免将全部历史文档一次性塞入上下文，也不在迭代中途增加重账本负担。

## 1. 检索协议（强制按需加载）

在 `harness-port-baseline`、`harness-analysis`、`harness-develop` 中使用本知识库时，遵循以下 3 步：

1. **先读 [INDEX.md](INDEX.md)**：根据当前任务所属的 **算子族** 或观测到的 **瓶颈症状**，定位最相关的 1–3 张卡片。**不要全量遍历读取所有目录。**
2. **按层次加载**：
   - **第 1 层 `foundations/`**：通用编写范式（显式 intrinsic vs 自动调度）、Roofline/仿真节奏、物理防呆评测铁律；首次编写、切换范式或设计验证脚本时读取。
   - **第 2 层 `patterns/`（含 `patterns/falsified/`）**：可组合微架构模式卡与已证伪反例卡；遇到具体硬件瓶颈（Vector-bound、Cube 空泡、L2 miss、核间长尾）或提出新优化假设前读取，避免重踩已知死胡同。
   - **第 3 层 `families/`**：按算子族组织的 Playbook（Attention、GEMM、Conv、Reduction/Norm）；移植初版或定位架构级差距时读取。
3. **以目标绑定工具链为准核实**：知识库记录的是经实卡/仿真验证的物理机理与工程模式；具体 API 名称、intrinsic 签名与容量限制必须以项目当前绑定的 TileLang 与 CANN 版本核实，尤其留意标注为 `kind: compiler-note` 的版本相关性条目。

与 [tilelang-ascend-generation](../tilelang-ascend-generation/SKILL.md) 的协作边界：
- `tilelang-ascend-generation` 聚焦 `T.Pipelined` 自动流水与 stage macro 组织；
- 本知识库提供 **显式 intrinsic / Stage 约束 / 自动调度三者选型**、通用评测防呆铁律、跨算子微架构模式与反例库，以及按算子族持续增长的 Playbook。

## 2. 经验沉淀协议（Distill：交付后提炼，不在迭代中途堆账）

当一个算子完成阶段验收或正式交付时，再从项目私有记录中提炼**可跨项目复用的通用经验**合入本知识库：

1. **只提炼跨算子/跨场景规律**：剥离具体项目 ID、私人机器路径、内部 run 编号与硬编码外部 commit。
2. **区分长寿规律与短寿绕过**：
   - 硬件微架构规律（数据复用、流水掩盖、Cube/Vector 平衡、原子写冲突）记为 `kind: hw-pattern` 或 `kind: falsified-direction`；
   - 针对特定编译器/Pass 限制的绕过记为 `kind: compiler-note`，注明适用范围，编译器修复后及时改为 `status: superseded` 或移除。
3. **选用标准模板**：
   - 新增微架构模式：基于 [pattern-card.md](templates/pattern-card.md) 放入 `patterns/`；
   - 新增已证伪方向：基于 [falsified-card.md](templates/falsified-card.md) 放入 `patterns/falsified/`；
   - 新增或扩充算子族：基于 [family-playbook.md](templates/family-playbook.md) 更新 `families/<family>/PLAYBOOK.md`。
4. **同步索引并校验**：在 [INDEX.md](INDEX.md) 登记新卡片，然后运行校验脚本确保零孤岛文件、零死链、零私有路径泄漏：

```bash
python3 assets/skills/tilelang-ascend-knowledge/scripts/validate_kb.py
```
