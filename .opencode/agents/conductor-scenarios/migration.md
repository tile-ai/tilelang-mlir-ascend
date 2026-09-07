---
disable: true
---

# conductor 场景文件：migration（公共规则 + plain 子模式）

> **适用范围**：`scenario=migration`。本文件含两个子模式**共享**的迁移规则（§1–§5）与 **plain 子模式专属规则**（§6，普通 GPU 仓）；harness 子模式专属规则另见 `conductor-scenarios/harness.md`（TileOPs 同构仓，Stage 0/5 激活）。**载入顺序**：plain 子模式只读本文件；harness 子模式先读本文件、再读 harness.md（路由探测出 `migration_mode` 后由主文件 `.opencode/agents/tilelang-op-conductor.md`「场景文件加载」指引按序 Read 载入）。

## 1. 迁移执行规则（公共项）

1. **严格**使用 `@tilelang.jit()` 所装饰的函数名作为算子名，不擅自裁剪变换。
2. `@tilelang.jit()` 装饰的函数（TileLang 内核函数）声明在迁移前后保持不变。
3. `@T.prim_func()` 装饰的函数（TIR 原语函数）参数名称及顺序在迁移前后保持不变。
4. 从用户提供的算子代码工程里推断输入张量规格，不用询问用户。
5. **编程模式默认 `developer`**（迁移类不问用户编程模式；用户显式指定时以用户为准）。
6. **源算子路径必须透传 Stage 1/2**：调度 designer 与 reviewer 时必须在 prompt 中传入 `source_op_path`（plain=用户给出的源文件路径；harness=`.migration_meta.json` 中该函数的 GPU 源码路径，缺失时从 `gpu_repo_root` 定位）。Stage 1 据此执行「源算子三问解读 → 算法调研（Phase R，源算法只是候选之一）→ 硬件耦合性判定 → NPU 算法重设计」（DESIGN.md §0 + §1.6.0），Stage 2 据此核对 §0 解读与源码一致性；两者均不得在未读源码时臆测语义。

> 子模式差异：harness 的多函数命名约定与 meta 读取见 harness.md §2；plain 的规格推断与精度门禁见本文件 §6。

## 2. Stage 1 增补（源算子解读与迁移分析）

在通用 Stage 1 输入（`op_requirements` 结构，格式见 `conductor-scenarios/new-op.md` §4）之上，迁移任务另传：

- `source_op_path`（源算子文件路径）与 `source_output_shape`（源算子输出 shape，如 (M, N)；无法从源码推断时由 designer 解读后回填并标注依据）。
- `DESIGN.md` 另含 **§0 源算子解读与迁移分析**（三问解读 + 算法调研 + 硬件耦合性判定 + NPU 算法重设计，**源算法已识别的优化手段不得静默丢弃**），§0 与 §1.6.0 的完整要求由 `tilelang-op-design` skill 的迁移执行流（Phase M0 → R → M1）承载，`statectl gate 1` 对 §0 章节做机械存在性校验。

## 3. Stage 2 增补（9 维度检视）

- 迁移任务执行 **9 维度检视**（非迁移任务 8 维度，维度 0 标 n/a）：新增**维度 0「源算子理解与迁移分析」**——语义/算法/优化手段解读核对、耦合性判定合理性、NPU 重设计可行性与设计一致性、golden 独立性（reviewer 须亲自 Read 源码核对 §0，输入需传 `source_op_path`）。
- 其余维度要求（含维度 8 算法优化分析、维度 3 分核策略核对）同通用规范，见主文件「各 Agent 交互规范」Stage 2。

## 4. 迁移类 `[DESIGN_ERROR]` 典型场景

权威清单见 `.agents/skills/_shared/standards/gate-and-retry.md` §4（表末三条迁移专属行：源算子语义理解偏差 / 照搬源硬件方案 / NPU 重设计不可行——识别信号均出自 Stage 2 维度 0 或 Stage 3 编译根因）。

## 5. 设计修订的迁移增补

- 设计修订（路径 A/B/C 任何一条）重调度 designer 时**必须传入 `source_op_path`**——若错误指向 §0，designer 需重新读源码重做迁移分析。
- 修订后的新 `DESIGN.md` 重新进入 Stage 2 时，重新执行**含维度 0 的 9 维度检视**（含维度 8 算法优化分析）。

## 6. plain 子模式（普通 GPU 仓，`migration_mode=plain`）

### 6.1 状态机与阶段计划

`stage_plan=[1, 2, 3, (4?)]`（同 new_op；Stage 4 可选）：

```
INIT --> DESIGN --> REVIEW --> DEVELOP --> TUNING(可选) --> DONE
  ^                 |
  |___ 修订循环 ____|  (retry_count < max_retry)
  |___ 超限 _______> FAILED
```

### 6.2 命名与预检

- 项目/算子命名解析**同 `conductor-scenarios/new-op.md` §2**（从用户消息解析；解析不出 `project = op`）。
- 规格从用户给出的 **GPU 源码直接推断**（不问用户，迁移公共规则见本文件 §1）。
- `source_op_path` = 用户给出的源算子文件路径（透传 Stage 1/2，见 §1 规则 6）。

### 6.3 阶段差异

- **无 Stage 0 / Stage 5**（不调度 scaffolder / integrator；`statectl start 0/5` 会被 `E-NOT-IN-PLAN` 拦截）。
- **无结构化用例仓**：精度门禁 = Stage 3 内嵌 L0/L1（L0/L1 失败阻塞，L2/Boundary 告警不阻塞）。
- Stage 3 返回 `[PRECISION_PASS]` 且二次校验通过后，**询问是否调优**：流程与调优信息收集同 `conductor-scenarios/new-op.md` §5（两场景共用）。
