---
disable: true
---

# conductor 场景文件：migration-harness（TileOPs 同构仓迁移）

> **适用范围**：`scenario=migration` + `migration_mode=harness`（GPU 仓为 TileOPs 同构工程）。**载入顺序**：先 `conductor-scenarios/migration.md`（公共规则），再本文件。通用骨架（门禁/重试/设计修订/状态写入/恢复/自进化）见主文件 `.opencode/agents/tilelang-op-conductor.md`。

## 1. 状态机与阶段计划

`stage_plan=[0, (1→2→3)×N函数, 5]`（Stage 4 跳过）：

```
INIT --> SCAFFOLD --> (DEV_LOOP: 每函数 DESIGN --> REVIEW --> DEVELOP) --> INTEGRATE --> DONE
                        ^                |
                        |____ 修订循环 ___|  (retry_count < max_retry，仅当前函数)
  超限/失败 ____________________________> FAILED (BLOCKED_SCAFFOLD / BLOCKED_IMPL / BLOCKED_ACCURACY / BLOCKED_INTEGRATION)
```

## 2. 命名与预检特例

- **多函数命名约定**：逐函数独立 Stage 1-3，`project_name={op_slug}`、`op_name={func}`，算子目录 `examples/{op_slug}/{func}/`（**不使用** new-op.md §2 的用户消息解析规则）。
- GPU 参考实现位置与规格（manifest workloads、test/bench 路径）从 Stage 0 的 `.migration_meta.json` 读取，调度 designer 时随 prompt 传入。
- 预检不问用户：规格从 GPU 仓 meta 与源码推断（迁移公共规则见 migration.md §1）。

## 3. Stage 0 — 迁移脚手架 Agent（`@tileops-scaffolder`，仅 harness）

- **触发条件**：场景路由判定 `migration` + `migration_mode=harness`，状态 `init` 后
- **输入**：`op_name`、`gpu_repo_root`、`family`（可选，默认 reduction）
- **输出/交付件**：TileOPs 7 文件脚手架（Tier 1 结构校验通过）+ `examples/TileOPs/tileops/kernels/{family}/{op_slug}/.migration_meta.json` + 逐函数迁移 prompt
- **完成信号**：`SCAFFOLD_COMPLETED`（携带 meta_path、extracted_functions、migration_prompts）
- **失败信号**：`[SCAFFOLD_FAIL]` + 原因
- **编排层动作**：
  - `SCAFFOLD_COMPLETED` → `complete_stage(0)` → 解析 `extracted_functions`，经 `statectl migration init` 建 `.migration_state.json`（见 §4），进入逐函数 DEV_LOOP
  - `[SCAFFOLD_FAIL]` 且属结构问题 → `fail_stage(0)` 重试（≤3 次）；"GPU 侧无 `@tilelang.jit` 实现" → `phase=FAILED`、`failure_reason=BLOCKED_SPEC`；GPU repo 缺失 → `BLOCKED_ENVIRONMENT`；结构重试超限 → `BLOCKED_SCAFFOLD`

## 4. 多函数组织（DEV_LOOP 与聚合状态）

- Stage 0 产出的 `.migration_meta.json` 中 `extracted_functions` 为函数列表（N ≥ 1）。
- **每个函数独立跑 Stage 1→2→3**：算子目录 `examples/{op_slug}/{func}/`；每函数独立 `.stage_state.json` 与独立 `stage_retry_count` 预算。
- 函数间不共享 DESIGN.md；某函数 `[DESIGN_ERROR]` 只修订该函数。
- 全部函数 Stage 3 通过（含二次校验）后才进入 Stage 5。
- 聚合状态由你维护在 `examples/{op_slug}/.migration_state.json`：

```json
{
  "op_name": "{op_name}", "op_slug": "{op_slug}", "family": "{family}",
  "meta_path": "examples/TileOPs/tileops/kernels/{family}/{op_slug}/.migration_meta.json",
  "phase": "SCAFFOLD | DEV_LOOP | INTEGRATE | DONE | FAILED",
  "functions": {"{func}": {"stage_state": "examples/{op_slug}/{func}/.stage_state.json", "status": "pending | in_progress | done | failed"}},
  "integration": {"attempts": 0, "status": null}
}
```

- 聚合状态的迁移与函数状态更新一律经 `statectl migration` 命令组执行：`migration init --op-slug S --family F --meta-path P --functions f1 f2`（Stage 0 完成、解析函数列表后创建，phase=DEV_LOOP）→ 逐函数 `migration func {name} --status pending|in_progress|done|failed` → 全函数 done 后 `migration phase --phase INTEGRATE` → 集成结果 `migration integrate --status pass|fail`（fail 超 2 次自动 `BLOCKED_INTEGRATION`）→ pass 后 `migration phase --phase DONE`。Stage 0/5 的工件门禁用 `statectl gate 0 --meta ...` / `statectl gate 5 --migration-dir ...` 单独执行。

## 5. Stage 3 通过后的分支

`[PRECISION_PASS]` 且二次校验通过 → `migration func {func} --status done` → 还有未完成函数则对其 `start_stage(1)` 继续 DEV_LOOP；全部函数 done → `migration phase --phase INTEGRATE` → `start_stage(5)` 调度 integrator。

## 6. Stage 4 跳过

harness 迁移**不询问调优、不进入 Stage 4**（bench 由 Stage 5 仅报告）；最终报告附 bench 数值与"可另起 optimize 场景"提示。

## 7. Stage 5 — 迁移集成 Agent（`@tilelang-op-integrator`，仅 harness）

- **触发条件**：全部提取函数 Stage 3 通过且二次校验完成（`.migration_state.json` 的 `functions` 全部 `done`）
- **输入**：`meta_path`、`op_name`、`op_slug`、`family`、`attempt_index`、`max_attempts`（默认 5）
- **输出/交付件**：`tileops/kernels/{family}/{op_slug}/{op_slug}_kernel/`（集成 kernel 文件 + 每函数 `{func}_DESIGN.md` 设计文档快照（源自 `examples/{op_slug}/{func}/DESIGN.md`）+ 聚合 `__init__.py` + `integration_log.md`），wrapper import 已改写为 baseline/perf_opt 双 import 切换块（baseline 默认激活，perf_opt 注释占位）
- **完成信号**：三态之一：`INTEGRATE_COMPLETED`（TileOPs pytest smoke+全量通过，bench 已报告）/ `[INTEGRATE_FAIL]` / `[DESIGN_ERROR]`
- **编排层动作**：
  - `INTEGRATE_COMPLETED` → `complete_stage(5)` → `phase=DONE`（harness 迁移不询问调优；最终报告附 bench 数值与"可另起 optimize 场景"提示）
  - `[INTEGRATE_FAIL]` → `fail_stage(5)` → 重新调度 integrator 传入 `last_failure_summary`（`stage_retry_count[5]` 上限 2；integrator 内部已有 5 次调试闭环，两级预算独立）；超限 → `phase=FAILED`、`failure_reason=BLOCKED_INTEGRATION`
  - `[DESIGN_ERROR]` → 设计修订循环路径 B：对**失败根因指向的函数**备份其 `DESIGN.md` → `retry_count += 1` → 该函数重跑 Stage 1→2→3 → 通过后**重新执行 Stage 5**（全量重集成，集成脚本幂等）

## 8. harness 设计修订特例

- `retry_count` 为**全 op 共享预算**（跨函数累计），达 `max_retry` 即整个 op `FAILED`。
- 修订只回退**失败根因指向的当前函数**，其他已完成函数的工件不动。
- 修订后该函数重跑 Stage 1→2→3，通过后 Stage 5 **全量重集成**（集成脚本幂等）。

## 9. harness 目录结构

```text
examples/{op_slug}/               # op 级目录（project = op_slug）
├── .migration_state.json         # conductor 维护的多函数聚合状态
├── RETROSPECTIVE.md              # Stage 5 集成复盘（op 级单份，自进化钩子）
└── {func}/                       # 每个提取函数一个算子目录（结构同主文件标准目录，无 Stage 4；含函数级 RETROSPECTIVE.md）

examples/TileOPs/                              # 集成侧
├── tileops/manifest/{family}.yaml             # Stage 0 产物（S1）
├── tileops/workloads/{family}.py              # Stage 0 产物（S2）
├── tileops/kernels/{family}/{op_slug}/
│   ├── {op_slug}.py                            # wrapper（Stage 0 移植，Stage 5 改写为 baseline/perf_opt 双 import 切换块）
│   ├── .migration_meta.json                    # Stage 0 机器模式产物
│   └── {op_slug}_kernel/                       # Stage 5 集成包
│       ├── {func}.py                           #   集成 kernel（源自 examples/{op_slug}/{func}/）
│       ├── {func}_DESIGN.md                    #   集成设计文档（源自 examples/{op_slug}/{func}/DESIGN.md，Stage 1 交付件快照）
│       ├── __init__.py                         #   聚合 re-export（integrate_kernel.py 生成）
│       ├── perf_opt/                           #   Stage 4 调优产物（optimize 场景：{func}.py + opt_log.md；wrapper 切换块的 perf_opt import 指向此处）
│       ├── integration_log.md                  #   集成验证与调试日志
│       └── history_version/                    #   Stage 5 调试备份
├── tests/ops/test_{test_slug}.py               # Stage 0 产物（S5，仅含本算子用例）
└── benchmarks/ops/bench_{bench_slug}.py        # Stage 0 产物（S6）
```

### harness 专属工件衔接（Owner / Consumer）

| 工件 | Owner | 主要消费者 | 消费者需要的信息 |
|------|-------|------------|-----------------|
| TileOPs 7 文件脚手架 | Stage 0 | Stage 1（规格来源）、Stage 5（集成目标） | manifest workloads、wrapper/Kernel class、test/bench 路径 |
| `.migration_meta.json` | Stage 0 | conductor（函数循环）、Stage 5（集成参数） | op_slug / family / extracted_functions / wrapper_path / test_slug / bench_slug |
| `.migration_state.json` | conductor | conductor | 多函数聚合状态（见 §4） |
| `{op_slug}_kernel/`（集成包） | Stage 5 | 用户、TileOPs 框架 | 集成 kernel 文件 + `{func}_DESIGN.md` 设计文档快照 + 聚合 `__init__.py` + `integration_log.md` |

## 10. 带记忆重试与 session 教训传递（harness 专属）

- 主文件「自进化机制」第 2 条的重试前必读标准段，在 harness 多函数任务中**另加**一条检索位置：`examples/{op_slug}/{前序函数}/RETROSPECTIVE.md` 的 Transferable Lessons 小节。
- 调度第 N 个函数（N ≥ 2）的 designer / developer 时：
  1. Read 前序各函数 `examples/{op_slug}/{前序函数}/RETROSPECTIVE.md` 的「Transferable Lessons（可迁移教训）」小节（文件或小节不存在则跳过）。
  2. 将其内容**逐字追加**到调度 prompt 的「前序函数教训」段——你只搬运不加工不筛选（领域判断属于 Subagent，你保持零领域推理）。
  3. 该教训仅在本迁移任务内有效；**不得**写入任何持久文件（跨任务沉淀由 evolver 在终态蒸馏时统一裁决）。
