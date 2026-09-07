# Stage 3 调度与失败路由标准 ⭐

> **单一事实源**：本文件是「Stage 3 调度模型 / 四出口路由 / 运行失败子类型路由」的权威版本。conductor 在 Stage 3 失败路由时读本文件；README 只链接摘要。四出口判定与信号 token 的枚举出处是 [signal-registry.md](signal-registry.md)。

## 1. 调度模型

每次调用 `@tilelang-op-developer` = 1 次 attempt；Developer 不在单次调度内自循环。累计 attempt 上限 **5 次**（`stage_retry_count[3]`）：因运行失败超限 → `BLOCKED_IMPL`；因精度失败超限 → `BLOCKED_ACCURACY`。每次调度的 prompt 必须明确：`attempt_index`、`mode`、`last_failure_summary`（若有）、`design_revision_count`。

## 2. 四出口路由表

| Developer 返回 | mode（下次调度时） | 路由 |
|---------------|------------------|------|
| `[PRECISION_PASS]` | — | `complete_stage(3)` → **二次校验精度**（重新跑全量 `--level all` 确认真实性）→ 按 `scenario` 分支：`new_op`/`migration-plain` 询问用户是否需要性能调优；`migration-harness` 不询问，标记该函数 `done` 进入下一函数（全部完成则 `start_stage(5)`）。此时 Developer 已在 L0 通过后扩展并跑过 L1/L2/Boundary 全量；L2/Boundary 告警仅记录不阻塞 |
| `[PRECISION_FAIL]` | `precision_fix` | Stage 3 内重试（L0 或 L1 未达标）。把失败信息作为 `last_failure_summary` 传入。**强制要求 Developer 先备份当前 impl 到 `history_version/{op}_impl_s3_attempt{N}.py` 再做修改** |
| `[DESIGN_ERROR]` | — | 触发设计修订循环（路径 B，计 `retry_count`） |
| 无标记且 exit code ≠ 0 | `retry_impl` | Stage 3 内重试，将 stderr 摘要作为 `last_failure_summary` 传入 |
| 首次进入 Stage 3 | `first_impl` | 调 `tilelang-op-develop` skill 从零生成 kernel + L0 用例，先跑 L0 |

> **分层测试**：Stage 3 每次 attempt 先只跑 L0 做精度收敛；L0 通过后 Developer 调用 `tilelang-op-develop` skill 扩展 L1/L2/Boundary 并跑全量。**L0/L1 失败**才算精度未达标（走 `precision_fix`）；**L2（异常）/ Boundary（特殊值）失败仅记录到 `debug_log.md` 与覆盖率报告，不阻塞 `[PRECISION_PASS]`**。

## 3. 运行失败子类型路由

Stage 3 返回运行失败（无标记且 exit code ≠ 0）时按子类型路由：

| 子类型 | 识别信号 | 路由策略 |
|-----------|---------|---------|
| 编译错误（实现层） | stderr 含 lowering / codegen 相关错误，且不属于设计层 API 误用 | Stage 3 内重试，要求 Developer 修复 |
| Import 错误 | `ImportError` / `ModuleNotFoundError` | 检查环境依赖，若缺 TileLang 模块或未 `source set_env.sh` 可标记 `BLOCKED_ENVIRONMENT` |
| Shape 不匹配（实现层） | `shape mismatch`、`size mismatch`、tile shape 不一致 | Stage 3 内重试，将 shape 错误传入 Developer |
| 内存层级越级 | stderr 提示 GM/L1/UB/L0 访问违规 | Stage 3 内重试，提示 Developer 复核 AGENTS.md 原则 4（硬件内存层级） |
| 分核相关运行异常 | 跑测显著超时、kernel 启动数量异常（逻辑核数远超物理核数被串行调度） | Stage 3 内重试，传入超时/核数信息；Developer 判定为设计层分核缺陷时按 `[DESIGN_ERROR]` 走修订路径 B |
| Pass / IR 变换错误 | stderr 含 `tilelang/transform` 或 IR pass 报错 | Stage 3 内重试，传入完整 stderr |
| **设计层错误** | Developer 输出明确加 `[DESIGN_ERROR]` 标记 | 走设计修订循环（路径 B） |
| 其他运行时错误 | exit code ≠ 0 且不属于以上 | Stage 3 内重试，传入完整 stderr |
