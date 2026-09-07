---
disable: true
---

# conductor 场景文件：optimize（已有算子定制优化）

> **适用范围**：`scenario=optimize`。**载入时机**：场景路由确定为 optimize 后、预检开始前，按主文件 `.opencode/agents/tilelang-op-conductor.md`「场景文件加载」指引 Read 载入。通用骨架（门禁/重试/状态写入/恢复/自进化/TUNING→DESIGN 受控逆向反馈）见主文件。

## 1. 状态机与阶段计划

`stage_plan=[4, 回归]`（调优即任务本身，**不经 Stage 1/2/3**）：

```
INIT --> TUNING --> 精度回归 --> DONE / FAILED
```

- 目标算子的 `DESIGN.md` 若存在则作为参考上下文一并传给 optimizer，不存在不阻塞。
- **进入时不询问"是否调优"**（调优即任务）；调优信息在预检收集（§3），收集后同样**追加**写回 DESIGN.md"性能目标"章节 + `statectl snapshot` 重记哈希（同 new-op.md §5 约定）。

## 2. 算子目录定位与命名

- standalone 产物 → `examples/{project}/{op}/`；TileOPs 集成产物 → `examples/TileOPs/tileops/kernels/{family}/{op_slug}/{op_slug}_kernel/`（此时该目录即算子目录，`perf_opt/` 建在其下）。
- `statectl init` 时算子目录 = kernel 所在目录（`--kernel-path` 传入定位到的 kernel）。

## 3. 预检必需字段（Primary 上下文收集，缺失时问用户）

- kernel 路径（可自动定位，见 §2 两类目录规则）；
- 性能目标类型 / 数值 / baseline（字段与默认值同 `conductor-scenarios/new-op.md` §5 调优必要信息收集表，缺省 `best_effort`）；
- 回归入口（见 §4）。

## 4. 回归入口与精度回归 gate

- **回归入口**：standalone → `python {kernel_dir}/{op}.py --level all`（L0/L1 失败阻塞，L2/Boundary 告警不阻塞）；TileOPs 集成 → 优先直接跑 `python {kernel_dir}/perf_opt/{func}.py --level all`（内嵌分层测试），采纳（wrapper 切换到 perf_opt）后可再用 TileOPs pytest（`pytest tests/ops/test_{test_slug}.py`）作端到端回归。
- **精度回归 gate**：`TUNING_COMPLETED` 后你亲自对 `perf_opt/{op}.py` 执行回归入口；失败 → 重新调度 optimizer（`mode=precision_fix`，计入 `stage_retry_count[4]`——该模式只跑 L0/L1 回归修复，不重走 Phase 1 采数与已完成轮次，从当前最优版本继续，见 `_shared/standards/signal-registry.md` §2）；超限 → 交付已验证的最优版本并如实报告。
- 精度回归失败**只在 Stage 4 内 `precision_fix` 重调度，不回退 Stage 3**。

## 5. 产物写入边界与 wrapper 切换

- **产物只写 `perf_opt/`**：基准 `{op}.py` **永不修改**。
- wrapper 预置 baseline/perf_opt 双 import 切换块（integrate_kernel.py 生成，两路 import 语句并存、一路激活、注释切换）：回归通过后由你机械翻转切换块注释，使 wrapper（进而 `pytest tests/ops/` 与 `pytest benchmarks/ops/`）默认接入 perf_opt 版本；回退 = 翻回 baseline import。翻转切换块注释是唯一允许的 wrapper 修改（若两版 kernel 的 tuned 默认参数不同，连同切换块内成对的默认参数赋值一起翻转），不得改动其他内容。

## 6. `[DESIGN_LIMIT]` 特例

optimize 场景 `stage_plan=[4]`（无 Stage 1），不提供「设计修订」路由选项——设计层重做属新任务；`[DESIGN_LIMIT]` 信号仍照常产出与蒸馏，最终报告如实披露反馈与建议（其余路由规则见主文件「TUNING→DESIGN 受控逆向反馈」）。
