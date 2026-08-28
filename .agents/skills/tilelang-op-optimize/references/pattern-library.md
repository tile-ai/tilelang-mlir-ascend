# NPU 性能模式库与证伪协议

> 本文件收录**已实测验证**的性能模式、代价数据与实验方法学规则。来源：真实调优任务的 msprof/NPU event 实测（首版素材：AvgPool2dFwdOp optimize 任务，2026-08，Ascend910B2C，数据可追溯至该任务 `perf_opt/opt_log.md`）。
>
> **维护规则**：每次调优任务产出的新模式/代价数据/证伪更正，由调优 Agent 在任务结束前追加到对应章节（含任务溯源与工具链版本）。模式库只有在"每次被读"的位置上才不会遗忘。

## 1. 向量化轴与布局模式（优先级最高的检索项）

**诊断触发器**：msprof 显示算子热点段**标量执行占比 > 50%** 时，强制触发"换向量化轴/换布局"分析——不要在原轴上微调（实测案例：窗口累加 98.3% 标量源于 `j·sW`（sW≠1）跨步系数阻碍向量化，原轴微调收益为 0，换轴后 5.6x）。

### 1.1 核内融合转置链（布局重排的正确形态）✅ 已验证

- **模式**：I/O 保持原生契约（如 NCHW），核内用 **T.transpose 二轴交换链**重排到目标布局（如 NHWC）。3D 置换 = 两次二轴交换：
  ```
  (CH, Hi, Wi) -[1,0,2]-> (Hi, CH, Wi) -[0,2,1]-> (Hi, Wi, CH)   # 输入侧
  (BH, WO, CH) -[0,2,1]-> (BH, CH, WO) -[1,0,2]-> (CH, BH, WO)   # 输出侧
  ```
- **实测代价**：4 次融合转置合计 **~5.4µs**（远低于算子本体）——"转置很慢"是错误印象，勿因直觉弃用。
- **硬约束**：① `T.transpose` 的 permutation **仅支持二轴交换**（如 `[1,0,2]`/`[0,2,1]`），**不支持 3-cycle**（`[1,2,0]`）——3D 置换必须拆成两次交换链；② **reshape→transpose 链会 mis-compile（数据错乱）**——必须始终在自然形状 buffer 之间做 transpose；③ 转置须在**自然形状** shared/UB buffer 间进行。
- **佐证**：`testing/npuir/broken/test_slice_transpose_dev.py`；AvgPool2d perf_opt v3a（`examples/TileOPs/tileops/kernels/pool/avg_pool2d/avg_pool2d_kernel/perf_opt/`）。

### 1.2 C 轴切片累加（channel-batched vector accumulation）✅ 已验证

- **模式**：`T.vadd(acc[i, j, :], in_f32[i*SH+ki, j*SW+kj, :], acc[i, j, :])`——整 C 轴切片作为向量操作数，**i/j 用 T.serial**（不是 Parallel），ki/kj serial 外层。
- **适用**：C 恒为向量宽度整数倍的 NCHW 类算子（C=64/96/128；96 用 CH=48/32/16 分块）。
- **实测收益**：AvgPool2d kernel 级 1.98–13.2x（vs 同代 H-collapse 版本）。

### 1.3 host permute 路线 ❌ 通常净亏（已量化证伪）

- 两次 host permute（NCHW↔NHWC）实测 **106–147µs** > 多数中等算子本体耗时；仅当算子本体远大于此（如 >500µs）且无法核内重排时才值得复测。

### 1.4 tiling 启发式（C 轴向量化形态）✅ 已验证

- **BH=1 + 最宽 CH 最优**：CH64 21.6µs < CH32 26.7µs < BH2/CH16 44.8µs——更宽 C 向量 + 更少 block 胜过更高空间并行度。
- **UB 约束**：CH×空间 tile 驻留超 192KB 时收缩 CH（实测 vis-3x3 的 CH=64 需 252.8KB 被排除，CH=32 落地）；buffer 生命周期不重叠时可探索 aliasing 复用（预估再省 10–20%，未验证）。
- **回退分发**：C%16≠0 或 shape 越界时回退到非重排路径（工厂层静态判定，无运行时开销）——保持全 shape 兼容。

### 1.5 其他已验证模式速查

| 模式 | 一句话 | 实测参考 |
|---|---|---|
| 乘编译期常数倒数 | divisor 恒定时 `T.vmul(acc, 1/d, acc)`，省逐元素除法 | AvgPool2d fast path |
| H-collapse | 把 kH 折叠进向量管道，消除一层 serial 累加 | AvgPool2d v2（对 sW 跨步问题的 W 轴解法） |
| fp32 求和序匹配 | kernel 累加序与 golden（F.avg_pool2d 等）一致时解锁高精度快路径 | AvgPool2d fp32 13.2x |
| launch 开销主导判定 | 小 shape 与大 shape 耗时同量级 → 固定开销主导；拆 pad-only/padded-in 定位 | bench_runner `--decompose` |

## 2. 已知编译器/运行时陷阱（工具链版本绑定 ⚠️）

> **证伪协议（强制）**：
> 1. **否定任何 API/模式前，必须用文档合法形态测试**——先查 `docs/Tilelang.language/` 确认 API 的合法参数/形式，穷举代表性写法后再下结论。实测教训：曾以非法 3-cycle permutation（`[1,2,0]`）与 Parallel 循环累加形式测出"编译失败"，误判"transpose 链/C 轴累加不可用"，掩盖了 2–13x 收益。
> 2. **一切编译器约束结论必须盖工具链版本戳**（tilelang commit/build 时间 + 来源任务），工具链变更（源码修改/重编译）后**自动视为待重验**，不得直接引用旧结论。
> 3. 证伪更正时须在 opt_log 写明"误判根因 + 合法形态 + 新数据"。

| # | 陷阱 | 状态（含版本戳） |
|---|---|---|
| C1 | 旧编译器 transpose/C-slice 形态不可用 | **已失效**（2026-08-27 tilelang 重编译后推翻；合法形态见 §1.1/1.2） |
| C9 | `TILELANG_ENABLE_TASKQUEUE=false` 异步 launch 损坏 fp32 数据 | 有效（截至 2026-08-28 build） |
| C10 | 跨步 gather 向量指令缺失（非 C 轴连续的跨步访问小心） | 有效（同上） |
| C11 | parser var-table 作用域问题 | 有效（同上） |
| C12 | `T.copy` 静默跨 dtype 转换 | 有效（同上） |
| — | UB dst 非零起点切片 + 32B 倍宽度触发 VEC 对齐错误 | 有效（同上；host pad 或 0 起点拷贝绕开） |
| — | vbrc 标量→shared 不可用；serial 变量条件 if_then_else select segfault | 有效（同上） |
| — | event 口径 ±15% session 漂移 | 方法学约束：**msprof kernel 真值决胜**，event 仅同 session 非回退检查；bench 前后检查设备空闲 |

## 3. 检索规则（调优 Agent 每轮必读）

1. **首轮必查项**：① 向量化轴/布局重估（对照 §1，即使上游设计已定——设计期结论可能基于旧工具链）；② 标量执行占比诊断（>50% 触发换轴）；③ 本模式库 §2 版本戳核对（工具链变更则重验）。
2. 新模式/新代价数据/证伪更正 → 任务结束前追加进本文件对应章节（含溯源）。
3. 本库模式是**起点不是终点**：每算子的最优 tiling（CH/BH/block）仍须实测扫描。
