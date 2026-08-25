# 技术约束清单（必须遵守）

本项目为 TileLang-NPUIR（后端为华为昇腾 NPU），与 GPU 版 TileLang 有显著差异。
**外部参考实现不可直接使用，必须转换为 NPUIR 兼容方案。**

## 目录

- [技术约束清单（必须遵守）](#技术约束清单必须遵守)
  - [目录](#目录)
  - [1. 本项目已知限制](#1-本项目已知限制)
  - [2. 强制检测规则](#2-强制检测规则)
  - [3. 警告输出格式](#3-警告输出格式)
  - [4. Host 侧 输入 操作约束](#4-host-侧-输入-操作约束)

---

## 1. 本项目已知限制

| 约束 | 说明 | 影响 | 替代方案 |
|------|------|------|----------|
| **不支持三维 Kernel** | `T.Kernel` 只接受一维 block 数 | 三维并行设计无法实现 | 将三维相乘结果作为 一维 block 数 |
| **部分 GPU API 不可用** | CUDA 专用 API 在 Ascend 不存在 | 直接移植 GPU 代码失败 | 查阅本项目 `examples/` 确认 Ascend API |
| **GEMM 要求 M,N 为 block 整数倍** | `M // block_M` 整除依赖；`M < block_M` 时零 block 启动 | 输出全零或除零编译崩溃 | 设计文档 §4/§5 必须明确处理策略：host 侧 padding+crop 或 Kernel 动态 block |
| **L0C 容量上限** | A2/A3 设备 L0C = 128KB | `block_M × block_N × sizeof(accum) > 128KB` 导致 segfault | 设计 block 时满足 `block_M × block_N ≤ 16384`（float32 accum） |
| **物理核数限制** | AI Core 物理核数有限（A2 系列 Cube 核约 20~24 个，Vector 核数量翻倍）；超发逻辑内核会被运行时串行调度并引入额外核启动开销 | 逻辑核数远超物理核数 → 串行调度性能急剧下降；内核总数非物理核数整数倍 → 负载不均（如启动 21 个内核将导致其中一个物理核执行两倍任务） | 逻辑核数 ≤ 物理核数：无需适配（附依据）；中等规模：调整 block_M/block_N 使内核总数接近物理核数整数倍（如 20/40/60）；极大规模：固定启动内核数 = 物理核数，核内 `T.serial` 串行处理多个逻辑块（`num_local_tasks = T.ceildiv(num_logical_kernels - kernel_id, num_physical_kernels)`，边界静态）。见 docs/开发指南.md §3.3 |

## 2. 强制检测规则

在设计文档生成前，**必须**执行以下检测：

| 检测项 | 触发条件 | 处理方式 |
|--------|----------|----------|
| 三维 Kernel | 参考实现包含 `T.Kernel(..., batch_count)` 或 3 个维度参数 | **立即警告**，提出 改成一维 方案 |
| GPU 专用 API | CUDA 相关 API（如 `T.gemm` 通用版） | **立即警告**，查阅本项目确认 Ascend API |
| GEMM 非整除风险 | `M` 或 `N` 不被 block size 整除（即 `M % block_M ≠ 0` 或 `N % block_N ≠ 0`） | **立即警告**，要求 design 中明确 padding 策略 |
| L0C 溢出风险 | block_M × block_N × sizeof(accum_dtype) > 131072 (128KB) | **立即警告**，建议减小 block 或拆分 |
| 分核策略缺失/不适配 | 逻辑核数远超目标设备物理核数，或内核总数非物理核数整数倍且无核内串行/对齐说明 | **立即警告**，要求 design §5 补分核策略三要素（逻辑核数计算、物理核数依据、规模判定与分核方案） |

## 3. 警告输出格式

```
⚠️ 技术限制检测警告

检测到参考实现包含本项目不支持的功能：

1. 三维 Kernel（本项目只支持一维 Kernel）
   - 参考实现：T.Kernel(m_num, n_num, batch_count)
   - 本项目方案：T.Kernel(total_blocks)
   - 参考：examples/gemm/matmul.py

建议：
- 先查阅本项目 examples/ 中的同类实现
- 确认 Ascend API 用法后再生成设计文档

是否继续生成设计文档？
```

## 4. Host 侧 输入 操作约束

> **⚠️ 核心原则：host 侧禁止改动输入 NPU 张量 内的真实内容，禁止触发任何 Padding 操作**
>
> 算子的所有核心计算逻辑（数据搬运、数学运算、归约、维度重排、padding 等）必须在 `@tilelang.jit` 装饰的 kernel 函数内部完成。**host 侧（kernel 外的 Python 代码）对 NPU 侧输入张量内的真实内容（数据值、物理排布、数据指针）一律不得改动**——只允许做只改 stride/shape 元数据的视图操作。**约束范围覆盖 kernel 调用前（输入预处理）和 kernel 调用后（输出后处理）的完整 host 代码路径。**
