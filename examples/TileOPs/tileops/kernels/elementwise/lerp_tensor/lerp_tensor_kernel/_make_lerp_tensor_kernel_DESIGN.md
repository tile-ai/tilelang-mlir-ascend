# _make_lerp_tensor_kernel 算子设计文档

> 迁移任务（harness 子模式）：源算子为 GPU 仓 TileOPs 的 `_make_lerp_tensor_kernel`（Pattern B 工厂闭包），迁往 TileLang-NPUIR（target="npuir"，华为昇腾 NPU）。本文档 §0 记录源算子三问解读 → 算法调研 → 硬件耦合性判定 → NPU 算法重设计的全过程，§1–§11 基于重设计后的 NPU 算法展开。

## 0. 源算子解读与迁移分析（迁移类任务必填）

### 0.1 源算子语义（做什么）

- **数学语义**：`out[i] = a[i] + w[i] · (b[i] − a[i])`，即 `torch.lerp(input, end, weight: Tensor)` 的 **Tensor-weight 三输入逐元素插值**变体。Op 层（`LerpTensorFwdOp`）先把 `input` / `end` / `weight` 三输入按 broadcast 规则 expand 到同一 shape 并物化为连续一维张量（`expand → contiguous → view(-1)`），kernel 只看三个长度 `N` 的连续一维张量；输出 `(N,)` 由 Op 层 `view(out_shape)` 还原。
- **I/O 契约**：三输入一输出，全部 `(N,)` 一维连续；三输入同 dtype；输出 dtype = 输入 dtype（`output_dtype` 缺省回退 `dtype`，本算子恒等）。`is_fp8` 参数被 `del`（fp8 不在 manifest 契约内，`SUPPORTED_DTYPES = (fp16, bf16, fp32)`，构造期拒绝 fp8）。
- **规约语义**：无规约、无跨元素依赖（每个输出元素只依赖同下标的三个输入元素），**不存在累加顺序问题**——并行组织的变化不影响数值结果。
- **dtype 语义**：计算**直接在输入 dtype 域**进行（fp16/bf16 的 `b−a`、`w·t`、`a+t` 三步各自在原生 dtype 舍入），源码无 fp32 中间提升。注意：这是 GPU 源算子的实际行为，NPU 侧对 bf16 的适配差异见 §0.6 R2。
- **边界语义**：逐元素 IEEE 语义——`w=NaN → out=NaN`；`a=+inf, b=−inf` 时 `b−a = −inf`，`w·(−inf)` 按 w 符号传播，最终 `a + (±inf/NaN)` 遵循 IEEE 754；`w=0 → out=a`、`w=1 → out=b`（精确）；`w∉[0,1]` 时外插（无钳位）。空张量（N=0）由上层 `N_total=1`（broadcast 全空时 `prod()=1` 的兜底）或 Op 层拦截，kernel 不单独处理。
- **host 侧语义**：预广播/展平/回 view 属于 Op 层契约（语义的一部分），由 NPU 侧 wrapper（`examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor.py` 的 `LerpTensorFwdKernel` + `tileops/ops/elementwise/lerp_tensor.py` 的 `LerpTensorFwdOp`）承接，**不属于本 kernel 的重设计范围**。

**语义保持基线**：§8.1 golden 函数以本节语义为唯一依据实现（`torch.lerp(input, end, weight)` 同 dtype 直算，与源仓测试基准一致），不复刻 NPU 算法。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `a`（input） | `(N,)` | float16 / bfloat16 / float32 | 起点张量（Op 层已展平物化），N = broadcast shape 元素总数 |
| `b`（end） | `(N,)` | same_as(a) | 终点张量 |
| `w`（weight） | `(N,)` | same_as(a) | 逐元素权重张量（Tensor-weight 变体） |
| `out` | `(N,)` | same_as(a) | 输出；**源码解读确认输出 shape 为 `(N_total,)` 一维**（`main` 的 `out: T.Tensor((N,), out_dtype)`，Op 层再 view 回 broadcast shape），无转置/布局变换 |

**输出 shape 依据**：源文件 `_lerp_tensor_fwd_kernels.py` L35–L39 的 `@T.prim_func main(a: T.Tensor((N,), dtype), …, out: T.Tensor((N,), out_dtype))`；GPU 仓 `tileops/ops/elementwise/arithmetic.py` `LerpTensorFwdOp._eager_forward` 中 `result.view(self.out_shape)` 佐证 kernel 输出为一维展平视图。

### 0.3 实现算法解读（怎么算）

源文件：`examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/_lerp_tensor_fwd_kernels.py`（Stage 0 从 GPU 仓 `tileops/kernels/elementwise.py` L2867–L2908 提取，Pattern B 工厂闭包）。

**计算步骤分解**（覆盖源码全部计算语句，无遗漏、无臆造）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| 1（host） | 三输入 `expand(out_shape) → contiguous() → view(-1)` | a/b/w 任意 broadcast shape | 三个 `(N,)` 连续张量 | 预广播（Op 层契约，wrapper 承接） |
| 2 | `block_size = threads * npt`；grid = `T.ceildiv(N, block_size)` | threads_arg, npt_arg | CUDA grid 配置 | 任务划分 |
| 3 | `T.copy(a[bx*bs:(bx+1)*bs], a_reg)` × 3 | GM a/b/w | register fragment a_reg/b_reg/w_reg | 数据搬入（连续段 coalesced） |
| 4 | `for i, j in T.Parallel(threads_arg, npt_arg): k = i*npt_arg + j; a_reg[k] = a_reg[k] + w_reg[k] * (b_reg[k] - a_reg[k])` | a_reg/b_reg/w_reg | a_reg（原地覆写） | 核心公式 `a + w·(b−a)` |
| 5 | `T.copy(a_reg, out[bx*bs:(bx+1)*bs])` | a_reg | GM out | 数据搬出 |
| 6（host） | `result.view(out_shape)` | `(N,)` 输出 | broadcast shape | 后处理（Op 层） |

**数据流与内存访问模式**（GPU 视角）：

```
GM[a] ──T.copy(连续 block_size 段)──> register a_reg ┐
GM[b] ──T.copy──────────────────────> register b_reg ┼─ thread 级标量算术 ──> a_reg ──T.copy──> GM[out]
GM[w] ──T.copy──────────────────────> register w_reg ┘
（无 shared memory 中转、无 bank conflict 处理、无异步流水；单阶段 load→compute→store）
```

**循环与并行结构**：CUDA grid（`ceildiv(N, block_size)` 个 block）× block 内 `T.Parallel(threads, npt)` 二维线程并行，每线程处理 `npt` 个连续元素（`k = i·npt + j` 的线程→元素映射）。工厂默认 `threads=256, npt=8`（block_size=2048）；GPU Kernel 类实际默认 `threads=512`，npt=4（fp32）/8（fp16/bf16）→ block_size = 2048 / 4096（NPU wrapper E7/E8 已按此折叠为 `block_size` 单参数）。

**host 侧逻辑**：`@tilelang.jit(out_idx=[3])` 工厂闭包按 `(N, dtype, output_dtype, is_fp8, threads, npt)` 特化编译（GPU 仓叠加 `functools.lru_cache(maxsize=32)` 缓存）；`del is_fp8`；`out_dtype = output_dtype or dtype`。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | register fragment 驻留（`alloc_fragment` + load→compute→store 单阶段） | 数据驻留寄存器、消除重复 GM 访问 | 三输入 tile 装入寄存器片段，算术全程寄存器内完成（`a_reg[k]` 原地覆写为输出） | GPU 大寄存器堆 + fragment 机制 | 硬件强相关 |
| 2 | 连续段 coalesced 搬运 | 访存带宽 | `T.copy` 按 block_size 连续段读写，尾轴连续 → CUDA memory coalescing | CUDA coalescing 硬件 | 硬件强相关 |
| 3 | thread × npt 两级并行 + 每线程多元素（ILP） | 并行度与指令级并行 | `T.Parallel(threads, npt)`，每线程连续处理 npt 个元素 | CUDA block/thread 线程模型 | 硬件强相关 |
| 4 | 一维展平（Op 层物化） | 消除多维索引开销、kernel 极简 | 三输入预广播展平为一维连续 | 无（通用） | 可移植 |
| 5 | 工厂闭包 + lru_cache 按 (N, dtype, …) 编译特化 | 摊薄 JIT 编译开销 | 每个参数组合只编译一次 | 无（host 层） | 可移植 |
| 6 | dtype 域直算（fp16/bf16 无 fp32 提升） | 少一半 cast 指令与带宽 | 原生 dtype 三步算术 | GPU 支持全 dtype fragment 算术 | 语义层（NPU bf16 需适配，见 §0.5/§0.6 R2） |

> 源码未使用 SMEM tiling / warp shuffle / 异步流水 / swizzle / persistent kernel——本表已穷尽源文件全部显式优化结构（对照 migration-analysis.md §4.1 清单逐项核对）。

### 0.5 硬件耦合性分析与 NPU 适配决策

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 数学语义 `a + w·(b−a)` | 语义 | 无 | — | **保留** | 语义层无条件保留（migration-analysis.md §5.4 规则 1）；§1.6.0 调研确认为最优公式形态 |
| I/O 契约（`(N,)`×3 进 `(N,)` 出，同 dtype） | 语义 | 无 | — | **保留** | 输出 shape 一维由 §0.2 源码解读确认 |
| dtype 契约（fp16/bf16/fp32；fp8 拒绝） | 语义 | 无 | — | **保留** | `SUPPORTED_DTYPES` 沿用（wrapper 已承接）；bf16 的**计算路径**因 NPU Vector API 限制需 fp32 中转（→ 重设计项 R2，dtype 契约本身不变） |
| host 预广播/展平/回 view | 语义（Op 契约） | 无 | — | **保留** | `LerpTensorFwdOp._eager_forward` + `LerpTensorFwdKernel` 已承接（`lerp_tensor.py`）；host 侧仅元数据/物化视图操作，符合 ascend-constraints.md §4 |
| 一维 grid `ceildiv(N, block_size)` | 算法 | 弱（CUDA grid） | 有（`T.Kernel(一维, is_npu=True)`） | **保留** | 一维 Kernel 为本项目支持形态（ascend-constraints.md §1 第 1 行；`examples/elementwise/vec_add_1d.py` 先例）；核数按分核策略适配（→ 重设计项 R3） |
| 工厂闭包 lru_cache 特化 | 算法（host） | 无 | — | **保留** | NPU wrapper `forward` 已实现按 block_size 惰性构建 + 缓存（`lerp_tensor.py` L163–L167） |
| register fragment 驻留 | 优化 | GPU 寄存器堆 | 有（UB） | **等价替换** | `T.alloc_fragment` → `T.alloc_ub`（migration-analysis.md §5.3「shared memory→UB」+「register 累加→UB 中间缓冲」行；v-prefix API 要求操作数在 UB：`docs/Tilelang.language/数学操作/T.vadd.md` §2.3 第 1 条）；意图承接：数据驻留片上、单阶段 load→compute→store |
| 连续段 coalesced 搬运 | 优化 | CUDA coalescing | 有（向量化 copy） | **等价替换** | `T.copy` GM↔UB 连续段搬运（`docs/Tilelang.language/内存操作/T.copy.md`；`vec_add_1d.py` 同款模式）；意图承接：带宽利用 |
| thread × npt CUDA 两级并行 | 优化 | CUDA 线程模型 | 无直接等价 | **重新设计** | → §0.6 重设计项 R1（一维 grid + Vector 128bit 向量化，npt 的"每线程多元素"意图由向量 lane 承接） |
| `T.Kernel(..., threads=threads_arg)` 的 threads kwarg | 优化 | CUDA | 无 | **舍弃** | npuir 的 `T.Kernel` 不接受 `threads=`；`is_npu=True` 取代（wrapper E3 适配注释；`example_elementwise_add.py` L21 的 `T.Kernel(..., is_npu=True)` 佐证）。舍弃理由：CUDA 线程维度概念在 NPU 不存在，threads/npt 双参数折叠为单一 `block_size`（K11/wrapper E2/E7/E8），无性能意图丢失（并行组织由 R1 重设计承接） |
| dtype 域直算 | 语义/优化 | GPU 全 dtype fragment 算术 | fp16/fp32 有（vadd/vsub/vmul ✓）；**bf16 无**（dtype 矩阵 bf16 ×） | **重新设计**（仅 bf16 路径） | → §0.6 重设计项 R2（bf16 经 `T.vcast` 升 fp32 计算、舍回 bf16）；依据 `docs/Tilelang.language/数学操作/T.vadd.md` §2.2.1（fp16 √ / fp32 √ / bf16 ×），`T.vsub.md`、`T.vmul.md` 同 |
| GPU grid 自由超发（逻辑核任意多） | 优化 | CUDA 硬件调度器 | 无（超发被串行调度 + 核启动开销） | **重新设计** | → §0.6 重设计项 R3（persistent 分核，依据 core-split-strategy.md §1） |

**判定统计**：保留 6 项、等价替换 2 项、重新设计 3 项（R1 并行结构 / R2 bf16 路径 / R3 分核策略）、舍弃 1 项（threads kwarg，理由见表）。

### 0.6 NPU 算法重设计

**重设计项 R1: 并行结构（CUDA block×thread 两级并行 → NPU 一维 Kernel + Vector 向量化）**

- **源方案**：`T.Kernel(ceildiv(N, block_size), threads=threads_arg)` 的 CUDA grid×block 两级并行；block 内 `T.Parallel(threads_arg, npt_arg)` 二维线程映射，每线程标量处理 npt 个连续元素（`k = i·npt + j` 索引 + 逐元素标量算术 `a_reg[k] + w_reg[k]·(b_reg[k]−a_reg[k])`）。性能意图：满并行 + 每线程多元素 ILP。
- **NPU 新算法**：一维 `T.Kernel(num_kernels, is_npu=True) as (cid, _)`（核数按 R3 分核策略取值）；核内 tile 的三步算术**全部替换为 v-prefix 向量 API**——`T.vsub(b_ub, a_ub, t_ub)`（t=b−a）→ `T.vmul(w_ub, t_ub, t_ub)`（t=w·t，原地）→ `T.vadd(a_ub, t_ub, out_ub)`（out=a+t），配合 `T.copy` GM↔UB 连续段搬运。源方案「每线程 npt 元素」的 ILP 意图由 Vector 单元 128bit 向量 lane 承接（fp16 ×8 / fp32 ×4 元素/指令）；「threads×npt=block_size」的 tile 粒度由单一 `block_size` 参数承接（wrapper E2/E4）。
- **语义保持论证**：逐元素映射不变（每元素独立计算 `a+w·(b−a)`，无跨元素依赖、无规约），并行组织与线程→元素映射的变化不改变任何输出值——数学上每个 `out[i]` 的计算表达式与求值顺序（先 `b−a`、再 `w·t`、后 `a+t`）与源码逐步一致。fp16/fp32 路径在原生 dtype 域计算，与源算子舍入路径一致（IEEE RN，逐步舍入位置相同）；无 nan/inf 边界行为差异（同样的三步算术）。

**重设计项 R2: bf16 计算路径（bf16 域直算 → vcast 升 fp32 计算 → 舍回 bf16）**

- **源方案**：bf16 fragment 直接三步算术（`b−a`、`w·t`、`a+t` 各自在 bf16 舍入）。
- **NPU 新算法**：`T.vcast` 三输入 bf16→fp32（round_mode="rint"，升精度无损）→ fp32 域 `vsub/vmul/vadd`（原地链）→ `T.vcast` fp32→bf16（round_mode="rint"）→ `T.copy` 回 GM。依据：`docs/Tilelang.language/数学操作/T.vadd.md`（及 vsub/vmul 同款）§2.2.1 dtype 矩阵 bf16 不支持；`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1 支持 `bf16→f32 (rint)` 与 `f32→bf16 (round/rint/floor/ceil/trunc)`。
- **语义保持论证**：数学恒等（同一公式）；数值为**容差内等价**——NPU 路径中间计算在 fp32 域（无逐步 bf16 舍入），仅最终一次舍回 bf16，结果相对源算子（三步 bf16 舍入）的差异上界为 ~1–2 ulp bf16（bf16 尾数 8 位，ε≈0.0078），且 NPU 路径精度**不低于**源路径（中间舍入更少）。源仓测试基准 `examples/TileOPs/tests/ops/test_lerp_tensor.py` `_lerp_tol`：bf16 atol=rtol=1e-2，覆盖该差异。边界语义：vcast 对 nan/inf 原样传播（IEEE 转换语义），`w=0/1` 的精确端点在 fp32 中转下同样精确（0/1 与有限值的乘加在两种精度域均精确表示后舍入一致）。fp16/fp32 路径不受影响（vadd/vsub/vmul 原生支持，保持 dtype 域直算、与源舍入路径一致）。
- **意图承接说明**：源优化手段 6（dtype 域直算，目的：省 cast 指令）在 bf16 路径无法原样保留（API 硬限制）；代价 = 每 tile 额外 4 次 vcast（3 输入升精度 + 1 结果舍回，UB 内向量指令，无额外 GM 流量），收益 = 中间精度提升。fp16/fp32 路径完整保留该优化意图。

**重设计项 R3: 分核策略（GPU grid 自由超发 → persistent 核数适配）**

- **源方案**：CUDA grid 大小 `ceildiv(N, block_size)` 任意取值（最大 workload 下 65536 个 block），超发 block 由 CUDA 硬件调度器吞吐（无额外软件开销）。
- **NPU 新算法**：NPU 物理 AI Core 有限且超发逻辑核会被运行时**串行调度并引入额外核启动开销**（core-split-strategy.md §1）。本算子 manifest workload 的逻辑核数 256–65536 全部超过物理 Vector 核数（实查 24×2=48，见 §5.5），属极大规模 → persistent 化：`num_kernels = min(num_logical, 48)`（编译期常量），核内 `for i in T.serial(num_local_tasks)`（**静态边界**，`num_local_tasks = ceildiv(num_logical, num_kernels)`）以 grid-stride 方式映射 `block_id = i * num_kernels + cid`，`if block_id < num_logical` 屏蔽越界任务。已验证先例：`docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.4 示例即此 `T.serial + block_id = i×BLOCK_SIZE + cid + if` 结构。
- **语义保持论证**：任务映射改变（grid-stride 分配 tile）不改逐元素结果——每个 tile 的计算与写出完全独立、无跨核依赖，tile 处理顺序对输出无影响（无竞写：`block_id ↔ tile` 一一对应）。静态边界由 N 编译期特化保证（工厂闭包按 N 特化，N 为 Python int → `num_logical`、`num_local_tasks` 均为编译期常量）。
- **意图承接说明**：源方案无 persistent 结构（CUDA 不需要）；R3 不是源优化的丢弃而是 NPU 硬件差异的必要适配（core-split-strategy.md §1 要素③ 极大规模分支的标准要求）。

**优化意图承接总表**（§0.4 → NPU 方案，无静默丢弃）：

| §0.4 优化手段 | NPU 承接 |
|---------------|----------|
| 1 register 驻留 | UB 驻留（`T.alloc_ub` + 原地 v-prefix 链），单阶段 load→compute→store 结构不变 |
| 2 coalesced 搬运 | `T.copy` 连续段 GM↔UB（同款语义） |
| 3 thread×npt ILP | Vector 128bit 向量 lane（fp16×8/fp32×4）+ 单一 block_size tile 粒度 |
| 4 一维展平 | 原样保留（wrapper/Op 层） |
| 5 lru_cache 特化 | 原样保留（wrapper forward 惰性构建 + 缓存） |
| 6 dtype 直算 | fp16/fp32 原样保留；bf16 因 API 限制改为 fp32 中转（R2，意图部分牺牲、精度反升，代价已量化） |

### 0.7 标杆实现

- **源算子文件**：`examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/_lerp_tensor_fwd_kernels.py`（Stage 0 提取件，接口契约的冻结基准）
- **GPU 仓原始上下文（交叉核对源）**：`/home/tilelang/zuochuanuong/TileOPs-fork/tileops/kernels/elementwise.py`（`_make_lerp_tensor_kernel` @ L2867、`LerpTensorFwdKernel` @ L2912）、`/home/tilelang/zuochuanuong/TileOPs-fork/tileops/ops/elementwise/arithmetic.py`（`LerpTensorFwdOp` @ L187，三输入预广播→展平→调度）
- **参考实现 / 测试基准**：`torch.lerp(input, end, weight)`（源仓测试 `examples/TileOPs/tests/ops/test_lerp_tensor.py` 的 ref 即此）；manifest 规格 `examples/TileOPs/tileops/manifest/elementwise.yaml` `LerpTensorFwdOp` 条目（4 workloads + roofline flops=3N / bytes=4N·elem_bytes）；性能基准 `examples/TileOPs/benchmarks/ops/bench_lerp_tensor.py`
- **NPU 集成目标（wrapper）**：`examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor.py`（`LerpTensorFwdKernel`：E1 工厂签名 `(N, dtype)`、E2 callable(block_size)、block_size 默认 fp32=2048 / fp16·bf16=4096）
- golden 以 §0.1 语义为依据（优先移植源仓参考 `torch.lerp`），见 §8.1

## 1. 概述

### 1.1 算子名称

`_make_lerp_tensor_kernel`（Op 层名 `LerpTensorFwdOp` / `lerp_tensor`）

### 1.2 功能描述

Tensor-weight 线性插值前向核：对三个已展平为一维 `(N,)` 的同 dtype 张量计算 `out = input + weight · (end − input)`，纯逐元素、无规约、无跨元素依赖。

### 1.3 数学公式

$$
\text{out}[i] = a[i] + w[i] \cdot \left( b[i] - a[i] \right), \quad i \in [0, N)
$$

参考 API：`torch.lerp(input, end, weight: Tensor)`（Tensor-weight overload）。

### 1.4 算法描述

**（迁移决策后的 NPU 侧算法，与源算法结构差异注明出处）**

单遍逐元素三步向量算法（§1.6.0 调研选定基线公式，无更优算法族替代）：

1. **搬入**：每 tile 将 `a/b/w` 三段连续 `block_size` 数据 `T.copy` GM→UB（§0.6 R1：fragment→UB 等价替换）；
2. **计算**：UB 内三步向量链 `t = b − a`（vsub）→ `t = w · t`（vmul，原地）→ `out = a + t`（vadd）——fp16/fp32 原生 dtype 域；bf16 先 vcast 升 fp32、算完舍回（§0.6 R2）；
3. **搬出**：`T.copy` UB→GM（尾块按 `tail_size` 截断，§6.4）。

与源算法的结构差异：① CUDA threads×npt 二维线程并行 → 一维 Kernel + v-prefix 向量化（R1）；② bf16 计算域 fp32 中转（R2）；③ grid 按逻辑核数自由超发 → persistent 核数适配（R3）。数据流骨架（单阶段 load→compute→store、片上驻留、连续段搬运）与源一致。

### 1.5 数据流图

```
fp16 / fp32 主路径（每 tile）：
GM[a] ──T.copy(tail 截断)──> UB[a_ub] ──────────────────┐
GM[b] ──T.copy──────────────> UB[b_ub] ──vsub(b,a,t)────┤（t 链原地覆写 b_ub）
GM[w] ──T.copy──────────────> UB[w_ub] ──vmul(w,t,t)────┤
                                   └──vadd(a,t,out)──> UB[out_ub] ──T.copy──> GM[out]

bf16 路径（R2）：
GM[a/b/w:bf16] ──T.copy──> UB[a/b/w_ub:bf16] ──vcast(rint)──> UB[a/b/w32:fp32]
  ──vsub/vmul/vadd（fp32 原地链）──> UB[t32:fp32] ──vcast(rint)──> UB[out_ub:bf16] ──T.copy──> GM[out]
```

### 1.6 算法调研与优化分析 ⭐

> 设计第一优先级：先调研（1.6.0，迁移任务在 Phase M0 后、M1 前完成，结论已驱动 §0.5/§0.6），再公式级优化（1.6.1），再循环/标量的向量化替代（1.6.2），最后向量化轴与布局决策（1.6.3）。本节分析对象 = §1.6.0 选定、经 §0.6 重设计落地的 NPU 算法。lerp_tensor 为**单步逐元素类**（无规约/窗口/矩阵/多步融合），按 algorithm-research.md §2 采用轻量调研深度（四问各一行结论 + 依据），但四问不可跳过。

#### 1.6.0 算法调研（Algorithm Research）⭐

**R1 等价化简公式候选**（基线在表；正式等价论证与收益量化在 §1.6.1 完成，此处只做初判）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | 基线（源算法/输入公式） | `a + w·(b−a)` = sub+mul+add 3 op/元素 | — | — | ✅（基线） |
| 1 | 仿射展开形式 | `a·(1−w) + b·w` | 数学恒等（分配律） | **负收益**：需额外算 `1−w`（+1 sub/元素）→ 4 op/元素，且多一个中间量 | ❌ op 数更多 |
| 2 | 差值直乘形式 | `b − (1−w)·(b−a)` | 数学恒等 | **负收益**：同样 4 op/元素 + 复用 `b−a` 无节省 | ❌ 同上 |
| 3 | FMA 融合形态 | `fma(w, b−a, a)`（mul+add 融合单指令） | 数学恒等 | op 数不变；融合与否取决于硬件指令与编译器 lowering，TileLang 前端无显式 FMA API 可控 | ❌ 非公式级优化，交由编译器（依据：`docs/Tilelang.language/数学操作/` 仅有 vadd/vsub/vmul 独立 API，无 vfma 条目——目录实查 2026-09-07） |

**R2 在线算法**：**天然单遍，无需在线变体**。结构依据：输出 `out[i]` 仅依赖 `(a[i], b[i], w[i])`，无 running 统计量（和/最大值/计数）、无跨元素依赖——逐元素映射本身即单遍流式（algorithm-research.md §3 R2 结构判据「逐元素映射天然单遍」）；源算法亦为单遍（§0.4 无 online 手段），一致。

**R3 复杂度对比**（带宽受限算子——算术强度 3·N / 4·N·B：fp16/bf16 = 0.375 flop/B、fp32 = 0.1875 flop/B，远低于 Vector 单元平衡点，**访存为主导项**）：

| 算法候选 | FLOPs | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-------|---------------|---------|-------------|---------------------|
| 基线 `a + w·(b−a)` | 3N（sub+mul+add 各 N；无超越函数） | **4N·B**（3 读 + 1 写，无中间 GM 往返——tile 驻留 UB）= I/O 下界 | 1（每元素每张量恰好读一次写一次） | O(block_size) tile 级（UB：32–96KB，§4.5） | N/block_size 个完全独立 tile，零跨核同步 |
| 候选 1 仿射 `a·(1−w)+b·w` | 4N（+1 sub） | 4N·B（相同） | 1 | 同上 | 同上 |
| （候选 3 FMA） | 3N（指令融合不改 FLOPs 口径） | 4N·B | 1 | 同上 | 同上 |

基线在四口径上全部并列最优（FLOPs 最少、访存达 I/O 下界、单遍、缓冲 tile 级）；候选 1/2 仅增加 FLOPs 无访存收益。

**R4 硬件亲和性评估**（逐候选对照检查清单）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| 基线（v-prefix 三步链） | 逐元素算术 → Vector 单元完美匹配（vadd/vsub/vmul：fp16 √ / fp32 √，`T.vadd.md` §2.2.1）；bf16 需 R2 中转（vcast bf16↔f32 √，`T.vcast.md` §2.2.1） | 4 buffer × block_size：fp16@4096=32KB、fp32@2048=32KB、bf16@4096=96KB（含 fp32 中转），均 ≤ UB 192KB | block_size×B 均 32B 整除（4096×2B=8KB / 2048×4B=8KB）；向量宽度整除（fp16×8、fp32×4 均整除 block_size）；N 非整除由尾块 tail_size 处理（§6.4） | N 编译期特化 → grid/循环边界全静态；尾块 tail 为运行时标量但仅 1 次/tile | 单阶段无需流水；三 op 同 buffer 原地链已是最简融合 | ✅ 选定 |
| 候选 1 仿射 | 同上（多 1 次 vsub/vmul 链长 25%） | 同上 | 同上 | 同上 | 同上 | ❌ R3 四口径无收益、FLOPs +33% |

**调研结论**：选定**基线公式 `out = a + w·(b−a)` 的单遍逐元素 Vector 算法**（vsub→vmul→vadd 原地链）。关键依据：R3 表中基线 FLOPs 最少（3N）且访存已达 I/O 下界（4N·B，与 manifest roofline `flops=3N / bytes=4N·elem_bytes` 一致）；R4 基线对 Vector 单元/UB 容量/对齐/静态边界全部亲和。与基线（源算法）的**结构差异**：仅实现载体变化（fragment 标量循环 → v-prefix 向量链；bf16 → fp32 中转），算法族本身无差异。**「无更优替代」的调研范围**：① algorithm-research.md §5 参考表「elementwise 链 / 激活」行（公共子表达式消除——本算子 a/b/w 各用一次、`b−a` 仅一次，无公共子式；除法乘倒数——无除法；cast 链合并——仅 bf16 路径有 cast，已按 R2 最小化为进出各一次）；② 恒等变形空间（分配律两个方向的展开形式均 op 更多，R1 候选 1/2）；③ `examples/elementwise/` 全部逐元素实现（vec_add 系列、exp2/log2——所用均为直排 v-prefix 单遍结构，无同类公式变形先例）；④ pattern-library.md §1 实测模式（乘倒数/H-collapse/C 轴累加等均针对池化/规约类，本算子无适用条目；§4 案例索引无 lerp/逐元素三输入同类条目）；⑤ 源算子结构分析（§0.3，单遍三 op）。源算法优化手段的意图承接见 §0.6 末尾总表，无静默丢弃。

#### 1.6.1 数学等价优化（公式级）

分析对象 = §1.6.0 选定的基线算法（经 §0.6 R1/R2 落地）。逐类别核查：

| # | 优化类别 | 核查结论 | 依据 |
|---|----------|----------|------|
| 1 | 恒等变形降代价（除法转乘倒数 / sqrt→rsqrt / 换底） | **无适用项**：公式无除法、无开方、无超越函数 | §1.3 公式；`docs/Tilelang.language/数学操作/` API 目录（T.rsqrt/T.vexp 等均与本算子无关） |
| 2 | 稳定化变形（减 max / logsumexp 改写） | **无需求**：lerp 无指数/求和结构，无溢出放大路径；`w∉[0,1]` 外插的溢出行为是语义本身（与源一致），非数值缺陷 | §0.1 边界语义 |
| 3 | 公共子表达式消除 | **无公共子式**：`a`/`b`/`w` 各被引用一次，`b−a` 仅出现一次（仿射变形不能产生可复用子式，反增 op，见 §1.6.0 R1 候选 1/2） | R1 候选表 |
| 4 | 算子降代换（exp→exp2 等） | **无高代价算子**：仅 sub/mul/add 三个基础向量 op | §1.3 |
| 5 | 访存量削减（中间驻留 / cast 最小化 / 重复读消除） | **已达下界**：三输入各读一次、输出写一次（4N·B），tile 全程 UB 驻留无 GM 往返；cast 仅 bf16 路径必需的进出各一次（R2 约束倒逼，非优化项；fp16/fp32 零 cast） | R3 口径表；§4.4 搬运路径 |
| 6 | 归约结构优化（多轮扫描合并） | **无规约** | §0.1 |

**优化结论**：**无公式级优化空间**。依据：单步逐元素三 op 映射，无公共子表达式、无高代价算子、无重复访存（R3 表 I/O 已达 4N·B 下界、扫描 1 遍）；两种代数等价展开形式（仿射/差值直乘）op 数均 +33% 无对应收益。**优化后公式 = 原式**：`out[i] = a[i] + w[i]·(b[i]−a[i])`（bf16 路径外加 `vcast(bf16→fp32)` 前处理与 `vcast(fp32→bf16)` 后处理，见 §0.6 R2）——此式即 §3.1 公式拆解的唯一输入。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

盘点实现方案（§0.6 重设计后的 NPU 算法）中**全部**循环与标量计算点：

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------|-------------|---------|------------------------------|
| 1 | 逐元素 `t = b − a` | 源码 `T.Parallel(threads, npt)` 标量循环逐元素减 | `T.vsub(b_ub, a_ub, t_ub)` | ✅ | —（佐证：`docs/Tilelang.language/数学操作/T.vsub.md`；`examples/elementwise/vec_add_1d.py` L37 同款 v-prefix 用法） |
| 2 | 逐元素 `t = w · t` | 同上标量乘 | `T.vmul(w_ub, t_ub, t_ub)`（原地，`T.vmul.md` §2.3 第 3 条支持 dst 与 src 同 buffer） | ✅ | —（佐证：`T.vmul.md`） |
| 3 | 逐元素 `out = a + t` | 同上标量加 | `T.vadd(a_ub, t_ub, out_ub)` | ✅ | —（佐证：`T.vadd.md`） |
| 4 | bf16↔fp32 逐元素 cast | （源无此步；R2 引入） | `T.vcast(src_ub, dst_ub, round_mode="rint")` | ✅ | —（佐证：`T.vcast.md` §2.2.1 bf16↔f32 行） |
| 5 | tile 起始偏移 `t0 = block_id × block_size` 等 block 索引 | 标量整数乘加 | 无逐元素等价 API（索引计算非数据计算） | ❌ | **block 级索引/任务映射计算**：每 tile 仅 3–4 次标量整数运算，非逐元素热点（占比 < 0.1% 指令量）；`T.vcast.md` §2.4 官方示例同款保留 |
| 6 | 尾块长度 `tail = T.min(block_size, N − t0)` | 标量比较 | 无 | ❌ | **依赖动态 shape 的边界处理**：N 非整除时逐 tile 不同的边界量；已采用 `examples/elementwise/vec_add_1d.py` L31–L33 的已验证 `T.min` 模式（`docs/Tilelang.language/比较操作/T.min.md`），无法静态化（N 为编译期常量但整除性随 N 变化，padding 方案会违反 host 侧禁止改输入约束，ascend-constraints.md §4） |
| 7 | 核内多 tile 调度循环 `for i in T.serial(num_local_tasks)` | 循环 | 无（tile 间无数据依赖但 persistent 结构需循环推进任务） | ❌ | **tile 级调度循环**（R3 persistent 分核的标准结构，`core-split-strategy.md` §1 要素③；`T.vcast.md` §2.4 示例同款）；循环边界 `num_local_tasks` 为编译期静态值，循环体一次处理一个完整 tile（内部全向量化），非逐元素热点 |
| 8 | host 侧 `num_kernels/num_logical` 计算、block_size 配置 | Python 标量 | — | ❌ | **host 元数据计算**（不在 kernel 内；工厂闭包编译期完成，符合决策树 §1「维度参数自推导」） |

**向量化结论**：4 类逐元素计算点（#1–#4）**已全部向量化**，替代 API 均有 docs + examples 双佐证；保留 4 处标量/循环（#5–#8），理由类别分别为 block 索引 / 动态边界 / tile 级调度循环 / host 元数据，逐项见上表——§6 循环结构与本结论一致（kernel 内无任何逐元素标量循环）。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

**豁免判定**：本算子为**一维展平单轴算子**（Op 层已将三输入物化为 `(N,)` 连续一维，`lerp_tensor.py` wrapper 契约），仅存在一个候选向量轴 → 按 SKILL.md Phase 2 第 3 条「纯 GEMV/单轴算子可写明仅一个候选轴豁免」处理。仍按模板要求记录候选矩阵与源码轴对照：

**候选矩阵**：

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益/代价 | 是否采纳 |
|---|---------|---------|------------|--------------|---------|
| 1 | 一维连续（I/O 原生，Op 层已展平） | 展平 N 轴（stride=1 连续） | 无 | 基线：整除性 4096%8=0（fp16）/2048%4=0（fp32）尾 lane 浪费 0%；累加链不存在（无累加）；UB 预算 §4.5 达标 | ✅ 采纳 |
| 2 | 还原 2D (rows, cols) 逐行处理 | 行内 cols 轴 | 无（view 层面） | 无收益：三输入同 shape 无广播可省（Op 层已物化展开），反增 2D 索引与行尾非对齐风险 | ❌（无广播需求，纯负收益） |
| 3 | 核内重排布局（转置/分块交织） | — | T.transpose 链 | 无意义：无窗口/无跨步/无规约轴可优化；pattern-library §1.1 转置链适用于 NCHW 类窗口算子，本算子无此结构 | ❌（无适用场景） |

**源码轴对照（迁移任务要求）**：GPU 源码的 `T.Parallel(threads, npt)` 二维轴是 **CUDA 硬件组织的输入而非结论**（thread/warp 映射与 NPU Vector 轴不对应）；其「每线程 npt 个连续元素」的隐式轴向选择（连续 N 轴、npt 为向量宽度雏形）与候选 #1 的 N 轴选择一致——源码轴选择经独立评估后确认采纳同一连续轴，Vector 128bit（fp16×8/fp32×4）承接 npt 的多元素意图。

**布局决策结论**：选定方案 #1——核内布局 = 一维 `(block_size,)` UB buffer（×3 输入 + ×1 输出，bf16 路径另加 fp32 中转 buffer），向量化轴 = 展平 N 轴（stride=1），无 repack。该结论同步落入 §3.3 伪代码（buffer 形状 `(block_size,)`、v-prefix 操作数即整 buffer）与 §6 循环结构（无逐元素循环，向量操作覆盖全 tile）；§4.5 UB 预算按此布局计算。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: Developer（`op_requirements.programming_mode` 指定，且与算子特征判定一致）

### 2.2 选型理由

| 特征 | 分析 | 结论 |
|------|------|------|
| 计算类型 | 纯 Vector（逐元素 sub/mul/add + bf16 cast），无 matmul、无规约 | 无 Cube/L0 需求，不需要 MixCV |
| 复杂度级别 | 单步算子（一次搬运 + 三 op 向量链 + 一次搬出） | 无多阶段流水/精细 buffer 生命周期控制需求 |
| 同步 | 每 tile 独立、无核间依赖 | 无手动 sync_block_set/wait 需求 |
| API | v-prefix（vsub/vmul/vadd/vcast）+ T.copy + T.alloc_ub 均为 Developer 可用形态 | `T.vadd.md` §2.4 示例即 `@tilelang.jit(target="npuir")` Developer 形态 |
| 先例 | `examples/elementwise/vec_add_1d.py`（一维逐元素 + alloc_ub + vadd，Developer 形态） | 同构算子已验证 |

决策树路径：纯 element-wise 单步 → Developer 模式（decision-tree.md §2）。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_ub((block_size,), dtype)` 显式 UB（v-prefix API 要求操作数在 UB，`T.vadd.md` §2.3 第 1 条） |
| 计算方式 | v-prefix 向量 API（vsub/vmul/vadd/vcast）+ 原地链；无 T.Parallel 标量循环 |
| 同步 | Developer 自动同步（copy→compute→copy 顺序依赖由编译器保证） |
| Kernel 启动 | `T.Kernel(num_kernels, is_npu=True) as (cid, _)` |
| 环境变量 | `TILELANG_ASCEND_MODE=Developer`（`examples/elementwise/example_elementwise_add.py` L41 先例） |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式 = §1.6.1 优化后公式（无变化）：`out = a + w·(b−a)`；bf16 路径含 R2 的 cast 前后处理。拆解与 §1.6.2 向量化结论一致。

**fp16 / fp32 路径**：

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `a_ub ← a[t0:t0+tail]`，`b_ub/w_ub` 同 | 三输入 tile 搬入 GM→UB（尾块截断） |
| 2 | `t = b − a` | vsub（原地：t 复用 b_ub） |
| 3 | `t = w · t` | vmul（原地） |
| 4 | `out = a + t` | vadd |
| 5 | `out[t0:t0+tail] ← out_ub[0:tail]` | 结果搬出 UB→GM |

**bf16 路径**（R2）：

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `a_ub/b_ub/w_ub ← a/b/w[t0:t0+tail]`（bf16） | 搬入 |
| 2 | `a32/b32/w32 ← vcast(a_ub/b_ub/w_ub, rint)` | 升 fp32（无损） |
| 3 | `t32 = b32 − a32` → `t32 = w32 · t32` → `t32 = a32 + t32` | fp32 域三步链（原地） |
| 4 | `out_ub ← vcast(t32, rint)` | 舍回 bf16 |
| 5 | `out[t0:t0+tail] ← out_ub[0:tail]` | 搬出 |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| tile 分派 | `block_id = i·num_kernels + cid` | `T.Kernel(num_kernels, is_npu=True) as (cid, _)` + `T.serial(num_local_tasks)` | num_kernels/num_local_tasks 为编译期常量（§5.5） | Developer |
| 搬入/搬出 | `x[t0:t0+tail] ↔ x_ub[0:tail]` | `T.copy(src_slice, dst_slice)` | 连续段切片拷贝（`T.copy.md`：gm-ub / ub-gm 均支持，bf16 √） | Developer |
| 尾块长度 | `tail = min(block_size, N − t0)` | `T.min(a, b)` | `docs/Tilelang.language/比较操作/T.min.md` | Developer |
| 减法 | `t = b − a` | `T.vsub(src0=b_ub, src1=a_ub, dst=t_ub)` | fp16 √ / fp32 √ / bf16 ×（dtype 矩阵 `T.vsub.md` §2.2.1） | Developer |
| 乘法 | `t = w · t` | `T.vmul(src1=w_ub, src2=t_ub, dst=t_ub)` | 原地合法（§2.3 第 3 条） | Developer |
| 加法 | `out = a + t` | `T.vadd(src0=a_ub, src1=t_ub, dst=out_ub)` | fp16 √ / fp32 √ / bf16 × | Developer |
| bf16 升精度 | `f32 ← bf16` | `T.vcast(src=a_ub, dst=a32, round_mode="rint")` | bf16→f32 仅 rint（`T.vcast.md` §2.2.1） | Developer |
| bf16 舍回 | `bf16 ← f32` | `T.vcast(src=t32, dst=out_ub, round_mode="rint")` | f32→bf16 支持 rint；rint = round-to-nearest-even，与 torch bf16 转换语义一致 | Developer |
| 整除向上取整 | `ceildiv(N, block_size)` | `T.ceildiv` | `docs/Tilelang.language/数学操作/T.ceildiv.md` | Developer |
| buffer 分配 | UB 片上缓冲 | `T.alloc_ub((block_size,), dtype)` | `docs/Tilelang.language/内存操作/T.alloc_ub.md`（1D 支持 ✓） | Developer |

### 3.3 计算伪代码

> dtype 位置传参（模板 §3.3 规则：`T.Tensor((N,), dtype)` 不用 `dtype=` 关键字，避免 false alarm）。

```python
def _make_lerp_tensor_kernel(N, dtype):                     # E1：工厂签名 (N, dtype)
    # 编译期常量（N 为 Python int，工厂特化）
    vector_cores = NPUUtils.get().get_aicore_num() * 2      # 纯 Vector 翻倍；实查见 §5.5

    @tilelang.jit(out_idx=[3], target="npuir")
    def kernel(block_size):                                  # E2：callable(block_size) 单参数
        num_logical = T.ceildiv(N, block_size)
        num_kernels = min(num_logical, vector_cores)         # persistent 分核（R3）
        num_local_tasks = T.ceildiv(num_logical, num_kernels)  # 静态边界

        @T.prim_func
        def main(
            a: T.Tensor((N,), dtype),                        # 位置传参
            b: T.Tensor((N,), dtype),
            w: T.Tensor((N,), dtype),
            out: T.Tensor((N,), dtype),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                # 1. 分配 UB buffer（§4.3/§4.5）
                a_ub = T.alloc_ub((block_size,), dtype)
                b_ub = T.alloc_ub((block_size,), dtype)
                w_ub = T.alloc_ub((block_size,), dtype)
                out_ub = T.alloc_ub((block_size,), dtype)

                # 2. persistent 核内串行处理多个 tile（静态边界）
                for i in T.serial(num_local_tasks):
                    block_id = i * num_kernels + cid
                    if block_id < num_logical:
                        t0 = block_id * block_size
                        tail = T.min(block_size, N - t0)      # 尾块截断（vec_add_1d 模式）

                        # 3. 数据搬入（GM → UB，尾块段拷贝，零起点 dst）
                        T.copy(a[t0 : t0 + tail], a_ub[0:tail])
                        T.copy(b[t0 : t0 + tail], b_ub[0:tail])
                        T.copy(w[t0 : t0 + tail], w_ub[0:tail])

                        # 4. 计算（fp16/fp32 主路径：三步向量链，原地覆写）
                        T.vsub(b_ub, a_ub, b_ub)              # t = b - a（复用 b_ub）
                        T.vmul(w_ub, b_ub, b_ub)              # t = w * t
                        T.vadd(a_ub, b_ub, out_ub)            # out = a + t

                        # 5. 数据搬出（UB → GM）
                        T.copy(out_ub[0:tail], out[t0 : t0 + tail])

        return main

    return kernel
```

**bf16 路径分支**（同结构，步骤 3–5 替换为）：

```python
                        # bf16（R2）：升 fp32 计算、舍回
                        a32 = T.alloc_ub((block_size,), "float32")
                        b32 = T.alloc_ub((block_size,), "float32")
                        w32 = T.alloc_ub((block_size,), "float32")
                        T.vcast(a_ub, a32, round_mode="rint")
                        T.vcast(b_ub, b32, round_mode="rint")
                        T.vcast(w_ub, w32, round_mode="rint")
                        T.vsub(b32, a32, b32)                 # t32 = b32 - a32（原地）
                        T.vmul(w32, b32, b32)                 # t32 = w32 * t32
                        T.vadd(a32, b32, b32)                 # t32 = a32 + t32
                        T.vcast(b32, out_ub, round_mode="rint")
                        T.copy(out_ub[0:tail], out[t0 : t0 + tail])
```

> 实现注记：bf16 分支以工厂内 `if dtype == "bfloat16"` 静态分派（编译期分支，与源算子单模板多 dtype 特化同构）；`b32` 承担 t32 角色省一个 buffer（§4.5 预算按含独立 t32 的保守上界计算）。v-prefix 整 buffer 操作在尾块（tail < block_size）时对 UB 中 [tail:block_size] 陈旧段一并计算，但搬出仅拷 [0:tail]，垃圾数据不落 GM——正确性无碍（`examples/elementwise/vec_add_1d.py` L34–L38 官方先例同款形态）。

### 3.4 API 可行性确认

| API | 来源确认 | 状态 |
|-----|----------|------|
| `T.Kernel(grid, is_npu=True)` | `examples/elementwise/vec_add_1d.py` L27；`T.vadd.md` §2.4 示例 | ✅ 已验证（examples 先例） |
| `T.alloc_ub(shape, dtype)` | `docs/Tilelang.language/内存操作/T.alloc_ub.md`（1D–2D 支持）；`vec_add_1d.py` L28–L30 | ✅ |
| `T.copy`（gm↔ub、切片、bf16） | `docs/Tilelang.language/内存操作/T.copy.md`（dtype 矩阵含 bf16 √；示例即切片拷贝）；`vec_add_1d.py` L34–L38 | ✅ |
| `T.vsub / T.vmul / T.vadd` | `docs/Tilelang.language/数学操作/T.vsub.md` / `T.vmul.md` / `T.vadd.md`（fp16 √ fp32 √ bf16 ×；原地合法 §2.3） | ✅ |
| `T.vcast`（bf16↔f32） | `docs/Tilelang.language/数据类型转换操作/T.vcast.md` §2.2.1 | ✅ |
| `T.min / T.ceildiv` | `docs/Tilelang.language/比较操作/T.min.md`；`docs/Tilelang.language/数学操作/T.ceildiv.md` | ✅ |
| `T.serial` + `block_id = i·K + cid` + `if` 边界 | `T.vcast.md` §2.4 示例（官方 persistent 分发模式） | ✅ |
| `@tilelang.jit(out_idx=[3], target="npuir")` | 源算子 `out_idx=[3]` 契约 + `examples/elementwise/example_elementwise_add.py` L13 `target="npuir"` | ✅ |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | No（源即一维 `ceildiv(N, block)`；NPU 一维 `T.Kernel(num_kernels)`） | 不涉及 |
| GPU 专用 API（`threads=` kwarg） | Yes（源 `T.Kernel(..., threads=threads_arg)`） | `is_npu=True` 取代（§0.5 舍弃项；wrapper E3） |
| GEMM 非整除（M/N block 整数倍） | No（无 GEMM；非 GEMM 的 N 非整除由尾块处理） | §6.4 尾块策略 |
| L0C 溢出 | No（纯 Vector，无 L0C） | 不涉及 |
| 物理核数限制（分核三要素） | Yes（逻辑核 256–65536 >> 48） | §5.5 persistent 分核 |
| bf16 Vector 算术缺失 | Yes（manifest 契约含 bf16） | §0.6 R2：vcast fp32 中转 |
| Host 侧输入操作约束 | Yes（Op 层预广播物化输入） | 预广播在 Op 层 wrapper（既有契约，非本 kernel 职责）；kernel 内 host 侧零数据操作，仅 N/block_size 元数据 → 符合 ascend-constraints.md §4 |

#### 3.5.2 参考实现差异说明

| 差异项 | 参考实现（GPU） | 本项目（Ascend） | 转换方案 |
|--------|----------------|-----------------|----------|
| Kernel 并行 | `T.Kernel(grid, threads=)` CUDA 两级 | 一维 `T.Kernel(n, is_npu=True)` | §0.6 R1 + wrapper E3/E4 |
| 元素并行 | `T.Parallel(threads, npt)` 标量循环 | v-prefix 向量 API（无逐元素循环） | §1.6.2 #1–#3 |
| 片上驻留 | `T.alloc_fragment`（寄存器） | `T.alloc_ub`（UB） | §0.5 等价替换 |
| bf16 计算 | bf16 域直算 | vcast→fp32→vcast | §0.6 R2 |
| grid 规模 | 任意（CUDA 调度器） | persistent min(逻辑核, 48) | §0.6 R3 / §5.5 |
| JIT 装饰 | `@tilelang.jit(out_idx=[3])` | `@tilelang.jit(out_idx=[3], target="npuir")` | K1 适配（`example_elementwise_add.py` 先例） |
| 工厂参数 | `(N, dtype, output_dtype, is_fp8, threads, npt)` | `(N, dtype)`；callable(block_size) | wrapper E1/E2（接口契约冻结，`lerp_tensor.py` 已按此实现） |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/elementwise/vec_add_1d.py` | 高度相似（一维逐元素 + 动态尾块 + v-prefix） | `T.Kernel(n, is_npu=True)`、`T.alloc_ub`、`tail_size = T.min(...)` 尾块模式、`T.vadd` 用法、`tilelang.compile(target="npuir")` |
| `examples/elementwise/example_elementwise_add.py` | 高度相似（二维形态 + jit 装饰） | `@tilelang.jit(out_idx=[-1], target="npuir")`、`TILELANG_ASCEND_MODE=Developer`、`T.ceildiv` 网格展开 |
| `examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_single/DESIGN.md` | 迁移流程同类（TileOPs harness 迁移先例） | §0 迁移分析组织方式、L0 测试计划结构 |
| `examples/TileOPs/tileops/kernels/elementwise/mish/mish.py` | 集成形态同类（NPU 侧 TileOPs elementwise 集成包） | harness 集成包结构（kernel + wrapper 协作） |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `a` | `(N,)` | float16 / bfloat16 / float32 | Op 层已预广播物化的连续一维张量 |
| `b` | `(N,)` | same_as(a) | 同上 |
| `w` | `(N,)` | same_as(a) | 同上 |

N 为编译期常量（工厂按 N 特化，`lru_cache` 语义由 wrapper 承接）；代表 workload N ∈ {2²⁰, 2²⁴, 2²⁶, 2²⁸}。

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `out` | `(N,)` | same_as(a) | `out_idx=[3]`；Op 层 view 回 broadcast shape |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `a_ub / b_ub / w_ub` | `(block_size,)` | 输入 dtype | UB | 输入 tile 驻留（b_ub 兼任 t 链原地缓冲，fp16/fp32 路径） |
| `out_ub` | `(block_size,)` | 输出 dtype | UB | 输出 tile |
| `a32 / b32 / w32`（仅 bf16） | `(block_size,)` | float32 | UB | R2 fp32 中转（b32 兼任 t32 原地缓冲） |

### 4.4 内存搬运路径

```
纯 Vector 单阶段（每 tile）：
GM[a] ──T.copy(尾块截断)──> UB[a_ub] ─┐
GM[b] ──T.copy────────────> UB[b_ub] ─┼─ vsub → vmul → vadd（原地链，全程 UB）──> UB[out_ub] ──T.copy──> GM[out]
GM[w] ──T.copy────────────> UB[w_ub] ─┘
bf16 变体：UB[a/b/w_ub:bf16] ──vcast──> UB[a/b/w32:fp32]（原地链）──vcast──> UB[out_ub:bf16]（cast 全程 UB 内，零额外 GM 流量）
无 L1/L0 参与纯 Vector 路径；无中间 GM 往返（tile 一次进出）
```

### 4.5 UB 内存预算

block_size 按 wrapper 默认（K11 承接：fp32=2048、fp16/bf16=4096）：

| Buffer | fp16 @4096 | fp32 @2048 | bf16 @4096 |
|--------|-----------|-----------|-----------|
| a_ub / b_ub / w_ub | 3 × 8192 B = 24576 | 3 × 8192 B = 24576 | 3 × 8192 B = 24576 |
| out_ub | 8192 | 8192 | 8192 |
| a32 / b32 / w32（+独立 t32 保守上界） | — | — | 4 × 16384 B = 65536 |
| **总计** | **32768 B (32KB)** | **32768 B (32KB)** | **98304 B (96KB)** |
| 目标平台 UB 容量 | 196608 B (192KB) | 196608 B | 196608 B |

三路径均 ≤ 192KB（余量 ≥ 96KB）；block_size 上限受 UB 约束为 fp32 ≤ 12288 / fp16·bf16 ≤ 24576（4 buffer 预算反推），默认值余量充足。

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 |
|--------|----------|-----------|
| `N_total` | 无 TIR 动态符号——工厂闭包按 N **编译期特化**（源算子同款模式，wrapper `lru_cache` 语义） | 不同 Op 实例 N 不同（2⁰ ~ 2²⁸+），每 N 触发一次 JIT 特化；kernel 边界全静态 |

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[3], target="npuir")
# out_idx=[3]：main(a, b, w, out) 的 out（index 3）为输出张量，承接源算子契约
# target="npuir"：K1 适配（GPU 版无 target，默认 CUDA；example_elementwise_add.py L13 先例）
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 纯 Vector

**判定依据**: 仅逐元素 sub/mul/add（+bf16 cast），无 matmul、无规约、无数据重排 → 全部计算落 Vector 单元，仅需 GM↔UB 搬运（决策树 §2 纯 element-wise 分支）。

### 5.2 Block 划分

```python
block_size = 4096   # fp16/bf16 默认（= GPU threads(512) × npt(8)，K11/wrapper E7 承接）
block_size = 2048   # fp32 默认（= GPU threads(512) × npt(4)）
# 选择理由：
# ① 尾轴 32B 对齐：4096×2B = 8192B、2048×4B = 8192B，均 32 整除 ✓
# ② 向量宽度整除：fp16×8 lane、fp32×4 lane 均整除 block_size → 尾 lane 浪费 0%
# ③ UB 预算：4 buffer 32KB（fp16/fp32）/ 96KB（bf16）≤ 192KB（§4.5）
# ④ tile 搬运粒度 8KB/buffer，兼顾 copy 带宽利用率与 persistent 任务粒度
# ⑤ GPU 语义承接：与源 Kernel 类默认 block_size（threads×npt）一致，跨平台行为可对照
num_logical = T.ceildiv(N, block_size)      # 逻辑 tile 数
```

一维划分（无 block_M/block_N 之分）——本算子为展平一维逐元素，GEMM 类分形/非整除策略不适用。

### 5.3 约束分析

- **对齐约束**: block_size×elem_bytes = 8KB（32B 的 256 倍）✓；GM 侧 tile 起点 `t0 = block_id × block_size` 恒为 8KB 对齐 ✓（尾块 tail 段长度非 32B 倍数时的拷贝行为依赖 `vec_add_1d.py` 先例形态，风险点 §9.2 R2 并入 L0 测试）
- **UB 容量**: 见 §4.5，三 dtype 路径 32–96KB ≤ 192KB ✓
- **L0 容量**: 不适用（纯 Vector）

### 5.4 注意事项

- 非整除处理：尾块 `tail = T.min(block_size, N − t0)` 截断拷贝（§6.4）；manifest workload 的 N 均为 2 的幂、被 block_size 整除（无尾块），尾块路径仅服务任意 broadcast shape 的动态 N
- block_size 是 wrapper config 可调参数（`{"block_size": ...}`），kernel 不硬编码——Stage 4 可扫描调优（§1.6.3 候选 #1 已锁定轴向，调优仅粒度维度）

### 5.5 分核策略（物理核数适配）⭐

> 权威标准：`.agents/skills/_shared/standards/core-split-strategy.md` §1 三要素表（依据 docs/开发指南.md §3.3）。

- **物理核数**: **24**（AI Core），纯 Vector 算子核数翻倍 = **48**。实查记录（2026-09-07，本机 Ascend 环境）：

  ```python
  from tilelang.utils.npu_utils import NPUUtils
  n = NPUUtils.get().get_aicore_num()   # → 24
  vector_cores = n * 2                  # → 48（纯 Vector 翻倍规则）
  ```

- **逻辑核数**: `num_logical = ceildiv(N, block_size)`，按 manifest workload：

  | workload | N | dtype（block） | num_logical |
  |----------|---|---------------|-------------|
  | smoke-1m（1024×1024） | 2²⁰ | fp32（2048） | 512 |
  | smoke-1m | 2²⁰ | fp16/bf16（4096） | 256 |
  | elementwise-16m（4096×4096） | 2²⁴ | fp32（2048） | 8192 |
  | elementwise-16m | 2²⁴ | fp16/bf16（4096） | 4096 |
  | elementwise-64m（8192×8192） | 2²⁶ | fp16/bf16（4096） | 16384 |
  | elementwise-256m（16384×16384） | 2²⁸ | fp16/bf16（4096） | 65536 |

- **规模判定**: **极大规模**——全部 manifest workload 的 num_logical（256–65536）均 > 物理 Vector 核数 48，且无法通过增大 block_size 缩减到 ≤ 48（UB 容量上限：fp32 block ≤ 12288 → 2²⁰ 需 86 核仍 > 48；256M 规模更不可能）。
- **分核方案**（极大规模 persistent，core-split-strategy.md §1 要素③）:
  - 固定启动内核数 `num_kernels = min(num_logical, 48)`（编译期常量；小 N 场景自动退化为 `num_kernels = num_logical ≤ 48`，同一模板全覆盖，工厂层静态判定无运行时开销）
  - 核内串行：`for i in T.serial(num_local_tasks)`，`num_local_tasks = ceildiv(num_logical, num_kernels)` —— **静态边界**（N/block_size/48 均为编译期常量）
  - 任务映射：`block_id = i × num_kernels + cid`（grid-stride），`if block_id < num_logical` 屏蔽尾部越界
  - 各 workload 任务分布：1M fp16 → 48 核 × ≤6 任务；16M fp16 → 48 × ≤86；64M → 48 × ≤342；256M → 48 × ≤1366——核启动次数固定 48 次（对比超发方案 65536 次串行调度），负载不均 ≤ 1 任务/核（≤2%，可接受）
  - **失败信号**（供 Stage 3/4 对照）：① `num_local_tasks` 依赖运行时值（非静态）→ 编译失败或违反标准；② msprof 显示核串行排队/启动开销占比异常 → 检查 num_kernels 是否误用逻辑核数（wrapper E3 字面形态）；③ 负载不均（末核任务数 2 倍）→ 确认 min() 取值与 ceildiv 上界

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化——本表循环仅 block/tile 级调度循环，无任何逐元素标量循环。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| tile 级（核间） | persistent 并行 | `T.Kernel(num_kernels, is_npu=True)` | §5.5 分核：48 核常驻，每核串行消化多个 tile |
| tile 级（核内多任务） | 串行 | `T.serial(num_local_tasks)`（静态边界） | R3 persistent 标准结构；摊薄核启动开销 |
| 元素级 | 向量化 | `T.vsub / T.vmul / T.vadd`（+`T.vcast` bf16） | §1.6.2 #1–#4；无 T.Parallel 逐元素循环（与源码差异：R1 重设计） |

### 6.2 循环伪代码

```python
with T.Kernel(num_kernels, is_npu=True) as (cid, _):        # 核间：persistent 48 核
    a_ub, b_ub, w_ub, out_ub = T.alloc_ub(...), ...          # §3.3 完整版
    for i in T.serial(num_local_tasks):                      # 核内：静态边界串行
        block_id = i * num_kernels + cid
        if block_id < num_logical:
            # tile 处理（T.copy ×3 → vsub/vmul/vadd → T.copy ×1，全向量化）
            ...
```

### 6.3 流水线优化

**不使用 T.Pipelined**。理由：单阶段 load→compute→store 算子（每 tile 一次进出），无 K 维迭代累加结构可供双缓冲；纯 Vector 简单结构下流水的收益上限低（搬运与计算的 overlap 可交由 Stage 4 按 msprof 数据评估——若 copy 占比显著可试验多 tile 双缓冲，属参数级调优不改变设计结构）。

### 6.4 尾块处理

`N % block_size ≠ 0` 时（任意 broadcast shape 的动态 N；manifest workload 全整除不触发）：

```python
t0 = block_id * block_size
tail = T.min(block_size, N - t0)      # 最后一个 tile 的有效长度
T.copy(a[t0 : t0 + tail], a_ub[0:tail])   # 零起点 dst 切片（规避 UB 非零起点对齐陷阱）
...（v-prefix 整 buffer 操作；UB 中 [tail:block_size] 陈旧段不落 GM）
T.copy(out_ub[0:tail], out[t0 : t0 + tail])
```

模式来源：`examples/elementwise/vec_add_1d.py` L31–L38（官方已验证形态）。不采用 host 侧 padding（违反 host 禁改输入约束，ascend-constraints.md §4）；不采用 Kernel 动态 block（N 已编译期特化，tail 标量一次/tile 开销可忽略）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**: 自动同步（Developer 模式）

### 7.2 同步点说明

无手动同步点。依据：① 每 tile 计算完全独立、无核间数据依赖（无 workspace、无跨核归约）→ 不需要 `T.sync_block_set/wait`；② tile 内 `T.copy → v-prefix → T.copy` 的顺序依赖由 Developer 模式自动同步保证（`T.vadd.md` 等 v-prefix 文档均为无显式同步示例）；③ persistent 核内 `T.serial` 顺序执行天然串行。

### 7.3 pass_configs 配置

无特殊 pass 配置（默认管线）。环境变量：`TILELANG_ASCEND_MODE=Developer`（§2.3；`example_elementwise_add.py` L41 先例）。

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 §0.1 源算子语义为唯一依据（优先移植源仓参考实现 `torch.lerp`，即源仓测试 `test_lerp_tensor.py` 的 ref），**不复刻 §0.6 的 NPU 算法**（不复用 vcast/fp32 中转/persistent 结构）——保证验证独立性。

```python
def golden_lerp_tensor(a, b, w):
    """Tensor-weight lerp 参考实现（源算子语义：out = a + w*(b-a)，同 dtype 直算）。

    与源仓测试基准一致（examples/TileOPs/tests/ops/test_lerp_tensor.py：ref = torch.lerp(a, b, w)）。
    输入为已展平的 (N,) 同 dtype 张量；输出 (N,)。
    """
    return torch.lerp(a, b, w)   # Tensor-weight overload：input + weight * (end - input)
```

### 8.2 精度标准

（承接源仓 `test_lerp_tensor.py` `_lerp_tol`，比模板默认更严）

| dtype | atol | rtol | 说明 |
|-------|------|------|------|
| float16 | 1e-3 | 1e-3 | NPU fp16 路径与源舍入路径一致，预期远优于容差 |
| bfloat16 | 1e-2 | 1e-2 | R2 fp32 中转差异 ≤1–2 ulp bf16（ε≈0.0078），容差覆盖 |
| float32 | 1e-6 | 1e-6 | 同公式同 dtype 域，预期 bit 级一致 |

### 8.3 L0 门槛测试计划

> L0 为 Stage 3 开发的精度门槛（golden 对比必须全过）；完整 L1/L2/Boundary 分层套件由 `tilelang-op-develop` 展开。

| # | 用例 | 输入 | 通过标准 |
|---|------|------|----------|
| L0-1 | smoke 全 dtype | N=2²⁰，fp16/bf16/fp32，`randn(a,b) + rand(w)` | vs golden，§8.2 容差 |
| L0-2 | 大规模抽检 | N=2²⁴ fp16 + bf16（elementwise-16m 代表） | vs golden |
| L0-3 | 尾块非整除 | N ∈ {1000, 4097, 5000}（非 block 整除、非 32B 对齐尾段），fp16/fp32 | vs golden（覆盖 §6.4/风险 R2） |
| L0-4 | 极小 shape | N ∈ {1, 7, 32}（单 tile 非满 / num_logical=1 退化） | vs golden |
| L0-5 | 端点与传播 | w=0（out=a）、w=1（out=b）、w∉[0,1] 外插、a/b 含 ±inf/NaN | vs golden（NaN/Inf 逐元素传播一致） |
| L0-6 | persistent 边界 | N 使 num_logical ∈ {47, 48, 49}（核数切换边界） | vs golden + 无越界（末 tile 屏蔽正确） |

---

## 9. 风险点与注意事项

### 9.1 已知约束

1. **bf16 无 Vector 算术**（`T.vadd/vsub/vmul.md` §2.2.1 dtype 矩阵 bf16 ×）：R2 vcast fp32 中转为唯一路径；cast 指令开销 ~4 次/tile（fp16 路径的 2 倍指令量），带宽口径不变（cast 在 UB 内）——bf16 性能预期略低于 fp16，属 API 硬约束。
2. **物理核数 24/48 为本机实查值**（2026-09-07）：跨设备部署时 `num_kernels` 需随 `NPUUtils.get().get_aicore_num()` 重算——工厂闭包每次 JIT 特化时实查（§3.3 伪代码），无硬编码。
3. **persistent 结构依赖 N 编译期特化**：动态 N 场景（每次新 shape）触发重编译，语义正确但首次调用有 JIT 延迟——与源算子 lru_cache 行为一致，非回退。

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| 尾块非 32B 对齐拷贝异常 | L0-3 的 N=1000/4097（tail 段长度非 32B 倍数） | T.copy 报错或数据错乱 | vec_add_1d.py 同款零起点切片先例；若复现按 pattern-library「UB dst 非零起点」陷阱条目排查（本设计 UB dst 恒零起点已规避主形态）；L0-3 强制覆盖 |
| `T.copy` 静默跨 dtype 转换（pattern-library C12） | copy 的 src/dst dtype 不一致时被静默 cast | 精度路径失控 | 设计中所有 T.copy 严格同 dtype（bf16↔fp32 一律显式 `T.vcast`）；实现期 lint 自查 |
| v-prefix 整段操作处理陈旧 UB 数据 | tail < block_size 时 [tail:] 段垃圾参与计算 | 无正确性影响（不写出）但 denormal 值可能拖慢向量单元 | 已接受（官方先例形态）；如 msprof 显示异常可在搬入前 `T.clear` 尾段（`docs/Tilelang.language/内存操作/T.clear.md`），Stage 4 评估 |
| num_kernels 误用逻辑核数（wrapper E3 字面形态） | 直接照抄 E3 注释 `T.ceildiv(N, block_size)` 作 grid | 65536 核串行调度、启动开销剧增 | §5.5 分核方案为准（min(num_logical, 48)）；失败信号③ |
| block_size 超 UB 上限 | wrapper config 传入 > 12288（fp32）/ > 24576（fp16） | UB 溢出编译失败 | §4.5 上界文档化；实现期 assert |

### 9.3 特殊场景处理

- **三输入不同 broadcast shape**（如 `(3,1)×(1,4)×(3,4)`）：Op 层预广播物化为 `(12,)`——kernel 无感知（源仓测试 `test_lerp_tensor_broadcast` 覆盖该路径，wrapper/Op 层契约不变）
- **N=0**：`LerpTensorFwdOp` 的 `prod()==1` 兜底或上层拦截，kernel 不触达（与源一致）
- **混合 dtype 输入**：Op 层 `forward` 已拒绝（dtype 一致性校验，`test_lerp_tensor_dtype_mismatch_rejected`）——kernel 无需防御

---

## 10. 交付清单

### 10.1 目录结构

```
examples/lerp_tensor/_make_lerp_tensor_kernel/
├── _make_lerp_tensor_kernel.py   # 算子实现（kernel 工厂 + golden + L0 测试入口）
├── DESIGN.md                      # 本设计文档
└── RETROSPECTIVE.md               # Stage 1 复盘（skill Phase 8）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | 已完成 | 本文档 |
| `_make_lerp_tensor_kernel.py` | 待实现（Stage 3） | E1/E2 契约：`_make_lerp_tensor_kernel(N, dtype)` → callable(block_size) → `@T.prim_func main(a,b,w,out)`；bf16 分支按 §3.3；集成目标 wrapper `examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor.py`（Stage 5 接入） |
| golden | 随实现文件交付（§8.1） | `torch.lerp` 同 dtype 直算 |

### 10.3 命名规范

- 项目目录：`lerp_tensor`（conductor 指定）；算子目录：`_make_lerp_tensor_kernel`（源算子函数名，harness 提取件同名）
- 实现文件：`_make_lerp_tensor_kernel.py`；测试入口内嵌（`if __name__ == "__main__"`，vec_add_1d.py 先例）

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）
2. ⬜ Golden 函数（§8.1，验证基准）
3. ⬜ 算子实现（`_make_lerp_tensor_kernel.py`：fp16/fp32 主路径 → bf16 分支 → L0 测试全绿）

## 11. 性能目标（optimize 场景追加，2026-09-07）

> 本章节由 conductor 在 optimize 场景预检后追加（不覆盖既有内容；optimize.md §1 约定）。基准 kernel 与本设计快照在调优期间冻结不改。

| 字段 | 值 |
|------|-----|
| 性能目标类型 | `best_effort`（用户显式要求"尽力调优"） |
| 目标数值 | N/A（best_effort，以 plateau / 迭代上限收敛） |
| Baseline | ① 集成基准 kernel（本包 `_make_lerp_tensor_kernel.py`，Stage 5 bench events 口径）；② `torch.lerp` NPU（msprof 口径） |
| 测试 shape | manifest workloads：1M (1024²) fp32/fp16/bf16、16M (4096²) fp16/bf16/fp32、64M (8192²) fp16/bf16、256M (16384²) fp16/bf16 |
| 噪声阈值 | 3%（默认采纳门槛） |
| 最大迭代数 | 10（默认） |
| 回归入口 | `python perf_opt/_make_lerp_tensor_kernel.py --level all`（L0/L1 失败阻塞）+ 采纳后 TileOPs pytest `tests/ops/test_lerp_tensor.py` |

### Stage 5 基线数值（Ascend910B2C，profile_run_msprof_20260907_023928.log）

| shape | dtype | block_size | latency_us | bandwidth_tbs | vs torch.lerp |
|-------|-------|-----------|-----------|--------------|---------------|
| 1M | fp32 / fp16 / bf16 | 2048/4096/4096 | 136.59 / 182.83 / 149.70 | 0.123 / 0.046 / 0.056 | 慢（派发开销主导） |
| 16M | fp16 / bf16 / fp32 | 4096/4096/2048 | 225.76 / 188.82 / 399.86 | 0.595 / 0.711 / 0.671 | 慢～接近 |
| 64M | fp16 / bf16 | 4096 | 571.78 / 577.50 | 0.939 / 0.930 | 1.21× |
| 256M | fp16 / bf16 | 4096 | 1988.22 / 1987.09 | 1.080 / 1.081 | 1.42× |

已知 headroom（Stage 5 记录）：中小 N 派发/开销主导慢于 torch；有效带宽峰值 1.08 TB/s ≈ 峰值基准 1.8 TB/s 的 60%。tileops=events / torch=msprof 计时口径差异仅作趋势参考。
