# Tilelang.language.copy

## 1. OP概述

简介：`tilelang.language.copy` 该算子用于在不同内存区域之间执行数据复制操作，是最通用的数据搬运接口。

支持的搬运方向：

| 搬运方向 | Developer 模式 | Expert 模式 | 说明 |
| --------- | :---: | :---: | ---- |
| GM -> UB  | √ | √ | 通用数据加载 |
| UB -> GM  | √ | √ | 通用数据写回 |
| UB -> UB（含 L0C fragment 与 UB 之间的局部拷贝） | √ | √ | 局部缓冲间拷贝 |
| GM -> L1  | ×（用 T.load_nd2nz） | √ | 自动转换为 ND2NZ 搬运（等价 `T.load_nd2nz`） |
| L0C -> GM | √（经 VCast 通用路径） | √ | 自动转换为 Fixpipe 搬运（等价 `T.store_fixpipe`，支持量化） |
| GM -> GM  | ×（编译期报错） | 未验证 | Developer 模式显式不支持；Expert 模式 codegen 未拦截但无使用用例，不建议使用 |

```python
T.copy(src, dst)                                  # 整缓冲拷贝 [Developer Op] [Expert Op]
T.copy(src[bx : bx + M, by : by + N], dst[0:M, 0:N])  # 切片语法 [Developer Op] [Expert Op]
T.copy(src[bx, by], dst, size=[M, N])             # 索引访问 + size [Developer Op] [Expert Op]
```

## 2. OP规格

### 2.1 参数说明

| 参数名    | 类型         | 说明       |
| ----------- | -------------- | ------------ |
| `src` | `tensor` | 源操作数，三种形式均可：整缓冲（`A_ub`）、索引访问（`A[bx, by]`）、切片（`A[bx : bx + M, by : by + N]`） |
| `dst` | `tensor` | 目的操作数，形式同 `src` |
| `coalesced_width` | `Optional[int]` | 合并访存宽度提示，仅 GPU 目标（cuda/hip/webgpu）使用；npuir 目标下传入会被忽略 |
| `size` | `Optional[List[PrimExpr]]` | 显式指定拷贝范围（legacy 兼容参数）。当两侧均为索引访问（如 `T.copy(A[bx, by], B[bx, by])`）时必须提供；不可与切片语法同时使用 |

拷贝范围（extent）推断规则（前端 `tilelang/language/copy.py`）：

1. 切片操作数：使用切片自身的 extents；
2. 整缓冲操作数：使用缓冲完整 shape；
3. 索引访问操作数（只携带起点，无范围信息）：借用另一侧操作数的 extents，或使用 `size`；
4. 同 rank 的整缓冲与切片配对时（如 `T.copy(A[bx : bx + M, by : by + N], A_ub)`），整缓冲一侧沿用切片一侧的 extents，用于尾块（tail tile）拷贝；
5. `size` 优先级最高；`size` 的 rank 允许大于缓冲 rank，此时取尾部维度对齐；
6. 两侧均为索引访问且未提供 `size` 时，无法推断拷贝范围，编译期报错（提示使用切片语法或 `size`）。

### 2.2 支持规格

#### 2.2.1 DataType支持

|        | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| -------- | ------- | ------ | -------- | ------- | -------- | ------- | -------- | ------- | ------ | ------ | ------ | ----------- |
| Ascend | √    | √   | √     | √    | √     | √    | √     | √    | √   |√   | √   | ×        |

dtype 约束（见 2.3）：

- Developer 模式：支持 src/dst 跨 dtype 拷贝，由编译器自动插入类型转换；
- Expert 模式（GM→L1 / L0C→GM 之外的通用路径）：要求 src 与 dst 的 dtype 一致；
- Expert 模式 L0C→GM：仅支持 fp32→fp16 / fp32→bf16 / int32→int8 三种量化组合。

#### 2.2.2 Shape支持

结论：src 与 dst 的拷贝区域需在"对齐后逻辑 shape 一致"，不要求字面 shape 完全相同：

1. 静态大小为 1 的维度会被自动折叠（rank 可不同，如 2D 拷贝到 3D 缓冲、含 1 的维度差异）；
2. 尾块拷贝（动态 tile 尺寸，`T.min(block_M, M - bx)` 等）支持：切片一侧的动态 extents 会被整缓冲一侧借用；
3. GM→L1（Expert）路径要求 GM 侧保持至少 2 维（min_rank=2），NZ 格式转换在搬运内完成；
4. Developer 模式 GM→UB：UB 侧缓冲 shape 须为静态（动态 range 以 UB 缓冲静态维为分配上界）。

### 2.3 特殊限制说明

1. **切片语法与 `size` 参数互斥**：同一调用中二者不可同时使用，否则前端报 `ValueError`；
2. **Expert 模式通用路径禁止跨 dtype**：src 与 dst dtype 不一致时报编译期错误 "T.copy does not support element type casting"，需先显式 `T.cast`；
3. **Expert 模式 L0C→GM 量化组合受限**：仅支持 fp32→fp16（F322F16）、fp32→bf16（F322BF16）、int32→int8（S322I8），其它跨 dtype 组合编译期报错；
4. **Developer 模式支持跨 dtype**：由编译器自动插入 `hivm::VCastOp`（RoundMode=RINT，四舍五入到偶数）完成类型转换，支持的类型对与 `T.vcast` 一致；
5. **Developer 模式不支持 GM→GM 拷贝**（memref 到 memref 直连，编译期报错 "Unsupported memref to memref copy yet"），需经 UB 中转。Expert 模式 codegen 未显式拦截 GM→GM，但仓库内无使用用例、后端支持未验证，不建议使用；
6. **Expert 模式 GM→L1 / L0C→GM 的 `T.copy` 等价于 `T.load_nd2nz` / `T.store_fixpipe`**（enable_nz2nd=true、channel_split=false、pre_relu_mode=no_relu 为固定默认值，不可通过 `T.copy` 定制；如需定制请使用对应显式接口）。

### 2.4 使用方法

示例1：实现了Expert Mode中将一个二维张量（Tensor）copy到A_ub中 (gm -> ub)

```python
@tilelang.jit(target="npuir")
def atomic_add_2d(M, N, block_M, block_N, dtype="float32"):
    m_blocks = M // block_M
    n_blocks = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_blocks * n_blocks, is_npu=True) as (cid, _):
            bx = (cid // n_blocks) * block_M
            by = (cid % n_blocks) * block_N
            A_ub = T.alloc_ub((block_M, block_N), dtype)
            tile_M = T.min(block_M, M - bx)
            tile_N = T.min(block_N, N - by)
            T.copy(
                A[bx : bx + tile_M, by : by + tile_N],
                A_ub[0:tile_M, 0:tile_N],
            )
            T.npuir_atomic_add(B[bx, by], A_ub, [tile_M, tile_N])

    return main
```

示例2：实现了Developer Mode中将一个二维张量（Tensor）的copy到A_shared中 (gm -> ub)

```python
@tilelang.jit(target="npuir")
def atomic_add_2d_dev(M, N, block_M, block_N, dtype="float32"):
    m_blocks = M // block_M
    n_blocks = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_blocks * n_blocks, is_npu=True) as (cid, _):
            bx = (cid // n_blocks) * block_M
            by = (cid % n_blocks) * block_N
            A_shared = T.alloc_shared((block_M, block_N), dtype)
            tile_M = T.min(block_M, M - bx)
            tile_N = T.min(block_N, N - by)
            T.copy(
                A[bx : bx + tile_M, by : by + tile_N],
                A_shared[0:tile_M, 0:tile_N],
            )
            T.npuir_atomic_add(B[bx, by], A_shared, [tile_M, tile_N])

    return main
```

示例3：索引访问 + `size` 写法，gm -> ub 加载与 ub -> gm 写回（Developer Mode）

```python
@tilelang.jit(target="npuir")
def reduce_sum_2d(M, N, block_M, dtype="float16"):
    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, 1), dtype)):
        with T.Kernel(T.ceildiv(M, block_M), is_npu=True) as (cid, _):
            offset = cid * block_M
            A_shared = T.alloc_shared((block_M, N), dtype)
            B_local = T.alloc_fragment((block_M, 1), dtype)
            # 索引访问 A[offset, 0] 只携带起点，size 显式给出拷贝范围 (gm -> ub)
            T.copy(A[offset, 0], A_shared, size=[block_M, N])
            T.reduce(A_shared, B_local, dims=1, reduce_mode="sum", clear=True)
            # ub -> gm 写回，size=[block_M, 1] 表示每行取 1 列
            T.copy(B_local, B[offset, 0], size=[block_M, 1])

    return main
```

示例4：Expert Mode Cube 路径，gm -> L1（自动 ND2NZ）与 L0C -> gm（自动 Fixpipe）

```python
with T.Scope("Cube"):
    l1_a = T.alloc_L1([block_m, block_share], dtype)
    l0_c = T.alloc_L0C([block_m, block_share], accum_dtype)

    T.copy(
        Q[offset_m : offset_m + tail_size_m, offset_k : offset_k + tail_size_k],
        l1_a[:tail_size_m, :tail_size_k],   # gm -> L1，自动转换为 ND2NZ
    )
    T.gemm(l1_a, l1_b, l0_c, initC=True, b_transpose=True)

    T.copy(
        l0_c[:tail_size_m, :tail_size_n],   # L0C -> gm，自动转换为 Fixpipe
        workspace[offset_m : offset_m + tail_size_m,
                  offset_n : offset_n + tail_size_n],
    )
```

## 3. Tilelang Op到Ascend NPU IR Op的转换

**1. 在expert 模式下：**

**GM -> L1：** `tilelang::copyOp` 将被转换为 `hivm::ND2NZOp`（GM 侧保持 min_rank=2，NZ 格式转换由 ND2NZ 完成）

**L0C -> GM：** `tilelang::copyOp` 将被转换为 `hivm::FixpipeOp`（enable_nz2nd=true、channel_split=false、pre_relu_mode=no_relu；src/dst dtype 一致时 pre_quant=NO_QUANT，fp32→fp16 / fp32→bf16 / int32→int8 时分别为 F322F16 / F322BF16 / S322I8）

**其它通用路径（GM↔UB、UB↔UB 等，要求 src/dst dtype 一致）：**

当src和dst的shape一致时：**tilelang::copyOp**将被转换为memref::CopyOp

否则：**tilelang::copyOp**将被转换为memref::ExtractStridedMetadataOp、memref::DimOp（动态shape）、arith::ConstantIndexOp（非动态shape）、memref::ReinterpretCastOp、memref::CopyOp（折叠静态 1 维并对齐逻辑 shape 后拷贝）

**2. 在developer 模式下（按 src/dst 的 IR 类型分发）：**

**GM -> UB:  tilelang::copyOp**将被转换为 memref::SubViewOp（GM 侧 rank 折叠视图）、memref::AllocOp（UB 分配）、memref::CopyOp、bufferization::ToTensorOp、（tensor::DimOp、tensor::EmptyOp、hivm::VCastOp）[for type cast]、tensor::InsertSliceOp

**UB/L0C -> GM（tensor -> memref）:  tilelang::copyOp**将被转换为 tensor::ExtractSliceOp、（tensor::DimOp、tensor::EmptyOp、hivm::VCastOp）[for type cast]、memref::SubViewOp（GM 侧目的视图）、bufferization::MaterializeInDestinationOp（全范围同 shape 时走快速路径，直接 MaterializeInDestination）

**UB <-> UB / L0C <-> UB（tensor -> tensor）:  tilelang::copyOp**将被转换为 tensor::ExtractSliceOp、（tensor::DimOp、tensor::EmptyOp、hivm::VCastOp）[for type cast]、tensor::InsertSliceOp（全范围同 shape 时走快速路径，仅 VCastOp）

**GM -> GM（memref -> memref）:** 不支持，编译期报错
