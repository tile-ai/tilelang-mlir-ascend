# Tilelang.language.arange

# 1. OP概述

简介：`tilelang.language.arange` 根据步长（strides）和偏移量（offset），向向量中填充从 0、1、2…… 开始的连续序列。

```markup
T.arange(dst, strides: Union[list, tuple], offset=0)
```

## 2. OP规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| - | - | - |
| `dst` | `tensor` | 输出tensor |
| `strides`                         | `Union[list,tuple]` | 输入步长 |
| `offset`                          | `int` | 输入偏移量 |

### 2.2 支持规格

#### 2.2.1 DataType支持

|   | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
| - | - | - | - | - | - | - | - | - | - | - | - | - |
| Ascend | × | × | × | √ | × | √ | × | √ | √ | √ | × | × |

#### 2.2.2 Shape支持

结论：在shape方面，arange无特殊要求；

### 2.3 特殊限制说明

#### 2.3.1 问题说明

二维 `T.arange` 的后端库实际使用 Scalar 流水逐元素写入 UB，但算子统一按 `PIPE_V` 建模；缺少 `PIPE_S -> PIPE_V` 跨流水同步时，后继向量操作可能提前读取尚未完全可见的数据。

`T.arange` 在 TileLang 中依次降级为 `tl.npuir_arange` 和 `hivm.hir.varange`。`hivmc` 随后将 `hivm.hir.varange` 转换为 arange 库调用。根据当前工具链中库实现的反汇编结果，一维和二维 arange 使用不同的实现：

- 二维 `arange_2d_core` 使用标量嵌套循环逐元素计算 `offset + i * stride0 + j * stride1`，并以标量 store 写入 UB；
- 一维连续 `arange_1d_core` 先以标量方式生成一个 32B 的种子块，再通过 `VADDS`（向量加标量）倍增扩展剩余序列。种子元素数为 `32 / sizeof(dtype)`，例如 float16 为 16 个、float32 为 8 个。若序列未超过一个种子块，则仅执行标量部分；一维非连续布局以及 int64 的扩展也不走上述 `VADDS` 路径。

`VArangeOp` 在 BishengIR 中继承 `HIVM_VectorOp`，因此统一按 `PIPE_V`建模。已检查的库实现中，二维 `arange_2d_core` 实际执行的是 `PIPE_S` 标量写，但函数内部没有一维实现所具有的 `PIPE_V -> PIPE_S`、`PIPE_S -> PIPE_V` 跨流水 `set_flag/wait_flag`。自动注入的 `pipe_barrier[PIPE_V]` 只能等待 Vector 流水，不能覆盖“标量写 UB -> 后继向量读”的真实依赖。这可能使二维 arange 的部分写入尚未对后继向量操作可见时，后继操作就开始读取，从而留下 stale UB 数据。

该问题的根因是 `hivm.varange` 的 `PIPE_V` 建模与二维库实现中的标量写入不一致，并且二维实现缺少必要的跨流水同步。

#### 2.3.2 修改方案说明

应尽可能避免使用二维 `T.arange`, 底层是通过标量实现，性能较差；在能够保持语义等价时，优先改为一维连续 `T.arange`，或者使用其他向量操作构造结果。

1. 目标形状为 `(1, N)` 时

   第一维只有一个元素，`stride0` 不影响结果。可以先生成一维 `(N,)` 序列，再通过 `T.reshape` 得到 `(1, N)`；如需得到 `(M, N)` 的重复行，再使用 `T.vbrc` 广播：

```python
idx_src = T.alloc_shared((N,), "float32")
idx_row = T.alloc_shared((1, N), "float32")
T.arange(idx_src, [stride1], offset)  # 原 strides=[0, 1] 时使用 [1], 0
T.reshape(idx_src, idx_row)     # (N,) -> (1, N)，纯元数据零拷贝
T.vbrc(idx_row, idx_j)
```

2. 目标形状为 `(N, 1)` 时

   第二维只有一个元素，`stride1` 不影响结果。可以先生成一维 `(N,)` 序列，再通过 `T.reshape` 得到 `(N, 1)`：

```python
idx_src = T.alloc_shared((N,), "float32")
idx_col = T.alloc_shared((N, 1), "float32")
T.arange(idx_src, [stride0], offset)  # 原 strides=[1, 0] 时使用 [1], 0
T.reshape(idx_src, idx_col)           # (N,) -> (N, 1)，纯元数据零拷贝
```

3. 目标形状为一般二维 `(M, N)` 时

   当 `M > 1` 且 `N > 1` 时，如果满足 `stride0 == N * stride1`，二维 `T.arange` 可以等价改写为一维 `T.arange` 再 `T.reshape`：

```python
idx_src = T.alloc_shared((M * N,), "float32")
idx_mat = T.alloc_shared((M, N), "float32")
T.arange(idx_src, [stride1], offset)
T.reshape(idx_src, idx_mat)
```

   该写法与 `T.arange(idx_mat, [N * stride1, stride1], offset)` 等价。例如 `stride1=1` 时，二维 strides 为 `[N, 1]`。

   其他不满足上述条件的 stride 组合也必须按目标数值关系重新构造。

如果无法进行语义等价改写，可以在二维 `T.arange` 与后继向量操作之间使用 `T.pipe_barrier("PIPE_ALL")` 作为覆盖 Scalar 和 Vector 流水的保守同步方案。

### 2.4 使用方法

以下示例实现了一个形状为(M,N)的tensor的arange功能

```python
@tilelang.jit(target="npuir")
def vec_arange(M, N, block_M, block_N, src_dtype="float32", dst_dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dst_dtype),
        B: T.Tensor((M, N), dst_dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx_ = cid // n_num
            bx = bx_ * block_M
            by_ = cid % n_num
            by = by_ * block_N

            A_VEC = T.alloc_ub((block_M, block_N), dst_dtype)
            B_VEC = T.alloc_ub((block_M, block_N), dst_dtype)
            strides = [1, 2]
            T.arange(A_VEC, strides, offset=1)
            T.arange(B_VEC, strides)
            T.copy(A_VEC, A[bx, by])
            T.copy(B_VEC, B[bx, by])

    return main
```

## 3. Tilelang Op到Ascend NPU IR Op的转换

**tilelang::arangeOp**将被转换为hivm::VArangeOp
