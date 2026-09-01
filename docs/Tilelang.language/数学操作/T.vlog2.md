# Tilelang.language.vlog2

## 1. OP概述

简介：`tilelang.language.vlog2`执行逐元素底数为2的对数计算 $log_2(src)$

由于底层不支持硬件级别的log2操作，实际上使用的是 $B = ln(A) \times \frac{1}{ln2}$。
该复合展开在 codegen 阶段完成，所需的中间结果 $ln(A)$ 由 codegen 根据输入 region 分析大小后自动分配临时 buffer，**不再体现在用户接口中**。

```python
T.vlog2(input, output)
```

## 2. OP规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| - | - | - |
| `input`  | `tensor` | 输入tensor |
| `output` | `tensor` | 输出tensor |

注意：旧版本需要第三个参数 `tmpBuffer`，现已废弃。临时 buffer 由 codegen 内部分配，用户无需也无法显式指定。

### 2.2 支持规格

#### 2.2.1 DataType支持

|   | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
| - | - | - | - | - | - | - | - | - | - | - | - | - |
| Ascend | × | × | × | × | × | × | × | × | √ | √ | × | × |

#### 2.2.2 Shape支持

input 与 output 形状需要一致。input 与 output 可以指向同一 buffer（原地计算），codegen 分配的临时 buffer 与 input/output 不存在别名。

### 2.3 特殊限制说明

无

### 2.4 使用方法

以下示例展示了对形状为(M,N)的输入tensor进行vlog2计算：

```python
@tilelang.jit(target="npuir")
def vec_log2(M, N, block_M, block_N):
    m_num = M // block_M
    n_num = N // block_N
    BLOCK_SIZE = 8

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            A_VEC = T.alloc_ub((block_M, block_N), dtype)
            B_VEC = T.alloc_ub((block_M, block_N), dtype)
            for i in T.serial(T.ceildiv(m_num * n_num, BLOCK_SIZE)):
                block_id = i * BLOCK_SIZE + cid
                if block_id < m_num * n_num:
                    block_id_m = block_id // n_num
                    block_id_n = block_id % n_num
                    bx = block_id_m * block_M
                    by = block_id_n * block_N
                    T.copy(A[bx, by], A_VEC)
                    T.vlog2(A_VEC, B_VEC)
                    T.copy(B_VEC, B[bx, by])

    return main
```

## 3. Tilelang Op到Ascend NPU IR Op的转换

`tilelang::npuir_log2` 在 frontend 仅产生一个 `tl.npuir_log2(A, B)` 调用。在 codegen 阶段，该 op 被展开为：

1. 由 codegen 分析 `A` 的 region，分配与 `A` 同 shape/dtype 的临时 buffer `tmp`；
2. `tmp = ln(A)`，对应 `hivm::VLnOp`（A5 dev 模式下对应 `math::LogOp`，A5 expert 模式下对应 `linalg::ElemwiseUnaryOp[log]`）；
3. `B = tmp * (1/ln2)`，对应 `hivm::VMulOp`（A5 dev 模式下对应 `arith::MulFOp`，A5 expert 模式下对应 `linalg::ElemwiseBinaryOp[mul]`）。

临时 buffer `tmp` 不出现在 TIR 语义中，仅存活于 codegen 生成的 MLIR 内，由后续 buffer 规划 pass 统一管理生命周期。
