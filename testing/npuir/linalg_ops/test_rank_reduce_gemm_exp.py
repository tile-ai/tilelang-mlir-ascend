# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor

pytestmark = [
    pytest.mark.op("gemm"),
    pytest.mark.mode("Expert"),
]


@tilelang.jit(
    out_idx=[-1],
    target="npuir",
    pass_configs={
        tilelang.PassConfigKey.NPUIR_ENABLE_AUTO_MULTI_BUFFER: False,
    },
)
def matmul(M, N, K, block_M, block_N, block_K, in_dtype, out_dtype, num_stages):
    m_num = M // block_M
    n_num = N // block_N

    multi_l1 = 2
    multi_l0 = 1

    @T.prim_func
    def sliceGemmExp(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor((K, N), in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num * block_M
            by = cid % n_num * block_N

            A_buf = T.alloc_L1((multi_l1, block_M, block_K), in_dtype)
            B_buf = T.alloc_L1((multi_l1, block_K, block_N), in_dtype)
            C_buf = T.alloc_L0C((multi_l0, block_M, block_N), out_dtype)

            for ki in T.serial(T.ceildiv(K, block_K)):
                offset_k = ki * block_K
                T.copy(
                    A[bx : bx + block_M, offset_k : offset_k + block_K],
                    A_buf[ki % multi_l1, :, :],
                )
                T.copy(
                    B[offset_k : offset_k + block_K, by : by + block_N],
                    B_buf[ki % multi_l1, :, :],
                )

                T.gemm(
                    A_buf[ki % multi_l1, :, :],
                    B_buf[ki % multi_l1, :, :],
                    C_buf[0, :, :],
                    initC=ki == 0,
                    b_transpose=False,
                )

                T.copy(C_buf[0, :, :], C[bx : bx + block_M, by : by + block_N])

    return sliceGemmExp


DTYPE_CASES = [("float16", "float32")]


@pytest.mark.parametrize("in_dtype,out_dtype", DTYPE_CASES)
def test_matmul_exp(in_dtype, out_dtype):
    M, K, N = 256, 512, 256
    A = gen_tensor((M, K), in_dtype, kind="randn")
    B = gen_tensor((K, N), in_dtype, kind="randn")
    ref_C = torch.matmul(A, B).to(torch.float32)

    kernel = matmul(
        M=M,
        N=N,
        K=K,
        block_M=32,
        block_K=32,
        block_N=32,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        num_stages=0,
    )
    C = kernel(A, B)

    assert_close(C, ref_C, dtype=out_dtype, rtol=1e-2, atol=1e-2)
