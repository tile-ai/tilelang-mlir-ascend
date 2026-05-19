# Copyright (c) Huawei Technologies Co., Ltd. 2026.
import torch
import tilelang
import tilelang.language as T
import pytest

from testcommon import assert_close, gen_tensor

pytestmark = [
    pytest.mark.op("gemm_bf16"),
    pytest.mark.mode("Developer"),
]

DTYPES = ["bfloat16"]


@tilelang.jit(target="npuir")
def gemm_bf16_func(
    M, K, N, block_M, block_K, block_N, dtype="bfloat16", accum_dtype="float32"
):
    bx = T.ceildiv(N, block_N)
    by = T.ceildiv(M, block_M)
    bk = T.ceildiv(K, block_K)

    @T.prim_func
    def gemm_bf16_kernel(
        A: T.Tensor((M, K), dtype),  # type: ignore
        B: T.Tensor((K, N), dtype),  # type: ignore
        C: T.Tensor((M, N), accum_dtype),  # type: ignore
    ):
        with T.Kernel(bx * by, is_npu=True) as (cid, _):
            local_a = T.alloc_shared((block_M, block_K), dtype)
            local_b = T.alloc_shared((block_K, block_N), dtype)
            local_c = T.alloc_shared((block_M, block_N), accum_dtype)
            T.clear(local_c)
            idx = cid % bx * block_N
            idy = cid // bx * block_M
            for k in T.Pipelined(bk):
                idk = k * block_K
                # Load A and B into shared memory
                T.copy(A[idy : idy + block_M, idk : idk + block_K], local_a)
                T.copy(B[idk : idk + block_K, idx : idx + block_N], local_b)
                # Compute local_c += local_a @ local_b
                T.gemm(local_a, local_b, local_c)

            # Write back the result to C
            T.copy(local_c, C[idy : idy + block_M, idx : idx + block_N])

    return gemm_bf16_kernel


@pytest.mark.parametrize("dtype", DTYPES)
def test_gemm_bf16(dtype):
    M, K, N = 64, 128, 64
    block_M, block_K, block_N = 16, 32, 16
    A = gen_tensor((M, K), dtype)
    B = gen_tensor((K, N), dtype)
    C = gen_tensor((M, N), "float32", kind="zeros")
    gemm_bf16_func(M, K, N, block_M, block_K, block_N)(A, B, C)
    expected = (A @ B).to(torch.float32)
    assert_close(C, expected, rtol=1e-2, atol=1e-2)
