# Copyright (c) Huawei Technologies Co., Ltd. 2026.
import torch
import tilelang
import tilelang.language as T
import pytest

from testcommon import assert_close, gen_tensor

# this kernel has three input A, B, C, and 1 out,
# first do tmp = A + B, then do out = tmp @ C

pytestmark = [
    pytest.mark.op("mix_kernel_bf16"),
    pytest.mark.mode("Developer"),
]
DTYPES = ["bfloat16"]


@tilelang.jit(target="npuir")
def mix_kernel_bf16_func(
    M, K, N, block_M, block_K, block_N, dtype="bfloat16", accum_dtype="float32"
):
    bx = T.ceildiv(N, block_N)
    by = T.ceildiv(M, block_M)
    bk = T.ceildiv(K, block_K)

    @T.prim_func
    def mix_kernel_bf16_kernel(
        A: T.Tensor((M, K), dtype),  # type: ignore
        B: T.Tensor((M, K), dtype),  # type: ignore
        C: T.Tensor((K, N), dtype),  # type: ignore
        out: T.Tensor((M, N), accum_dtype),  # type: ignore
    ):
        with T.Kernel(bx * by, is_npu=True) as (cid, _):
            local_a = T.alloc_shared((block_M, block_K), dtype)
            local_b = T.alloc_shared((block_M, block_K), dtype)
            local_c = T.alloc_shared((block_K, block_N), dtype)
            local_out = T.alloc_shared((block_M, block_N), accum_dtype)
            T.clear(local_out)
            idx = cid % bx * block_N
            idy = cid // bx * block_M
            for k in T.Pipelined(bk):
                idk = k * block_K
                # Load A, B and C into shared memory
                T.copy(A[idy : idy + block_M, idk : idk + block_K], local_a)
                T.copy(B[idy : idy + block_M, idk : idk + block_K], local_b)
                T.copy(C[idk : idk + block_K, idx : idx + block_N], local_c)
                # Compute local_out += (local_a + local_b) @ local_c
                local_tmp = T.alloc_shared((block_M, block_K), dtype)
                T.vadd(local_a, local_b, local_tmp)
                T.gemm(local_tmp, local_c, local_out)

            # Write back the result to out
            T.copy(local_out, out[idy : idy + block_M, idx : idx + block_N])

    return mix_kernel_bf16_kernel


@pytest.mark.parametrize("dtype", DTYPES)
def test_mix_kernel_bf16(dtype):
    M, K, N = 64, 128, 64
    block_M, block_K, block_N = 16, 32, 16
    A = gen_tensor((M, K), dtype)
    B = gen_tensor((M, K), dtype)
    C = gen_tensor((K, N), dtype)
    out = gen_tensor((M, N), "float32", kind="zeros")

    mix_kernel_bf16_func(M, K, N, block_M, block_K, block_N)(A, B, C, out)
    expected = ((A + B) @ C).to(torch.float32)
    assert_close(out, expected, rtol=1e-2, atol=1e-2)
