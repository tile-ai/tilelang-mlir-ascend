# Copyright (c) Huawei Technologies Co., Ltd. 2026.
import torch
import tilelang
import tilelang.language as T
import pytest

from testcommon import assert_close, gen_tensor

# this kernel has three input A, B, C, and 1 out,
# calculate out = A * B + C

pytestmark = [
    pytest.mark.op("vv_bf16"),
    pytest.mark.mode("Developer"),
]
DTYPES = ["bfloat16"]


@tilelang.jit(target="npuir")
def vv_bf16_func(M, N, block_M, block_N, dtype="bfloat16"):
    bx = T.ceildiv(N, block_N)
    by = T.ceildiv(M, block_M)

    @T.prim_func
    def vv_bf16_kernel(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
        C: T.Tensor((M, N), dtype),  # type: ignore
        out: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(bx * by, is_npu=True) as (cid, _):
            local_a = T.alloc_shared((block_M, block_N), dtype)
            local_b = T.alloc_shared((block_M, block_N), dtype)
            local_c = T.alloc_shared((block_M, block_N), dtype)
            local_out = T.alloc_shared((block_M, block_N), dtype)
            idx = cid % bx * block_N
            idy = cid // bx * block_M
            T.copy(A[idy : idy + block_M, idx : idx + block_N], local_a)
            T.copy(B[idy : idy + block_M, idx : idx + block_N], local_b)
            local_tmp = T.alloc_shared((block_M, block_N), dtype)
            T.vmul(local_a, local_b, local_tmp)
            T.copy(C[idy : idy + block_M, idx : idx + block_N], local_c)
            T.vadd(local_tmp, local_c, local_out)
            T.copy(local_out, out[idy : idy + block_M, idx : idx + block_N])

    return vv_bf16_kernel


@pytest.mark.parametrize("dtype", DTYPES)
def test_vv_bf16(dtype):
    M, N = 64, 64
    block_M, block_N = 16, 16
    A = gen_tensor((M, N), dtype)
    B = gen_tensor((M, N), dtype)
    C = gen_tensor((M, N), dtype)
    out = gen_tensor((M, N), dtype, kind="zeros")
    vv_bf16_func(M, N, block_M, block_N)(A, B, C, out)
    expected = A * B + C
    out = out.to(torch.float32)
    expected = expected.to(torch.float32)
    assert_close(out, expected, rtol=1e-2, atol=1e-2)
