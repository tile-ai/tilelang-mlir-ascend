# Copyright (c) Huawei Technologies Co., Ltd. 2026.
import tilelang
import tilelang.language as T
import pytest

from testcommon import assert_close, gen_tensor

pytestmark = [
    pytest.mark.op("reduce_add_bf16"),
    pytest.mark.mode("Developer"),
]
DTYPES = ["bfloat16"]


@tilelang.jit(target="npuir")
def reduce_add_bf16_func(M, block_M, N, dtype="bfloat16", accum_dtype="float32"):
    by = T.ceildiv(M, block_M)

    @T.prim_func
    def reduce_add_bf16_kernel(
        A: T.Tensor((M, N), dtype),  # type: ignore
        out: T.Tensor((block_M, 1), accum_dtype),  # type: ignore
    ):
        with T.Kernel(by, is_npu=True) as (cid, _):
            idy = cid * block_M
            local_a = T.alloc_shared((block_M, N), dtype)
            local_out = T.alloc_shared((block_M, 1), accum_dtype)
            T.clear(local_out)
            T.copy(A[idy : idy + block_M, :], local_a)
            T.reduce_sum(local_a, local_out, dim=1)
            T.copy(local_out, out[idy : idy + block_M, 0])

    return reduce_add_bf16_kernel


@pytest.mark.parametrize("dtype", DTYPES)
def test_reduce_add_bf16(dtype):
    M, N = 64, 64
    block_M = 16
    A = gen_tensor((M, N), dtype)
    out = gen_tensor((M, 1), "float32", kind="zeros")
    reduce_add_bf16_func(M, block_M, N)(A, out)
    expected = A.float().sum(dim=1, keepdim=True)
    assert_close(out, expected, rtol=1e-2, atol=1e-2)
