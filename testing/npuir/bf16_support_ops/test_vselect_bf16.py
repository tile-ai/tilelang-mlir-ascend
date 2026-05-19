# Copyright (c) Huawei Technologies Co., Ltd. 2026.
import torch
import tilelang
import tilelang.language as T
import pytest

from testcommon import assert_close, gen_tensor

pytestmark = [
    pytest.mark.op("vselect_bf16"),
    pytest.mark.mode("Developer"),
]

DTYPES = ["bfloat16"]


@tilelang.jit(target="npuir")
def vselect_bf16_func(M, N, block_M, block_N, dtype="bfloat16"):
    bx = T.ceildiv(N, block_N)
    by = T.ceildiv(M, block_M)

    @T.prim_func
    def vselect_bf16_kernel(
        A: T.Tensor((M, N), dtype),  # type: ignore
        out: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(bx * by, is_npu=True) as (cid, _):
            idx = cid % bx * block_N
            idy = cid // bx * block_M
            local_a = T.alloc_shared((block_M, block_N), dtype)
            T.copy(A[idy : idy + block_M, idx : idx + block_N], local_a)
            bool_mask = T.alloc_shared((block_M, block_N), "bool")

            value = 0.5
            T.vcmp(local_a, value, bool_mask, "gt")
            local_out = T.alloc_shared((block_M, block_N), dtype)
            local_tmp = T.alloc_shared((block_M, block_N), dtype)
            T.clear(local_tmp)
            T.vselect(bool_mask, local_a, local_tmp, local_out)
            T.copy(local_out, out[idy : idy + block_M, idx : idx + block_N])

    return vselect_bf16_kernel


@pytest.mark.parametrize("dtype", DTYPES)
def test_vselect_bf16(dtype):
    M, N = 64, 64
    block_M, block_N = 16, 16
    A = gen_tensor((M, N), dtype)
    out = gen_tensor((M, N), dtype, kind="zeros")
    vselect_bf16_func(M, N, block_M, block_N)(A, out)
    expected = torch.where(A > 0.5, A, torch.zeros_like(A))
    out = out.to(torch.float32)
    expected = expected.to(torch.float32)
    assert_close(out, expected, rtol=1e-2, atol=1e-2)
