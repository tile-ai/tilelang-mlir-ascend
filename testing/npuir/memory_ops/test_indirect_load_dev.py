# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor

pytestmark = [
    pytest.mark.op("indirect_load"),
    pytest.mark.mode("Developer"),
]

DTYPES = ["float32"]
INDIRECT_LOAD_1D_CASES = [(1024, 256)]


@tilelang.jit(target="npuir")
def indirect_load_1d_ub(N, block_N, dtype="float32"):
    @T.prim_func
    def main(
        X: T.Tensor((N * 2,), dtype),
        IDX_GM: T.Tensor((N,), "int32"),
        OUT_GM: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), is_npu=True) as (pid, _):
            start = pid * block_N
            valid = T.min(block_N, N - start)

            IDX_UB = T.alloc_shared((block_N,), "int32")
            O_UB = T.alloc_shared((block_N,), dtype)

            T.copy(IDX_GM[start:start + valid], IDX_UB[0:valid])

            for i in T.Parallel(block_N):
                if i < valid:
                    O_UB[i] = X[IDX_UB[i]]

            T.copy(O_UB[0:valid], OUT_GM[start:start + valid])

    return main


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N, block_N", INDIRECT_LOAD_1D_CASES)
def test_indirect_load_1d_ub_dev(dtype, N, block_N):
    kernel = indirect_load_1d_ub(N, block_N, dtype=dtype)

    x = gen_tensor((N * 2,), dtype, kind="randn")
    idx = torch.randint(0, N * 2, (N,), device="npu", dtype=torch.int32)
    out = gen_tensor((N,), dtype, kind="zeros")

    kernel(x, idx, out)

    assert_close(out.cpu(), x[idx.long()].cpu(), dtype=dtype, rtol=1e-3, atol=1e-3)
