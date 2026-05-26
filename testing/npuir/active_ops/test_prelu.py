# Copyright (c) Huawei Technologies Co., Ltd. 2025.
"""PReLU activation test(Developer mode).
y = max(x, 0) + weight * min(x, 0)  (weight shape (N,) broadcast per row)
Reference: ``torch.nn.functional.prelu``.
"""
import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
import tilelang
import tilelang.language as T
from testcommon import gen_tensor

pytestmark = [pytest.mark.op("prelu"), pytest.mark.mode("Developer")]
DTYPES = ["float16", "float32"]
_DTYPE_MAP = {"float16": torch.float16, "float32": torch.float32}


def prelu_kernel_dev(M, N, block_m, block_n, dtype):
    @T.prim_func
    def prelu_dev(X: T.Tensor((M, N), dtype), Weight: T.Tensor((N,), dtype),
                   Y: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, block_m) * T.ceildiv(N, block_n), is_npu=True) as (cid, _):
            bx = (cid // T.ceildiv(N, block_n)) * block_m
            by = (cid % T.ceildiv(N, block_n)) * block_n
            x_sh = T.alloc_shared((block_m, block_n), dtype)
            w_1d = T.alloc_shared((block_n,), dtype)
            w_sh = T.alloc_shared((1, block_n), dtype)
            pos = T.alloc_shared((block_m, block_n), dtype)
            neg = T.alloc_shared((block_m, block_n), dtype)
            y_sh = T.alloc_shared((block_m, block_n), dtype)
            T.copy(X[bx:bx+block_m, by:by+block_n], x_sh)
            T.copy(Weight[by:by+block_n], w_1d)
            for j in T.Parallel(block_n):
                w_sh[0, j] = w_1d[j]
            T.vmax(x_sh, 0.0, pos)
            T.vmin(x_sh, 0.0, neg)
            T.vmul(neg, w_sh, neg)
            T.vadd(pos, neg, y_sh)
            T.copy(y_sh, Y[bx:bx+block_m, by:by+block_n])
    return prelu_dev


def _run_test(*, M, N, block_m, block_n, dtype, device="npu", rtol=1e-2, atol=1e-2):
    td = _DTYPE_MAP[dtype]
    X = gen_tensor((M, N), dtype, kind="randn", device=device)
    Weight = gen_tensor((N,), dtype, kind="randn", device=device)
    Y = torch.zeros((M, N), dtype=td, device=device)
    ref = F.prelu(X.cpu().float(), Weight.cpu().float()).to(td)
    compiled = tilelang.compile(prelu_kernel_dev(M, N, block_m, block_n, dtype), target="npuir")
    compiled(X, Weight, Y)
    torch.testing.assert_close(Y.cpu().float(), ref.float(), rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", DTYPES)
def test_prelu_dev_basic(dtype):
    _run_test(M=16, N=16, block_m=16, block_n=16, dtype=dtype)

@pytest.mark.parametrize("dtype", DTYPES)
def test_prelu_dev_larger(dtype):
    _run_test(M=64, N=128, block_m=32, block_n=32, dtype=dtype)
