# Copyright (c) Huawei Technologies Co., Ltd. 2025.
"""Leaky ReLU activation test(Developer mode).
y = max(x, 0) + negative_slope * min(x, 0)
Reference: ``torch.nn.functional.leaky_relu``.
"""
import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
import tilelang
import tilelang.language as T
from testcommon import gen_tensor

pytestmark = [pytest.mark.op("leaky_relu"), pytest.mark.mode("Developer")]
DTYPES = ["float16", "float32"]
_DTYPE_MAP = {"float16": torch.float16, "float32": torch.float32}
NEGATIVE_SLOPE = 0.01


def leaky_relu_kernel_dev(M, N, block_m, block_n, dtype, negative_slope):
    @T.prim_func
    def leaky_relu_dev(X: T.Tensor((M, N), dtype), Y: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, block_m) * T.ceildiv(N, block_n), is_npu=True) as (cid, _):
            bx = (cid // T.ceildiv(N, block_n)) * block_m
            by = (cid % T.ceildiv(N, block_n)) * block_n
            x_sh = T.alloc_shared((block_m, block_n), dtype)
            y_sh = T.alloc_shared((block_m, block_n), dtype)
            T.copy(X[bx:bx+block_m, by:by+block_n], x_sh)
            T.vmax(x_sh, 0.0, y_sh)
            T.vmin(x_sh, 0.0, x_sh)
            T.vmul(x_sh, negative_slope, x_sh)
            T.vadd(y_sh, x_sh, y_sh)
            T.copy(y_sh, Y[bx:bx+block_m, by:by+block_n])
    return leaky_relu_dev


def _run_test(*, M, N, block_m, block_n, dtype, negative_slope=NEGATIVE_SLOPE,
              device="npu", rtol=1e-2, atol=1e-2):
    td = _DTYPE_MAP[dtype]
    X = gen_tensor((M, N), dtype, kind="randn", device=device)
    Y = torch.zeros((M, N), dtype=td, device=device)
    ref = F.leaky_relu(X.cpu().float(), negative_slope).to(td)
    compiled = tilelang.compile(
        leaky_relu_kernel_dev(M, N, block_m, block_n, dtype, negative_slope), target="npuir")
    compiled(X, Y)
    torch.testing.assert_close(Y.cpu().float(), ref.float(), rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", DTYPES)
def test_leaky_relu_dev_basic(dtype):
    _run_test(M=16, N=16, block_m=16, block_n=16, dtype=dtype)

@pytest.mark.parametrize("dtype", DTYPES)
def test_leaky_relu_dev_larger(dtype):
    _run_test(M=64, N=128, block_m=32, block_n=32, dtype=dtype)
