# Copyright (c) Huawei Technologies Co., Ltd. 2025.
"""HardTanh activation test(Developer mode).
y = clamp(x, min_val, max_val)
Reference: ``torch.nn.functional.hardtanh``.
"""
import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
import tilelang
import tilelang.language as T
from testcommon import gen_tensor

pytestmark = [pytest.mark.op("hardtanh"), pytest.mark.mode("Developer")]
DTYPES = ["float16", "float32"]
_DTYPE_MAP = {"float16": torch.float16, "float32": torch.float32}


def hardtanh_kernel_dev(M, N, block_m, block_n, dtype, min_val=-1.0, max_val=1.0):
    @T.prim_func
    def hardtanh_dev(X: T.Tensor((M, N), dtype), Y: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, block_m) * T.ceildiv(N, block_n), is_npu=True) as (cid, _):
            bx = (cid // T.ceildiv(N, block_n)) * block_m
            by = (cid % T.ceildiv(N, block_n)) * block_n
            x_sh = T.alloc_shared((block_m, block_n), dtype)
            y_sh = T.alloc_shared((block_m, block_n), dtype)
            T.copy(X[bx:bx+block_m, by:by+block_n], x_sh)
            T.vclamp(x_sh, y_sh, min_val, max_val)
            T.copy(y_sh, Y[bx:bx+block_m, by:by+block_n])
    return hardtanh_dev


def _run_test(*, M, N, block_m, block_n, dtype, min_val=-1.0, max_val=1.0,
              device="npu", rtol=1e-2, atol=1e-2):
    td = _DTYPE_MAP[dtype]
    X = gen_tensor((M, N), dtype, kind="randn", device=device)
    Y = torch.zeros((M, N), dtype=td, device=device)
    ref = F.hardtanh(X.cpu().float(), min_val, max_val).to(td)
    compiled = tilelang.compile(
        hardtanh_kernel_dev(M, N, block_m, block_n, dtype, min_val, max_val), target="npuir")
    compiled(X, Y)
    torch.testing.assert_close(Y.cpu().float(), ref.float(), rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", DTYPES)
def test_hardtanh_dev_basic(dtype):
    _run_test(M=16, N=16, block_m=16, block_n=16, dtype=dtype)

@pytest.mark.parametrize("dtype", DTYPES)
def test_hardtanh_dev_larger(dtype):
    _run_test(M=64, N=128, block_m=32, block_n=32, dtype=dtype)
