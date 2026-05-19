# Copyright (c) Huawei Technologies Co., Ltd. 2025.
"""LogSumExp operator test.

Covers:
  - Expert mode (alloc_ub)
  - Developer mode (alloc_shared)

Algorithm (numerically stable):
  1. m = max(x, dim=1)
  2. e = exp(x - m)
  3. s = sum(e, dim=1)
  4. y = m + ln(s)

Output shape: (M, 1) when reducing dim=1 with keepdim=True.
"""

import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor

pytestmark = [
    pytest.mark.op("logsumexp"),
]

DTYPES = ["float16", "float32"]


def logsumexp_kernel_exp(M, N, dtype, accum_dtype):
    BLOCK_SIZE = 1

    @T.prim_func
    def logsumexp_exp(
        X: T.Tensor((M, N), dtype),
        Y: T.Tensor((M, 1), accum_dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            x_ub = T.alloc_ub((M, N), dtype)
            max_ub = T.alloc_ub((M, 1), accum_dtype)
            exp_ub = T.alloc_ub((M, N), dtype)
            sum_ub = T.alloc_ub((M, 1), accum_dtype)
            log_ub = T.alloc_ub((M, 1), accum_dtype)

            T.copy(X, x_ub)
            # m = max(x, dim=1)
            T.reduce_max(x_ub, max_ub, dim=1)
            # e = exp(x - m)
            T.vsub(x_ub, max_ub, exp_ub)
            T.vexp(exp_ub, exp_ub)
            # s = sum(e, dim=1)
            T.reduce(exp_ub, sum_ub, dims=[1], reduce_mode="sum")
            # y = m + ln(s)
            T.vln(sum_ub, log_ub)
            T.vadd(max_ub, log_ub, sum_ub)
            T.copy(sum_ub, Y)

    return logsumexp_exp


def logsumexp_kernel_dev(M, N, dtype, accum_dtype):
    BLOCK_SIZE = 1

    @T.prim_func
    def logsumexp_dev(
        X: T.Tensor((M, N), dtype),
        Y: T.Tensor((M, 1), accum_dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            x_sh = T.alloc_shared((M, N), dtype)
            max_sh = T.alloc_shared((M, 1), accum_dtype)
            exp_sh = T.alloc_shared((M, N), dtype)
            sum_sh = T.alloc_shared((M, 1), accum_dtype)
            log_sh = T.alloc_shared((M, 1), accum_dtype)

            T.copy(X, x_sh)
            T.reduce_max(x_sh, max_sh, dim=1)
            T.vsub(x_sh, max_sh, exp_sh)
            T.vexp(exp_sh, exp_sh)
            T.reduce(exp_sh, sum_sh, dims=[1], reduce_mode="sum")
            T.vln(sum_sh, log_sh)
            T.vadd(max_sh, log_sh, sum_sh)
            T.copy(sum_sh, Y)

    return logsumexp_dev


def _ref_logsumexp(x: torch.Tensor, dim: int = 1, keepdim: bool = True) -> torch.Tensor:
    return torch.logsumexp(x.float(), dim=dim, keepdim=keepdim).to(x.dtype)


@pytest.mark.mode("Expert")
@pytest.mark.parametrize("dtype", DTYPES)
def test_logsumexp_exp(dtype):
    M, N = 16, 16
    accum_dtype = dtype
    X = gen_tensor((M, N), dtype, kind="randn")
    Y = gen_tensor((M, 1), accum_dtype, kind="zeros")
    ref = _ref_logsumexp(X.cpu())

    func = logsumexp_kernel_exp(M=M, N=N, dtype=dtype, accum_dtype=accum_dtype)
    compiled = tilelang.compile(func, target="npuir")
    compiled(X, Y)

    assert_close(Y.cpu(), ref, dtype=accum_dtype, rtol=1e-2, atol=1e-2)


@pytest.mark.mode("Developer")
@pytest.mark.parametrize("dtype", DTYPES)
def test_logsumexp_dev(dtype):
    M, N = 16, 16
    accum_dtype = dtype
    X = gen_tensor((M, N), dtype, kind="randn")
    Y = gen_tensor((M, 1), accum_dtype, kind="zeros")
    ref = _ref_logsumexp(X.cpu())

    func = logsumexp_kernel_dev(M=M, N=N, dtype=dtype, accum_dtype=accum_dtype)
    compiled = tilelang.compile(func, target="npuir")
    compiled(X, Y)

    assert_close(Y.cpu(), ref, dtype=accum_dtype, rtol=1e-2, atol=1e-2)
