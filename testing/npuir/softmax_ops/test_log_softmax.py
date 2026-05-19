# Copyright (c) Huawei Technologies Co., Ltd. 2025.
"""Log-softmax operator test.

Covers:
  - Expert mode (alloc_ub)
  - Developer mode (alloc_shared)

Algorithm (numerically stable):
  1. m = max(x, dim=1)
  2. s = x - m
  3. e = exp(s)
  4. sum_e = sum(e, dim=1)
  5. log_sum = ln(sum_e)
  6. y = s - log_sum
"""

import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor

pytestmark = [
    pytest.mark.op("log_softmax"),
]

DTYPES = ["float16", "float32"]


def log_softmax_kernel_exp(M, N, dtype, accum_dtype):
    BLOCK_SIZE = 1

    @T.prim_func
    def log_softmax_exp(
        X: T.Tensor((M, N), dtype),
        Y: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            x_ub = T.alloc_ub((M, N), dtype)
            max_ub = T.alloc_ub((M, 1), accum_dtype)
            sub_ub = T.alloc_ub((M, N), dtype)
            exp_ub = T.alloc_ub((M, N), dtype)
            sum_ub = T.alloc_ub((M, 1), accum_dtype)
            log_ub = T.alloc_ub((M, 1), accum_dtype)

            T.copy(X, x_ub)
            # m = max(x, dim=1)
            T.reduce_max(x_ub, max_ub, dim=1)
            # s = x - m
            T.vsub(x_ub, max_ub, sub_ub)
            # e = exp(s)
            T.vexp(sub_ub, exp_ub)
            # sum_e = sum(e, dim=1)
            T.reduce(exp_ub, sum_ub, dims=[1], reduce_mode="sum")
            # log_sum = ln(sum_e)
            T.vln(sum_ub, log_ub)
            # y = s - log_sum
            T.vsub(sub_ub, log_ub, x_ub)
            T.copy(x_ub, Y)

    return log_softmax_exp


def log_softmax_kernel_dev(M, N, dtype, accum_dtype):
    BLOCK_SIZE = 1

    @T.prim_func
    def log_softmax_dev(
        X: T.Tensor((M, N), dtype),
        Y: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            x_sh = T.alloc_shared((M, N), dtype)
            max_sh = T.alloc_shared((M, 1), accum_dtype)
            sub_sh = T.alloc_shared((M, N), dtype)
            exp_sh = T.alloc_shared((M, N), dtype)
            sum_sh = T.alloc_shared((M, 1), accum_dtype)
            log_sh = T.alloc_shared((M, 1), accum_dtype)

            T.copy(X, x_sh)
            T.reduce_max(x_sh, max_sh, dim=1)
            T.vsub(x_sh, max_sh, sub_sh)
            T.vexp(sub_sh, exp_sh)
            T.reduce(exp_sh, sum_sh, dims=[1], reduce_mode="sum")
            T.vln(sum_sh, log_sh)
            T.vsub(sub_sh, log_sh, x_sh)
            T.copy(x_sh, Y)

    return log_softmax_dev


def _ref_log_softmax(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    return torch.nn.functional.log_softmax(x.float(), dim=dim).to(x.dtype)


@pytest.mark.mode("Expert")
@pytest.mark.parametrize("dtype", DTYPES)
def test_log_softmax_exp(dtype):
    M, N = 16, 16
    accum_dtype = dtype
    X = gen_tensor((M, N), dtype, kind="randn")
    Y = gen_tensor((M, N), dtype, kind="zeros")
    ref = _ref_log_softmax(X.cpu())

    func = log_softmax_kernel_exp(M=M, N=N, dtype=dtype, accum_dtype=accum_dtype)
    compiled = tilelang.compile(func, target="npuir")
    compiled(X, Y)

    assert_close(Y.cpu(), ref, dtype=dtype, rtol=1e-2, atol=1e-2)


@pytest.mark.mode("Developer")
@pytest.mark.parametrize("dtype", DTYPES)
def test_log_softmax_dev(dtype):
    M, N = 16, 16
    accum_dtype = dtype
    X = gen_tensor((M, N), dtype, kind="randn")
    Y = gen_tensor((M, N), dtype, kind="zeros")
    ref = _ref_log_softmax(X.cpu())

    func = log_softmax_kernel_dev(M=M, N=N, dtype=dtype, accum_dtype=accum_dtype)
    compiled = tilelang.compile(func, target="npuir")
    compiled(X, Y)

    assert_close(Y.cpu(), ref, dtype=dtype, rtol=1e-2, atol=1e-2)
