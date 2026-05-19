# Copyright (c) Huawei Technologies Co., Ltd. 2025.
"""Softmax operator test.

Covers:
  - Expert mode (alloc_ub)
  - Developer mode (alloc_shared)

Algorithm (numerically stable):
  1. m = max(x, dim=1)
  2. e = exp(x - m)
  3. s = sum(e, dim=1)
  4. y = e / s
"""

import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor

pytestmark = [
    pytest.mark.op("softmax"),
]

DTYPES = ["float16", "float32"]


def softmax_kernel_exp(M, N, dtype, accum_dtype):
    BLOCK_SIZE = 1

    @T.prim_func
    def softmax_exp(
        X: T.Tensor((M, N), dtype),
        Y: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            x_ub = T.alloc_ub((M, N), dtype)
            max_ub = T.alloc_ub((M, 1), accum_dtype)
            exp_ub = T.alloc_ub((M, N), dtype)
            sum_ub = T.alloc_ub((M, 1), accum_dtype)

            T.copy(X, x_ub)
            T.reduce_max(x_ub, max_ub, dim=1)
            T.vsub(x_ub, max_ub, exp_ub)
            T.vexp(exp_ub, exp_ub)
            T.reduce(exp_ub, sum_ub, dims=[1], reduce_mode="sum")
            T.vdiv(exp_ub, sum_ub, x_ub)
            T.copy(x_ub, Y)

    return softmax_exp


def softmax_kernel_dev(M, N, dtype, accum_dtype):
    BLOCK_SIZE = 1

    @T.prim_func
    def softmax_dev(
        X: T.Tensor((M, N), dtype),
        Y: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            x_sh = T.alloc_shared((M, N), dtype)
            max_sh = T.alloc_shared((M, 1), accum_dtype)
            exp_sh = T.alloc_shared((M, N), dtype)
            sum_sh = T.alloc_shared((M, 1), accum_dtype)

            T.copy(X, x_sh)
            T.reduce_max(x_sh, max_sh, dim=1)
            T.vsub(x_sh, max_sh, exp_sh)
            T.vexp(exp_sh, exp_sh)
            T.reduce(exp_sh, sum_sh, dims=[1], reduce_mode="sum")
            T.vdiv(exp_sh, sum_sh, x_sh)
            T.copy(x_sh, Y)

    return softmax_dev


def _ref_softmax(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    return torch.nn.functional.softmax(x.float(), dim=dim).to(x.dtype)


@pytest.mark.mode("Expert")
@pytest.mark.parametrize("dtype", DTYPES)
def test_softmax_exp(dtype):
    M, N = 16, 16
    accum_dtype = dtype
    X = gen_tensor((M, N), dtype, kind="randn")
    Y = gen_tensor((M, N), dtype, kind="zeros")
    ref = _ref_softmax(X.cpu())

    func = softmax_kernel_exp(M=M, N=N, dtype=dtype, accum_dtype=accum_dtype)
    compiled = tilelang.compile(func, target="npuir")
    compiled(X, Y)

    assert_close(Y.cpu(), ref, dtype=dtype, rtol=1e-2, atol=1e-2)


@pytest.mark.mode("Developer")
@pytest.mark.parametrize("dtype", DTYPES)
def test_softmax_dev(dtype):
    M, N = 16, 16
    accum_dtype = dtype
    X = gen_tensor((M, N), dtype, kind="randn")
    Y = gen_tensor((M, N), dtype, kind="zeros")
    ref = _ref_softmax(X.cpu())

    func = softmax_kernel_dev(M=M, N=N, dtype=dtype, accum_dtype=accum_dtype)
    compiled = tilelang.compile(func, target="npuir")
    compiled(X, Y)

    assert_close(Y.cpu(), ref, dtype=dtype, rtol=1e-2, atol=1e-2)
