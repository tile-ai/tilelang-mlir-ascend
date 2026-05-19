# Copyright (c) Huawei Technologies Co., Ltd. 2025.
"""BatchNorm forward operator test for npuir target (Developer mode).

BatchNorm normalizes each channel independently over (N, *spatial) elements.

Simplified 2-D form (assumes spatial_size = 1, training-like statistics):
    Input:   X        (C, L)   where  C = channels,
                                       L = batch * spatial_size
    Weight:  w        (C, 1)    per-channel affine scale
    Bias:    b        (C, 1)    per-channel affine shift
    Output:  Y        (C, L)

The reference delegates to ``torch.nn.functional.batch_norm`` (training mode).
"""

import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import gen_tensor

pytestmark = [
    pytest.mark.op("batch_norm"),
    pytest.mark.mode("Developer"),
]

DTYPES = ["float16", "float32"]
EPS = 1e-5

_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
}


def batch_norm_kernel_dev(C, L, block_c, eps, dtype):
    @T.prim_func
    def batch_norm_dev(
        X: T.Tensor((C, L), dtype),
        Weight: T.Tensor((C, 1), dtype),
        Bias: T.Tensor((C, 1), dtype),
        Y: T.Tensor((C, L), dtype),
    ):
        with T.Kernel(T.ceildiv(C, block_c), is_npu=True) as (pid_c, _):
            x_sh = T.alloc_shared((block_c, L), dtype)
            w_sh = T.alloc_shared((block_c, 1), dtype)
            b_sh = T.alloc_shared((block_c, 1), dtype)
            y_sh = T.alloc_shared((block_c, L), dtype)
            x_f32 = T.alloc_shared((block_c, L), "float32")
            sq_f32 = T.alloc_shared((block_c, L), "float32")
            w_f32 = T.alloc_shared((block_c, 1), "float32")
            b_f32 = T.alloc_shared((block_c, 1), "float32")
            acc = T.alloc_shared((block_c, 1), "float32")
            mean = T.alloc_shared((block_c, 1), "float32")
            rstd = T.alloc_shared((block_c, 1), "float32")

            T.copy(X[pid_c * block_c, 0], x_sh)
            T.copy(Weight[pid_c * block_c, 0], w_sh)
            T.copy(Bias[pid_c * block_c, 0], b_sh)

            # Pre-cast all inputs to float32 to avoid mixed-type vector ops
            for i, j in T.Parallel(block_c, L):
                x_f32[i, j] = T.cast(x_sh[i, j], "float32")
            for i in T.Parallel(block_c):
                w_f32[i, 0] = T.cast(w_sh[i, 0], "float32")
                b_f32[i, 0] = T.cast(b_sh[i, 0], "float32")

            # Mean
            T.reduce_sum(x_f32, acc, dim=1)
            for i in T.Parallel(block_c):
                mean[i, 0] = acc[i, 0] / float(L)

            # Variance: use separate sq_f32 to keep x_f32 intact
            for i, j in T.Parallel(block_c, L):
                sq_f32[i, j] = (x_f32[i, j] - mean[i, 0]) * (x_f32[i, j] - mean[i, 0])
            T.reduce_sum(sq_f32, acc, dim=1)
            for i in T.Parallel(block_c):
                rstd[i, 0] = acc[i, 0] / float(L) + eps
            T.vrsqrt(rstd, rstd)

            # Output: all operands are float32; only final cast touches dtype
            for i, j in T.Parallel(block_c, L):
                y_sh[i, j] = T.cast(
                    (x_f32[i, j] - mean[i, 0]) * rstd[i, 0] * w_f32[i, 0] + b_f32[i, 0],
                    dtype,
                )

            T.copy(y_sh, Y[pid_c * block_c, 0])

    return batch_norm_dev


def _ref_batch_norm(
    x_2d: torch.Tensor,
    weight_2d: torch.Tensor,
    bias_2d: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Reference via ``torch.nn.functional.batch_norm`` (training mode).

    The 2-D (C, L) layout is mapped to 4-D (L, C, 1, 1) for PyTorch.
    ``weight_2d`` (C, 1) and ``bias_2d`` (C, 1) are squeezed to (C,).
    """
    C, L = x_2d.shape
    x_4d = x_2d.float().T.reshape(L, C, 1, 1)
    w = weight_2d.float().squeeze()
    b = bias_2d.float().squeeze()
    rm = torch.zeros(C, dtype=torch.float32)
    rv = torch.ones(C, dtype=torch.float32)
    ref = F.batch_norm(x_4d, rm, rv, w, b, training=True, eps=eps)
    return ref.reshape(L, C).T.to(x_2d.dtype)


def _run_test(
    *,
    C: int,
    L: int,
    block_c: int,
    eps: float,
    dtype: str,
    device: str = "npu",
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> None:
    """Unified test runner (mirrors the style of examples/norm/layer_norm.py).

    Generates inputs, compiles and executes the kernel, then validates
    against the PyTorch reference.
    """
    torch_dtype = _DTYPE_MAP[dtype]

    X = gen_tensor((C, L), dtype, kind="randn", device=device)
    Weight = gen_tensor((C, 1), dtype, kind="randn", device=device)
    Bias = gen_tensor((C, 1), dtype, kind="randn", device=device)
    Y = torch.zeros((C, L), dtype=torch_dtype, device=device)

    ref = _ref_batch_norm(X.cpu(), Weight.cpu(), Bias.cpu(), eps)

    func = batch_norm_kernel_dev(C=C, L=L, block_c=block_c, eps=eps, dtype=dtype)
    compiled = tilelang.compile(func, target="npuir")
    compiled(X, Weight, Bias, Y)

    torch.testing.assert_close(
        Y.cpu().float(),
        ref.float(),
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_batch_norm_dev_basic(dtype):
    _run_test(
        C=8,  # channels
        L=32,  # batch * spatial_size
        block_c=4,
        eps=EPS,
        dtype=dtype,
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_batch_norm_dev_larger(dtype):
    _run_test(
        C=16,
        L=128,
        block_c=8,
        eps=EPS,
        dtype=dtype,
    )
