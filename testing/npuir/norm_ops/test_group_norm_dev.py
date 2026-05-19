# Copyright (c) Huawei Technologies Co., Ltd. 2025.
"""GroupNorm forward operator test for npuir target (Developer mode).

GroupNorm partitions channels into groups and normalizes each
(C/num_groups, *spatial) slice independently.

Simplified 2-D form (assumes spatial_size = 1 for clarity):
    Input:   X        (M, N)   where  M = batch * num_groups,
                                       N = C // num_groups
    Weight:  w        (N,)      per-element-in-group affine (tiled across groups)
    Bias:    b        (N,)
    Output:  Y        (M, N)

The reference delegates to ``torch.nn.functional.group_norm``.
"""

import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import gen_tensor

pytestmark = [
    pytest.mark.op("group_norm"),
    pytest.mark.mode("Developer"),
]

DTYPES = ["float16", "float32"]
EPS = 1e-5

_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
}


def group_norm_kernel_dev(M, N, block_m, eps, dtype):
    @T.prim_func
    def group_norm_dev(
        X: T.Tensor((M, N), dtype),
        Weight: T.Tensor((N,), dtype),
        Bias: T.Tensor((N,), dtype),
        Y: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_m), is_npu=True) as (pid_m, _):
            x_sh = T.alloc_shared((block_m, N), dtype)
            w_sh = T.alloc_shared((N,), dtype)
            b_sh = T.alloc_shared((N,), dtype)
            y_sh = T.alloc_shared((block_m, N), dtype)
            x_f32 = T.alloc_shared((block_m, N), "float32")
            sq_f32 = T.alloc_shared((block_m, N), "float32")
            w_f32 = T.alloc_shared((N,), "float32")
            b_f32 = T.alloc_shared((N,), "float32")
            acc = T.alloc_shared((block_m, 1), "float32")
            mean = T.alloc_shared((block_m, 1), "float32")
            rstd = T.alloc_shared((block_m, 1), "float32")

            T.copy(X[pid_m * block_m, 0], x_sh)
            T.copy(Weight, w_sh)
            T.copy(Bias, b_sh)

            # Pre-cast all inputs to float32 to avoid mixed-type vector ops
            for i, j in T.Parallel(block_m, N):
                x_f32[i, j] = T.cast(x_sh[i, j], "float32")
            for j in T.Parallel(N):
                w_f32[j] = T.cast(w_sh[j], "float32")
                b_f32[j] = T.cast(b_sh[j], "float32")

            # Mean
            T.reduce_sum(x_f32, acc, dim=1)
            for i in T.Parallel(block_m):
                mean[i, 0] = acc[i, 0] / float(N)

            # Variance: use separate sq_f32 to keep x_f32 intact
            for i, j in T.Parallel(block_m, N):
                sq_f32[i, j] = (x_f32[i, j] - mean[i, 0]) * (x_f32[i, j] - mean[i, 0])
            T.reduce_sum(sq_f32, acc, dim=1)
            for i in T.Parallel(block_m):
                rstd[i, 0] = acc[i, 0] / float(N) + eps
            T.vrsqrt(rstd, rstd)

            # Output: all operands are float32; only final cast touches dtype
            for i, j in T.Parallel(block_m, N):
                y_sh[i, j] = T.cast(
                    (x_f32[i, j] - mean[i, 0]) * rstd[i, 0] * w_f32[j] + b_f32[j],
                    dtype,
                )

            T.copy(y_sh, Y[pid_m * block_m, 0])

    return group_norm_dev


def _ref_group_norm(
    x_2d: torch.Tensor,
    weight_1d: torch.Tensor,
    bias_1d: torch.Tensor,
    num_groups: int,
    eps: float,
) -> torch.Tensor:
    """Reference via ``torch.nn.functional.group_norm``.

    The 2-D layout is lifted back to a 4-D (N, C, 1, 1) tensor.
    Since ``weight_1d`` is per-element-in-group and shared across groups,
    we tile it ``num_groups`` times to obtain the per-channel (C,) form.
    """
    M, N_dim = x_2d.shape
    batch = M // num_groups
    C = num_groups * N_dim
    x_4d = x_2d.float().reshape(batch, C, 1, 1)
    w = weight_1d.float().repeat(num_groups)
    b = bias_1d.float().repeat(num_groups)
    ref = F.group_norm(x_4d, num_groups, w, b, eps)
    return ref.reshape(M, N_dim).to(x_2d.dtype)


def _run_test(
    *,
    M: int,
    N: int,
    num_groups: int,
    block_m: int,
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

    X = gen_tensor((M, N), dtype, kind="randn", device=device)
    Weight = gen_tensor((N,), dtype, kind="randn", device=device)
    Bias = gen_tensor((N,), dtype, kind="randn", device=device)
    Y = torch.zeros((M, N), dtype=torch_dtype, device=device)

    ref = _ref_group_norm(X.cpu(), Weight.cpu(), Bias.cpu(), num_groups, eps)

    func = group_norm_kernel_dev(M=M, N=N, block_m=block_m, eps=eps, dtype=dtype)
    compiled = tilelang.compile(func, target="npuir")
    compiled(X, Weight, Bias, Y)

    torch.testing.assert_close(
        Y.cpu().float(),
        ref.float(),
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_group_norm_dev_basic(dtype):
    _run_test(
        M=8,  # batch * num_groups
        N=16,  # C // num_groups
        num_groups=4,
        block_m=4,
        eps=EPS,
        dtype=dtype,
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_group_norm_dev_larger(dtype):
    _run_test(
        M=16,
        N=64,
        num_groups=4,
        block_m=8,
        eps=EPS,
        dtype=dtype,
    )
