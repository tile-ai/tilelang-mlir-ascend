# Copyright (c) Huawei Technologies Co., Ltd. 2025.
"""MaxPool3d operator test(Developer mode)."""

import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor

pytestmark = [
    pytest.mark.op("max_pool3d"),
    pytest.mark.mode("Developer"),
]

DTYPES = ["float16", "float32"]


def max_pool3d_kernel(D, H, W, kernel_d, kernel_h, kernel_w, dtype="float16"):
    """Max-pool3d kernel (stride=1).

    Vectorized implementation using slicing and T.vmax to avoid scalar loops,
    eliminate expensive division/modulo operations, and reduce shared memory.
    """
    assert kernel_d <= D, (
        f"D ({D}) must be greater than or equal to kernel_d ({kernel_d})"
    )
    assert kernel_h <= H, (
        f"H ({H}) must be greater than or equal to kernel_h ({kernel_h})"
    )
    assert kernel_w <= W, (
        f"W ({W}) must be greater than or equal to kernel_w ({kernel_w})"
    )
    D_out = D - kernel_d + 1
    H_out = H - kernel_h + 1
    W_out = W - kernel_w + 1

    @T.prim_func
    def kernel_fn(
        In: T.Tensor((D, H, W), dtype),
        Out: T.Tensor((D_out, H_out, W_out), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            out_sh = T.alloc_shared((D_out, H_out, W_out), dtype)
            tmp_sh = T.alloc_shared((D_out, H_out, W_out), dtype)

            # Shift (0, 0, 0): read first window directly from global.
            T.copy(In[0:D_out, 0:H_out, 0:W_out], out_sh)

            # Remaining shifts: read from global, element-wise max.
            for kd in T.serial(kernel_d):
                for kh in T.serial(kernel_h):
                    for kw in T.serial(kernel_w):
                        if kd != 0 or kh != 0 or kw != 0:
                            T.copy(
                                In[kd : kd + D_out, kh : kh + H_out, kw : kw + W_out],
                                tmp_sh,
                            )
                            T.vmax(out_sh, tmp_sh, out_sh)

            T.copy(out_sh, Out)

    return kernel_fn


def _ref_max_pool3d(
    x: torch.Tensor, kernel_d: int, kernel_h: int, kernel_w: int
) -> torch.Tensor:
    """Reference via ``torch.nn.functional.max_pool3d``."""
    x_5d = x.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    ref_5d = F.max_pool3d(x_5d.float(), (kernel_d, kernel_h, kernel_w), stride=1)
    return ref_5d.squeeze().to(x.dtype)


def _compile_and_run(
    D, H, W, kernel_d, kernel_h, kernel_w, dtype, rtol=1e-2, atol=1e-2
):
    dtype_t = getattr(torch, dtype)
    x = gen_tensor((D, H, W), dtype, kind="randn")
    out = torch.zeros(
        (D - kernel_d + 1, H - kernel_h + 1, W - kernel_w + 1),
        dtype=dtype_t,
        device="npu",
    )
    ref = _ref_max_pool3d(x.cpu(), kernel_d, kernel_h, kernel_w)

    func = max_pool3d_kernel(
        D=D,
        H=H,
        W=W,
        kernel_d=kernel_d,
        kernel_h=kernel_h,
        kernel_w=kernel_w,
        dtype=dtype,
    )
    compiled = tilelang.compile(func, target="npuir")
    compiled(x, out)

    assert_close(out.cpu(), ref, dtype=dtype, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_max_pool3d_small(dtype):
    """Small cubic input, kernel 2x2x2."""
    _compile_and_run(D=8, H=16, W=16, kernel_d=2, kernel_h=2, kernel_w=2, dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_max_pool3d_square(dtype):
    """Cubic input, kernel 3x3x3."""
    _compile_and_run(D=8, H=16, W=16, kernel_d=3, kernel_h=3, kernel_w=3, dtype=dtype)
