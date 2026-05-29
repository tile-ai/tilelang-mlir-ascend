# Copyright (c) Huawei Technologies Co., Ltd. 2025.
"""AvgPool2d operator test(Developer mode)."""

import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor

pytestmark = [
    pytest.mark.op("avg_pool2d"),
    pytest.mark.mode("Developer"),
]

DTYPES = ["float16", "float32"]


def avg_pool2d_kernel(H, W, kernel_h, kernel_w, dtype="float16", accum_dtype="float32"):
    """Avg-pool2d kernel (stride=1).  All cross-dtype conversion uses explicit
    T.cast; no shared-to-shared T.copy.  Loop nesting is partitioned so that
    no conditional (if) appears inside T.serial.
    """
    assert kernel_h <= H, (
        f"H ({H}) must be greater than or equal to kernel_h ({kernel_h})"
    )
    assert kernel_w <= W, (
        f"W ({W}) must be greater than or equal to kernel_w ({kernel_w})"
    )
    H_out = H - kernel_h + 1
    W_out = W - kernel_w + 1
    inv_area = 1.0 / float(kernel_h * kernel_w)

    @T.prim_func
    def kernel_fn(
        In: T.Tensor((H, W), dtype),
        Out: T.Tensor((H_out, W_out), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            # --- same-dtype buffers for I/O ---
            in_sh = T.alloc_shared((H, W), dtype)
            out_sh = T.alloc_shared((H_out, W_out), dtype)

            # --- accum_dtype buffers for compute ---
            in_f32 = T.alloc_shared((H, W), accum_dtype)
            acc_f32 = T.alloc_shared((H_out, W_out), accum_dtype)

            # 1. Load
            T.copy(In, in_sh)

            # 2. Cast input → f32
            for i, j in T.Parallel(H, W):
                in_f32[i, j] = T.cast(in_sh[i, j], accum_dtype)

            # 3. Sliding-window accumulation (no if inside serial)
            for oi, oj in T.Parallel(H_out, W_out):
                # ---- window (0, 0) ----
                acc_f32[oi, oj] = in_f32[oi, oj]

                # ---- kh = 0, kw = 1 .. kernel_w-1 ----
                for kj in T.serial(kernel_w - 1):
                    acc_f32[oi, oj] = acc_f32[oi, oj] + in_f32[oi, oj + kj + 1]

                # ---- kh = 1 .. kernel_h-1, kw = 0 .. kernel_w-1 ----
                for ki in T.serial(kernel_h - 1):
                    for kj in T.serial(kernel_w):
                        acc_f32[oi, oj] = acc_f32[oi, oj] + in_f32[oi + ki + 1, oj + kj]

            # 4. SIMD division
            T.vmul(acc_f32, inv_area, acc_f32)

            # 5. Cast back → dtype
            for i, j in T.Parallel(H_out, W_out):
                out_sh[i, j] = T.cast(acc_f32[i, j], dtype)

            # 6. Store
            T.copy(out_sh, Out)

    return kernel_fn


def _ref_avg_pool2d(x: torch.Tensor, kernel_h: int, kernel_w: int) -> torch.Tensor:
    """Reference via ``torch.nn.functional.avg_pool2d``."""
    x_4d = x.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    ref_4d = F.avg_pool2d(x_4d.float(), (kernel_h, kernel_w), stride=1)
    return ref_4d.squeeze().to(x.dtype)


def _compile_and_run(H, W, kernel_h, kernel_w, dtype, rtol=1e-2, atol=1e-2):
    dtype_t = getattr(torch, dtype)
    x = gen_tensor((H, W), dtype, kind="randn")
    out = torch.zeros((H - kernel_h + 1, W - kernel_w + 1), dtype=dtype_t, device="npu")
    ref = _ref_avg_pool2d(x.cpu(), kernel_h, kernel_w)

    func = avg_pool2d_kernel(
        H=H, W=W, kernel_h=kernel_h, kernel_w=kernel_w, dtype=dtype
    )
    compiled = tilelang.compile(func, target="npuir")
    compiled(x, out)

    assert_close(out.cpu(), ref, dtype=dtype, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_avg_pool2d_small(dtype):
    _compile_and_run(H=16, W=16, kernel_h=2, kernel_w=2, dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_avg_pool2d_square(dtype):
    _compile_and_run(H=32, W=32, kernel_h=3, kernel_w=3, dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_avg_pool2d_rect(dtype):
    _compile_and_run(H=32, W=64, kernel_h=3, kernel_w=3, dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_avg_pool2d_larger(dtype):
    _compile_and_run(H=64, W=64, kernel_h=5, kernel_w=5, dtype=dtype)
