# Copyright (c) Huawei Technologies Co., Ltd. 2025.
"""AvgPool3d operator test(Developer mode)."""

import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor

pytestmark = [
    pytest.mark.op("avg_pool3d"),
    pytest.mark.mode("Developer"),
]

DTYPES = ["float16", "float32"]


def avg_pool3d_kernel(
    D, H, W, kernel_d, kernel_h, kernel_w, dtype="float16", accum_dtype="float32"
):
    """Avg-pool3d kernel (stride=1).  All cross-dtype conversion uses explicit
    T.cast; no shared-to-shared T.copy.  Loop nesting is partitioned so that
    no conditional (if) appears inside T.serial.
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
    inv_area = 1.0 / float(kernel_d * kernel_h * kernel_w)

    @T.prim_func
    def kernel_fn(
        In: T.Tensor((D, H, W), dtype),
        Out: T.Tensor((D_out, H_out, W_out), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            in_sh = T.alloc_shared((D, H, W), dtype)
            out_sh = T.alloc_shared((D_out, H_out, W_out), dtype)
            in_f32 = T.alloc_shared((D, H, W), accum_dtype)
            acc_f32 = T.alloc_shared((D_out, H_out, W_out), accum_dtype)

            # 1. Load
            T.copy(In, in_sh)

            # 2. Cast → f32
            for di, hi, wi in T.Parallel(D, H, W):
                in_f32[di, hi, wi] = T.cast(in_sh[di, hi, wi], accum_dtype)

            # 3. Sliding-window accumulation (no if inside serial)
            for doi, hoi, woi in T.Parallel(D_out, H_out, W_out):
                # ---- window (0, 0, 0) ----
                acc_f32[doi, hoi, woi] = in_f32[doi, hoi, woi]

                # ---- kd=0, kh=0, kw=1..kernel_w-1 ----
                for kwi in T.serial(kernel_w - 1):
                    acc_f32[doi, hoi, woi] = (
                        acc_f32[doi, hoi, woi] + in_f32[doi, hoi, woi + kwi + 1]
                    )

                # ---- kd=0, kh=1..kernel_h-1, kw=0..kernel_w-1 ----
                for khi in T.serial(kernel_h - 1):
                    for kwi in T.serial(kernel_w):
                        acc_f32[doi, hoi, woi] = (
                            acc_f32[doi, hoi, woi]
                            + in_f32[doi, hoi + khi + 1, woi + kwi]
                        )

                # ---- kd=1..kernel_d-1, kh=0..kernel_h-1, kw=0..kernel_w-1 ----
                for kdi in T.serial(kernel_d - 1):
                    for khi in T.serial(kernel_h):
                        for kwi in T.serial(kernel_w):
                            acc_f32[doi, hoi, woi] = (
                                acc_f32[doi, hoi, woi]
                                + in_f32[doi + kdi + 1, hoi + khi, woi + kwi]
                            )

            # 4. SIMD division
            T.vmul(acc_f32, inv_area, acc_f32)

            # 5. Cast back → dtype
            for i, j, k in T.Parallel(D_out, H_out, W_out):
                out_sh[i, j, k] = T.cast(acc_f32[i, j, k], dtype)

            # 6. Store
            T.copy(out_sh, Out)

    return kernel_fn


def _ref_avg_pool3d(
    x: torch.Tensor, kernel_d: int, kernel_h: int, kernel_w: int
) -> torch.Tensor:
    """Reference via ``torch.nn.functional.avg_pool3d``."""
    x_5d = x.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    ref_5d = F.avg_pool3d(x_5d.float(), (kernel_d, kernel_h, kernel_w), stride=1)
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
    ref = _ref_avg_pool3d(x.cpu(), kernel_d, kernel_h, kernel_w)

    func = avg_pool3d_kernel(
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
def test_avg_pool3d_small(dtype):
    _compile_and_run(D=8, H=16, W=16, kernel_d=2, kernel_h=2, kernel_w=2, dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_avg_pool3d_square(dtype):
    _compile_and_run(D=8, H=16, W=16, kernel_d=3, kernel_h=3, kernel_w=3, dtype=dtype)
