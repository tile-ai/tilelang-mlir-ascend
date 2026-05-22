# Copyright (c) Huawei Technologies Co., Ltd. 2025.
"""AvgPool1d operator test(Developer mode)."""

import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor

pytestmark = [
    pytest.mark.op("avg_pool1d"),
    pytest.mark.mode("Developer"),
]

DTYPES = ["float16", "float32"]


def avg_pool1d_kernel(L, kernel_size, dtype="float16", accum_dtype="float32"):
    """Avg-pool kernel that avoids cross-dtype / shared-to-shared T.copy.

    Algorithm:
        L_out = L - kernel_size + 1
        for o in [0 .. L_out):
            acc = cast(in[o], f32)
            for k in [1 .. kernel_size):
                acc += cast(in[o + k], f32)
            out[o] = cast(acc / kernel_size, dtype)
    """
    assert kernel_size <= L, (
        f"L ({L}) must be greater than or equal to kernel_size ({kernel_size})"
    )
    L_out = L - kernel_size + 1

    @T.prim_func
    def kernel_fn(
        In: T.Tensor((L,), dtype),
        Out: T.Tensor((L_out,), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            # --- same-dtype buffers for global <-> shared copies ---
            in_sh = T.alloc_shared((L,), dtype)
            out_sh = T.alloc_shared((L_out,), dtype)

            # --- accum_dtype buffers for computation ---
            in_f32 = T.alloc_shared((L,), accum_dtype)
            acc_f32 = T.alloc_shared((L_out,), accum_dtype)

            # 1. Load input (same dtype)
            T.copy(In, in_sh)

            # 2. Cast input to accum_dtype element by element
            for i in T.Parallel(L):
                in_f32[i] = T.cast(in_sh[i], accum_dtype)

            # 3. Sliding-window accumulation
            for o in T.Parallel(L_out):
                acc_f32[o] = in_f32[o]
                for k in T.serial(kernel_size - 1):
                    acc_f32[o] = acc_f32[o] + in_f32[o + k + 1]

            # 4. SIMD division to obtain the mean
            T.vmul(acc_f32, 1.0 / float(kernel_size), acc_f32)

            # 5. Cast back to output dtype element by element
            for i in T.Parallel(L_out):
                out_sh[i] = T.cast(acc_f32[i], dtype)

            # 6. Copy to global (same dtype)
            T.copy(out_sh, Out)

    return kernel_fn


def _ref_avg_pool1d(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Reference via ``torch.nn.functional.avg_pool1d``.

    F.avg_pool1d expects (N, C, L) input; we lift the 1-d input accordingly
    and squeeze the result back.  The reference is computed in float32 and
    cast back to avoid internal fp16 rounding differences in torch.
    """
    x_3d = x.unsqueeze(0).unsqueeze(0)  # (1, 1, L)
    ref_3d = F.avg_pool1d(x_3d.float(), kernel_size, stride=1)
    return ref_3d.squeeze().to(x.dtype)


def _compile_and_run(L, kernel_size, dtype, rtol=1e-2, atol=1e-2):
    dtype_t = getattr(torch, dtype)
    x = gen_tensor((L,), dtype, kind="randn")
    out = torch.zeros((L - kernel_size + 1,), dtype=dtype_t, device="npu")
    ref = _ref_avg_pool1d(x.cpu(), kernel_size)

    func = avg_pool1d_kernel(L=L, kernel_size=kernel_size, dtype=dtype)
    compiled = tilelang.compile(func, target="npuir")
    compiled(x, out)

    assert_close(out.cpu(), ref, dtype=dtype, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_avg_pool1d_small(dtype):
    """Kernel size 3 on a short input."""
    _compile_and_run(L=16, kernel_size=3, dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_avg_pool1d_medium(dtype):
    """Kernel size 5 on a larger input."""
    _compile_and_run(L=64, kernel_size=5, dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_avg_pool1d_k2(dtype):
    """Kernel size 2 - minimal window."""
    _compile_and_run(L=32, kernel_size=2, dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_avg_pool1d_k4(dtype):
    """Kernel size 4."""
    _compile_and_run(L=48, kernel_size=4, dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_avg_pool1d_k8(dtype):
    """Kernel size 8 - wider window."""
    _compile_and_run(L=128, kernel_size=8, dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_avg_pool1d_large(dtype):
    """Larger input with kernel size 7."""
    _compile_and_run(L=512, kernel_size=7, dtype=dtype)
