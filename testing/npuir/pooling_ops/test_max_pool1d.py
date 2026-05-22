# Copyright (c) Huawei Technologies Co., Ltd. 2025.
"""MaxPool1d operator test(Developer mode)."""

import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor

pytestmark = [
    pytest.mark.op("max_pool1d"),
    pytest.mark.mode("Developer"),
]

DTYPES = ["float16", "float32"]


def max_pool1d_kernel(L, kernel_size, dtype="float16"):
    """Max-pool1d kernel (stride=1).

    Reads each shifted window directly from global memory (In), then uses
    T.vmax for element-wise maximum.  No shared-to-shared T.copy is needed.
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
            out_sh = T.alloc_shared((L_out,), dtype)
            tmp_sh = T.alloc_shared((L_out,), dtype)

            # Shift 0: read first window directly from global.
            T.copy(In[0:L_out], out_sh)

            # Shifts 1 .. kernel_size-1: read from global, element-wise max.
            for i in T.serial(kernel_size - 1):
                shift = i + 1
                T.copy(In[shift : shift + L_out], tmp_sh)
                T.vmax(out_sh, tmp_sh, out_sh)

            T.copy(out_sh, Out)

    return kernel_fn


def _ref_max_pool1d(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Reference via ``torch.nn.functional.max_pool1d``.

    F.max_pool1d expects (N, C, L) input; lift the 1-d input accordingly
    and squeeze the result back.
    """
    x_3d = x.unsqueeze(0).unsqueeze(0)  # (1, 1, L)
    ref_3d = F.max_pool1d(x_3d.float(), kernel_size, stride=1)
    return ref_3d.squeeze().to(x.dtype)


def _compile_and_run(L, kernel_size, dtype, rtol=1e-2, atol=1e-2):
    dtype_t = getattr(torch, dtype)
    x = gen_tensor((L,), dtype, kind="randn")
    out = torch.zeros((L - kernel_size + 1,), dtype=dtype_t, device="npu")
    ref = _ref_max_pool1d(x.cpu(), kernel_size)

    func = max_pool1d_kernel(L=L, kernel_size=kernel_size, dtype=dtype)
    compiled = tilelang.compile(func, target="npuir")
    compiled(x, out)

    assert_close(out.cpu(), ref, dtype=dtype, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_max_pool1d_small(dtype):
    """Kernel size 3 on a short input."""
    _compile_and_run(L=16, kernel_size=3, dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_max_pool1d_medium(dtype):
    """Kernel size 5 on a larger input."""
    _compile_and_run(L=64, kernel_size=5, dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_max_pool1d_k2(dtype):
    """Kernel size 2 - minimal window."""
    _compile_and_run(L=32, kernel_size=2, dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_max_pool1d_k4(dtype):
    """Kernel size 4."""
    _compile_and_run(L=48, kernel_size=4, dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_max_pool1d_k8(dtype):
    """Kernel size 8 - wider window."""
    _compile_and_run(L=128, kernel_size=8, dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_max_pool1d_large(dtype):
    """Larger input with kernel size 7."""
    _compile_and_run(L=512, kernel_size=7, dtype=dtype)
