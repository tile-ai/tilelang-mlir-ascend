import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor

pytestmark = [
    pytest.mark.op("max_pool2d"),
    pytest.mark.mode("Developer"),
]

DTYPES = ["float16", "float32"]


def max_pool2d_kernel(H, W, kernel_h, kernel_w, dtype="float16"):
    """Max-pool2d kernel (stride=1).

    Reads each shifted 2D window directly from global memory, then uses
    T.vmax for element-wise maximum.  No shared-to-shared T.copy is needed.
    """
    assert kernel_h <= H, (
        f"H ({H}) must be greater than or equal to kernel_h ({kernel_h})"
    )
    assert kernel_w <= W, (
        f"W ({W}) must be greater than or equal to kernel_w ({kernel_w})"
    )
    H_out = H - kernel_h + 1
    W_out = W - kernel_w + 1

    @T.prim_func
    def kernel_fn(
        In: T.Tensor((H, W), dtype),
        Out: T.Tensor((H_out, W_out), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            out_sh = T.alloc_shared((H_out, W_out), dtype)
            tmp_sh = T.alloc_shared((H_out, W_out), dtype)

            # Shift (0, 0): read first window directly from global.
            T.copy(In[0:H_out, 0:W_out], out_sh)

            # Remaining shifts: read from global, element-wise max.
            for kh in T.serial(kernel_h):
                for kw in T.serial(kernel_w):
                    if kh != 0 or kw != 0:
                        T.copy(In[kh : kh + H_out, kw : kw + W_out], tmp_sh)
                        T.vmax(out_sh, tmp_sh, out_sh)

            T.copy(out_sh, Out)

    return kernel_fn


def _ref_max_pool2d(x: torch.Tensor, kernel_h: int, kernel_w: int) -> torch.Tensor:
    """Reference via ``torch.nn.functional.max_pool2d``."""
    x_4d = x.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    ref_4d = F.max_pool2d(x_4d.float(), (kernel_h, kernel_w), stride=1)
    return ref_4d.squeeze().to(x.dtype)


def _compile_and_run(H, W, kernel_h, kernel_w, dtype, rtol=1e-2, atol=1e-2):
    dtype_t = getattr(torch, dtype)
    x = gen_tensor((H, W), dtype, kind="randn")
    out = torch.zeros((H - kernel_h + 1, W - kernel_w + 1), dtype=dtype_t, device="npu")
    ref = _ref_max_pool2d(x.cpu(), kernel_h, kernel_w)

    func = max_pool2d_kernel(
        H=H, W=W, kernel_h=kernel_h, kernel_w=kernel_w, dtype=dtype
    )
    compiled = tilelang.compile(func, target="npuir")
    compiled(x, out)

    assert_close(out.cpu(), ref, dtype=dtype, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_max_pool2d_small(dtype):
    """Small square input, kernel 2x2."""
    _compile_and_run(H=16, W=16, kernel_h=2, kernel_w=2, dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_max_pool2d_square(dtype):
    """Square input, kernel 3x3."""
    _compile_and_run(H=32, W=32, kernel_h=3, kernel_w=3, dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_max_pool2d_rect(dtype):
    """Rectangular input, kernel 3x3."""
    _compile_and_run(H=32, W=64, kernel_h=3, kernel_w=3, dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_max_pool2d_larger(dtype):
    """Larger input, kernel 5x5."""
    _compile_and_run(H=64, W=64, kernel_h=5, kernel_w=5, dtype=dtype)
