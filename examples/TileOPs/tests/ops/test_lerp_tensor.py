"""Correctness tests for LerpTensorFwdOp (NPU).

Adapted from TileOPs ``tests/ops/test_binary_arith.py`` — LerpTensor
subset only (the shared GPU file also covers add/sub/mul/div/lerp/...,
which are not migrated).  Covers same-shape, 3-way broadcast, fp8 dtype
rejection, and dtype-mismatch rejection at forward().

NPU adaptations (T1-T4):

- T1: manual tensor creation uses the device backend (``_device()``)
  instead of hard-coded ``device="cuda"``.
- T2: ``@pytest.mark.skipif(not torch.cuda.is_available(), ...)``
  guards removed (NPU availability is a runtime concern).
- T3/T4: no ``tune`` parametrization / autotune try-except in the GPU
  LerpTensor tests — nothing to remove.
"""

import pytest
import torch

from tileops.device import get_device_backend
from tileops.ops.elementwise.lerp_tensor import LerpTensorFwdOp

_LERP_TENSOR_DTYPES = [torch.float16, torch.bfloat16, torch.float32]


def _device() -> str:
    return get_device_backend().name


def _lerp_tol(dtype: torch.dtype) -> dict:
    if dtype == torch.float16:
        return {"atol": 1e-3, "rtol": 1e-3}
    if dtype == torch.bfloat16:
        return {"atol": 1e-2, "rtol": 1e-2}
    return {"atol": 1e-6, "rtol": 1e-6}


@pytest.mark.smoke
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float32, marks=[pytest.mark.smoke, pytest.mark.packaging]),
        pytest.param(torch.float16, marks=[pytest.mark.smoke]),
        pytest.param(torch.bfloat16, marks=[pytest.mark.smoke]),
    ],
)
def test_lerp_tensor_same_shape(dtype: torch.dtype) -> None:
    """LerpTensorFwdOp matches torch.lerp on same-shape inputs."""
    shape = (4, 8)
    device = _device()
    a = torch.randn(shape, device=device, dtype=dtype)
    b = torch.randn(shape, device=device, dtype=dtype)
    w = torch.rand(shape, device=device, dtype=dtype)
    op = LerpTensorFwdOp(input=shape, end=shape, weight=shape, dtype=dtype)
    out = op(a, b, w)
    ref = torch.lerp(a, b, w)
    torch.testing.assert_close(out, ref, **_lerp_tol(dtype))


@pytest.mark.smoke
def test_lerp_tensor_broadcast() -> None:
    """LerpTensorFwdOp supports the manifest's 3-way broadcast rule."""
    a_shape, b_shape, w_shape = (3, 1), (1, 4), (3, 4)
    dtype = torch.float32
    device = _device()
    a = torch.randn(a_shape, device=device, dtype=dtype)
    b = torch.randn(b_shape, device=device, dtype=dtype)
    w = torch.rand(w_shape, device=device, dtype=dtype)
    op = LerpTensorFwdOp(
        input=a_shape,
        end=b_shape,
        weight=w_shape,
        dtype=dtype,
    )
    out = op(a, b, w)
    ref = torch.lerp(a, b, w)
    torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-6)
    assert tuple(out.shape) == (3, 4)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "bad_dtype",
    [torch.float8_e4m3fn, torch.float8_e5m2],
)
def test_lerp_tensor_rejects_fp8_dtype(bad_dtype: torch.dtype) -> None:
    """LerpTensorFwdOp must reject fp8 dtypes (manifest declares no fp8)."""
    shape = (4, 8)
    with pytest.raises((ValueError, TypeError)):
        LerpTensorFwdOp(
            input=shape,
            end=shape,
            weight=shape,
            dtype=bad_dtype,
        )


@pytest.mark.smoke
def test_lerp_tensor_dtype_mismatch_rejected() -> None:
    """forward() must reject inputs whose dtype disagrees with __init__."""
    shape = (4, 8)
    device = _device()
    op = LerpTensorFwdOp(
        input=shape,
        end=shape,
        weight=shape,
        dtype=torch.float32,
    )
    a = torch.randn(shape, device=device, dtype=torch.float32)
    b = torch.randn(shape, device=device, dtype=torch.float32)
    w_bad = torch.rand(shape, device=device, dtype=torch.float16)
    with pytest.raises(ValueError, match="weight.dtype"):
        op(a, b, w_bad)


@pytest.mark.smoke
def test_lerp_tensor_eval_roofline() -> None:
    """eval_roofline() matches the GPU formula (E14).

    flops = 3 * N_total (sub + mul + add per element);
    bytes = 4 * N_total * elem_bytes (3 reads + 1 write).
    """
    shape = (64, 128)
    dtype = torch.float16
    op = LerpTensorFwdOp(input=shape, end=shape, weight=shape, dtype=dtype)
    n_total = 64 * 128
    flops, mem_bytes = op.eval_roofline()
    elem_bytes = dtype.itemsize
    assert flops == 3 * n_total, f"flops {flops} != 3 * N = {3 * n_total}"
    assert mem_bytes == 4 * n_total * elem_bytes, (
        f"bytes {mem_bytes} != 4 * N * elem_bytes = {4 * n_total * elem_bytes}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
