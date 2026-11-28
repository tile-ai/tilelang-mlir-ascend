"""Independent encoding/contract tests and opt-in full-manifest NPU regression.

Run: TILELANG_ASCEND_MODE=Expert python -m pytest example/gemm_w4a8 -q
Set RUN_W4A8_MANIFEST=1 to execute all 98 large source shapes on an NPU.
"""

import contextlib
import json
import math
import os
from pathlib import Path

import pytest
import torch

from .w4a8 import (
    FP8,
    GEMMW4A8FwdOp,
    _device_tables,
    _host_tables,
    _make_decode,
    _make_dequant,
    _validate,
)

with contextlib.suppress(ImportError):
    import torch_npu  # noqa: F401

HAS_NPU = hasattr(torch, "npu") and torch.npu.is_available()
npu_only = pytest.mark.skipif(not HAS_NPU, reason="requires Ascend NPU and torch_npu")
CASES = json.loads(Path(__file__).with_name("manifest_cases.json").read_text())["cases"]


@pytest.fixture(autouse=True)
def expert_mode(monkeypatch):
    monkeypatch.setenv("TILELANG_ASCEND_MODE", "Expert")


def decode_byte(code):
    """Independent E4M3FN bit-field definition, with a bias of seven."""
    exponent, mantissa = (code >> 3) & 15, code & 7
    if exponent == 15 and mantissa == 7:
        return float("nan")
    magnitude = (
        mantissa * 2.0**-9
        if exponent == 0
        else (1 + mantissa / 8) * 2.0 ** (exponent - 7)
    )
    return -magnitude if code & 128 else magnitude


def round_e4m3(value):
    """Nearest-even over the finite codebook, independently of torch FP8 casts."""
    if math.isnan(value):
        return float("nan")
    if abs(value) > 464:
        # PyTorch releases differ in overflow saturation. Match the source's
        # torch conversion contract; finite rounding below is checked separately.
        overflow = torch.tensor(1e6).to(FP8).float().item()
        return math.copysign(overflow, value)
    code = min(range(127), key=lambda c: (abs(abs(value) - decode_byte(c)), c % 2))
    return math.copysign(decode_byte(code), value)


@pytest.mark.parametrize("code", range(256))
def test_all_encodings_and_signed_nibbles(code):
    decode, dequant = _host_tables()
    torch.testing.assert_close(
        decode[0, code].float(),
        torch.tensor(decode_byte(code)),
        atol=0,
        rtol=0,
        equal_nan=True,
    )
    expected = [
        round_e4m3(decode_byte(code) * (q if q < 8 else q - 16)) for q in range(16)
    ]
    torch.testing.assert_close(
        dequant.reshape(256, 16)[code].float(),
        torch.tensor(expected),
        atol=0,
        rtol=0,
        equal_nan=True,
    )


def inputs(m=3, n=128, k=256, device="cpu"):
    generator = torch.Generator().manual_seed(82)
    a = (torch.randn(m, k, generator=generator) * 0.25).to(FP8)
    q = torch.randint(-8, 8, (n, k), dtype=torch.int16, generator=generator)
    w = ((q[:, ::2] & 15) | ((q[:, 1::2] & 15) << 4)).to(torch.uint8)
    gs = (
        (torch.rand(n, k // 128, generator=generator) * 0.05 + 0.03)
        * torch.where(torch.arange(n)[:, None] % 2 == 0, 1, -1)
    ).to(FP8)
    channel = torch.linspace(-1.5, 1.5, n)
    token = torch.linspace(-0.5, 1.5, m)
    channel[0] = 0
    token[-1] = 0
    return tuple(t.to(device) for t in (a, w, gs, channel, token))


def reference(a, packed, group, channel, token):
    q = torch.empty(
        (packed.shape[0], packed.shape[1] * 2), dtype=torch.int32, device=a.device
    )
    q[:, ::2] = packed.to(torch.int32) & 15
    q[:, 1::2] = packed.to(torch.int32) >> 4
    q = torch.where(q < 8, q, q - 16)
    w = (q.float() * group.float().repeat_interleave(128, dim=1)).to(FP8).float()
    return ((a.float() @ w.T) * token[:, None] * channel[None, :]).to(torch.bfloat16)


def assert_accuracy(actual, expected, k):
    actual, expected = actual.float(), expected.float()
    torch.testing.assert_close(
        actual, expected, atol=5e-5 * k, rtol=2e-2, equal_nan=True
    )
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    error = ((actual[finite] - expected[finite]).double() ** 2).sum()
    energy = (actual[finite].double() ** 2 + expected[finite].double() ** 2).sum()
    assert error <= 5e-5 * energy


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["label"])
def test_manifest_shape_contract(case):
    """Metadata-only coverage; this does not claim device correctness for a shape."""
    m, n, k = case["M"], case["N"], case["K"]
    tensors = (
        torch.empty(m, k, dtype=FP8, device="meta"),
        torch.empty(n, k // 2, dtype=torch.uint8, device="meta"),
        torch.empty(n, k // 128, dtype=FP8, device="meta"),
        torch.empty(n, dtype=torch.float32, device="meta"),
        torch.empty(m, dtype=torch.float32, device="meta"),
    )
    assert _validate(*tensors) == (m, n, k)


@pytest.mark.parametrize("group_size", [0, 64, 256, 128.0, True])
def test_unsupported_groups(group_size):
    with pytest.raises(ValueError, match="group_size"):
        GEMMW4A8FwdOp(group_size)


@pytest.mark.parametrize("index", range(5))
def test_reject_wrong_dtype(index):
    data = list(inputs())
    data[index] = data[index].to(torch.float16)
    with pytest.raises(ValueError):
        GEMMW4A8FwdOp()(*data)


@pytest.mark.parametrize("index", range(5))
def test_reject_wrong_shape(index):
    data = list(inputs())
    data[index] = data[index][:-1]
    with pytest.raises(ValueError):
        GEMMW4A8FwdOp()(*data)


@pytest.mark.parametrize("index", range(5))
def test_reject_noncontiguous(index):
    data = list(inputs())
    x = data[index]
    larger = torch.empty((*x.shape[:-1], x.shape[-1] * 2), dtype=x.dtype)
    larger[..., ::2] = x
    data[index] = larger[..., ::2]
    with pytest.raises(ValueError, match="contiguous"):
        GEMMW4A8FwdOp()(*data)


def test_reject_cpu_execution():
    with pytest.raises(RuntimeError, match="NPU"):
        GEMMW4A8FwdOp()(*inputs())


def test_repack_preserves_order_without_aliasing():
    original = (
        torch.arange(128 * 128, dtype=torch.int32)
        .to(torch.uint8)
        .reshape(128, 128)[:, ::2]
    )
    result = GEMMW4A8FwdOp().repack(original)
    assert result.is_contiguous() and result.data_ptr() != original.data_ptr()
    torch.testing.assert_close(result, original)
    result.zero_()
    assert torch.count_nonzero(original) > 0


@npu_only
@pytest.mark.parametrize("m", [1, 17, 32, 33, 65, 129])
@pytest.mark.parametrize("n,k", [(128, 128), (256, 384)])
def test_npu_tail_scales_and_fresh_weights(m, n, k):
    data = inputs(m, n, k, device="npu")
    before = [x.cpu().clone() for x in data]
    op = GEMMW4A8FwdOp()
    actual = op(data[0], op.repack(data[1]), *data[2:])
    expected = reference(*before)
    assert_accuracy(actual.cpu(), expected, k)
    assert (actual[:, 0] == 0).all() and (actual[-1] == 0).all()
    for x, original in zip(data, before, strict=True):
        torch.testing.assert_close(
            x.cpu().view(torch.uint8), original.view(torch.uint8), atol=0, rtol=0
        )
    # No cached dequantization may conceal new packed weights on a later call.
    data[1].zero_()
    assert (op(*data) == 0).all()


@npu_only
def test_npu_storage_offsets():
    data = inputs(7, 128, 256)
    shifted = []
    for value in data:
        base = torch.empty(value.numel() + 1, dtype=value.dtype)
        view = base[1:].view(value.shape)
        view.copy_(value)
        device_base = base.to("npu")
        shifted.append(device_base[1:].view(value.shape))
    assert_accuracy(GEMMW4A8FwdOp()(*shifted).cpu(), reference(*data), 256)


@npu_only
def test_npu_all_byte_encodings():
    """Exercise byte widening, sign bits and the full scale/nibble codebook."""
    device = torch.device("npu", torch.npu.current_device())
    decode, dequant = _device_tables(device.index)
    codes = torch.arange(256).to(torch.uint8)
    decoded = torch.empty(1, 256, dtype=torch.float16, device=device)
    _make_decode(1, 256, device.index)(
        codes.reshape(1, 256).to(device), decode, decoded
    )
    torch.testing.assert_close(
        decoded.cpu(),
        codes.view(FP8).half().reshape(1, 256),
        atol=0,
        rtol=0,
        equal_nan=True,
    )
    packed = codes.repeat(256, 1)
    scales = codes[:, None].expand(256, 4).contiguous()
    unpacked = torch.empty(256, 512, dtype=torch.float16, device=device)
    _make_dequant(256, 512, device.index)(
        packed.to(device), scales.to(device), dequant, unpacked
    )
    q = torch.empty(256, 512, dtype=torch.int16)
    q[:, ::2] = packed.to(torch.int16) & 15
    q[:, 1::2] = packed.to(torch.int16) >> 4
    q = torch.where(q < 8, q, q - 16)
    expected = (
        (q.float() * scales.view(FP8).float().repeat_interleave(128, 1)).to(FP8).half()
    )
    torch.testing.assert_close(unpacked.cpu(), expected, atol=0, rtol=0, equal_nan=True)


@npu_only
@pytest.mark.skipif(
    os.environ.get("RUN_W4A8_MANIFEST") != "1",
    reason="set RUN_W4A8_MANIFEST=1 for large manifest cases",
)
@pytest.mark.parametrize("case", CASES, ids=lambda c: c["label"])
def test_npu_manifest(case):
    m, n, k = case["M"], case["N"], case["K"]
    data = inputs(m, n, k)
    # Keep reference arithmetic independent of the NPU implementation.
    expected = reference(*data)
    actual = GEMMW4A8FwdOp()(*(x.to("npu") for x in data)).cpu()
    assert_accuracy(actual, expected, k)
