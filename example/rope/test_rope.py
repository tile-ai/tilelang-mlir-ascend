"""RoPE parity tests adapted from TileOPs commit 86b5b058.

Run from the repository root:
    TILELANG_ASCEND_MODE=Developer python -m pytest example/rope/test_rope.py -q

NPU execution tests skip without torch_npu/hardware; frequency and validation
tests still run on CPU. The five variants and both input layouts are retained.
"""

import contextlib
import math

import pytest
import torch

from .ops import (
    LongRoPEFwdOp,
    RoPEFwdOp,
    RoPELlama31FwdOp,
    RoPENeoxPositionIdsFwdOp,
    YaRNFwdOp,
)

with contextlib.suppress(ImportError):
    import torch_npu  # noqa: F401

HAS_NPU = hasattr(torch, "npu") and torch.npu.is_available()
npu_only = pytest.mark.skipif(not HAS_NPU, reason="requires Ascend NPU and torch_npu")


@pytest.fixture(autouse=True)
def developer_mode(monkeypatch):
    monkeypatch.setenv("TILELANG_ASCEND_MODE", "Developer")


def run_device():
    return "npu" if HAS_NPU else "cpu"


def rope_verification():
    return None


def compare_outputs(actual, expected, _verification=None):
    tol = {torch.float16: 1e-3, torch.bfloat16: 8e-3, torch.float32: 1e-5}[
        expected.dtype
    ]
    torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)


RoPEBasicFixture = pytest.mark.parametrize(
    "batch,seq_len,num_heads,head_dim,dtype",
    [
        (2, 128, 8, 64, torch.float16),
        (2, 128, 8, 64, torch.bfloat16),
        (2, 128, 8, 64, torch.float32),
        (1, 256, 4, 128, torch.float16),
    ],
)
RoPEEdgeFixture = pytest.mark.parametrize(
    "batch,seq_len,num_heads,head_dim,dtype",
    [
        (1, 1, 1, 16, torch.float32),
        (1, 1, 1, 16, torch.float16),
        (1, 4, 2, 24, torch.float16),
        (2, 512, 8, 64, torch.float16),
    ],
)


class RoPEWorkload:
    def verification(self, *inputs):
        return rope_verification()

    def __init__(
        self,
        variant: str,
        input_layout: str,
        batch: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        extra_kwargs: dict | None = None,
    ):
        self.variant = variant
        self.input_layout = input_layout
        self.batch = batch
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.extra_kwargs = extra_kwargs or {}

    def gen_inputs(self) -> tuple[torch.Tensor]:
        """Generate only x; cos/sin are computed by the op internally."""
        if self.input_layout == "1d":
            x = torch.randn(
                self.seq_len, self.head_dim, device=run_device(), dtype=self.dtype
            )
        else:
            x = torch.randn(
                self.batch,
                self.seq_len,
                self.num_heads,
                self.head_dim,
                device=run_device(),
                dtype=self.dtype,
            )
        return (x,)


def rope_frequency_tables(
    head_dim: int,
    seq_len: int,
    base: float = 10000.0,
    dtype: torch.dtype = torch.float32,
    device: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute standard RoPE cos/sin tables.

    Returns:
        (cos, sin) each of shape (seq_len, head_dim // 2).
    """
    device = device or run_device()
    half = head_dim // 2
    freqs = 1.0 / base ** (
        torch.arange(0, half, device=device, dtype=torch.float32) / half
    )
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    cos_vals = torch.cos(angles).to(dtype)
    sin_vals = torch.sin(angles).to(dtype)
    return (cos_vals, sin_vals)


def llama31_frequency_tables(
    head_dim: int,
    seq_len: int,
    base: float = 10000.0,
    scale_factor: float = 8.0,
    low_freq_factor: float = 1.0,
    high_freq_factor: float = 4.0,
    original_max_position: int = 8192,
    dtype: torch.dtype = torch.float32,
    device: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Llama 3.1 scaled frequency computation."""
    device = device or run_device()
    half = head_dim // 2
    freqs = 1.0 / base ** (
        torch.arange(0, half, device=device, dtype=torch.float32) / half
    )
    low_freq_wavelen = original_max_position / low_freq_factor
    high_freq_wavelen = original_max_position / high_freq_factor
    scaled_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq.item()
        if wavelen < high_freq_wavelen:
            scaled_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            scaled_freqs.append(freq / scale_factor)
        else:
            smooth = (original_max_position / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            scaled_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    freqs = torch.stack(scaled_freqs)
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    cos_vals = torch.cos(angles).to(dtype)
    sin_vals = torch.sin(angles).to(dtype)
    return (cos_vals, sin_vals)


def _yarn_find_correction_dim(
    num_rotations: float, dim: int, base: float, max_position_embeddings: int
) -> float:
    """Canonical yarn_find_correction_dim from TVM position_embedding.py."""
    return (
        dim
        * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))
        / (2 * math.log(base))
    )


def _yarn_find_correction_range(
    beta_fast: float,
    beta_slow: float,
    dim: int,
    base: float,
    max_position_embeddings: int,
) -> tuple[int, int]:
    """Canonical yarn_find_correction_range from TVM position_embedding.py."""
    low = math.floor(
        _yarn_find_correction_dim(beta_fast, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        _yarn_find_correction_dim(beta_slow, dim, base, max_position_embeddings)
    )
    return (max(low, 0), min(high, dim - 1))


def yarn_frequency_tables(
    head_dim: int,
    seq_len: int,
    base: float = 10000.0,
    scale: float = 16.0,
    original_max_position: int = 4096,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    attn_factor: float = 1.0,
    dtype: torch.dtype = torch.float32,
    device: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Canonical YaRN frequency computation with NTK-aware interpolation.

    Reference: TVM ``rope_freq_yarn`` in position_embedding.py.

    Key formula:
    - freq_extra = 1 / (base ^ (2k/d))  (original, for extrapolation)
    - freq_inter = 1 / ((scale * base) ^ (2k/d))  (NTK-aware, for interpolation)
    - Linear ramp mask between correction dims
    - inv_freq = freq_inter * (1 - mask) + freq_extra * mask
    """
    device = device or run_device()
    half = head_dim // 2
    dim_indices = torch.arange(0, half, device=device, dtype=torch.float32)
    freq_extra = 1.0 / base ** (dim_indices / half)
    freq_inter = 1.0 / (scale * base) ** (dim_indices / half)
    low, high = _yarn_find_correction_range(
        beta_fast, beta_slow, half, base, original_max_position
    )
    if low == high:
        high = high + 1
    inv_freq_mask = 1.0 - torch.clamp((dim_indices - low) / (high - low), 0.0, 1.0)
    inv_freq = freq_inter * (1.0 - inv_freq_mask) + freq_extra * inv_freq_mask
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(t, inv_freq)
    cos_vals = (torch.cos(angles) * attn_factor).to(dtype)
    sin_vals = (torch.sin(angles) * attn_factor).to(dtype)
    return (cos_vals, sin_vals)


def longrope_frequency_tables(
    head_dim: int,
    seq_len: int,
    base: float = 10000.0,
    rescale_factors: torch.Tensor | None = None,
    max_position_embeddings: int = 4096,
    original_max_position_embeddings: int = 4096,
    dtype: torch.dtype = torch.float32,
    device: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Canonical LongRoPE frequency computation with amplitude scaling.

    Reference: TVM ``rope_freq_longrope`` in position_embedding.py.

    Key formula:
    - divisor = ext_factors[k] * base^(2k/d)  (ext_factors multiply divisor)
    - scaling_factor = sqrt(1 + log(scale) / log(orig_max_pos)) if scale > 1
    - cos/sin are multiplied by scaling_factor (amplitude factor)
    """
    device = device or run_device()
    half = head_dim // 2
    dim_indices = torch.arange(0, half, device=device, dtype=torch.float32)
    divisor = base ** (dim_indices / half)
    if rescale_factors is not None:
        rf = rescale_factors.to(device=device, dtype=torch.float32)
        divisor = rf * divisor
    freqs = 1.0 / divisor
    scale = max_position_embeddings / original_max_position_embeddings
    if scale > 1.0:
        scaling_factor = math.sqrt(
            1.0 + math.log(scale) / math.log(original_max_position_embeddings)
        )
    else:
        scaling_factor = 1.0
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    cos_vals = (torch.cos(angles) * scaling_factor).to(dtype)
    sin_vals = (torch.sin(angles) * scaling_factor).to(dtype)
    return (cos_vals, sin_vals)


def ref_rope_neox(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Reference neox RoPE: full-dim cos/sin broadcast with half-rotation."""
    cos_full = torch.cat([cos, cos], dim=-1)
    sin_full = torch.cat([sin, sin], dim=-1)
    if x.ndim == 2:
        return (
            x.float() * cos_full.float()
            + _rotate_half_neox(x).float() * sin_full.float()
        ).to(x.dtype)
    elif x.ndim == 4:
        cos_full = cos_full.unsqueeze(0).unsqueeze(2)
        sin_full = sin_full.unsqueeze(0).unsqueeze(2)
        return (
            x.float() * cos_full.float()
            + _rotate_half_neox(x).float() * sin_full.float()
        ).to(x.dtype)
    elif x.ndim == 3:
        return (
            x.float() * cos_full.float()
            + _rotate_half_neox(x).float() * sin_full.float()
        ).to(x.dtype)
    else:
        raise ValueError(f"Unsupported ndim={x.ndim}")


def ref_rope_non_neox(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Reference non-neox (RoFormer) RoPE: adjacent pair rotation."""
    cos_interleaved = cos.repeat_interleave(2, dim=-1)
    sin_interleaved = sin.repeat_interleave(2, dim=-1)
    if x.ndim == 2:
        return (
            x.float() * cos_interleaved.float()
            + _rotate_half_non_neox(x).float() * sin_interleaved.float()
        ).to(x.dtype)
    elif x.ndim == 4:
        cos_interleaved = cos_interleaved.unsqueeze(0).unsqueeze(2)
        sin_interleaved = sin_interleaved.unsqueeze(0).unsqueeze(2)
        return (
            x.float() * cos_interleaved.float()
            + _rotate_half_non_neox(x).float() * sin_interleaved.float()
        ).to(x.dtype)
    else:
        raise ValueError(f"Unsupported ndim={x.ndim}")


class RoPECase(RoPEWorkload):
    """Generic test fixture for RoPE ops.

    The op computes cos/sin internally; the test generates only x as input
    and computes the reference rotation using independently generated
    frequency tables.
    """

    def _compute_cos_sin(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Independently compute cos/sin for the reference implementation."""
        if self.variant in ("neox", "non_neox"):
            return rope_frequency_tables(self.head_dim, self.seq_len, dtype=self.dtype)
        elif self.variant == "rope_llama31":
            return llama31_frequency_tables(
                self.head_dim, self.seq_len, dtype=self.dtype, **self.extra_kwargs
            )
        elif self.variant == "yarn_rope":
            return yarn_frequency_tables(
                self.head_dim, self.seq_len, dtype=self.dtype, **self.extra_kwargs
            )
        elif self.variant == "longrope":
            return longrope_frequency_tables(
                self.head_dim, self.seq_len, dtype=self.dtype, **self.extra_kwargs
            )
        else:
            raise ValueError(f"Unknown variant: {self.variant}")

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        """Pure-PyTorch reference: independently computes cos/sin and applies rotation."""
        cos, sin = self._compute_cos_sin()
        if self.variant in ("neox", "rope_llama31", "yarn_rope", "longrope"):
            return ref_rope_neox(x, cos, sin)
        elif self.variant == "non_neox":
            return ref_rope_non_neox(x, cos, sin)
        else:
            raise ValueError(f"Unknown variant: {self.variant}")


def _rotate_half_neox(x: torch.Tensor) -> torch.Tensor:
    """Neox-style rotation: split at midpoint and negate first half."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def _rotate_half_non_neox(x: torch.Tensor) -> torch.Tensor:
    """Non-neox (RoFormer) rotation: adjacent pairs."""
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack([-x_odd, x_even], dim=-1)
    return rotated.flatten(-2)


def ref_rope_neox_position_ids(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    rotary_dim: int | None = None,
) -> torch.Tensor:
    """Reference neox RoPE for packed THD tensors with explicit positions."""
    rotary_dim = x.shape[-1] if rotary_dim is None else rotary_dim
    cos_full = torch.cat([cos, cos], dim=-1)[position_ids].unsqueeze(1)
    sin_full = torch.cat([sin, sin], dim=-1)[position_ids].unsqueeze(1)
    x_rot = x[..., :rotary_dim]
    y_rot = (
        x_rot.float() * cos_full.float()
        + _rotate_half_neox(x_rot).float() * sin_full.float()
    ).to(x.dtype)
    if rotary_dim == x.shape[-1]:
        return y_rot
    return torch.cat([y_rot, x[..., rotary_dim:]], dim=-1)


class RoPETest(RoPECase):
    def check(self, op, x):
        compare_outputs(op(x), self.ref_program(x))


@npu_only
@pytest.mark.parametrize(
    "rope_layout, variant", [("neox", "neox"), ("interleaved", "non_neox")]
)
@RoPEBasicFixture
def test_rope_1d(
    batch: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    rope_layout: str,
    variant: str,
) -> None:
    test = RoPETest(variant, "1d", batch, seq_len, num_heads, head_dim, dtype)
    op = RoPEFwdOp(rope_layout=rope_layout, input_layout="1d")
    test.check(op, *test.gen_inputs())


@npu_only
@pytest.mark.parametrize(
    "rope_layout, variant", [("neox", "neox"), ("interleaved", "non_neox")]
)
@RoPEBasicFixture
def test_rope_2d(
    batch: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    rope_layout: str,
    variant: str,
) -> None:
    test = RoPETest(variant, "2d", batch, seq_len, num_heads, head_dim, dtype)
    op = RoPEFwdOp(rope_layout=rope_layout, input_layout="2d")
    test.check(op, *test.gen_inputs())


@npu_only
@pytest.mark.parametrize(
    "rotary_dim, dtype",
    [
        (None, torch.float16),
        (32, torch.float16),
        (None, torch.bfloat16),
        (None, torch.float32),
    ],
)
def test_rope_neox_position_ids_thd(rotary_dim: int | None, dtype: torch.dtype) -> None:
    num_tokens, num_heads, head_dim, max_position = (96, 8, 64, 512)
    table_dim = head_dim if rotary_dim is None else rotary_dim
    x = torch.randn(num_tokens, num_heads, head_dim, device=run_device(), dtype=dtype)
    position_ids = (
        torch.arange(num_tokens, device=run_device(), dtype=torch.int32) * 3 + 17
    ) % max_position
    cos, sin = rope_frequency_tables(
        table_dim, max_position, dtype=dtype, device=run_device()
    )
    ref = ref_rope_neox_position_ids(
        x, cos, sin, position_ids.long(), rotary_dim=rotary_dim
    )
    op = RoPENeoxPositionIdsFwdOp(max_position=max_position, rotary_dim=rotary_dim)
    output = op(x, position_ids)
    compare_outputs(output, ref, rope_verification())


@npu_only
@pytest.mark.parametrize("num_tokens", [2, 32768], ids=["one-wave", "multi-wave"])
def test_rope_neox_position_ids_validates_range(num_tokens: int) -> None:
    op = RoPENeoxPositionIdsFwdOp(max_position=8)
    x = torch.randn(num_tokens, 1, 16, device=run_device(), dtype=torch.float16)
    position_ids = torch.zeros(num_tokens, device=run_device(), dtype=torch.int32)
    position_ids[-1] = 8
    with pytest.raises(ValueError, match="position_ids"):
        op(x, position_ids)


@npu_only
@pytest.mark.parametrize("num_tokens", [96, 2048], ids=["one-wave", "multi-wave"])
def test_rope_neox_position_ids_input_off_a_vector_boundary(num_tokens: int) -> None:
    """A contiguous input starting one element past a 16-byte boundary matches torch."""
    shape, max_position = ((num_tokens, 8, 64), 512)
    x = torch.randn(math.prod(shape) + 1, device=run_device(), dtype=torch.float16)[1:]
    x = x.view(shape)
    position_ids = torch.randint(0, max_position, (num_tokens,), device=run_device())
    position_ids = position_ids.to(torch.int32)
    cos, sin = rope_frequency_tables(
        64, max_position, dtype=x.dtype, device=run_device()
    )
    ref = ref_rope_neox_position_ids(x, cos, sin, position_ids.long())
    op = RoPENeoxPositionIdsFwdOp(max_position=max_position)
    compare_outputs(op(x, position_ids), ref, rope_verification())


@npu_only
@pytest.mark.parametrize(
    "rope_layout, variant", [("neox", "neox"), ("interleaved", "non_neox")]
)
def test_rope_input_off_a_vector_boundary(rope_layout: str, variant: str) -> None:
    """A contiguous input starting one element past a 16-byte boundary matches the reference."""
    test = RoPETest(variant, "2d", 2, 64, 4, 128, torch.float16)
    (aligned,) = test.gen_inputs()
    x = torch.empty(aligned.numel() + 1, device=run_device(), dtype=aligned.dtype)[1:]
    x = x.view(aligned.shape).copy_(aligned)
    test.check(RoPEFwdOp(rope_layout=rope_layout, input_layout="2d"), x)


@npu_only
def test_rope_longrope_rejects_a_zero_rescale_factor() -> None:
    rescale = torch.tensor([1.0, 0.0, 2.0, 1.5], device=run_device())
    with pytest.raises(ValueError, match="rescale_factors"):
        LongRoPEFwdOp(rescale_factors=rescale)


@npu_only
def test_rope_neox_position_ids_none_rotary_dim_reinfers_head_dim() -> None:
    max_position = 64
    position_ids = torch.arange(8, device=run_device(), dtype=torch.int32)
    op = RoPENeoxPositionIdsFwdOp(max_position=max_position, rotary_dim=None)
    x1 = torch.randn(8, 2, 16, device=run_device(), dtype=torch.float16)
    cos1, sin1 = rope_frequency_tables(
        16, max_position, dtype=x1.dtype, device=run_device()
    )
    ref1 = ref_rope_neox_position_ids(
        x1, cos1, sin1, position_ids.long(), rotary_dim=None
    )
    compare_outputs(op(x1, position_ids), ref1, rope_verification())
    x2 = torch.randn(8, 2, 32, device=run_device(), dtype=torch.float16)
    cos2, sin2 = rope_frequency_tables(
        32, max_position, dtype=x2.dtype, device=run_device()
    )
    ref2 = ref_rope_neox_position_ids(
        x2, cos2, sin2, position_ids.long(), rotary_dim=None
    )
    compare_outputs(op(x2, position_ids), ref2, rope_verification())


@npu_only
@RoPEBasicFixture
def test_rope_llama31_1d(
    batch: int, seq_len: int, num_heads: int, head_dim: int, dtype: torch.dtype
) -> None:
    extra = {
        "scale_factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position": 8192,
    }
    test = RoPETest(
        "rope_llama31",
        "1d",
        batch,
        seq_len,
        num_heads,
        head_dim,
        dtype,
        extra_kwargs=extra,
    )
    op = RoPELlama31FwdOp(input_layout="1d", **extra)
    test.check(op, *test.gen_inputs())


@npu_only
@RoPEBasicFixture
def test_rope_llama31_2d(
    batch: int, seq_len: int, num_heads: int, head_dim: int, dtype: torch.dtype
) -> None:
    extra = {
        "scale_factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position": 8192,
    }
    test = RoPETest(
        "rope_llama31",
        "2d",
        batch,
        seq_len,
        num_heads,
        head_dim,
        dtype,
        extra_kwargs=extra,
    )
    op = RoPELlama31FwdOp(input_layout="2d", **extra)
    test.check(op, *test.gen_inputs())


@npu_only
@RoPEBasicFixture
def test_yarn_1d(
    batch: int, seq_len: int, num_heads: int, head_dim: int, dtype: torch.dtype
) -> None:
    extra = {
        "scale": 16.0,
        "original_max_position": 4096,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "attn_factor": 1.0,
    }
    test = RoPETest(
        "yarn_rope",
        "1d",
        batch,
        seq_len,
        num_heads,
        head_dim,
        dtype,
        extra_kwargs=extra,
    )
    op = YaRNFwdOp(input_layout="1d", **extra)
    test.check(op, *test.gen_inputs())


@npu_only
@RoPEBasicFixture
def test_yarn_2d(
    batch: int, seq_len: int, num_heads: int, head_dim: int, dtype: torch.dtype
) -> None:
    extra = {
        "scale": 16.0,
        "original_max_position": 4096,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "attn_factor": 1.0,
    }
    test = RoPETest(
        "yarn_rope",
        "2d",
        batch,
        seq_len,
        num_heads,
        head_dim,
        dtype,
        extra_kwargs=extra,
    )
    op = YaRNFwdOp(input_layout="2d", **extra)
    test.check(op, *test.gen_inputs())


@npu_only
@RoPEBasicFixture
def test_rope_longrope_1d(
    batch: int, seq_len: int, num_heads: int, head_dim: int, dtype: torch.dtype
) -> None:
    half = head_dim // 2
    rescale = torch.linspace(1.0, 2.0, half, device=run_device())
    max_pos = 16384
    orig_max_pos = 4096
    extra = {
        "rescale_factors": rescale,
        "max_position_embeddings": max_pos,
        "original_max_position_embeddings": orig_max_pos,
    }
    test = RoPETest(
        "longrope", "1d", batch, seq_len, num_heads, head_dim, dtype, extra_kwargs=extra
    )
    op = LongRoPEFwdOp(
        input_layout="1d",
        rescale_factors=rescale,
        max_position_embeddings=max_pos,
        original_max_position_embeddings=orig_max_pos,
    )
    test.check(op, *test.gen_inputs())


@npu_only
@RoPEBasicFixture
def test_rope_longrope_2d(
    batch: int, seq_len: int, num_heads: int, head_dim: int, dtype: torch.dtype
) -> None:
    half = head_dim // 2
    rescale = torch.linspace(1.0, 2.0, half, device=run_device())
    max_pos = 16384
    orig_max_pos = 4096
    extra = {
        "rescale_factors": rescale,
        "max_position_embeddings": max_pos,
        "original_max_position_embeddings": orig_max_pos,
    }
    test = RoPETest(
        "longrope", "2d", batch, seq_len, num_heads, head_dim, dtype, extra_kwargs=extra
    )
    op = LongRoPEFwdOp(
        input_layout="2d",
        rescale_factors=rescale,
        max_position_embeddings=max_pos,
        original_max_position_embeddings=orig_max_pos,
    )
    test.check(op, *test.gen_inputs())


@npu_only
@pytest.mark.parametrize(
    "rope_layout, variant", [("neox", "neox"), ("interleaved", "non_neox")]
)
@RoPEEdgeFixture
def test_rope_edge(
    batch: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    rope_layout: str,
    variant: str,
) -> None:
    """Edge cases: seq_len=1 and longer sequences."""
    test = RoPETest(variant, "2d", batch, seq_len, num_heads, head_dim, dtype)
    op = RoPEFwdOp(rope_layout=rope_layout, input_layout="2d")
    test.check(op, *test.gen_inputs())


@npu_only
def test_rope_noncontiguous_1d_works() -> None:
    """A non-contiguous 1D view must produce correct results after contiguity normalization."""
    seq_len, head_dim = (4, 8)
    op = RoPEFwdOp(input_layout="1d")
    base = torch.randn(head_dim, seq_len, device=run_device(), dtype=torch.float32)
    x_nc = base.t()
    assert not x_nc.is_contiguous()
    x_c = x_nc.contiguous()
    out_nc = op(x_nc)
    out_c = op(x_c)
    assert torch.equal(out_nc, out_c)


@npu_only
@pytest.mark.parametrize("invalid_position", [-1, 8])
def test_rope_position_ids_multihead_range_recovery(invalid_position: int) -> None:
    """Multi-head rotation rejects either range boundary and accepts the next valid call."""
    max_position = 8
    x = torch.randn(2, 32, 128, device=run_device(), dtype=torch.bfloat16)
    positions = torch.tensor(
        [0, invalid_position], device=run_device(), dtype=torch.int64
    )
    op = RoPENeoxPositionIdsFwdOp(max_position=max_position)
    with pytest.raises(ValueError, match="position_ids"):
        op(x, positions)
    positions[-1] = max_position - 1
    cos, sin = rope_frequency_tables(
        128, max_position, dtype=x.dtype, device=run_device()
    )
    ref = ref_rope_neox_position_ids(x, cos, sin, positions)
    compare_outputs(op(x, positions), ref, rope_verification())


@pytest.mark.parametrize(
    "variant", ["neox", "non_neox", "rope_llama31", "yarn_rope", "longrope"]
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("head_dim", [16, 24, 128])
def test_frequency_tables_on_cpu(variant, dtype, head_dim):
    """Check every host frequency scheme independently of compiler/hardware."""
    kwargs = {"dtype": dtype, "device": "cpu"}
    seq_len = 19
    if variant in ("neox", "non_neox"):
        op = RoPEFwdOp(rope_layout="interleaved" if variant == "non_neox" else "neox")
        expected = rope_frequency_tables(head_dim, seq_len, **kwargs)
    elif variant == "rope_llama31":
        op = RoPELlama31FwdOp()
        expected = llama31_frequency_tables(head_dim, seq_len, **kwargs)
    elif variant == "yarn_rope":
        op = YaRNFwdOp(attn_factor=1.25)
        expected = yarn_frequency_tables(head_dim, seq_len, attn_factor=1.25, **kwargs)
    else:
        factors = torch.linspace(1.0, 2.0, head_dim // 2)
        extra = {"rescale_factors": factors, "max_position_embeddings": 16384}
        op = LongRoPEFwdOp(**extra)
        expected = longrope_frequency_tables(head_dim, seq_len, **extra, **kwargs)
    actual = op._get_cos_sin(seq_len, head_dim, dtype, torch.device("cpu"))
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, atol=0, rtol=0)


@pytest.mark.parametrize("position", [-1, 8, 2**32, -(2**32)])
def test_position_ids_rejected_before_int32_cast_on_cpu(position):
    op = RoPENeoxPositionIdsFwdOp(max_position=8)
    x = torch.zeros(2, 3, 16)
    ids = torch.tensor([0, position], dtype=torch.int64)
    with pytest.raises(ValueError, match="position_ids"):
        op(x, ids)


@pytest.mark.parametrize("rotary_dim", [0, 3, 18])
def test_invalid_rotary_dimension_on_cpu(rotary_dim):
    with pytest.raises(ValueError, match="rotary_dim"):
        op = RoPENeoxPositionIdsFwdOp(max_position=8, rotary_dim=rotary_dim)
        op(torch.zeros(2, 3, 16), torch.tensor([0, 1], dtype=torch.int32))


def test_zero_longrope_rescale_on_cpu():
    with pytest.raises(ValueError, match="rescale_factors"):
        LongRoPEFwdOp(rescale_factors=torch.tensor([1.0, 0.0]))


@npu_only
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dim", [24, 128])
@pytest.mark.parametrize("layout", ["neox", "interleaved"])
def test_head_block_tail_on_npu(dtype, dim, layout):
    """Nine heads exercises the tail of the eight-head staged block."""
    x = torch.randn(2, 7, 9, dim, device="npu", dtype=dtype)
    cos, sin = rope_frequency_tables(dim, 7, dtype=dtype, device="npu")
    ref = ref_rope_non_neox if layout == "interleaved" else ref_rope_neox
    compare_outputs(
        RoPEFwdOp(rope_layout=layout, input_layout="2d")(x), ref(x, cos, sin)
    )


@npu_only
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("head_dim", [64, 65])
def test_partial_rotary_suffix_is_exact_on_npu(dtype, head_dim):
    x = torch.randn(5, 9, head_dim, device="npu", dtype=dtype)
    positions = torch.tensor([3, 1, 7, 0, 3], device="npu", dtype=torch.int64)
    result = RoPENeoxPositionIdsFwdOp(max_position=8, rotary_dim=24)(x, positions)
    cos, sin = rope_frequency_tables(24, 8, dtype=dtype, device="npu")
    compare_outputs(
        result, ref_rope_neox_position_ids(x, cos, sin, positions, rotary_dim=24)
    )
    torch.testing.assert_close(result[..., 24:], x[..., 24:], atol=0, rtol=0)
