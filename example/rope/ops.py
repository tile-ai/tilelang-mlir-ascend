"""Standalone RoPE operators; frequency formulas follow TileOPs PR #35.

No TileOPs installation or backend registration is required. Kernels run in
NPUIR Developer mode; set TILELANG_ASCEND_MODE=Developer before execution.
"""

import math
from typing import Optional

import torch

from .rotation import apply_rotation


def _validate_shape(x, rank, *, even_head=True):
    if x.ndim != rank or any(size <= 0 for size in x.shape):
        raise ValueError(f"expected a nonempty rank-{rank} input")
    if even_head and x.shape[-1] % 2:
        raise ValueError("head_dim must be even")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("input must be float16, bfloat16 or float32")


class _RoPEOpBase:
    rope_layout = "neox"

    def __init__(self, input_layout="1d", base=10000.0):
        if input_layout not in ("1d", "2d"):
            raise ValueError("input_layout must be '1d' or '2d'")
        if not math.isfinite(base) or base <= 0:
            raise ValueError("base must be finite and positive")
        self.input_layout = input_layout
        self.base = base
        self._freq_cache = {}

    def _get_cos_sin(self, seq_len, head_dim, dtype, device):
        key = (seq_len, head_dim, dtype, device)
        if key not in self._freq_cache:
            self._freq_cache[key] = self._compute_cos_sin(
                seq_len, head_dim, dtype, device
            )
        return self._freq_cache[key]

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        _validate_shape(x, 2 if self.input_layout == "1d" else 4)
        seq_len = x.shape[0] if x.ndim == 2 else x.shape[1]
        cos, sin = self._get_cos_sin(seq_len, x.shape[-1], x.dtype, x.device)
        return apply_rotation(
            x, cos, sin, interleaved=self.rope_layout == "interleaved"
        )


class RoPENeoxPositionIdsFwdOp:
    """Packed [tokens, heads, dim] NeoX RoPE, including partial rotation."""

    def __init__(self, max_position, base=10000.0, rotary_dim=None):
        if not isinstance(max_position, int) or not 0 < max_position <= 2**31 - 1:
            raise ValueError("max_position must be a positive int32 extent")
        if rotary_dim is not None and (
            not isinstance(rotary_dim, int) or rotary_dim <= 0 or rotary_dim % 2
        ):
            raise ValueError("rotary_dim must be positive and even")
        if not math.isfinite(base) or base <= 0:
            raise ValueError("base must be finite and positive")
        self.max_position = max_position
        self.base = base
        self.rotary_dim = rotary_dim
        self._freq_cache = {}

    def __call__(self, x, position_ids):
        return self.forward(x, position_ids)

    def forward(self, x, position_ids):
        _validate_shape(x, 3, even_head=False)
        rotary_dim = x.shape[-1] if self.rotary_dim is None else self.rotary_dim
        if rotary_dim > x.shape[-1] or rotary_dim % 2:
            raise ValueError("rotary_dim must be even and cannot exceed head_dim")
        if position_ids.shape != (x.shape[0],) or position_ids.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError(
                "position_ids must be an int32/int64 vector with one entry per token"
            )
        if position_ids.device != x.device:
            raise ValueError("position_ids and input must be on the same device")
        key = (rotary_dim, x.dtype, x.device)
        if key not in self._freq_cache:
            self._freq_cache[key] = base_freqs(
                rotary_dim, self.max_position, self.base, x.dtype, x.device
            )
        cos, sin = self._freq_cache[key]
        return apply_rotation(
            x, cos, sin, position_ids=position_ids, rotary_dim=rotary_dim
        )


def base_freqs(
    head_dim: int,
    seq_len: int,
    base: float = 10000.0,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    half = head_dim // 2
    freqs = 1.0 / base ** (
        torch.arange(0, half, device=device, dtype=torch.float32) / half
    )
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    return (torch.cos(angles).to(dtype), torch.sin(angles).to(dtype))


def _yarn_find_correction_dim(
    num_rotations: float, dim: int, base: float, max_position_embeddings: int
) -> float:
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
    low = math.floor(
        _yarn_find_correction_dim(beta_fast, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        _yarn_find_correction_dim(beta_slow, dim, base, max_position_embeddings)
    )
    return (max(low, 0), min(high, dim - 1))


class RoPEFwdOp(_RoPEOpBase):
    def __init__(
        self, rope_layout: str = "neox", input_layout: str = "1d", base: float = 10000.0
    ):
        if rope_layout not in ("neox", "interleaved"):
            raise ValueError(
                f"rope_layout must be 'neox' or 'interleaved', got '{rope_layout}'"
            )
        self.rope_layout = rope_layout
        super().__init__(input_layout, base)

    def _compute_cos_sin(
        self, seq_len: int, head_dim: int, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return base_freqs(head_dim, seq_len, base=self.base, dtype=dtype, device=device)


class RoPELlama31FwdOp(_RoPEOpBase):
    @staticmethod
    def _llama31_freqs(
        head_dim: int,
        seq_len: int,
        base: float = 10000.0,
        scale_factor: float = 8.0,
        low_freq_factor: float = 1.0,
        high_freq_factor: float = 4.0,
        original_max_position: int = 8192,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        return (torch.cos(angles).to(dtype), torch.sin(angles).to(dtype))

    def __init__(
        self,
        input_layout: str = "1d",
        base: float = 10000.0,
        scale_factor: float = 8.0,
        low_freq_factor: float = 1.0,
        high_freq_factor: float = 4.0,
        original_max_position: int = 8192,
    ):
        self.scale_factor = scale_factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.original_max_position = original_max_position
        super().__init__(input_layout, base)

    def _compute_cos_sin(
        self, seq_len: int, head_dim: int, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return RoPELlama31FwdOp._llama31_freqs(
            head_dim,
            seq_len,
            base=self.base,
            scale_factor=self.scale_factor,
            low_freq_factor=self.low_freq_factor,
            high_freq_factor=self.high_freq_factor,
            original_max_position=self.original_max_position,
            dtype=dtype,
            device=device,
        )


class YaRNFwdOp(_RoPEOpBase):
    @staticmethod
    def _yarn_freqs(
        head_dim: int,
        seq_len: int,
        base: float = 10000.0,
        scale: float = 16.0,
        original_max_position: int = 4096,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        attn_factor: float = 1.0,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        return (
            (torch.cos(angles) * attn_factor).to(dtype),
            (torch.sin(angles) * attn_factor).to(dtype),
        )

    def __init__(
        self,
        input_layout: str = "1d",
        base: float = 10000.0,
        scale: float = 16.0,
        original_max_position: int = 4096,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        attn_factor: float = 1.0,
    ):
        self.scale = scale
        self.original_max_position = original_max_position
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.attn_factor = attn_factor
        super().__init__(input_layout, base)

    def _compute_cos_sin(
        self, seq_len: int, head_dim: int, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return YaRNFwdOp._yarn_freqs(
            head_dim,
            seq_len,
            base=self.base,
            scale=self.scale,
            original_max_position=self.original_max_position,
            beta_fast=self.beta_fast,
            beta_slow=self.beta_slow,
            attn_factor=self.attn_factor,
            dtype=dtype,
            device=device,
        )


class LongRoPEFwdOp(_RoPEOpBase):
    @staticmethod
    def _longrope_freqs(
        head_dim: int,
        seq_len: int,
        base: float = 10000.0,
        rescale_factors: Optional[torch.Tensor] = None,
        max_position_embeddings: int = 4096,
        original_max_position_embeddings: int = 4096,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        return (
            (torch.cos(angles) * scaling_factor).to(dtype),
            (torch.sin(angles) * scaling_factor).to(dtype),
        )

    def __init__(
        self,
        input_layout: str = "1d",
        base: float = 10000.0,
        rescale_factors: Optional[torch.Tensor] = None,
        max_position_embeddings: int = 4096,
        original_max_position_embeddings: int = 4096,
    ):
        if rescale_factors is not None and (not rescale_factors.is_meta):
            defined = (rescale_factors != 0) & ~torch.isnan(rescale_factors)
            if not bool(defined.all()):
                raise ValueError("rescale_factors must be nonzero and not NaN")
        self.rescale_factors = rescale_factors
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        super().__init__(input_layout, base)

    def _compute_cos_sin(
        self, seq_len: int, head_dim: int, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return LongRoPEFwdOp._longrope_freqs(
            head_dim,
            seq_len,
            base=self.base,
            rescale_factors=self.rescale_factors,
            max_position_embeddings=self.max_position_embeddings,
            original_max_position_embeddings=self.original_max_position_embeddings,
            dtype=dtype,
            device=device,
        )
