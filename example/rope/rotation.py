"""Validate inputs and dispatch standalone RoPE to the NPUIR vector kernel."""

import torch


def apply_rotation(
    x, cos, sin, *, interleaved=False, position_ids=None, rotary_dim=None
):
    """Rotate SD/BSHD inputs, or packed THD inputs with explicit position IDs.

    The unrotated suffix is copied by the kernel. Noncontiguous inputs are
    normalized, including contiguous views with a nonzero storage offset.
    """
    indexed = position_ids is not None
    if x.ndim not in ((3,) if indexed else (2, 4)) or any(n <= 0 for n in x.shape):
        raise ValueError("expected nonempty SD/BSHD input or indexed THD input")
    dim = x.shape[-1]
    rotary = dim if rotary_dim is None else rotary_dim
    if not isinstance(rotary, int) or rotary <= 0 or rotary > dim or rotary % 2:
        raise ValueError(
            "rotary_dim must be positive, even and no larger than head_dim"
        )
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("input must be float16, bfloat16 or float32")
    heads = 1 if x.ndim == 2 else x.shape[-2]
    tokens = x.numel() // (heads * dim)
    if (
        cos.ndim != 2
        or cos.shape != sin.shape
        or cos.shape[1] != rotary // 2
        or cos.shape[0] <= 0
    ):
        raise ValueError(
            "cos and sin must have matching [positions, rotary_dim/2] shapes"
        )
    if any(t.device != x.device or t.dtype != x.dtype for t in (cos, sin)):
        raise ValueError("frequency tables must have the input device and dtype")
    if not indexed:
        seq_len = x.shape[0] if x.ndim == 2 else x.shape[1]
        if cos.shape[0] != seq_len:
            raise ValueError("frequency table length must equal seq_len")
    else:
        if position_ids.shape != (tokens,) or position_ids.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError(
                "position_ids must be an int32/int64 vector with one entry per token"
            )
        if position_ids.device != x.device:
            raise ValueError("position_ids must have the input device")
        if bool(((position_ids < 0) | (position_ids >= cos.shape[0])).any()):
            raise ValueError("position_ids must be in the frequency table range")
    if x.device.type != "npu":
        raise ValueError("RoPE execution requires an Ascend NPU")

    # Import lazily so frequency/reference tests do not require the NPU compiler.
    from .staged import build_staged_rotation

    positions = (
        position_ids.to(torch.int32).contiguous()
        if indexed
        else torch.zeros(1, dtype=torch.int32, device=x.device)
    )
    kernel = build_staged_rotation(
        tokens,
        heads,
        dim,
        rotary,
        cos.shape[0],
        str(x.dtype).split(".")[-1],
        interleaved,
        indexed,
    )
    output = kernel(
        x.contiguous().view(tokens, heads, dim),
        cos.contiguous(),
        sin.contiguous(),
        positions,
    )
    return output.view_as(x)
