"""Standalone Ascend 950 Dense Attention reference skeletons (TileLang)."""

from .prefill_bwd_dense import (
    dense_attention_bwd,
    dense_attention_bwd_kernel,
    reference_dense_attention_bwd,
)
from .prefill_fwd_dense import (
    dense_attention_fwd,
    dense_attention_fwd_kernel,
    reference_dense_attention_fwd,
)

__all__ = [
    "dense_attention_fwd",
    "dense_attention_fwd_kernel",
    "reference_dense_attention_fwd",
    "dense_attention_bwd",
    "dense_attention_bwd_kernel",
    "reference_dense_attention_bwd",
]
