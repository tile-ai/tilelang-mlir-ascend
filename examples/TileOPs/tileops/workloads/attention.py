"""Workload definitions for the attention op family (NPU).

Ported from the GPU TileOPs ``workloads/attention/mha.py`` -- the
``MhaFwdWorkload`` class only (``MhaBwdWorkload`` /
``MhaDecodeWorkload`` / ``MhaDecodePagedWorkload`` belong to ops that are
not migrated yet, and the Bwd workload additionally depends on the
unmigrated GQA square-LSE helper).

The only adaptation (W1) is device placement: ``device='cuda'`` is
resolved through :func:`tileops.device.get_device_backend` instead of the
hard-coded ``"cuda"``.
"""

import torch

from tileops.device import get_device_backend
from tileops.workloads.workload_base import WorkloadBase

__all__ = ["MhaFwdWorkload"]


class MhaFwdWorkload(WorkloadBase):
    """Workload for MultiHeadAttentionFwdOp.

    Generates the q/k/v triple in BSHD layout:
    ``(batch, seq_len, heads, dim)`` per tensor.
    """

    def __init__(
        self, batch: int, heads: int, seq_len: int, dim: int, is_causal: bool, dtype: torch.dtype
    ):
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # W1: device resolved through the device backend (GPU used "cuda").
        device = get_device_backend().name
        q = torch.randn(
            self.batch, self.seq_len, self.heads, self.dim, device=device, dtype=self.dtype
        )
        k = torch.randn(
            self.batch, self.seq_len, self.heads, self.dim, device=device, dtype=self.dtype
        )
        v = torch.randn(
            self.batch, self.seq_len, self.heads, self.dim, device=device, dtype=self.dtype
        )
        return q, k, v
