"""Mamba-2 State-Space Dual (SSD) fused chunk output operator.

Adaptation from GPU (TileOPs) to NPU:

- ``_validate``/``forward``: ``x.is_cuda`` device check -> ``backend.is_device_tensor(x)`` (O1).
- ``tune`` parameter removed (O3); kernel constructor takes
  ``(batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype,
  config=None)``.
- ``eval_roofline``: the GPU
  ``tileops.perf.formulas.ssd_chunk_scan_fwd_roofline`` re-implemented inline
  (the NPU project has no tileops.perf module; roofline arithmetic is
  device-agnostic).
- Forward flow (validate -> kernel dispatch -> contiguous inputs) preserved
  unchanged (O6).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from tileops.device import get_device_backend
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.mamba import SSDChunkScanFwdKernel
from tileops.ops.op_base import Op

__all__ = ["SSDChunkScanFwdOp"]


class SSDChunkScanFwdOp(Op):
    """Mamba-2 State-Space Dual (SSD) fused chunk output operator.

    Fuses the history (prev_states) contribution and intra-chunk causal decay
    into a single pass, computing:

      out[l, p] = exp(dA_cumsum[l]) * (C[l] @ prev_states)
                + sum_{s <= l} cb[l, s] * exp(dA_cumsum[l] - dA_cumsum[s]) * dt[s] * x[s, p]

    Args:
        kernel_map: Optional override for kernel dispatch.
    """

    def __init__(
        self,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        self.batch = None
        self.num_chunks = None
        self.chunk_len = None
        self.n_heads = None
        self.d_head = None
        self.d_state = None
        self.n_groups = None
        self.dtype = None
        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[tuple, Kernel] = {}
        self.kernel = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"ssd_chunk_scan_fwd": SSDChunkScanFwdKernel}

    def _get_kernel(
        self,
        batch: int,
        num_chunks: int,
        chunk_len: int,
        n_heads: int,
        d_head: int,
        d_state: int,
        n_groups: int,
        dtype: torch.dtype,
        device_index: int | None,
    ) -> Kernel:
        key = (
            batch,
            num_chunks,
            chunk_len,
            n_heads,
            d_head,
            d_state,
            n_groups,
            dtype,
            device_index,
        )
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map["ssd_chunk_scan_fwd"](
                batch,
                num_chunks,
                chunk_len,
                n_heads,
                d_head,
                d_state,
                n_groups,
                dtype,
            )
        return self._kernel_cache[key]

    def forward(
        self,
        x: torch.Tensor,
        cb: torch.Tensor,
        dA_cumsum: torch.Tensor,
        C: torch.Tensor,
        prev_states: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        """Run the fused SSD chunk scan.

        Args:
            x:           (batch, seqlen, n_heads, d_head)                    dtype
            cb:          (batch, num_chunks, n_groups, chunk_len, chunk_len)  dtype
            dA_cumsum:   (batch, n_heads, num_chunks, chunk_len)              float32
            C:           (batch, seqlen, n_groups, d_state)                   dtype
            prev_states: (batch, num_chunks, n_heads, d_head, d_state)        dtype
            dt:          (batch, n_heads, num_chunks, chunk_len)              dtype

        Returns:
            out: (batch, seqlen, n_heads, d_head)  float32
        """
        backend = get_device_backend()
        if not backend.is_device_tensor(x):
            raise ValueError(f"x must be a {backend.name} tensor, got device {x.device}")
        if x.ndim != 4:
            raise ValueError("x must have shape [batch, seq_len, n_heads, d_head]")
        batch, seq_len, n_heads, d_head = x.shape
        if dA_cumsum.ndim != 4:
            raise ValueError("dA_cumsum must have shape [batch, n_heads, num_chunks, chunk_len]")
        if dA_cumsum.shape[0] != batch or dA_cumsum.shape[1] != n_heads:
            raise ValueError("dA_cumsum must match x batch and n_heads")
        num_chunks, chunk_len = dA_cumsum.shape[2], dA_cumsum.shape[3]
        if seq_len != num_chunks * chunk_len:
            raise ValueError("x seq_len must equal num_chunks * chunk_len")
        if C.ndim != 4 or C.shape[0] != batch or C.shape[1] != seq_len:
            raise ValueError("C must have shape [batch, seq_len, n_groups, d_state]")
        n_groups, d_state = C.shape[2], C.shape[3]
        if n_heads % n_groups != 0:
            raise ValueError("n_heads must be divisible by n_groups")
        if cb.shape != (batch, num_chunks, n_groups, chunk_len, chunk_len):
            raise ValueError(
                "cb must have shape [batch, num_chunks, n_groups, chunk_len, chunk_len]"
            )
        if prev_states.shape != (batch, num_chunks, n_heads, d_head, d_state):
            raise ValueError(
                "prev_states must have shape [batch, num_chunks, n_heads, d_head, d_state]"
            )
        if dt.shape != (batch, n_heads, num_chunks, chunk_len):
            raise ValueError("dt must have shape [batch, n_heads, num_chunks, chunk_len]")

        self.batch = batch
        self.num_chunks = num_chunks
        self.chunk_len = chunk_len
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.n_groups = n_groups
        self.dtype = x.dtype
        self.kernel = self._get_kernel(
            batch,
            num_chunks,
            chunk_len,
            n_heads,
            d_head,
            d_state,
            n_groups,
            x.dtype,
            x.device.index,
        )

        x = x.contiguous()
        cb = cb.contiguous()
        dA_cumsum = dA_cumsum.contiguous()
        C = C.contiguous()
        prev_states = prev_states.contiguous()
        dt = dt.contiguous()

        return self.kernel(x, cb, dA_cumsum, C, prev_states, dt)

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for this op instance.

        Inlined from the GPU ``tileops.perf.formulas.ssd_chunk_scan_fwd_roofline``:
        history path ``2 * tokens * d_state * d_head`` plus the intra-chunk
        causal path ``batch * num_chunks * n_heads * chunk_len**2 * d_head``.
        """
        if self.batch is None or self.dtype is None:
            raise RuntimeError(
                f"{type(self).__name__}.eval_roofline() requires a prior forward() "
                "call to bind dynamic input shape"
            )
        batch = int(self.batch)
        num_chunks = int(self.num_chunks)
        chunk_len = int(self.chunk_len)
        n_heads = int(self.n_heads)
        d_head = int(self.d_head)
        d_state = int(self.d_state)
        n_groups = int(self.n_groups)
        elem_bytes = self.dtype.itemsize

        seq_len = num_chunks * chunk_len
        tokens = batch * seq_len * n_heads
        flops = 2 * tokens * d_state * d_head + batch * num_chunks * n_heads * chunk_len**2 * d_head
        nbytes = (
            tokens * d_head * elem_bytes
            + batch * num_chunks * n_groups * chunk_len**2 * elem_bytes
            + tokens * 4
            + batch * seq_len * n_groups * d_state * elem_bytes
            + batch * num_chunks * n_heads * d_head * d_state * 4
            + tokens * elem_bytes
            + tokens * d_head * 4
        )
        return int(flops), int(nbytes)
