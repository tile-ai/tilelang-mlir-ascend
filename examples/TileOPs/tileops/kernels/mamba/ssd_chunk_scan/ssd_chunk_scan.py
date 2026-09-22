"""Mamba-2 State-Space Dual (SSD) fused chunk output forward kernel (NPU-adapted).

History + intra-chunk paths in one pass:

  out[l, p] = exp(dA_cumsum[l]) * (C[l] @ prev_states)
            + sum_{s <= l} cb[l, s] * exp(dA_cumsum[l] - dA_cumsum[s]) * dt[s] * x[s, p]

Official-aligned interface (matches _chunk_scan_fwd in mamba_ssm):

Inputs:
  x:           [B, S, H, P]        dtype       seqlen-fused
  cb:          [B, C, G, L, L]     dtype       group-owned
  dA_cumsum:   [B, H, C, L]        float32
  C:           [B, S, G, N]        dtype       seqlen-fused, group-owned
  prev_states: [B, C, H, P, N]     float32     P before N
  dt:          [B, H, C, L]        dtype

Output:
  out:         [B, S, H, P]        float32     seqlen-fused

Adaptation summary (GPU -> NPU):

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang kernel factory (``_ssd_chunk_scan_fwd_kernel``) is
    extracted from the GPU repo via ``extract_tl_kernel.py`` and imported
    as-is.  It serves as the reference for the NPU kernel component to
    reimplement for ``target="npuir"``.  K1-K4 adaptations (decorator,
    grid/sync semantics, ``threads`` removal, padding strategy) are
    handled by the NPU kernel component during re-implementation.

  **Part B -- custom_op wrapper + Kernel class** (fully ported):
    K5: ``supported_archs = None`` (was ``[80, 86, 89, 90]``).
    K7: ``custom_op("npub::ssd_chunk_scan_fwd")`` (was ``"top::..."``).
    K8: ``autotune_configs`` / ``tune`` param removed; heuristic config
        selection only (``init_config(config)``).
    K9: ``threads`` removed from the wrapper signature, ``default_config``
        and the ``forward`` call.  ``num_stages`` is retained (pipeline
        depth for ``T.Pipelined``; not a CUDA thread parameter).

  Pre-Stage-3 note: the imported factory is still the GPU (CUDA target)
  implementation whose inner callable takes
  ``(block_l, block_p, block_n, block_s, threads, num_stages)``.  The
  dispatch below passes the NPU-shaped argument list (no ``threads``);
  it becomes callable once the NPU kernel component rewrites the factory
  for ``target="npuir"`` and drops ``threads`` from the callable
  signature (K3).  Running ``forward`` before that rewrite fails -- this
  is the expected scaffold state.
"""

from typing import Optional

import torch

from tileops.kernels.kernel_base import Kernel

# Part A: GPU TileLang kernel factory extracted by extract_tl_kernel.py
# (pattern B) from {gpu_repo_root}/tileops/kernels/mamba/ssd_chunk_scan.py.
# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .ssd_chunk_scan_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
# from .ssd_chunk_scan_kernel import _ssd_chunk_scan_fwd_kernel
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
from .ssd_chunk_scan_kernel.perf_opt._ssd_chunk_scan_fwd_kernel import _ssd_chunk_scan_fwd_kernel

__all__ = ["SSDChunkScanFwdKernel"]


# ---------------------------------------------------------------------------
# custom_op wrapper (K7: top:: -> npub::; K9: threads removed)
# ---------------------------------------------------------------------------


@torch.library.custom_op("npub::ssd_chunk_scan_fwd", mutates_args=())
def _ssd_chunk_scan_fwd_wrapped(
    batch: int,
    num_chunks: int,
    chunk_len: int,
    n_heads: int,
    d_head: int,
    d_state: int,
    n_groups: int,
    dtype: str,
    block_l: int,
    block_p: int,
    block_n: int,
    block_s: int,
    num_stages: int,
    x: torch.Tensor,
    cb: torch.Tensor,
    dA_cumsum: torch.Tensor,
    C: torch.Tensor,
    prev_states: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    # GPU callable signature (pre-Stage-3): (block_l, block_p, block_n,
    # block_s, threads, num_stages).  The NPU-shaped call below omits
    # `threads` (K3/K9); it binds 1:1 once the NPU kernel component
    # rewrites the factory for target="npuir".
    return _ssd_chunk_scan_fwd_kernel(
        batch,
        num_chunks,
        chunk_len,
        n_heads,
        d_head,
        d_state,
        n_groups,
        dtype,
    )(block_l, block_p, block_n, block_s, num_stages)(
        x,
        cb,
        dA_cumsum,
        C,
        prev_states,
        dt,
    )


@_ssd_chunk_scan_fwd_wrapped.register_fake
def _(
    batch: int,
    num_chunks: int,
    chunk_len: int,
    n_heads: int,
    d_head: int,
    d_state: int,
    n_groups: int,
    dtype: str,
    block_l: int,
    block_p: int,
    block_n: int,
    block_s: int,
    num_stages: int,
    x: torch.Tensor,
    cb: torch.Tensor,
    dA_cumsum: torch.Tensor,
    C: torch.Tensor,
    prev_states: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    # output: [B, S, H, P]
    return x.new_empty(
        (batch, num_chunks * chunk_len, n_heads, d_head),
        dtype=torch.float32,
    )


# ---------------------------------------------------------------------------
# Kernel class (K5/K8/K9 adaptations)
# ---------------------------------------------------------------------------


class SSDChunkScanFwdKernel(Kernel):
    """Mamba-2 SSD fused chunk output forward kernel.

    Official-aligned interface (matches _chunk_scan_fwd in mamba_ssm):

    Inputs:
      x:           [B, S, H, P]        dtype       seqlen-fused
      cb:          [B, C, G, L, L]     dtype       group-owned
      dA_cumsum:   [B, H, C, L]        float32
      C:           [B, S, G, N]        dtype       seqlen-fused, group-owned
      prev_states: [B, C, H, P, N]     float32     P before N
      dt:          [B, H, C, L]        dtype

    Output:
      out:         [B, S, H, P]        float32     seqlen-fused

    NPU adaptations:
      K5: ``supported_archs = None`` (all architectures; GPU pinned CUDA
          SM ints).
      K8: autotune removed -- heuristic config selection only;
          ``init_config(config)`` takes no ``tune`` argument.
      K9: ``threads`` removed from ``default_config`` and the
          ``forward`` dispatch (GPU default was ``threads=128``).
    """

    ascend_mode = "Expert"

    # K5: [80, 86, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

    def __init__(
        self,
        batch: int,
        num_chunks: int,
        chunk_len: int,
        n_heads: int,
        d_head: int,
        d_state: int,
        n_groups: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        device_index: int | None = None,
    ) -> None:
        super().__init__()
        self.batch = batch
        self.num_chunks = num_chunks
        self.chunk_len = chunk_len
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.n_groups = n_groups
        self.dtype = dtype
        self.device_index = device_index
        self.kernel = _ssd_chunk_scan_fwd_kernel(
            batch,
            num_chunks,
            chunk_len,
            n_heads,
            d_head,
            d_state,
            n_groups,
            self.dtype_str,
        )
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        # NPU tuned default, aligned with the kernel `TUNED_DEFAULT_CONFIG`
        # (integration_log documented intent: block_n=min(128,N)/num_stages=2).
        # block_n=128 collapses the d_state=128 manifest cases to a single
        # n-block (DESIGN §5.2, v6_bn128: w2 −10.5% / w3 −5.0% / w4 −15.7%);
        # the kernel clamps block_n internally to min(block_n, N), so N<128
        # falls back safely.  num_stages is a factory-signature placeholder:
        # the NPU Expert kernel hardcodes the depth-2 task pipeline and does
        # not consume num_stages (kept =2 for source-compatible signatures).
        return {
            "block_l": 64,
            "block_p": 64,
            "block_n": min(128, self.d_state),
            "block_s": 64,
            "num_stages": 2,
        }

    def forward(
        self,
        x: torch.Tensor,
        cb: torch.Tensor,
        dA_cumsum: torch.Tensor,
        C: torch.Tensor,
        prev_states: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:           [B, S, H, P]        dtype
            cb:          [B, C, G, L, L]     dtype
            dA_cumsum:   [B, H, C, L]        float32
            C:           [B, S, G, N]        dtype
            prev_states: [B, C, H, P, N]     float32     P before N
            dt:          [B, H, C, L]        dtype

        Returns:
            out: [B, S, H, P]  float32
        """
        # K9: no `threads` in the dispatch.
        return _ssd_chunk_scan_fwd_wrapped(
            self.batch,
            self.num_chunks,
            self.chunk_len,
            self.n_heads,
            self.d_head,
            self.d_state,
            self.n_groups,
            self.dtype_str,
            self.config["block_l"],
            self.config["block_p"],
            self.config["block_n"],
            self.config["block_s"],
            self.config["num_stages"],
            x.contiguous(),
            cb.contiguous(),
            dA_cumsum.contiguous(),
            C.contiguous(),
            prev_states.contiguous(),
            dt.contiguous(),
        )
