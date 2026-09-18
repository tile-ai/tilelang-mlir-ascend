"""Adaptive LayerNorm (AdaLN / AdaLN-Zero) kernel using TileLang (NPU-adapted).

AdaLN:      y = scale * LayerNorm(x) + shift
AdaLN-Zero: y = gate * (scale * LayerNorm(x) + shift)

scale, shift (and optionally gate) are per-token tensors of shape (M, N),
pre-computed by the caller from a conditioning signal.  The ``has_gate``
parameter controls the variant (False -> AdaLN, True -> AdaLN-Zero); this
module is ported for ``AdaLayerNormFwdOp`` (``has_gate=False``) and keeps
the gated branch of the shared GPU kernel class for the later
``AdaLayerNormZeroFwdOp`` migration.

The kernels accept natural ``(M, N)`` tensors.  For non-aligned ``N``,
boundary handling stays on device: loads zero-fill the logical reduction
tail and stores write only real output columns.  The centered two-pass
variance subtracts the padded-zero contribution explicitly.

Adaptation summary (GPU -> NPU), per docs/gpu_to_npu_adaptation.md:

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang kernel factory (``_ada_layer_norm_kernel``) is
    extracted from the GPU repo via ``extract_tl_kernel.py`` and imported
    as-is, together with its ``_align_up`` helper and the ``ALIGNMENT``
    constant.  It serves as the reference for the NPU kernel component to
    reimplement for ``target="npuir"``.  K1-K4 adaptations (decorator,
    grid/sync semantics, ``threads`` removal, padding strategy, and the
    CUDA-only ``T.ptx_cp_async`` prefetch path) are handled by the NPU
    component during re-implementation.

  **Part B -- custom_op wrappers + Kernel class** (fully ported):
    K5: ``supported_archs = None`` (was ``[80, 86, 89, 90]``).
    K7: ``custom_op("npub::...")`` (was ``"top::..."``).
    K8: ``autotune_configs`` property and ``tune`` parameter removed;
        heuristic config selection only, ``init_config(config)``.
    K9: ``threads`` removed from the custom_op wrapper signatures, the
        factory-callable invocation, ``default_config``, and the
        ``forward`` dispatch (row-reduction family: the meaningful knobs
        are ``block_m`` and the shape policy, not CUDA thread counts).
"""

from typing import Optional

import torch

from tileops.kernels.kernel_base import Kernel

# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .ada_layer_norm_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
# from .ada_layer_norm_kernel import ALIGNMENT, _ada_layer_norm_kernel, _align_up
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
from .ada_layer_norm_kernel.perf_opt._ada_layer_norm_kernel import (
    ALIGNMENT,
    _ada_layer_norm_kernel,
    _align_up,
)
from .ada_layer_norm_kernel.perf_opt._ada_layer_norm_kernel import (
    select_row_config as _select_row_config_tuned,
)

__all__ = ["AdaLayerNormKernel"]


def _should_use_cp_async(
    n: int,
    dtype: torch.dtype,
    has_gate: bool = False,
) -> bool:
    """Select async prefetch when lowering and shared-memory limits allow it."""
    n_padded = _align_up(n, ALIGNMENT)
    row_bytes = n * dtype.itemsize
    num_buffers = 4 if has_gate else 3
    shared_bytes = num_buffers * n_padded * dtype.itemsize
    return n_padded != n and row_bytes % 4 == 0 and shared_bytes <= 48 * 1024


# ---------------------------------------------------------------------------
# Row-config selection.
#
# Ported from GPU ``tileops/kernels/norm/_config.py:select_row_config``.
# The GPU helper pins ``block_m=1`` (with one row per CTA the per-row
# uniformity the row reduction needs always holds, so the cross-thread
# AllReduce collapse is structurally impossible) plus ``threads=128``
# (measured CUDA optimum).  NPU adaptation K9 drops the ``threads`` key,
# leaving the structural ``block_m=1`` default.  The GPU
# ``select_row_configs`` autotune-space generator is not ported (K8).
# ---------------------------------------------------------------------------


# --- baseline (Stage 3): structural block_m=1 --------------------------------
# _DEFAULT_BLOCK_M = 1
#
#
# def _select_row_config(M: int, N: int, dtype) -> dict:
#     """Structurally collapse-free default ``{block_m}`` for a row reduction."""
#     return {"block_m": _DEFAULT_BLOCK_M}
# --- perf_opt (Stage 4 tuned): per-(M, N) dispatch table ---------------------
def _select_row_config(M: int, N: int, dtype) -> dict:
    """Tuned per-(M, N) ``block_m`` (perf_opt TUNED_DEFAULT_BLOCK_M + UB budget)."""
    use_fp32_transit = dtype in (torch.float16, torch.bfloat16)
    return _select_row_config_tuned(M, N, use_fp32_transit)


# ---------------------------------------------------------------------------
# custom_op wrappers (K7: top:: -> npub::, K9: threads removed)
# ---------------------------------------------------------------------------


@torch.library.custom_op("npub::ada_layer_norm_fwd", mutates_args=())
def _ada_layer_norm_wrapped(
    M: int,
    N: int,
    eps: float,
    dtype_str: str,
    block_m: int,
    use_cp_async: bool,
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> torch.Tensor:
    dummy = torch.empty(1, dtype=x.dtype, device=x.device)
    return _ada_layer_norm_kernel(
        M,
        N,
        eps,
        dtype_str,
        has_gate=False,
        use_cp_async=use_cp_async,
    )(block_m)(x, scale, shift, dummy)


@_ada_layer_norm_wrapped.register_fake
def _(M, N, eps, dtype_str, block_m, use_cp_async, x, scale, shift):
    return torch.empty((M, N), dtype=x.dtype, device=x.device)


@torch.library.custom_op("npub::ada_layer_norm_zero_fwd", mutates_args=())
def _ada_layer_norm_zero_wrapped(
    M: int,
    N: int,
    eps: float,
    dtype_str: str,
    block_m: int,
    use_cp_async: bool,
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    dummy = torch.empty(1, dtype=x.dtype, device=x.device)
    return _ada_layer_norm_kernel(
        M,
        N,
        eps,
        dtype_str,
        has_gate=True,
        use_cp_async=use_cp_async,
    )(block_m)(x, scale, shift, gate, dummy)


@_ada_layer_norm_zero_wrapped.register_fake
def _(M, N, eps, dtype_str, block_m, use_cp_async, x, scale, shift, gate):
    return torch.empty((M, N), dtype=x.dtype, device=x.device)


# ---------------------------------------------------------------------------
# Kernel class (K5-K9 adaptations)
# ---------------------------------------------------------------------------


class AdaLayerNormKernel(Kernel):
    """Adaptive LayerNorm kernel.

    Supports both AdaLN and AdaLN-Zero variants via the `has_gate` parameter.
    Uses 256-element alignment (512 bytes for fp16/bf16) for shared memory copies.

    NPU adaptations:

    - K5: ``supported_archs = None`` (all architectures).
    - K8: autotune removed (no ``autotune_configs``, no ``tune`` parameter);
      heuristic config selection only, ``init_config(config)``.
    - K9: ``threads`` removed from ``default_config`` and the ``forward``
      dispatch.

    Args:
        M: Number of rows (product of all dims except last).
        N: Hidden dimension (last dim).
        eps: Epsilon for numerical stability.
        dtype: Data type (float32, float16, or bfloat16).
        has_gate: If True, uses the AdaLN-Zero variant with gating.
        config: Optional kernel config override.
    """

    # K5: [80, 86, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

    # Developer-form vector kernel; invocations are scoped to Developer
    # mode regardless of ambient TILELANG_ASCEND_MODE (CG-2026-0010).
    ascend_mode = "Developer"

    def __init__(
        self,
        M: int,
        N: int,
        eps: float,
        dtype: torch.dtype,
        has_gate: bool = False,
        config: Optional[dict] = None,
    ):
        super().__init__()
        self.M = M
        self.N = N
        self.eps = eps
        self.dtype = dtype
        self.has_gate = has_gate
        self.N_padded = _align_up(N, ALIGNMENT)
        # Shape policy is benchmarked independently from block/thread tuning.
        self.use_cp_async = _should_use_cp_async(N, dtype, has_gate)
        self.kernel = _ada_layer_norm_kernel(
            self.M,
            self.N,
            self.eps,
            self.dtype_str,
            has_gate=self.has_gate,
            use_cp_async=self.use_cp_async,
        )
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        """Default config (K9: ``threads`` key removed; switch-block paired)."""
        return _select_row_config(self.M, self.N, self.dtype)

    def forward(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
        gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the AdaLN (or AdaLN-Zero) kernel.

        K9: ``threads`` removed from the dispatch.
        """
        if self.has_gate:
            if gate is None:
                raise ValueError("gate tensor is required when has_gate=True")
            return _ada_layer_norm_zero_wrapped(
                self.M,
                self.N,
                self.eps,
                self.dtype_str,
                self.config["block_m"],
                self.use_cp_async,
                x,
                scale,
                shift,
                gate,
            )
        else:
            return _ada_layer_norm_wrapped(
                self.M,
                self.N,
                self.eps,
                self.dtype_str,
                self.config["block_m"],
                self.use_cp_async,
                x,
                scale,
                shift,
            )
