"""Multi-head attention forward kernel (NPU-adapted wrapper + Kernel class).

MHA forward (prefill) is flash-attention style online-softmax attention
over BSHD-layout q/k/v tensors with ``heads_kv == heads``:

    o = softmax(scale * q @ k^T + causal_mask) @ v    (shape (B, S, H, D))
    lse = per-row log-sum-exp                          (shape (B, H, S))

The GPU op routes MHA through the GQA prefill dispatcher
(``MultiHeadAttentionFwdOp`` -> ``GroupedQueryAttentionFwdOp`` ->
``GQAPrefillFwdKernel``); the NPU port keeps the same kernel class, the
same kernel key (``gqa_prefill_fwd_kernel``) and the historical MHA
return contract ``(output, lse)``.

Adaptation summary (GPU -> NPU):

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang kernel factory (``_gqa_prefill_fwd_kernel``) is
    extracted from the GPU repo via ``extract_tl_kernel.py`` and imported
    as-is.  It serves as the reference for the NPU kernel component to
    reimplement for ``target="npuir"``.  K1-K4 adaptations (decorator,
    grid/sync semantics, ``threads`` removal, boundary-mask strategy) are
    handled by the NPU component during re-implementation.

  **Part B -- custom_op wrapper + Kernel class** (fully ported):
    K5: ``supported_archs = None`` (was ``[80, 89, 90]``).
    K7: ``custom_op("npub::...")`` (was ``"top::..."``).
    K8: ``autotune_configs`` (``tile_stage_thread_configs``) / ``tune``
        param -- removed; heuristic config selection only.
    K9: ``threads`` removed from the wrapper signature,
        ``default_config`` and the ``forward`` call.

  **Scope note (K5)**: the GPU kernel selection (``gqa_prefill_fwd_kernel``
  -> ``GQAPrefillFwdKernel`` | ``GQAPrefillFwdWsPersistentCausalKernel``
  and the H200 square dense fast path
  ``gqa_prefill_square_fwd_kernel`` -> ``GQAFwdWsPersistentCausalKernel``)
  is GPU arch specialization; NPU collapses to the architecture-generic
  dense prefill kernel.
"""

from typing import Optional, Tuple

import torch

from tileops.kernels.kernel_base import Kernel

# Part A: the GPU TileLang kernel factory extracted by extract_tl_kernel.py
# (single output file ``_multi_head_attention_fwd_kernels.py``, shared by all
# per-kernel migration prompts for this op).  It is the GPU (CUDA-target)
# implementation imported as-is -- the reference the NPU kernel component
# rewrites for ``target="npuir"``.  A later Stage 5 integration replaces this
# import with its baseline/perf_opt source-selection block.
# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .multi_head_attention_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
# from .multi_head_attention_kernel import _gqa_prefill_fwd_kernel
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
from .multi_head_attention_kernel.perf_opt._gqa_prefill_fwd_kernel import _gqa_prefill_fwd_kernel

__all__ = ["GQAPrefillFwdKernel"]


# ---------------------------------------------------------------------------
# custom_op wrapper (K7: top:: -> npub::, K9: threads removed)
# ---------------------------------------------------------------------------


@torch.library.custom_op("npub::gqa_prefill_fwd_wrapped_kernel", mutates_args=())
def _gqa_prefill_fwd_wrapped_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len_q: int,
    seq_len_kv: int,
    dim: int,
    is_causal: bool,
    sm_scale: float,
    softcap: float,
    dtype: str,
    block_m: int,
    block_n: int,
    num_stages: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # K3/K9: the factory callable is invoked without ``threads`` (the CUDA
    # execution detail is dropped on NPU); the NPU kernel component removes
    # the ``threads`` parameter from the extracted GPU factory when
    # rewriting it for ``target="npuir"``.
    return _gqa_prefill_fwd_kernel(
        batch, heads, heads_kv, seq_len_q, seq_len_kv, dim, is_causal, sm_scale, softcap, dtype
    )(block_m, block_n, num_stages)(q, k, v)


@_gqa_prefill_fwd_wrapped_kernel.register_fake
def _(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len_q: int,
    seq_len_kv: int,
    dim: int,
    is_causal: bool,
    sm_scale: float,
    softcap: float,
    dtype: str,
    block_m: int,
    block_n: int,
    num_stages: int,
    *inputs: Tuple[torch.Tensor, ...],
) -> Tuple[torch.Tensor, torch.Tensor]:
    fake_o = torch.empty_like(inputs[0])
    fake_lse = fake_o.new_empty([batch, heads, seq_len_q])
    return fake_o, fake_lse


# ---------------------------------------------------------------------------
# Kernel class (K5-K9 adaptations)
# ---------------------------------------------------------------------------


class GQAPrefillFwdKernel(Kernel):
    """GQA/MHA prefill forward kernel (BSHD layout, flash-attention style).

    Computes online-softmax attention with a running max / rescaled
    accumulator over KV tiles, emitting both the attention output and the
    per-row log-sum-exp.  Handles non-tile-aligned ``seq_len_q`` /
    ``seq_len_kv`` via masked loads and predicated stores inside the
    kernel.

    NPU adaptations:

    - K5: ``supported_archs = None`` (all architectures; was ``[80, 89, 90]``).
    - K8: autotune removed; heuristic config selection only.
      ``init_config(config)`` takes no ``tune`` argument.
    - K9: ``threads`` removed from ``default_config`` and ``forward``.

    Args:
        batch: Batch size.
        heads: Number of query heads.
        heads_kv: Number of key/value heads (``heads % heads_kv == 0``).
        seq_len_q: Query sequence length.
        seq_len_kv: Key/value sequence length
            (``seq_len_q <= seq_len_kv`` required when causal).
        dim: Head dimension.
        is_causal: Whether to apply a causal mask.
        dtype: Data type (float16 or bfloat16).
        sm_scale: Softmax scale (default ``dim ** -0.5``).
        softcap: Score softcap (``0.0`` disables).
        config: Optional dict with "block_m", "block_n", "num_stages".

    Raises:
        ValueError: If ``heads % heads_kv != 0`` or causal with
            ``seq_len_q > seq_len_kv``.
    """

    # K5: [80, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

    # Two-phase Cube/Vector Expert-form kernel (Scope/alloc_L1/L0C/ub,
    # sync_block_set/wait): every invocation is scoped to Expert mode
    # regardless of ambient TILELANG_ASCEND_MODE (CG-2026-0010).
    ascend_mode = "Expert"

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seq_len_q: int,
        seq_len_kv: int,
        dim: int,
        is_causal: bool,
        dtype: torch.dtype,
        sm_scale: Optional[float] = None,
        softcap: float = 0.0,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.batch = batch
        self.heads = heads
        if heads % heads_kv != 0:
            raise ValueError("heads must be divisible by heads_kv")
        if is_causal and seq_len_q > seq_len_kv:
            raise ValueError("causal prefill requires seq_len_q <= seq_len_kv")
        self.heads_kv = heads_kv
        self.seq_len_q = seq_len_q
        self.seq_len_kv = seq_len_kv
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype
        self.sm_scale = dim**-0.5 if sm_scale is None else sm_scale
        self.softcap = softcap

        # Build the factory callable (no TileLang JIT compilation yet --
        # compilation happens when the returned func is invoked inside the
        # custom_op wrapper).  The imported function is the GPU
        # implementation; the NPU component rewrites it for
        # ``target="npuir"``.
        self.kernel = _gqa_prefill_fwd_kernel(
            self.batch,
            self.heads,
            self.heads_kv,
            self.seq_len_q,
            self.seq_len_kv,
            self.dim,
            self.is_causal,
            self.sm_scale,
            self.softcap,
            self.dtype_str,
        )

        self.init_config(config)

    @property
    def default_config(self) -> dict:
        """Default tiling config (GPU defaults minus the CUDA ``threads``).

        K9: ``threads`` (CUDA execution detail) dropped from the config
        dict; the tiling knobs ``block_m`` / ``block_n`` / ``num_stages``
        are preserved from the GPU defaults.

        NPU config-contract note (S5 integration, 2026-09-15): the
        integrated two-phase kernel (E6) reinterprets the legacy wrapper
        defaults ``(64, 64, 1)`` / ``(64, 32, 1)`` as the design default
        ``(64, 256)``; that traced variant exceeds the 192KB per-AIV UB
        budget on causal ``dim=128`` shapes (BiShengHIR "ub overflow",
        ~204KB required; DESIGN §4.5 budget 178.2KB underestimates the
        actual allocation).  Passing ``num_stages=2`` selects the kernel's
        verbatim-config path (E6 respects any non-default config
        verbatim), restoring the historical effective 64x64 tiling
        (``bn_eff = max(64, ceil16(ceildiv(S_kv, 15)))`` in {64, 144}) --
        the exact configuration Stage 3's L1 gate validated on the
        manifest causal workload domain.  ``num_stages`` is a reserved,
        structurally inert knob in this kernel (single-slot two-phase,
        stage-invariant per DESIGN R-7), so 1 -> 2 has no semantic
        effect on the traced kernel.
        """
        return {
            "block_m": 64,
            "block_n": 64 if self.dim <= 128 else 32,
            "num_stages": 2,
        }

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the GQA/MHA prefill kernel.

        Accepts BSHD-layout q ``(B, S_q, H, D)`` and k/v
        ``(B, S_kv, H_kv, D)``.  Returns ``(output, lse)`` per the MHA
        return contract: ``output`` has the q shape and ``lse`` has shape
        ``[batch, heads, seq_len_q]`` in float32.

        K9: ``threads`` removed from the call.
        """
        return _gqa_prefill_fwd_wrapped_kernel(
            self.batch,
            self.heads,
            self.heads_kv,
            self.seq_len_q,
            self.seq_len_kv,
            self.dim,
            self.is_causal,
            self.sm_scale,
            self.softcap,
            self.dtype_str,
            self.config["block_m"],
            self.config["block_n"],
            self.config["num_stages"],
            q,
            k,
            v,
        )
