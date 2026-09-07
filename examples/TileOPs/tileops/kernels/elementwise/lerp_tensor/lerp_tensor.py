"""LerpTensor forward kernel using TileLang (NPU-adapted).

Implements the Tensor-weight overload of ``torch.lerp``:
    out = input + weight * (end - input)

The Op layer pre-broadcasts ``input`` / ``end`` / ``weight`` to the flat
output shape, so the kernel sees three contiguous 1-D tensors of size
``N`` and writes one contiguous 1-D output.

Adaptation summary (GPU -> NPU), per docs/gpu_to_npu_adaptation.md §3:

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang kernel factory (``_make_lerp_tensor_kernel``) is
    extracted from the GPU repo via ``extract_tl_kernel.py`` and imported
    as-is.  It serves as the reference for the NPU kernel component to
    reimplement for ``target="npuir"``.  K1-K4 and K11 adaptations
    (decorator, grid/sync, ``threads``/``npt`` collapse, padding strategy)
    are handled by the NPU component during re-implementation:
      - E1: factory signature drops ``output_dtype`` / ``is_fp8`` /
        ``threads`` / ``npt`` -> ``_make_lerp_tensor_kernel(N, dtype)``.
      - E2: the returned callable takes a single ``block_size``
        (the GPU ``threads * npt`` product) instead of
        ``(threads_arg, npt_arg)``.
      - E3: ``T.Kernel(T.ceildiv(N, block_size), is_npu=True)`` replaces
        the CUDA ``threads=`` dimension.
      - E4: ``T.Parallel(block_size)`` replaces
        ``T.Parallel(threads_arg, npt_arg)``.

  **Part B -- Kernel class** (fully ported from GPU ``LerpTensorFwdKernel``):
    E5:  ``supported_archs = None`` (was ``[80, 86, 89, 90]``).
    E6:  GPU ``ParametricUnaryKernel`` base inlined (the NPU project does
         not ship that base class); config selection, factory call, and
         ``forward`` dispatch live directly on ``LerpTensorFwdKernel``.
    E7/E8: GPU config ``{"threads": 512, "num_per_thread": npt}`` collapsed
         into ``{"block_size": threads * npt}`` (K11).
    E9:  ``autotune_configs`` / ``autotune()`` / ``tune`` param -- removed;
         heuristic config selection only.
    E10: no ``@torch.library.custom_op`` wrapper (the GPU kernel has none
         either); ``forward`` calls the compiled kernel directly.
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
# .lerp_tensor_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
# from .lerp_tensor_kernel import _make_lerp_tensor_kernel
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
from .lerp_tensor_kernel.perf_opt._make_lerp_tensor_kernel import _make_lerp_tensor_kernel

__all__ = ["LerpTensorFwdKernel"]

# ---------------------------------------------------------------------------
# K11 constants: GPU ``threads * npt`` defaults collapsed into ``block_size``.
#
# The GPU ``LerpTensorFwdKernel`` overrides ``ParametricUnaryKernel``'s
# ``_DEFAULT_THREADS`` to 512 (vs. the base's 256) and keeps the base's
# npt heuristic: npt = 4 for fp32, npt = 8 for fp16/bf16 (fp8 is not in
# the manifest contract for this op).  Per the E7/E8 rule
# ("block_size = the original threads * npt product"), the defaults are:
#   fp32:      512 * 4 = 2048
#   fp16/bf16: 512 * 8 = 4096
# (The 1024/2048 numbers in the adaptation guide's E7 illustrate the
# generic base-class default of 256 threads; this kernel's GPU source
# of truth runs 512 threads, so the faithful product is 2048/4096.)
# ---------------------------------------------------------------------------

_BLOCK_SIZE_FP32 = 8192  # Stage 4 tuned (perf_opt): 2048 -> 8192, 16M fp32 -3.1%
_BLOCK_SIZE_NON_FP32 = 4096  # GPU threads(512) * npt(8)


# ---------------------------------------------------------------------------
# Kernel class (E5-E10 adaptations)
# ---------------------------------------------------------------------------


class LerpTensorFwdKernel(Kernel):
    """Tensor-weight lerp: out = input + weight * (end - input).

    Implements the Tensor-weight overload of ``torch.lerp`` —
    ``torch.lerp(input, end, weight: Tensor)`` — where all three operands
    are float tensors of the same dtype broadcast together by the Op
    layer to a flat ``N``-element view.

    Manifest declares ``float16 | bfloat16 | float32``; fp8 is rejected
    at construction. The Op layer is responsible for broadcasting the
    three inputs to ``N_total`` before dispatch.

    NPU adaptations:

    - E5: ``supported_archs = None`` (all architectures).
    - E6: ``ParametricUnaryKernel`` logic inlined (NPU project has no
      such base class).
    - E7/E8 (K11): GPU ``threads * npt`` product collapsed into a single
      ``block_size`` config key; ``default_config`` returns
      ``{"block_size": 2048}`` (fp32) or ``{"block_size": 4096}``
      (fp16/bf16).
    - E9: autotune removed; heuristic config selection only.
      ``init_config(config)`` takes no ``tune`` argument.
    - E10: no custom_op wrapper; ``forward`` dispatches directly (the
      GPU kernel class also has no wrapper).

    Args:
        N_total: Total number of (post-broadcast, flattened) elements.
        dtype: Data type (float16, bfloat16, or float32).
        config: Optional kernel configuration dict (e.g.
            ``{"block_size": 4096}``).
    """

    # E5: [80, 86, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

    # Manifest: float16 | bfloat16 | float32 (fp8 rejected, GPU parity).
    SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

    def __init__(
        self,
        N_total: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
    ):
        super().__init__()
        if dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports dtypes [{supported}], got {dtype}"
            )
        self.N_total = N_total
        self.dtype = dtype
        self.output_dtype = dtype  # same_as(input); fp8 output path skipped
        # Build the factory callable (does not compile yet — compilation
        # is deferred to forward()).  The imported function is the GPU
        # implementation; the NPU component rewrites it for
        # target="npuir" with the E1 signature ``(N, dtype)``.
        self.kernel = _make_lerp_tensor_kernel(self.N_total, self.dtype_str)
        self._compiled_fn = None
        self._compiled_block_size: Optional[int] = None
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        """Return the default config (K11: block_size = threads * npt).

        GPU defaults (LerpTensorFwdKernel): threads=512, npt=4 (fp32) or
        npt=8 (fp16/bf16).  Collapsed per E7/E8:
        block_size = 2048 (fp32) or 4096 (fp16/bf16).
        """
        # K11: collapse threads * npt into block_size.
        if self.dtype == torch.float32:
            return {"block_size": _BLOCK_SIZE_FP32}
        return {"block_size": _BLOCK_SIZE_NON_FP32}

    def forward(self, a: torch.Tensor, b: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Run the lerp_tensor kernel.

        Accepts three contiguous 1-D tensors of ``N_total`` elements.
        E10: direct kernel call (no custom_op wrapper, GPU parity).  The
        compiled callable is built lazily with the configured
        ``block_size`` (E2: the NPU-reimplemented factory callable takes
        a single ``block_size`` argument) and cached per block_size.
        """
        block_size = self.config["block_size"]
        if self._compiled_fn is None or self._compiled_block_size != block_size:
            self._compiled_fn = self.kernel(block_size)
            self._compiled_block_size = block_size
        return self._compiled_fn(a, b, w)
