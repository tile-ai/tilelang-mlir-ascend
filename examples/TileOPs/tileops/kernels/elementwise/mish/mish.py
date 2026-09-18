"""Mish forward kernel using TileLang (NPU-adapted).

Implements the elementwise Mish activation:
    y = x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))

Adaptation summary (GPU -> NPU):

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang kernel function (``mish_fwd_kernel``) is extracted
    from the GPU repo via ``extract_tl_kernel.py`` and imported as-is.
    It serves as the reference for the NPU kernel component to
    reimplement for ``target="npuir"``.  K1-K4 and K11 adaptations
    (decorator, grid/sync, ``threads``/``npt`` removal, padding strategy)
    are handled by the NPU component during re-implementation.

  **Part B -- custom_op wrapper + Kernel class** (fully ported):
    K5:  ``supported_archs = None`` (was ``[80, 86, 89, 90]``).
    K7:  ``custom_op("npub::...")`` (was ``"top::..."``).
    K8:  ``autotune_configs`` / ``autotune()`` / ``tune`` param -- removed.
    K9:  ``threads`` removed from ``default_config`` and ``forward`` call.
    K11: ``threads * npt`` collapsed into a single ``block_size`` param
         (elementwise ops only).

  **Kernel source selection** (baseline vs perf_opt):
    The kernel factory is imported from exactly one of two paths via the
    comment toggle below: the Stage 3 baseline (``mish_kernel/mish.py``)
    or the Stage 4 tuned version (``mish_kernel/perf_opt/mish.py``).
    ``pytest tests/ops/test_mish.py`` and
    ``pytest benchmarks/ops/bench_mish.py`` therefore exercise whichever
    source is active.
"""

from typing import Optional

import torch

from tileops.kernels.kernel_base import Kernel

# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source is active once the tuned kernel
# (``mish_kernel/perf_opt/mish.py``, tuned on Ascend 910B2C -- see
# ``mish_kernel/perf_opt/opt_log.md``: yolo-p3 fp16 103.42 -> 78.30 us,
# -24.3%) has passed its L0/L1 regression; the baseline (Stage 3) source
# is active otherwise.
#
# NOTE: each source block also pins ``_DEFAULT_BLOCK_SIZE`` -- the two
# kernels ship different tuned block_size defaults (Stage 3 heuristic
# 1024/2048 vs perf_opt 8192), so the assignment must be toggled together
# with its import.  ``pytest tests/ops/test_mish.py`` and
# ``pytest benchmarks/ops/bench_mish.py`` dispatch through whichever
# source is active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
# from .mish_kernel.mish import mish_fwd_kernel
# _DEFAULT_BLOCK_SIZE = {"float32": 1024, "float16": 2048, "bfloat16": 2048}
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
from .mish_kernel.perf_opt.mish import (
    BLOCK_SIZE_CAST,
    BLOCK_SIZE_FP32,
    mish_fwd_kernel,
)

_DEFAULT_BLOCK_SIZE = {
    "float32": BLOCK_SIZE_FP32,
    "float16": BLOCK_SIZE_CAST,
    "bfloat16": BLOCK_SIZE_CAST,
}

__all__ = ["MishFwdKernel"]


# ---------------------------------------------------------------------------
# custom_op wrapper (K7: top:: -> npub::, K9: threads removed, K11: block_size)
# ---------------------------------------------------------------------------


@torch.library.custom_op("npub::mish_fwd", mutates_args=())
def _mish_fwd_wrapped(
    N: int,
    dtype_str: str,
    block_size: int,
    x: torch.Tensor,
) -> torch.Tensor:
    return mish_fwd_kernel(N, dtype_str, block_size=block_size)(x, N)


@_mish_fwd_wrapped.register_fake
def _(N: int, dtype_str: str, block_size: int, x: torch.Tensor):
    return torch.empty((N,), dtype=x.dtype, device=x.device)


# ---------------------------------------------------------------------------
# Kernel class (K5-K9, K11 adaptations)
# ---------------------------------------------------------------------------


class MishFwdKernel(Kernel):
    """Mish forward kernel: y = x * tanh(softplus(x)).

    A single-input, single-output elementwise activation.  The kernel
    flattens the input to a 1-D vector of ``N_total`` elements and
    applies the pointwise Mish formula.

    NPU adaptations:

    - K5: ``supported_archs = None`` (all architectures).
    - K8: autotune removed; heuristic config selection only.
      ``init_config(config)`` takes no ``tune`` argument.
    - K9: ``threads`` removed from ``default_config`` and ``forward``.
    - K11: GPU ``threads * npt`` product collapsed into ``block_size``.

    Args:
        N_total: Total number of elements (flattened input).
        dtype: Data type (float16, bfloat16, or float32).
        config: Optional kernel configuration dict (e.g.
            ``{"block_size": 2048}``).
    """

    # K5: [80, 86, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

    # Developer-form vector kernel; invocations are scoped to Developer
    # mode regardless of ambient TILELANG_ASCEND_MODE (CG-2026-0010).
    ascend_mode = "Developer"

    # Float-only dtypes (inherited from GPU FloatUnaryKernel).
    SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

    def __init__(
        self,
        N_total: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
    ):
        super().__init__()
        if self.SUPPORTED_DTYPES is not None and dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports dtypes [{supported}], got {dtype}"
            )
        self.N_total = N_total
        self.dtype = dtype
        self.output_dtype = dtype  # same_as(input)
        # Build the factory callable (does not compile yet — compilation
        # is deferred to forward() via the custom_op wrapper).  The
        # factory comes from the ACTIVE kernel source (baseline or
        # perf_opt, see the source-selection toggle at the top of this
        # file); forward() rebuilds it with the configured block_size.
        self.kernel = mish_fwd_kernel(self.N_total, self.dtype_str)
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        """Return the default config (K11: block_size = threads * npt).

        The default block_size follows the ACTIVE kernel source (see the
        source-selection toggle at the top of this file): the Stage 3
        heuristic (1024 fp32 / 2048 fp16-bf16) or the Stage 4 perf_opt
        tuned value (8192, taken from the tuned module constants).

        GPU defaults: threads=256, npt=4 (fp32) or npt=8 (fp16/bf16).
        Collapsed: block_size = 1024 (fp32) or 2048 (fp16/bf16).
        """
        # K11: collapse threads * npt into block_size.
        return {"block_size": _DEFAULT_BLOCK_SIZE[self.dtype_str]}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the Mish kernel.

        Accepts a 1-D contiguous tensor of ``N_total`` elements.
        K9: ``threads`` removed from the call; K11: ``block_size`` used.
        """
        return _mish_fwd_wrapped(
            self.N_total,
            self.dtype_str,
            self.config["block_size"],
            x,
        )
