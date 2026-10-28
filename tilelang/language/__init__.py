"""Default TileLang language facade.

``tilelang.language`` re-exports the CUDA dialect so that ``import
tilelang.language as T`` yields the common surface plus CUDA extensions. Other
backends are reached explicitly via ``tilelang.<backend>.language`` (which build
on ``tilelang.language.common``).
"""

from __future__ import annotations

from tilelang.cuda.language import *  # noqa: F401,F403
from tilelang.cuda.language import __all__ as __all__  # noqa: F401

# Imported by name so static type checkers resolve the CUDA-typed signatures
# through this facade (they cannot evaluate the dynamic __all__).
from tilelang.cuda.language import (  # noqa: F401
    Kernel,
    Parallel,
    Unroll,
    atomic_add,
    copy,
    gemm,
    gemm_blockscaled,
    gemm_sp,
    im2col,
    reduce_absmax,
    reduce_max,
    reduce_min,
    unroll,
)

__tilelang_dialect__ = "cuda"
# ---- Ascend 910 (A2) compatibility facade -------------------------------
# Examples use `import tilelang.language as T` and select A2 with
# `T.Kernel(..., is_npu=True)`; keep the CUDA surface intact and dispatch only
# while an A2 kernel frame is active.
from tilelang.cuda.language import Kernel as _CoreKernel  # noqa: E402
from tilelang.language.copy_op import copy as _CoreCopy  # noqa: E402
from tilelang.language.allocate import alloc_shared as _core_alloc_shared  # noqa: E402


def Kernel(*blocks, is_npu: bool = False, **kwargs):  # noqa: F811
    if is_npu:
        from tilelang.ascend910.language.kernel import Kernel as _A2Kernel

        return _A2Kernel(*blocks, is_npu=True, **kwargs)
    return _CoreKernel(*blocks, **kwargs)


def _a3_active() -> bool:
    """Whether the Ascend 910 (A3) language surface should handle the call.

    True inside an ``is_npu=True`` kernel frame *or* while lowering with the
    ``ascend910`` target: the legacy A2 examples also use plain
    ``T.Kernel(...)`` and rely on ``T.copy`` lowering to ``tl.copy`` with
    ``tl.region`` operands (the v0.1.15 default routes to ``tl.tileop.copy``,
    which the A3 code generators do not implement).
    """

    from tilelang.ascend910.language.kernel import is_active

    if is_active():
        return True
    try:
        from tvm.target import Target

        current = Target.current(allow_none=True)
    except Exception:
        return False
    return current is not None and current.kind.name == "ascend910"


def copy(*args, **kwargs):  # noqa: F811
    if _a3_active():
        from tilelang.ascend910.language.copy import copy as _a2_copy

        return _a2_copy(*args, **kwargs)
    return _CoreCopy(*args, **kwargs)


def print(obj=None, msg="", **kwargs):  # noqa: F811,A001
    """Route ``T.print`` to the Ascend dialect while lowering an A2 kernel.

    The CUDA/default implementation gates the print on ``threadIdx``, which the
    ``ascend910`` target rejects ("no SIMT threads").  The Ascend dialect prints
    from every core, matching the A2 execution model.
    """

    if _a3_active():
        from tilelang.ascend.language.print import print as _ascend_print

        return _ascend_print(obj, msg)
    from tilelang.cuda.language.print import print as _core_print

    return _core_print(obj, msg, **kwargs)


def alloc_ub(shape, dtype, scope="shared"):
    return _core_alloc_shared(shape, dtype, scope=scope)


_A2_OP_NAMES = [
    "npuir_abs",
    "npuir_add",
    "npuir_and",
    "npuir_arange",
    "npuir_atomic_add",
    "npuir_bitcast",
    "npuir_brc",
    "npuir_cast",
    "npuir_cmp",
    "npuir_concat",
    "npuir_cumsum",
    "npuir_deinterleave",
    "npuir_div",
    "npuir_dot",
    "npuir_exp",
    "npuir_exp2",
    "npuir_flip",
    "npuir_gather",
    "npuir_interleave",
    "npuir_ln",
    "npuir_load_nd2nz",
    "npuir_max",
    "npuir_min",
    "npuir_mul",
    "npuir_not",
    "npuir_or",
    "npuir_pad",
    "npuir_pow",
    "npuir_rec",
    "npuir_reduce",
    "npuir_relu",
    "npuir_reshape",
    "npuir_rsqrt",
    "npuir_select",
    "npuir_shl",
    "npuir_shr",
    "npuir_sigmoid",
    "npuir_sort",
    "npuir_sqrt",
    "npuir_store_fixpipe",
    "npuir_store_nz2nd",
    "npuir_sub",
    "npuir_transpose",
    "npuir_vcos",
    "npuir_verf",
    "npuir_vsin",
    "npuir_vtanh",
    "npuir_xor",
]


def _make_a2_op(name):
    def _op(*args, **kwargs):
        from tilelang.ascend910.language import npuir

        return getattr(npuir, name)(*args, **kwargs)

    _op.__name__ = name
    return _op


for _name in _A2_OP_NAMES:
    globals()[_name] = _make_a2_op(_name)

_A2_ALIASES = {
    "vadd": "npuir_add",
    "vsub": "npuir_sub",
    "vmul": "npuir_mul",
    "vdiv": "npuir_div",
    "vmax": "npuir_max",
    "vmin": "npuir_min",
    "vor": "npuir_or",
    "vand": "npuir_and",
    "vxor": "npuir_xor",
    "vpow": "npuir_pow",
    "vshl": "npuir_shl",
    "vshr": "npuir_shr",
    "vexp": "npuir_exp",
    "vexp2": "npuir_exp2",
    "vrelu": "npuir_relu",
    "vsigmoid": "npuir_sigmoid",
    "vln": "npuir_ln",
    "vlog2": "npuir_log2",
    "vsqrt": "npuir_sqrt",
    "vrsqrt": "npuir_rsqrt",
    "vabs": "npuir_abs",
    "vrec": "npuir_rec",
    "vnot": "npuir_not",
    "vselect": "npuir_select",
    "vcmp": "npuir_cmp",
    "vdot": "npuir_dot",
    "vbrc": "npuir_brc",
    "vcast": "npuir_cast",
    "vreduce": "npuir_reduce",
    "vcumsum": "npuir_cumsum",
    "vsort": "npuir_sort",
    "vtranspose": "npuir_transpose",
    "vinterleave": "npuir_interleave",
    "vdeinterleave": "npuir_deinterleave",
    "varange": "npuir_arange",
    "vconcat": "npuir_concat",
    "vpad": "npuir_pad",
    "vflip": "npuir_flip",
    "vbitcast": "npuir_bitcast",
    "vcos": "npuir_vcos",
    "vsin": "npuir_vsin",
    "verf": "npuir_verf",
    "vtanh": "npuir_vtanh",
}
for _alias, _target in _A2_ALIASES.items():
    globals()[_alias] = _make_a2_op(_target)


from tilelang.ascend910.language.gemm import dot as dot, gemm as gemm  # noqa: E402,F811
from tilelang.ascend910.language.npuir import Scope, resource_specialize, rs  # noqa: E402,F401
from tilelang.ascend910.language.npuir import (  # noqa: E402,F401
    block_barrier,
    pipe_barrier,
    set_flag,
    subblock_barrier,
    sync_block_set,
    sync_block_wait,
    wait_flag,
)

from tilelang.ascend910.language import (  # noqa: E402,F401
    alloc_L0A,
    alloc_L0B,
    alloc_L0C,
    alloc_L1,
)

from tilelang.ascend910.language.npuir import (  # noqa: E402,F401,F811
    npuir_reduce as reduce,
    reduce_absmax,
    reduce_abssum,
    reduce_max,
    reduce_min,
    reduce_sum,
)

from tilelang.ascend910.language.npuir import (  # noqa: E402,F401
    npuir_load_nd2nz as load_nd2nz,
    npuir_store_fixpipe as store_fixpipe,
    npuir_store_nz2nd as store_nz2nd,
)

from tilelang.ascend910.language.npuir import atomic_add  # noqa: E402,F811,F401

from tilelang.ascend910.language.npuir import (  # noqa: E402,F811,F401
    npuir_arange as arange,
    npuir_clamp as vclamp,
)

from tilelang.ascend910.language.npuir import (  # noqa: E402,F401,F811
    npuir_clear as clear,
    npuir_fill as fill,
)

# Legacy ``npuir_*`` spellings stay importable next to their v-prefix aliases
# (the rebase renamed the public surface to ``clear``/``fill``/``atomic_add``).
from tilelang.ascend910.language.npuir import (  # noqa: E402,F401
    npuir_atomic_addx4,
    npuir_clear,
    npuir_fill,
)

# v-prefix aliases used by the A2 examples.
from tilelang.ascend910.language.npuir import (  # noqa: E402,F401,F811
    npuir_atomic_addx4 as atomic_addx4,
)
from tilelang.ascend910.language.npuir import (  # noqa: E402,F401,F811
    npuir_deinterleave as deinterleave,
)
from tilelang.ascend910.language.npuir import (  # noqa: E402,F401,F811
    npuir_interleave as interleave,
)

from tilelang.ascend910.language.npuir import npuir_reshape as reshape  # noqa: E402,F811,F401
