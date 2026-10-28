"""Ascend 910 GEMM surface.

The legacy A2 frontend exposes ``T.gemm`` as ``T.npuir_dot`` itself (see
``tilelang/language/__init__.py`` on the ``main`` branch::

    from .customize_npuir import npuir_dot, npuir_dot as gemm

so ``T.gemm(A, B, C, size=..., initC=..., a_transpose=..., b_transpose=...)``
maps one-to-one onto ``tl.npuir_dot``.  In particular ``initC`` is forwarded
*unchanged*: it is either a Python bool or a runtime predicate such as
``k == 0``, and the codegen emits the L0C clear/accumulate flag from it
(``arith.cmpi eq, %arg, %c0`` vs ``arith.constant false``).

This module keeps that contract and additionally tolerates the CUDA-style
spellings (``transpose_A``/``transpose_B``/``clear_accum``/``policy``/
``k_pack``/``wg_wait``) used by a few library-level helpers.
"""

from __future__ import annotations

from typing import Optional

from tvm import tirx as tir

from . import npuir

__all__ = ["gemm", "dot"]


def gemm(
    A,
    B,
    C,
    size: Optional[list] = None,
    initC=False,
    a_transpose: bool = False,
    b_transpose: bool = False,
    **kwargs,
):
    """Emit ``tl.npuir_dot`` (C = A @ B, optionally accumulating into C).

    Args:
        A, B: input tiles (buffers or regions).
        C: accumulator tile.
        size: optional ``[m, k, n]`` override.
        initC: whether the L0C accumulator starts from zero.  May be a
            :class:`tir.PrimExpr` such as ``k == 0``; it is forwarded as-is.
        a_transpose, b_transpose: layout flags (Python bools).
    """

    if "transpose_A" in kwargs:
        a_transpose = kwargs.pop("transpose_A")
    if "transpose_B" in kwargs:
        b_transpose = kwargs.pop("transpose_B")
    if "clear_accum" in kwargs:
        clear_accum = kwargs.pop("clear_accum")
        if initC is False:
            initC = clear_accum
    for unused in ("policy", "k_pack", "wg_wait"):
        kwargs.pop(unused, None)
    if kwargs:
        raise TypeError(f"T.gemm got unexpected keyword arguments: {sorted(kwargs)}")

    return npuir.npuir_dot(
        A,
        B,
        C,
        size=size,
        initC=initC,
        a_transpose=a_transpose,
        b_transpose=b_transpose,
    )


# ``main`` exposes the same intrinsic under both names for the A2 target.
dot = gemm