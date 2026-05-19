"""Symbolic variable helpers exposed on the TileLang language surface.

Backported from upstream tile-ai/tilelang for mlir-ascend so kernels that
declare dynamic shape dimensions via ``T.dynamic("batch")`` can be parsed.
"""

from __future__ import annotations

import re

from tvm import tir

from tilelang.utils import deprecated

__all__ = ["dynamic", "symbolic"]


def dynamic(name: str, dtype: str = "int32"):
    """Create one or more TIR dynamic symbolic variables.

    Accepts comma- or whitespace-separated names to declare several vars
    at once: ``B, M, N = T.dynamic("B, M, N")``.
    """
    if "," in name:
        names = re.split(r"\s*,\s*", name)
        return tuple(tir.Var(n, dtype) for n in names)
    if " " in name:
        names = re.split(r"\s+", name)
        return tuple(tir.Var(n, dtype) for n in names)
    return tir.Var(name, dtype)


@deprecated("T.symbolic(...)", "T.dynamic(...)")
def symbolic(name: str, dtype: str = "int32"):
    """Deprecated alias for :func:`dynamic`."""
    return dynamic(name, dtype)
