"""NPUIR RoPE examples adapted from TileOPs PR #35, commit 86b5b058.

The original package exports/registration are replaced by these local exports
and ``ops.py``. ``rotation.py``, ``staged.py`` and ``test_rope.py`` provide the
kernel dispatch, staged vector implementation and standalone regression suite.

Use TILELANG_ASCEND_MODE=Developer and an installed NPUIR TileLang build.
"""

from .ops import (
    LongRoPEFwdOp,
    RoPEFwdOp,
    RoPELlama31FwdOp,
    RoPENeoxPositionIdsFwdOp,
    YaRNFwdOp,
)
from .rotation import apply_rotation

__all__ = [
    "LongRoPEFwdOp",
    "RoPEFwdOp",
    "RoPELlama31FwdOp",
    "RoPENeoxPositionIdsFwdOp",
    "YaRNFwdOp",
    "apply_rotation",
]
