# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""
TileLangIR transforms: transformation passes by dialect.

- mlir: canonicalize, cse, sccp
- tilelangir: insert_cross_core_scope, vectorize
- bishengir: adapt_triton_kernel
"""

from . import mlir, tilelangir, bishengir

__all__ = ["mlir", "tilelangir", "bishengir"]
