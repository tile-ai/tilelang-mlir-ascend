"""Standalone NPUIR Grouped GEMM adapted from TileOPs PR #21."""

from .grouped import AscendGroupedGemmKernel, GroupedGemmFwdOp

__all__ = ["AscendGroupedGemmKernel", "GroupedGemmFwdOp"]
