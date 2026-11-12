"""Ascend 950 channel-first convolution kernels."""

from tileops._kernels.ascend.convolution.forward import (
    AscendConv2dKernel,
    AscendConv3dKernel,
)

__all__ = ["AscendConv2dKernel", "AscendConv3dKernel"]
