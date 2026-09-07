"""Workload definitions for the elementwise op family."""

from math import prod

import torch

from tileops.device import get_device_backend
from tileops.workloads.workload_base import RandnWorkload


class ShapedRandnWorkload(RandnWorkload):
    """One ``randn`` tensor of arbitrary rank, with its element count.

    Adaptation point: ``device`` is resolved from :func:`get_device_backend`
    via ``RandnWorkload.gen_inputs`` instead of hard-coded ``"cuda"``.
    """

    def __init__(self, shape: tuple, dtype):
        super().__init__(tuple(shape), dtype)
        self.n_total = prod(self.shape)


class MishWorkload(ShapedRandnWorkload):
    """Workload definition for MishFwdOp (shape + dtype, randn input)."""


class LerpTensorManifestWorkload:
    """Workload definition for LerpTensorFwdOp (Tensor-weight torch.lerp).

    Ported from TileOPs ``workloads/elementwise.py``.  All three operands
    (``input`` / ``end`` / ``weight``) share the manifest ``input_shape``;
    the Op layer performs the 3-way broadcast.  ``weight`` uses ``rand``
    (interpolation weights are naturally in [0, 1]) while ``input`` /
    ``end`` use ``randn``.

    Adaptation point W1: ``device`` is resolved from
    :func:`get_device_backend` instead of hard-coded ``"cuda"``.
    """

    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype):
        self.input_shape = shape
        self.end_shape = shape
        self.weight_shape = shape
        self.shape = shape
        self.n_total = prod(shape)
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = get_device_backend().name
        x = torch.randn(self.input_shape, device=device, dtype=self.dtype)
        end = torch.randn(self.end_shape, device=device, dtype=self.dtype)
        weight = torch.rand(self.weight_shape, device=device, dtype=self.dtype)
        return x, end, weight
