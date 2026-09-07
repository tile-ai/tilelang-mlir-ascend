"""LerpTensorFwdOp -- Tensor-weight lerp (NPU-adapted).

Adaptation from GPU (TileOPs) to NPU, per docs/gpu_to_npu_adaptation.md §3.4:

- E11 (O1): ``input.is_cuda and end.is_cuda and weight.is_cuda`` device
  check -> ``backend.is_device_tensor(...)`` for all three inputs.
- E12: ``_OP_REGISTRY`` / ``_wrapped`` custom_op wrapper /
  ``compile_boundary.register_instance`` removed — ``forward`` calls
  ``_eager_forward`` directly.
- E13 (O3): ``tune`` parameter removed; kernel constructor takes
  ``(N_total, dtype, config=None)``.
- E14: ``eval_roofline()`` formula identical to the GPU roofline func
  ``tileops.perf.formulas.lerp_tensor_fwd_roofline``:
  ``flops = 3 * N_total`` (sub + mul + add per element),
  ``bytes = 4 * N_total * elem_bytes`` (3 reads + 1 write).
- E15 (O6): 3-way broadcast -> flatten -> kernel dispatch -> reshape
  flow preserved unchanged.
"""

from __future__ import annotations

from math import prod
from typing import Dict, Optional

import torch

from tileops.device import get_device_backend
from tileops.kernels.elementwise.lerp_tensor import LerpTensorFwdKernel
from tileops.kernels.kernel_base import Kernel
from tileops.ops.op_base import Op

__all__ = ["LerpTensorFwdOp"]


class LerpTensorFwdOp(Op):
    """Tensor-weight lerp: out = input + weight * (end - input).

    Conforms to the Tensor-weight overload of ``torch.lerp`` —
    ``torch.lerp(input, end, weight: Tensor)`` where ``weight`` is a
    Tensor that broadcasts together with ``input`` and ``end`` to the
    output shape. The Op layer expands the three inputs to the broadcast
    shape and dispatches the flat ``LerpTensorFwdKernel`` on
    ``N_total = product(broadcast_shape)`` elements. The scalar-weight
    overload is handled separately by ``LerpFwdOp`` (not migrated).

    Args:
        input: Shape of the start tensor.
        end: Shape of the end tensor.
        weight: Shape of the per-element weight tensor.
        dtype: Torch dtype for all three operands.
        kernel_map: Optional kernel dispatch override.
    """

    _op_name = "lerp_tensor"

    # Manifest declares all three operands as ``float16 | bfloat16 | float32``;
    # fp8 dtypes are rejected at the op-layer signature so the impl matches
    # the manifest contract (the kernel also rejects fp8 independently).
    _SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

    def __init__(
        self,
        *,
        input: tuple,
        end: tuple,
        weight: tuple,
        dtype: torch.dtype,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        if dtype not in self._SUPPORTED_DTYPES:
            names = ", ".join(str(dt) for dt in self._SUPPORTED_DTYPES)
            raise ValueError(
                f"LerpTensorFwdOp does not support dtype {dtype}. Supported: [{names}]"
            )
        self.input_shape = tuple(input)
        self.end_shape = tuple(end)
        self.weight_shape = tuple(weight)
        self.dtype = dtype
        self.output_dtype = dtype  # same_as(input)
        self.out_shape = tuple(
            torch.broadcast_shapes(
                self.input_shape,
                self.end_shape,
                self.weight_shape,
            )
        )
        self.N_total = prod(self.out_shape) if self.out_shape else 1
        self.dispatch_kernel(kernel_map)
        # E13: ``tune`` removed; the kernel constructor takes
        # (N_total, dtype, config=None).
        self.kernel = self.kernel_map[self._op_name](
            self.N_total,
            dtype,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"lerp_tensor": LerpTensorFwdKernel}

    @staticmethod
    def _expand_flat(t: torch.Tensor, target_shape: tuple) -> torch.Tensor:
        """Expand ``t`` to ``target_shape`` and return a contiguous flat view."""
        if tuple(t.shape) != tuple(target_shape):
            t = t.expand(target_shape)
        return t.contiguous().view(-1)

    def _eager_forward(
        self,
        input: torch.Tensor,
        end: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        out_shape = self.out_shape if self.out_shape else (1,)
        a_flat = self._expand_flat(input, out_shape)
        b_flat = self._expand_flat(end, out_shape)
        w_flat = self._expand_flat(weight, out_shape)
        result = self.kernel(a_flat, b_flat, w_flat)
        return result.view(self.out_shape if self.out_shape else ())

    def forward(
        self,
        input: torch.Tensor,
        end: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        # E11: three-way device check via the device backend.
        backend = get_device_backend()
        if not (
            backend.is_device_tensor(input)
            and backend.is_device_tensor(end)
            and backend.is_device_tensor(weight)
        ):
            raise ValueError(f"Inputs must be {backend.name} tensors")
        for name, t, expected in [
            ("input", input, self.input_shape),
            ("end", end, self.end_shape),
            ("weight", weight, self.weight_shape),
        ]:
            if t.dtype != self.dtype:
                raise ValueError(f"Expected {name}.dtype {self.dtype}, got {t.dtype}")
            if tuple(t.shape) != expected:
                raise ValueError(f"Expected {name}.shape {expected}, got {tuple(t.shape)}")
        # E12: no _wrapped custom_op dispatch; call the eager path directly.
        return self._eager_forward(input, end, weight)

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for this op instance (E14).

        Mirrors the GPU roofline func
        ``tileops.perf.formulas.lerp_tensor_fwd_roofline``:
        per output element 3 flops (sub + mul + add); 3 reads + 1 write
        at post-broadcast ``N_total``.
        """
        n_total = int(self.N_total)
        elem_bytes = self.dtype.itemsize
        return 3 * n_total, 4 * n_total * elem_bytes
