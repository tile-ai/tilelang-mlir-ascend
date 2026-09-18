"""Test base class for correctness testing (device-agnostic).

Extracted from TileOPs ``tests/test_base.py``.  No GPU/NPU adaptation needed —
``TestBase.check`` runs ``op(*inputs)`` under ``torch.no_grad()`` and compares
against ``ref_program`` using ``torch.testing.assert_close``.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from functools import partial
from typing import Any

import torch

from tileops.workloads.workload_base import WorkloadBase

_logger = logging.getLogger("tileops.testing")

_check_result = threading.local() if (threading := __import__("threading")) else None


def _to_tuple(outputs):
    if isinstance(outputs, torch.Tensor):
        return (outputs,)
    if isinstance(outputs, list):
        return tuple(outputs)
    if isinstance(outputs, tuple):
        return outputs
    raise ValueError(f"Unsupported output type: {type(outputs)}")


def allclose_compare(
    output: torch.Tensor,
    output_ref: torch.Tensor,
    atol: float = 1e-8,
    rtol: float = 1e-5,
) -> None:
    output, output_ref = torch.broadcast_tensors(output, output_ref)
    torch.testing.assert_close(
        output,
        output_ref,
        atol=atol,
        rtol=rtol,
        equal_nan=True,
    )


def exact_compare(output: torch.Tensor, output_ref: torch.Tensor) -> None:
    assert torch.equal(output, output_ref), "output does not exactly match reference"


class _CheckResult:
    """Thread-local container for conftest hook to pick up test metadata."""

    op_name: str | None = None
    op_module: str | None = None
    max_abs_err: float | None = None


_check_result = _CheckResult()


def get_check_result() -> _CheckResult:
    return _check_result


def record_check_result(op, outputs: Any, outputs_ref: Any) -> float:
    """Record operator identity and maximum absolute error for pytest reports."""
    outputs = _to_tuple(outputs)
    outputs_ref = _to_tuple(outputs_ref)
    assert len(outputs) == len(outputs_ref), (
        f"outputs: {len(outputs)} and outputs_ref: {len(outputs_ref)} have different size"
    )

    max_abs_err = 0.0
    for output, output_ref in zip(outputs, outputs_ref, strict=True):
        if output_ref is not None:
            wide = torch.complex64 if output.is_complex() else torch.float32
            err = (output.to(wide) - output_ref.to(wide)).abs().max().item()
            max_abs_err = max(max_abs_err, err)

    _check_result.op_name = op.__class__.__name__
    _check_result.op_module = op.__class__.__module__
    _check_result.max_abs_err = max_abs_err
    return max_abs_err


class TestBase(WorkloadBase):
    """Abstract base class for op correctness testing.

    Inherits ``gen_inputs()`` from ``WorkloadBase``.
    Subclasses must implement ``ref_program()``.
    """

    __test__ = False

    @abstractmethod
    def ref_program(self, *inputs: Any) -> Any:
        raise NotImplementedError

    def check(
        self,
        op,
        *inputs: torch.Tensor,
        compare=None,
        atol: float = 1e-08,
        rtol: float = 1e-05,
    ) -> None:
        if compare is None:
            compare = partial(allclose_compare, atol=atol, rtol=rtol)

        op_name = op.__class__.__name__
        op_module = op.__class__.__module__

        try:
            outputs_ref = self.ref_program(*inputs)
        except RuntimeError as e:
            if "out of memory" in str(e):
                _logger.warning("op=%s module=%s status=skip_oom", op_name, op_module)
                return
            raise e

        outputs_ref = _to_tuple(outputs_ref)

        with torch.no_grad():
            outputs = op(*inputs)

        outputs = _to_tuple(outputs)

        max_abs_err = record_check_result(op, outputs, outputs_ref)

        comparators = [compare] * len(outputs) if callable(compare) else list(compare)
        for output, output_ref, cmp in zip(outputs, outputs_ref, comparators, strict=True):
            if output_ref is not None:
                cmp(output, output_ref)

        _logger.info(
            "op=%s module=%s status=pass max_abs_err=%.2e",
            op_name,
            op_module,
            max_abs_err,
        )
