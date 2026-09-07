"""Manifest-driven benchmark for LerpTensorFwdOp (NPU).

Adapted from TileOPs ``benchmarks/ops/bench_elementwise_manifest.py`` —
LerpTensor subset only (the shared GPU file covers every implemented
elementwise manifest entry; only this op's benchmark is ported).

Multi-input note: ``LerpTensorFwdOp`` has 3 tensor inputs, so
``workloads_to_params`` cannot be used (its single-input workload
contract raises ``KeyError`` for multi-input ops).  A custom
parametrization helper reads the manifest workloads instead — see the
multi-input pattern in the add-npu-op skill (S6).
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import (
    BenchmarkReport,
    ManifestBenchmark,
)
from tileops.manifest import load_workloads
from tileops.ops.elementwise.lerp_tensor import LerpTensorFwdOp
from tileops.workloads.elementwise import LerpTensorManifestWorkload

_LERP_TENSOR_OP = "LerpTensorFwdOp"


def _mark(idx: int, dtype: torch.dtype):
    return pytest.mark.smoke if idx == 0 and dtype is torch.float16 else pytest.mark.full


def _shape_dtype_params(workloads: list[dict]) -> list:
    """Parametrize (shape, dtype) from manifest workloads (multi-input).

    All three operands share the manifest ``input_shape`` (the
    ``LerpTensorManifestWorkload`` fans it out to input/end/weight), so a
    single shape parameter suffices — same as the GPU benchmark.
    """
    params = []
    for idx, w in enumerate(workloads):
        shape = tuple(w["input_shape"])
        label = w.get("label", "x".join(str(dim) for dim in shape))
        for dtype_name in w["dtypes"]:
            dtype = getattr(torch, dtype_name)
            params.append(
                pytest.param(shape, dtype, id=f"{label}-{dtype_name}", marks=_mark(idx, dtype))
            )
    return params


@pytest.mark.parametrize("shape, dtype", _shape_dtype_params(load_workloads(_LERP_TENSOR_OP)))
def test_lerp_tensor_manifest_bench(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    test = LerpTensorManifestWorkload(shape, dtype)
    x, end, weight = test.gen_inputs()
    op = LerpTensorFwdOp(input=shape, end=shape, weight=shape, dtype=dtype)
    bm = ManifestBenchmark(_LERP_TENSOR_OP, op, test)
    result = bm.profile(op, x, end, weight)
    BenchmarkReport.record(op, locals(), result, tag="tileops")
    result_bl = bm.profile(torch.lerp, x, end, weight)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
