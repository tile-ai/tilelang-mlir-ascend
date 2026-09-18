"""Manifest-driven benchmark for AdaLayerNormFwdOp (NPU).

Adapted from TileOPs ``benchmarks/ops/bench_ada_layer_norm.py`` --
``test_ada_layer_norm_bench`` only.  The shared GPU file also benchmarks
``AdaLayerNormZeroFwdOp`` (gated variant, ``test_ada_layer_norm_zero_bench``),
which is not migrated; its imports and bench function are dropped.

Multi-input note: ``AdaLayerNormFwdOp`` has 3 tensor inputs (x / scale /
shift), so ``workloads_to_params`` cannot be used (its single-input
workload contract raises ``KeyError`` for multi-input ops).  The GPU
``_to_params`` helper -- which reads the manifest ``x_shape`` -- is the
custom parametrization helper (same pattern as ``bench_lerp_tensor.py``).
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops.norm.ada_layer_norm import AdaLayerNormFwdOp
from tileops.workloads.norm import AdaLayerNormWorkload

_ADA_OP_NAME = "AdaLayerNormFwdOp"


def _to_params(workloads):
    params = []
    for w in workloads:
        m, n = w["x_shape"]
        label = w.get("label", f"{m}x{n}")
        for dtype_str in w["dtypes"]:
            dtype = getattr(torch, dtype_str)
            params.append(pytest.param(m, n, dtype, id=f"{label}-{dtype_str}"))
    return params


@pytest.mark.parametrize("m, n, dtype", _to_params(load_workloads(_ADA_OP_NAME)))
def test_ada_layer_norm_bench(m: int, n: int, dtype: torch.dtype) -> None:
    test = AdaLayerNormWorkload(m, n, dtype)
    inputs = test.gen_inputs()

    op = AdaLayerNormFwdOp(dtype=dtype)
    bm = ManifestBenchmark(_ADA_OP_NAME, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
