"""Benchmarks for MishFwdOp (NPU).

Adapted from TileOPs ``benchmarks/ops/bench_elementwise_manifest.py`` — Mish subset.
Workloads and roofline formulas are loaded from the standalone manifest.
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import (
    BenchmarkReport,
    ManifestBenchmark,
    workloads_to_params,
)
from tileops.ops.elementwise.mish import MishFwdOp
from tileops.workloads.elementwise import MishWorkload

_MISH_OP = "MishFwdOp"


@pytest.mark.parametrize(
    "shape, dtype, op_params",
    workloads_to_params(_MISH_OP, include_extra=True),
)
def test_mish_bench(shape: tuple, dtype: torch.dtype, op_params: dict) -> None:
    test = MishWorkload(shape, dtype)
    inputs = test.gen_inputs()

    op = MishFwdOp(N_total=test.n_total, dtype=dtype, **op_params)
    bm = ManifestBenchmark(_MISH_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
