"""Benchmarks for LogSumExpFwdOp (NPU).

Adapted from TileOPs ``benchmarks/ops/bench_softmax.py`` — LogSumExp subset.
Workloads and roofline formulas are loaded from the standalone manifest.
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import (
    BenchmarkReport,
    ManifestBenchmark,
    workloads_to_params,
)
from tileops.ops.reduction.softmax import LogSumExpFwdOp
from tileops.workloads.reduction import LogSumExpWorkload

_LOGSUMEXP_OP = "LogSumExpFwdOp"


@pytest.mark.parametrize(
    "shape, dtype, op_params",
    workloads_to_params(_LOGSUMEXP_OP, include_extra=True),
)
def test_logsumexp_bench(shape: tuple, dtype: torch.dtype, op_params: dict) -> None:
    test = LogSumExpWorkload(shape, dtype)
    inputs = test.gen_inputs()

    op_params.setdefault("dim", -1)
    op = LogSumExpFwdOp(dtype=dtype, **op_params)
    bm = ManifestBenchmark(_LOGSUMEXP_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
