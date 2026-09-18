"""Benchmarks for MultiHeadAttentionFwdOp (NPU).

Adapted from TileOPs ``benchmarks/ops/attention/bench_mha.py`` -- forward
subset only.  The GPU bench file is shared with ``MultiHeadAttentionBwdOp``
(not migrated yet), so the backward bench function and its imports are
intentionally not ported.  Performance benchmarks profile only the TileOps
kernel; PyTorch remains the correctness reference in ``tests/ops``.

MultiHeadAttentionFwdOp is a 3-input op (q, k, v), so the single-input
``workloads_to_params`` contract does not apply; the GPU
``manifest_params`` / ``mha_qkv_args`` helpers (from
``benchmarks/ops/attention/manifest_params.py``) are inlined instead --
the same multi-input pattern as ``bench_ada_layer_norm.py`` /
``bench_fp8_lightning_indexer.py``.

Adaptations (T3-T4):

- T3: the ``tune`` bench-run policy knob is removed (NPU uses heuristic
  config selection only).
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops.attention.multi_head_attention import MultiHeadAttentionFwdOp
from tileops.workloads.attention import MhaFwdWorkload

_MHA_FWD_OP = "MultiHeadAttentionFwdOp"


def _mha_qkv_args(workload: dict) -> tuple[int, int, int, int, bool]:
    """Extract ``(batch, seq_len, heads, dim, is_causal)`` from a workload."""
    batch, seq_len, heads, dim = workload["q_shape"]
    return batch, seq_len, heads, dim, workload.get("is_causal", True)


def _manifest_params(workloads: list[dict]) -> list:
    """Params from manifest workloads (GPU ``manifest_params`` inlined).

    T3: the GPU helper's trailing ``tune`` flag is dropped.
    """
    params = []
    for workload in workloads:
        label = workload.get("label", "manifest")
        marks = ()
        if reason := workload.get("bench_skip_reason"):
            marks = (pytest.mark.skip(reason=reason),)
        for dtype_name in workload["dtypes"]:
            dtype = getattr(torch, dtype_name)
            params.append(
                pytest.param(
                    *_mha_qkv_args(workload), dtype, id=f"{label}-{dtype_name}", marks=marks
                )
            )
    return params


_MHA_FWD_BENCH_PARAMS = _manifest_params(load_workloads(_MHA_FWD_OP))


@pytest.mark.parametrize("batch, seq_len, heads, dim, causal, dtype", _MHA_FWD_BENCH_PARAMS)
def test_mha_fwd_bench(
    batch: int, seq_len: int, heads: int, dim: int, causal: bool, dtype: torch.dtype
) -> None:
    test = MhaFwdWorkload(batch, heads, seq_len, dim, causal, dtype)
    inputs = test.gen_inputs()

    op = MultiHeadAttentionFwdOp(batch, heads, seq_len, dim, causal, dtype)
    bm = ManifestBenchmark(_MHA_FWD_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
