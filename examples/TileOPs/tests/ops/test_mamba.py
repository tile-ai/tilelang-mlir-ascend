"""Correctness tests for the mamba (Mamba-2 / SSD) family (NPU).

Adapted from TileOPs ``tests/ops/test_mamba.py`` -- SSDChunkScanFwdOp subset
only (the shared GPU file also covers DaCumsum / CBProducer / SSDChunkState /
SSDStatePassing / SSDDecode / Mamba2Fwd, which are not migrated).

NPU adaptations (T1-T4):

- T1: manual tensor creation uses the device backend (via the workload's
  ``gen_inputs``) instead of hard-coded ``device="cuda"``.
- T2: no ``torch.cuda`` availability guard in the GPU SSDChunkScan tests --
  nothing to remove.
- T3: ``tune`` parametrization dimension removed.
"""

import pytest
import torch

from tileops.ops.mamba.ssd_chunk_scan import SSDChunkScanFwdOp
from tileops.testing.mamba2_reference import ssd_chunk_scan_fwd_ref
from tileops.testing.test_base import TestBase
from tileops.workloads.mamba import SSDChunkScanFwdFixture, SSDChunkScanFwdWorkload


class SSDChunkScanFwdTest(SSDChunkScanFwdWorkload, TestBase):
    def ref_program(self, x, cb, dA_cumsum, C, prev_states, dt):
        return ssd_chunk_scan_fwd_ref(x, cb, dA_cumsum, C, prev_states, dt, self.n_groups)


@SSDChunkScanFwdFixture
def test_ssd_chunk_scan_fwd(
    batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype
):
    test = SSDChunkScanFwdTest(
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype
    )
    op = SSDChunkScanFwdOp()
    inputs = test.gen_inputs()
    atol = 1e-3 if dtype == torch.float16 else 2e-3
    rtol = 1e-5
    test.check(op, *inputs, atol=atol, rtol=rtol)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
