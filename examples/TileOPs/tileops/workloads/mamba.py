"""Workload definitions for the mamba (Mamba-2 / SSD) op family.

Ported from TileOPs ``workloads/mamba.py`` -- SSDChunkScanFwdOp subset only
(the shared GPU file also defines DaCumsum / SSDChunkState / SSDDecode /
SSDStatePassing fixtures and workloads, which are not migrated).

Adaptation point W1: ``device`` is resolved from :func:`get_device_backend`
instead of the hard-coded ``"cuda"`` in the original TileOPs code.
"""

import torch

from tileops.device import get_device_backend
from tileops.workloads.workload_base import FixtureBase, WorkloadBase


class SSDChunkScanFwdFixture(FixtureBase):
    @classmethod
    def get_params(cls):
        import pytest

        return [
            (
                "batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype",
                [
                    pytest.param(1, 2, 64, 4, 64, 32, 1, torch.float16, marks=pytest.mark.smoke),
                    pytest.param(1, 2, 128, 4, 128, 32, 1, torch.bfloat16, marks=pytest.mark.smoke),
                    pytest.param(2, 4, 64, 8, 64, 64, 2, torch.float16, marks=pytest.mark.full),
                    pytest.param(2, 2, 64, 4, 64, 32, 2, torch.bfloat16, marks=pytest.mark.full),
                ],
            ),
        ]


class SSDChunkScanFwdWorkload(WorkloadBase):
    def __init__(
        self,
        batch: int,
        num_chunks: int,
        chunk_len: int,
        n_heads: int,
        d_head: int,
        d_state: int,
        n_groups: int,
        dtype: torch.dtype,
    ):
        self.batch = batch
        self.num_chunks = num_chunks
        self.chunk_len = chunk_len
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.n_groups = n_groups
        self.dtype = dtype

    def gen_inputs(self):
        b, c, L, h, p, n, g = (
            self.batch,
            self.num_chunks,
            self.chunk_len,
            self.n_heads,
            self.d_head,
            self.d_state,
            self.n_groups,
        )
        S = c * L
        device = get_device_backend().name

        # Official layouts (aligned with _chunk_scan_fwd in mamba_ssm)
        x = torch.randn(b, S, h, p, dtype=self.dtype, device=device) * 0.1
        cb = torch.randn(b, c, g, L, L, dtype=self.dtype, device=device) * 0.1
        dA_cumsum = -torch.rand(b, h, c, L, dtype=torch.float32, device=device).cumsum(-1)
        C = torch.randn(b, S, g, n, dtype=self.dtype, device=device) * 0.1
        prev_states = torch.randn(b, c, h, p, n, dtype=torch.float32, device=device) * 0.1
        dt = torch.rand(b, h, c, L, dtype=self.dtype, device=device) * 0.1 + 0.01
        return x, cb, dA_cumsum, C, prev_states, dt
