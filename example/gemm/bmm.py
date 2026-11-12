"""Ascend 950 implementations for BmmFwdOp."""

import functools
from typing import Optional

import tilelang
import tilelang.ascend.language as T
import torch
from tilelang.carver.arch.driver import get_num_cube_cores

import tileops.backend._registry
from tileops._kernels.gemm import BMMCall, BMMFwdInterface
from tileops._kernels.kernel_base import Entry, Kernel

_BLOCK_M = 128
_BLOCK_N = 128
_BLOCK_K = 128
_NUM_STAGES = 2


def _dtype_str(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


@functools.lru_cache(maxsize=8)
def _calibrated_num_cube_cores(device_index: int, fallback: int = 36) -> int:
    """The Cube core count of ``device_index``, queried live (not hardcoded).

    The device is part of the key: a heterogeneous host can report a different count per
    card, and caching one card's answer for another would size the grid from the wrong
    board. A failed query falls back rather than refusing the call, so the diagnostic is
    lost; the fallback is the smallest 950 board these schedules were tuned on.
    """
    try:
        queried = get_num_cube_cores(device_id=device_index)
    except Exception:
        queried = None
    return int(queried) if queried else fallback


@functools.lru_cache(maxsize=64)
def _make_bmm_kernel(batch: int, m: int, n: int, k: int, dtype: str, device_index: int):
    """Build (and cache) the persistent batched-GEMM kernel for one ``(batch, m, n, k,
    dtype)`` combination."""
    block_m, block_n, block_k = (_BLOCK_M, _BLOCK_N, _BLOCK_K)
    m_tiles = -(-m // block_m)
    n_tiles = -(-n // block_n)
    num_k_blocks = -(-k // block_k)
    total_tiles = batch * m_tiles * n_tiles
    num_blocks = min(_calibrated_num_cube_cores(device_index), total_tiles)

    @T.prim_func
    def main(
        A: T.Tensor((batch, m, k), dtype),
        B: T.Tensor((batch, k, n), dtype),
        D: T.Tensor((batch, m, n), dtype),
    ):
        with T.Kernel(num_blocks) as bx:
            sched = T.AscendBatchedTileScheduler(
                block_m=block_m,
                block_n=block_n,
                num_cores=num_blocks,
                shape_m=m,
                shape_n=n,
                num_groups=batch,
            )
            a_l1 = T.alloc_l1((block_m, block_k), dtype)
            b_l1 = T.alloc_l1((block_n, block_k), dtype)
            a_l0 = T.alloc_l0a((block_m, block_k), dtype)
            b_l0 = T.alloc_l0b((block_n, block_k), dtype)
            res = T.alloc_l0c((block_m, block_n), "float32")
            T.annotate_buffer_versions(
                {a_l1: _NUM_STAGES, b_l1: _NUM_STAGES, a_l0: _NUM_STAGES, b_l0: _NUM_STAGES}
            )
            sched.init(bx)
            while sched.valid():
                batch_idx = sched.batch()
                m_idx, n_idx, actual_m, actual_n = sched.tile()
                for k_block_idx in T.Pipelined(num_k_blocks, num_stages=_NUM_STAGES):
                    k_idx = k_block_idx * block_k
                    actual_k = T.min(k - k_idx, block_k)
                    T.copy(
                        A[batch_idx, m_idx : m_idx + actual_m, k_idx : k_idx + actual_k],
                        a_l1[0:actual_m, 0:actual_k],
                    )
                    T.copy(
                        B[batch_idx, k_idx : k_idx + actual_k, n_idx : n_idx + actual_n],
                        b_l1[0:actual_n, 0:actual_k],
                        transpose=True,
                    )
                    T.copy(a_l1[0:actual_m, 0:actual_k], a_l0[0:actual_m, 0:actual_k])
                    T.copy(b_l1[0:actual_n, 0:actual_k], b_l0[0:actual_n, 0:actual_k])
                    T.gemm(
                        a_l0[0:actual_m, 0:actual_k],
                        b_l0[0:actual_n, 0:actual_k],
                        res[0:actual_m, 0:actual_n],
                        transpose_B=True,
                        clear_accum=k_block_idx == 0,
                    )
                T.copy(
                    res[0:actual_m, 0:actual_n],
                    D[batch_idx, m_idx : m_idx + actual_m, n_idx : n_idx + actual_n],
                )
                sched.next_block()

    return tilelang.compile(main, target="ascend", out_idx=-1)


_L1_BLOCK_K = (128, 256)
_L1_NUM_STAGES = (1, 2, 3)
_MAX_CUBE_BLOCKS = 36


@functools.lru_cache(maxsize=64)
def _make_bmm_aligned_l1_kernel(
    batch: int,
    m: int,
    n: int,
    k: int,
    dtype: str,
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    num_blocks: int,
):
    """Build (and cache) one aligned L1-direct GEMM candidate."""
    if block_m not in (128, 256) or block_n not in (128, 256):
        raise ValueError(f"block_m/block_n must be 128 or 256, got {block_m}/{block_n}")
    if m % block_m or n % block_n or k % _BLOCK_K:
        raise ValueError(
            f"aligned L1 builder requires M, N and K divisible by {_BLOCK_K}, got M={m}, N={n}, K={k}"
        )
    if block_k not in _L1_BLOCK_K:
        raise ValueError(f"block_k must be one of {_L1_BLOCK_K}, got {block_k}")
    if k % block_k:
        raise ValueError(f"block_k={block_k} must divide K={k}")
    if num_stages not in _L1_NUM_STAGES:
        raise ValueError(f"num_stages must be one of {_L1_NUM_STAGES}, got {num_stages}")
    if not 1 <= num_blocks <= _MAX_CUBE_BLOCKS:
        raise ValueError(f"num_blocks must be in [1, {_MAX_CUBE_BLOCKS}], got {num_blocks}")
    m_tiles = m // block_m
    n_tiles = n // block_n
    num_k_blocks = k // block_k
    total_tiles = batch * m_tiles * n_tiles
    grid = min(num_blocks, total_tiles)
    if grid < 1:
        raise ValueError("aligned L1 builder has an empty output tile grid")

    @T.prim_func
    def main(
        A: T.Tensor((batch, m, k), dtype),
        B: T.Tensor((batch, k, n), dtype),
        D: T.Tensor((batch, m, n), dtype),
    ):
        with T.Kernel(grid) as bx:
            sched = T.AscendBatchedTileScheduler(
                block_m=block_m,
                block_n=block_n,
                num_cores=grid,
                shape_m=m,
                shape_n=n,
                num_groups=batch,
            )
            a_l1 = T.alloc_l1((block_m, block_k), dtype)
            b_l1 = T.alloc_l1((block_n, block_k), dtype)
            res = T.alloc_l0c((block_m, block_n), "float32")
            T.annotate_buffer_versions({a_l1: num_stages, b_l1: num_stages})
            sched.init(bx)
            while sched.valid():
                batch_idx = sched.batch()
                m_idx, n_idx, actual_m, actual_n = sched.tile()
                for k_block_idx in T.Pipelined(num_k_blocks, num_stages=num_stages):
                    k_idx = k_block_idx * block_k
                    T.copy(
                        A[batch_idx, m_idx : m_idx + actual_m, k_idx : k_idx + block_k],
                        a_l1[0:actual_m, 0:block_k],
                    )
                    T.copy(
                        B[batch_idx, k_idx : k_idx + block_k, n_idx : n_idx + actual_n],
                        b_l1[0:actual_n, 0:block_k],
                        transpose=True,
                    )
                    T.gemm(a_l1, b_l1, res, transpose_B=True, clear_accum=k_block_idx == 0)
                T.copy(
                    res[0:actual_m, 0:actual_n],
                    D[batch_idx, m_idx : m_idx + actual_m, n_idx : n_idx + actual_n],
                )
                sched.next_block()

    return tilelang.compile(main, target="ascend", out_idx=-1)


@functools.lru_cache(maxsize=1)
def _make_bmm_small_fp16_kernel():
    """Split each 128-square batch item into four independent Cube tiles."""

    @T.prim_func
    def main(
        A: T.Tensor((8, 128, 128), "float16"),
        B: T.Tensor((8, 128, 128), "float16"),
        D: T.Tensor((8, 128, 128), "float16"),
    ):
        with T.Kernel(32) as bx:
            a = T.alloc_l1((64, 128), "float16")
            b = T.alloc_l1((64, 128), "float16")
            acc = T.alloc_l0c((64, 64), "float32")
            batch = bx // 4
            mi = (bx % 4) // 2 * 64
            ni = bx % 2 * 64
            T.copy(A[batch, mi : mi + 64, 0:128], a)
            T.copy(B[batch, 0:128, ni : ni + 64], b, transpose=True)
            T.gemm(a, b, acc, transpose_B=True, clear_accum=True)
            T.copy(acc, D[batch, mi : mi + 64, ni : ni + 64])

    return tilelang.compile(main, target="ascend", out_idx=-1)


_ALIGNED_L1_SELECTIONS = {
    (64, 128, 2048, 128, "float16"): (128, 256, 128, 2, 32),
    (64, 128, 2048, 128, "bfloat16"): (128, 256, 128, 2, 32),
    (128, 512, 512, 2048, "bfloat16"): (256, 256, 128, 2, 32),
    (8, 1024, 1024, 1024, "float16"): (256, 256, 128, 2, 32),
    (8, 1024, 1024, 1024, "bfloat16"): (256, 256, 128, 2, 32),
    (8, 2048, 2048, 2048, "float16"): (256, 256, 128, 2, 32),
    (8, 2048, 2048, 2048, "bfloat16"): (256, 256, 128, 2, 32),
    (4, 4096, 4096, 4096, "bfloat16"): (256, 256, 128, 2, 32),
}


class AscendBMMKernel(Kernel, BMMFwdInterface):
    """Batched dense GEMM on Ascend NPUs: ``d[i] = a[i] @ b[i]`` per batch item."""

    devices = frozenset({"npu"})

    @classmethod
    def refusal(cls, call: BMMCall) -> Optional[str]:
        return None if call.k % 16 == 0 else f"requires k a multiple of 16, got k={call.k}"

    @classmethod
    def entry_for(cls, call: BMMCall) -> Entry:
        device_index = call.device.index if call.device is not None else None
        return (
            call,
            lambda: cls(call.batch, call.m, call.n, call.k, call.dtype, device_index),
        )

    def __init__(
        self,
        batch: int,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        device_index: Optional[int] = None,
    ):
        super().__init__(device_index=device_index)
        self.batch = batch
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        if k == 0:
            # An empty contraction produces zeros; no accumulator is initialized.
            self._compiled_fn = None
            return
        dtype_name = _dtype_str(dtype)
        selection = _ALIGNED_L1_SELECTIONS.get((batch, m, n, k, dtype_name))
        if (batch, m, n, k, dtype_name) == (8, 128, 128, 128, "float16"):
            self._compiled_fn = _make_bmm_small_fp16_kernel()
        elif selection is None:
            self._compiled_fn = _make_bmm_kernel(batch, m, n, k, dtype_name, device_index or 0)
        else:
            block_m, block_n, block_k, num_stages, num_blocks = selection
            self._compiled_fn = _make_bmm_aligned_l1_kernel(
                batch, m, n, k, dtype_name, block_m, block_n, block_k, num_stages, num_blocks
            )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.k == 0:
            return torch.zeros((self.batch, self.m, self.n), dtype=a.dtype, device=a.device)
        return self._compiled_fn(a, b)


tileops.backend._registry.register_kernel_type("BMMFwdOp", "ascend", AscendBMMKernel)
