"""Packed grouped GEMM with device-side routing, FP32 accumulation and NPUIR.

NT/NN split output rows; TN splits the contraction and returns [G, N, K].
Adapted from TileOPs PR #21 and this repository's examples/gemm/matmul.py.
Set TILELANG_ASCEND_MODE=Expert before importing TileLang. No TileOPs registry,
host routing readback, or torch.matmul fallback is used by these entry points.
"""

import os
from functools import lru_cache

import torch

BLOCK_M, BLOCK_N, BLOCK_K = 64, 128, 64
BLOCKS = 32
ZERO_K = 16


def _compile(program):
    import tilelang

    return tilelang.compile(
        program,
        target="npuir",
        pass_configs={
            tilelang.PassConfigKey.NPUIR_ENABLE_AUTO_MULTI_BUFFER: False,
            tilelang.PassConfigKey.NPUIR_DISABLE_HIVM_AUTO_INJECT_SYNC: False,
        },
    )


@lru_cache(maxsize=32)
def _build_tile_offsets(experts, n_tiles, device_index):
    """Exclusive tile counts, on device; the extra launch orders the consumers."""
    import tilelang.language as T

    @T.prim_func
    def main(
        Sizes: T.Tensor((experts,), "int32"),
        TileOffsets: T.Tensor((experts + 1,), "int32"),
    ):
        with T.Kernel(1, is_npu=True) as (pid, _):
            with T.Scope("Cube"):
                TileOffsets[0] = 0
                for group in T.serial(experts):
                    TileOffsets[group + 1] = (
                        TileOffsets[group] + T.ceildiv(Sizes[group], BLOCK_M) * n_tiles
                    )

    return _compile(main)


@lru_cache(maxsize=64)
def _build_m_grouped(numel, experts, n, k, trans_b, dtype, device_index):
    """Group-major flat tile space; narrow every copy, MAD and store to live lanes."""
    import tilelang.language as T

    n_tiles = (n + BLOCK_N - 1) // BLOCK_N
    b_shape = (experts, n, k) if trans_b else (experts, k, n)
    b_tile_shape = (BLOCK_N, BLOCK_K) if trans_b else (BLOCK_K, BLOCK_N)

    @T.prim_func
    def main(
        A: T.Tensor((numel, k), dtype),
        B: T.Tensor(b_shape, dtype),
        Sizes: T.Tensor((experts,), "int32"),
        Offsets: T.Tensor((experts,), "int32"),
        TileOffsets: T.Tensor((experts + 1,), "int32"),
        Out: T.Tensor((numel, n), dtype),
    ):
        with T.Kernel(BLOCKS, is_npu=True) as (pid, _):
            with T.Scope("Cube"):
                a_l1 = T.alloc_L1((BLOCK_M, BLOCK_K), dtype)
                b_l1 = T.alloc_L1(b_tile_shape, dtype)
                acc = T.alloc_L0C((BLOCK_M, BLOCK_N), "float32")
                for work in T.serial(T.ceildiv(TileOffsets[experts], BLOCKS)):
                    tile = work * BLOCKS + pid
                    for group in T.serial(experts):
                        with (
                            T.If(
                                (tile >= TileOffsets[group])
                                & (tile < TileOffsets[group + 1])
                            ),
                            T.Then(),
                        ):
                            local_tile = tile - TileOffsets[group]
                            m_start = local_tile // n_tiles * BLOCK_M
                            n_start = local_tile % n_tiles * BLOCK_N
                            row = Offsets[group] + m_start
                            rows = T.min(BLOCK_M, Sizes[group] - m_start)
                            cols = T.min(BLOCK_N, n - n_start)
                            for step in T.serial(T.ceildiv(k, BLOCK_K)):
                                k_start = step * BLOCK_K
                                depth = T.min(BLOCK_K, k - k_start)
                                T.copy(
                                    A[row : row + rows, k_start : k_start + depth],
                                    a_l1[:rows, :depth],
                                )
                                if trans_b:
                                    T.copy(
                                        B[
                                            group,
                                            n_start : n_start + cols,
                                            k_start : k_start + depth,
                                        ],
                                        b_l1[:cols, :depth],
                                    )
                                else:
                                    T.copy(
                                        B[
                                            group,
                                            k_start : k_start + depth,
                                            n_start : n_start + cols,
                                        ],
                                        b_l1[:depth, :cols],
                                    )
                                T.gemm(
                                    a_l1,
                                    b_l1,
                                    acc,
                                    size=[rows, depth, cols],
                                    initC=(step == 0),
                                    b_transpose=trans_b,
                                )
                            T.store_fixpipe(
                                acc,
                                Out[row, n_start],
                                size=[rows, cols],
                                enable_nz2nd=True,
                            )

    return _compile(main)


@lru_cache(maxsize=64)
def _build_k_grouped(numel, experts, n, k, dtype, device_index):
    """Group the contraction; initialize every output tile with a zero MAD."""
    import tilelang.language as T

    m_tiles = (n + BLOCK_M - 1) // BLOCK_M
    n_tiles = (k + BLOCK_N - 1) // BLOCK_N
    total = experts * m_tiles * n_tiles

    @T.prim_func
    def main(
        A: T.Tensor((numel, n), dtype),
        B: T.Tensor((numel, k), dtype),
        Sizes: T.Tensor((experts,), "int32"),
        Offsets: T.Tensor((experts,), "int32"),
        ZeroA: T.Tensor((ZERO_K, BLOCK_M), dtype),
        ZeroB: T.Tensor((ZERO_K, BLOCK_N), dtype),
        Out: T.Tensor((experts, n, k), dtype),
    ):
        with T.Kernel(BLOCKS, is_npu=True) as (pid, _):
            with T.Scope("Cube"):
                a_l1 = T.alloc_L1((BLOCK_K, BLOCK_M), dtype)
                b_l1 = T.alloc_L1((BLOCK_K, BLOCK_N), dtype)
                acc = T.alloc_L0C((BLOCK_M, BLOCK_N), "float32")
                for work in T.serial(T.ceildiv(total, BLOCKS)):
                    tile = work * BLOCKS + pid
                    with T.If(tile < total), T.Then():
                        group = tile // (m_tiles * n_tiles)
                        m_start = tile % (m_tiles * n_tiles) // n_tiles * BLOCK_M
                        n_start = tile % n_tiles * BLOCK_N
                        rows = T.min(BLOCK_M, n - m_start)
                        cols = T.min(BLOCK_N, k - n_start)
                        T.copy(ZeroA, a_l1[:ZERO_K, :BLOCK_M])
                        T.copy(ZeroB, b_l1[:ZERO_K, :BLOCK_N])
                        T.gemm(
                            a_l1,
                            b_l1,
                            acc,
                            size=[rows, ZERO_K, cols],
                            initC=True,
                            a_transpose=True,
                        )
                        for step in T.serial(T.ceildiv(Sizes[group], BLOCK_K)):
                            row = Offsets[group] + step * BLOCK_K
                            depth = T.min(BLOCK_K, Sizes[group] - step * BLOCK_K)
                            T.copy(
                                A[row : row + depth, m_start : m_start + rows],
                                a_l1[:depth, :rows],
                            )
                            T.copy(
                                B[row : row + depth, n_start : n_start + cols],
                                b_l1[:depth, :cols],
                            )
                            T.gemm(
                                a_l1,
                                b_l1,
                                acc,
                                size=[rows, depth, cols],
                                initC=False,
                                a_transpose=True,
                            )
                        T.store_fixpipe(
                            acc,
                            Out[group, m_start, n_start],
                            size=[rows, cols],
                            enable_nz2nd=True,
                        )

    return _compile(main)


def _validate(a, b, sizes, offsets, transpose_a, transpose_b):
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise ValueError("a and b must be tensors")
    if a.dtype not in (torch.float16, torch.bfloat16) or b.dtype != a.dtype:
        raise ValueError("a and b must have the same float16 or bfloat16 dtype")
    if a.ndim != 2:
        raise ValueError("a must have rank 2")
    for name, value in (("batch_sizes", sizes), ("batch_offsets", offsets)):
        if not isinstance(value, torch.Tensor) or value.dtype != torch.int32:
            raise ValueError(f"{name} must be an int32 tensor")
        if value.ndim != 1:
            raise ValueError(f"{name} must have rank 1")
    experts = sizes.numel()
    if experts == 0 or offsets.shape != sizes.shape:
        raise ValueError(
            "batch_sizes and batch_offsets must have equal, nonempty shapes"
        )
    for name, value in (
        ("a", a),
        ("b", b),
        ("batch_sizes", sizes),
        ("batch_offsets", offsets),
    ):
        if value.device != a.device:
            raise ValueError(f"{name} must be on {a.device}")
        if not value.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    numel = a.shape[0]
    if transpose_a:
        if b.ndim != 2 or b.shape[0] != numel:
            raise ValueError("TN requires a=[M,N] and b=[M,K]")
        n, k = a.shape[1], b.shape[1]
    else:
        if b.ndim != 3 or b.shape[0] != experts:
            raise ValueError("NT/NN requires one rank-3 weight tensor with G groups")
        k = a.shape[1]
        n = b.shape[1] if transpose_b else b.shape[2]
        if (b.shape[2] if transpose_b else b.shape[1]) != k:
            raise ValueError("a and b contraction dimensions differ")
    if n <= 0 or k <= 0:
        raise ValueError("N and K must be positive")
    max_tiles = (numel + BLOCK_M - 1) // BLOCK_M + experts - 1
    max_tiles *= (n + BLOCK_N - 1) // BLOCK_N
    if transpose_a:
        max_tiles = (
            experts * ((n + BLOCK_M - 1) // BLOCK_M) * ((k + BLOCK_N - 1) // BLOCK_N)
        )
    if numel >= 2**31 or max_tiles >= 2**31:
        raise ValueError("packed rows and tile counts must fit int32")
    return numel, experts, n, k


class GroupedGemmFwdOp:
    """One GEMM per packed group, with runtime routing metadata on the NPU.

    NT (default): A[M,K], B[G,N,K] -> D[M,N].
    NN: A[M,K], B[G,K,N] -> D[M,N].
    TN: A[M,N], B[M,K] -> D[G,N,K].

    Sizes[G] and Offsets[G] are contiguous int32 tensors on the operand device.
    Sizes must be nonnegative and sum to M; Offsets must be its exclusive prefix
    sum. The caller owns these value invariants: checking them here would add a
    host synchronization. Shape, dtype, layout and device checks are explicit.
    Inputs are never modified. Groups may be empty; no alignment/padding of their
    packed row offsets is required. TT is outside the source PR's contract.
    """

    def __init__(self, *, transpose_a=False, transpose_b=True):
        if not isinstance(transpose_a, bool) or not isinstance(transpose_b, bool):
            raise ValueError("transpose flags must be bool")
        if transpose_a and transpose_b:
            raise ValueError("TT layout is not supported; use NT, NN or TN")
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b

    def __call__(self, a, b, batch_sizes, batch_offsets):
        numel, experts, n, k = _validate(
            a, b, batch_sizes, batch_offsets, self.transpose_a, self.transpose_b
        )
        if a.device.type != "npu":
            raise RuntimeError("Grouped GEMM requires an Ascend NPU; no CPU fallback")
        if os.environ.get("TILELANG_ASCEND_MODE", "Expert") != "Expert":
            raise RuntimeError(
                "Set TILELANG_ASCEND_MODE=Expert before importing TileLang"
            )
        shape = (experts, n, k) if self.transpose_a else (numel, n)
        if numel == 0:
            return torch.zeros(shape, dtype=a.dtype, device=a.device)
        dtype = str(a.dtype).removeprefix("torch.")
        device_index = a.device.index
        out = torch.empty(shape, dtype=a.dtype, device=a.device)
        # Compile and launch on the input device; include it in the cache key.
        with torch.npu.device(a.device):
            if self.transpose_a:
                zero_a = torch.zeros((ZERO_K, BLOCK_M), dtype=a.dtype, device=a.device)
                zero_b = torch.zeros((ZERO_K, BLOCK_N), dtype=a.dtype, device=a.device)
                kernel = _build_k_grouped(numel, experts, n, k, dtype, device_index)
                kernel(a, b, batch_sizes, batch_offsets, zero_a, zero_b, out)
            else:
                tile_offsets = torch.empty(
                    experts + 1, dtype=torch.int32, device=a.device
                )
                _build_tile_offsets(
                    experts, (n + BLOCK_N - 1) // BLOCK_N, device_index
                )(batch_sizes, tile_offsets)
                kernel = _build_m_grouped(
                    numel, experts, n, k, self.transpose_b, dtype, device_index
                )
                kernel(a, b, batch_sizes, batch_offsets, tile_offsets, out)
        return out

    forward = __call__


# Keep the source kernel's public name without importing the TileOPs registry.
AscendGroupedGemmKernel = GroupedGemmFwdOp
