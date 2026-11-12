"""Ascend 950 W4A8 GEMM: FP8 activations times a group-scaled signed INT4 weight.

Three passes. The weight is unpacked from INT4, multiplied by its group scale and
rounded to ``float8_e4m3fn`` — the value the reference's FP8 tensor-core product
consumes — then a native FP8 GEMM accumulates in FP32, and a vector pass applies the
token and channel scales. Unpacking is a vector-unit job, so it runs as its own pass
over the weight; the GEMM then reads plain FP8 and keeps the cube's FP8 rate.
"""

import functools
from typing import Optional

import tilelang
import tilelang.ascend.language as T
import torch
from tilelang.carver.arch.driver import get_num_cube_cores

import tileops.backend._registry
from tileops._kernels.gemm import (
    GEMMW4A8Call,
    GEMMW4A8FwdInterface,
    W4A8RepackCall,
    W4A8RepackFwdInterface,
)
from tileops._kernels.kernel_base import Kernel

__all__ = ["AscendGEMMW4A8Kernel", "AscendW4A8RepackKernel"]

_FP8 = "float8_e4m3fn"
_OUT_DTYPE = "bfloat16"
_NUM_AIV_CORES = 64
_CAST_STAGES = 2


def _tiles(m: int, n: int, k: int) -> tuple[int, int, int, int, int]:
    """The tile ``(tile_m, tile_n, tile_k, stages, blocks)`` one call runs with.

    A 256-deep K stage with two operand versions is what this GEMM wants at every row
    count, so the row tile is what gives: it grows with the call up to the 256 rows
    the L0C budget allows, and shrinks by 64 rows when the L1 operand budget cannot
    hold the deeper stage at 256. Row tiles below that stay narrow so a short call
    does not compute rows it has no tokens for. The column tile prefers 224, which
    splits 7168 into exactly 32 tiles — one per Cube core.
    """
    tile_m = 16 if m <= 16 else 64 if m <= 64 else 128 if m <= 256 else 256
    tile_n = 224 if n % 224 == 0 else 256 if n % 256 == 0 else 128
    tile_k = 256 if k % 256 == 0 else 128
    # Two operand versions cost 2 * (rows + columns) * tile_k * stages bytes in L1.
    while tile_m > 16 and 2 * (tile_m + tile_n) * tile_k * 2 >= 512 * 1024:
        tile_m -= 64
    return tile_m, tile_n, tile_k, 2, 32


def _dequant_program(n: int, k: int, group_size: int):
    """Build the unpack pass: ``wb[n, k] = fp8(q * group_scale[n, k // group])``.

    The vector unit takes one whole group per step. ``UNPK4_B8`` puts a packed byte in
    the low byte of each lane, so the low nibbles are the even K values and the high
    nibbles the odd ones; ``vintlv`` pairs them back into K order and ``PK4_B32``
    stores the FP8 lanes back to back. Group scales arrive as FP8, which the vector
    unit cannot widen, so a scalar prologue converts them to FP16 for the broadcast.

    Args:
        n: Weight rows.
        k: Contracted length.
        group_size: Values per scale group; the vector step consumes exactly one.

    Raises:
        ValueError: ``group_size`` is not the 128 values one vector step covers, or
            the shapes do not divide into whole steps.
    """
    scale_cols = k // group_size
    rows_per_block = 2  # two rows per block keep the packed read and FP8 write in flight
    if group_size != 128:
        raise ValueError(f"this unpack step consumes one 128-value group, got {group_size}")
    if k % group_size or n % rows_per_block:
        raise ValueError(f"requires whole groups and even N, got n={n}, k={k}")

    @T.prim_func
    def main(
        W: T.Tensor((n, k // 2), "uint8"),
        GS: T.Tensor((n, scale_cols), _FP8),
        WB: T.Tensor((n, k), _FP8),
    ):
        T.func_attr({"tilelang_callee_allocated_outputs": 0})
        with T.Kernel(_NUM_AIV_CORES) as core:
            for blk in T.Persistent(
                [n // rows_per_block], _NUM_AIV_CORES, core, num_stages=_CAST_STAGES
            ):
                packed_ub = T.alloc_shared((rows_per_block, k // 2), "uint8")
                scale_ub = T.alloc_shared((rows_per_block, scale_cols), _FP8)
                scale_f16 = T.alloc_shared((rows_per_block, scale_cols), "float16")
                out_ub = T.alloc_shared((rows_per_block, k), _FP8)
                T.annotate_buffer_versions(
                    {
                        packed_ub: _CAST_STAGES,
                        scale_ub: _CAST_STAGES,
                        scale_f16: _CAST_STAGES,
                        out_ub: _CAST_STAGES,
                    }
                )
                T.copy(W[blk * rows_per_block : (blk + 1) * rows_per_block, 0 : k // 2], packed_ub)
                T.copy(
                    GS[blk * rows_per_block : (blk + 1) * rows_per_block, 0:scale_cols],
                    scale_ub,
                )
                with T.SimtVF(threads=scale_cols):
                    for r, j in T.Parallel(rows_per_block, scale_cols):
                        scale_f16[r, j] = T.Cast("float16", scale_ub[r, j])
                with T.SimdVF():
                    m_all_b8 = T.simd.pset(8, "PAT_ALL")
                    m_all_b16 = T.simd.pset(16, "PAT_ALL")
                    m_all_b32 = T.simd.pset(32, "PAT_ALL")
                    fifteen = T.simd.vdup(T.uint8(15), "uint8")
                    eight8 = T.simd.vdup(T.uint8(8), "uint8")
                    for r in T.serial(rows_per_block):
                        src = T.simd.make_ubuf_ptr(
                            T.access_ptr(packed_ub[r, 0], "r", extent=k // 2), "uint8"
                        )
                        for g in T.serial(scale_cols):
                            scale = T.simd.vcvt(
                                T.simd.vld(scale_f16[r, g], "BRC_B16"), "float32", m_all_b16
                            )
                            packed, src = T.simd.vld(src, "UNPK4_B8", post_inc=group_size // 2)
                            low = T.simd.vsub(
                                T.simd.vxor(
                                    T.simd.vand(packed, fifteen, m_all_b8), eight8, m_all_b8
                                ),
                                eight8,
                                m_all_b8,
                            )
                            high = T.simd.vsub(
                                T.simd.vxor(T.simd.vshrs(packed, 4, m_all_b8), eight8, m_all_b8),
                                eight8,
                                m_all_b8,
                            )
                            even = _scaled_fp8(low, scale, m_all_b8, m_all_b16, m_all_b32)
                            odd = _scaled_fp8(high, scale, m_all_b8, m_all_b16, m_all_b32)
                            lo, hi = T.simd.vintlv(
                                T.reinterpret(even, "uint32x64"),
                                T.reinterpret(odd, "uint32x64"),
                            )
                            T.simd.vsts(out_ub[r, g * group_size], lo, m_all_b32, "PK4_B32")
                            T.simd.vsts(out_ub[r, g * group_size + 64], hi, m_all_b32, "PK4_B32")
                T.copy(out_ub, WB[blk * rows_per_block : (blk + 1) * rows_per_block, 0:k])

    return main


def _scaled_fp8(nibbles, scale, m_all_b8, m_all_b16, m_all_b32):
    """Widen one nibble vector to FP32 lanes, apply the group scale, round to FP8."""
    return T.simd.vcvt(
        T.simd.vmul(
            T.simd.vcvt(
                T.simd.vcvt(T.reinterpret(nibbles, "int8x256"), "int16", m_all_b8, part=0),
                "float32",
                m_all_b16,
            ),
            scale,
            m_all_b32,
        ),
        _FP8,
        m_all_b32,
    )


def _gemm_program(
    m: int, n: int, k: int, tile_m: int, tile_n: int, tile_k: int, stages: int, blocks: int
):
    """Build the FP8 GEMM: one persistent block per Cube core, tiles from the scheduler."""
    mad_k = 64  # one MAD step: two L0 versions of a 128-row FP8 operand take 16 KiB
    k_tiles = -(-k // tile_k)
    if k % tile_k or tile_k % mad_k or n % tile_n:
        raise ValueError(f"this schedule requires whole K and N tiles, got n={n}, k={k}")
    if 2 * (tile_m + tile_n) * tile_k * stages > 512 * 1024:
        raise ValueError("L1 operand versions exceed the 512 KiB capacity")
    if 4 * tile_m * tile_n > 256 * 1024:
        raise ValueError("FP32 accumulator exceeds the 256 KiB L0C capacity")

    @T.prim_func
    def main(
        A: T.Tensor((m, k), _FP8),
        B: T.Tensor((n, k), _FP8),
        D: T.Tensor((m, n), "float32"),
    ):
        T.func_attr({"tilelang_callee_allocated_outputs": 0})
        with T.Kernel(blocks) as bx:
            a_l1 = T.alloc_l1((tile_m, tile_k), _FP8)
            b_l1 = T.alloc_l1((tile_n, tile_k), _FP8)
            a_l0 = T.alloc_l0a((tile_m, mad_k), _FP8)
            b_l0 = T.alloc_l0b((tile_n, mad_k), _FP8)
            acc = T.alloc_l0c((tile_m, tile_n), "float32")
            T.annotate_buffer_versions({a_l1: stages, b_l1: stages, a_l0: 2, b_l0: 2})
            sched = T.AscendTileScheduler(
                block_m=tile_m,
                block_n=tile_n,
                num_cores=blocks,
                shape_m=m,
                shape_n=n,
            )
            sched.init(bx)
            while sched.valid():
                m_start, n_start, rows, cols = sched.tile()
                for ki in T.Pipelined(k_tiles, num_stages=stages):
                    start = ki * tile_k
                    T.copy(
                        A[m_start : m_start + rows, start : start + tile_k], a_l1[0:rows, 0:tile_k]
                    )
                    T.copy(
                        B[n_start : n_start + cols, start : start + tile_k], b_l1[0:cols, 0:tile_k]
                    )
                    for m_slice in range(tile_k // mad_k):
                        k_slice = m_slice * mad_k
                        T.copy(a_l1[0:rows, k_slice : k_slice + mad_k], a_l0[0:rows, 0:mad_k])
                        T.copy(b_l1[0:cols, k_slice : k_slice + mad_k], b_l0[0:cols, 0:mad_k])
                        T.gemm(
                            a_l0[0:rows, 0:mad_k],
                            b_l0[0:cols, 0:mad_k],
                            acc[0:rows, 0:cols],
                            transpose_B=True,
                            clear_accum=ki == 0 and m_slice == 0,
                        )
                T.copy(acc[0:rows, 0:cols], D[m_start : m_start + rows, n_start : n_start + cols])
                sched.next_block()

    return main


def _scale_program(m: int, n: int):
    """Build the scale pass: ``out = bf16(acc * token[m] * channel[n])``.

    One vector core takes a row at a time in column tiles, so the row buffers stay
    small however wide the output is; the channel scales stay in unified memory for
    the whole pass and each row carries one token scale.
    """
    # Row buffers cost 12 bytes per column (double-buffered FP32 in, BF16 out) on top
    # of the resident channel scales, so a wide output is walked in column tiles: the
    # widest divisor of N that keeps the pass inside the vector unit's 192 KiB.
    ub_budget = 176 * 1024
    if 16 * n <= ub_budget:
        tile_n = n
    else:
        cap = (ub_budget - 4 * n) // 12
        divisors = [d for d in range(128, min(n, cap) + 1, 128) if n % d == 0]
        if not divisors:
            raise ValueError(f"this scale pass cannot hold N={n}: its channel row is too wide")
        tile_n = divisors[-1]
    threads = min(1024, tile_n)

    @T.prim_func
    def main(
        X: T.Tensor((m, n), "float32"),
        Token: T.Tensor((m,), "float32"),
        Channel: T.Tensor((n,), "float32"),
        Y: T.Tensor((m, n), _OUT_DTYPE),
    ):
        T.func_attr({"tilelang_callee_allocated_outputs": 0})
        with T.Kernel(_NUM_AIV_CORES) as core:
            channel_ub = T.alloc_shared((n,), "float32")
            T.copy(Channel[0:n], channel_ub)
            for row in T.Persistent([m], _NUM_AIV_CORES, core, num_stages=_CAST_STAGES):
                x_ub = T.alloc_shared((tile_n,), "float32")
                out_ub = T.alloc_shared((tile_n,), _OUT_DTYPE)
                T.annotate_buffer_versions({x_ub: _CAST_STAGES, out_ub: _CAST_STAGES})
                for col in T.serial(n // tile_n):
                    T.copy(X[row, col * tile_n : (col + 1) * tile_n], x_ub)
                    with T.SimtVF(threads=threads):
                        for j in T.Parallel(tile_n):
                            out_ub[j] = T.Cast(
                                _OUT_DTYPE,
                                x_ub[j] * Token[row] * channel_ub[col * tile_n + j],
                            )
                    T.copy(out_ub, Y[row, col * tile_n : (col + 1) * tile_n])

    return main


@functools.lru_cache(maxsize=256)
def _make_dequant(n: int, k: int, group_size: int):
    """Compile the unpack pass for ``(n, k)`` and its group size."""
    return tilelang.compile(
        _dequant_program(n, k, group_size),
        target="ascend",
        out_idx=-1,
        pass_configs={"tl.enable_fast_math": False},
    )


@functools.lru_cache(maxsize=256)
def _make_gemm(m: int, n: int, k: int):
    """Compile the FP8 GEMM serving ``(m, n, k)``."""
    tile_m, tile_n, tile_k, stages, blocks = _tiles(m, n, k)
    return tilelang.compile(
        _gemm_program(m, n, k, tile_m, tile_n, tile_k, stages, blocks),
        target="ascend",
        out_idx=-1,
        pass_configs={"tl.enable_fast_math": False},
    )


@functools.lru_cache(maxsize=256)
def _make_scale(m: int, n: int):
    """Compile the token and channel scale pass for ``(m, n)``."""
    return tilelang.compile(
        _scale_program(m, n),
        target="ascend",
        out_idx=-1,
        pass_configs={"tl.enable_fast_math": False},
    )


class AscendGEMMW4A8Kernel(Kernel, GEMMW4A8FwdInterface):
    """W4A8 NT matmul on the Ascend 950 matrix unit.

    The signed INT4 weight is unpacked to its group-scaled ``float8_e4m3fn`` value, the
    FP8 product accumulates in FP32, and the token and channel scales are applied on
    the vector unit when the result leaves.

    The unpacked operand is bit-identical to the reference's own rounding, so the
    result differs from the reference only by FP32 accumulation order; the unpack
    contributes no error of its own.
    """

    devices = frozenset({"npu"})

    @classmethod
    def refusal(cls, call: GEMMW4A8Call) -> Optional[str]:
        if call.group_size != 128:
            return f"this schedule serves a weight group of 128, got {call.group_size}"
        if call.k % 128 or call.n % 128:
            return f"requires K and N multiples of 128, got n={call.n}, k={call.k}"
        return super().refusal(call)

    def __init__(self, call: GEMMW4A8Call):
        device_index = call.device.index
        super().__init__(device_index=device_index)
        name = torch.npu.get_device_name(device_index)
        if "Ascend950" not in name:
            raise RuntimeError(f"These W4A8 schedules require Ascend 950, got {name}")
        # The schedules launch a fixed grid of 32 blocks. Larger devices can run
        # that grid unchanged; do not enlarge it without retuning the schedule.
        if get_num_cube_cores(device_id=device_index or 0) < 32:
            raise RuntimeError("These W4A8 schedules require at least 32 Cube cores")
        self._dequant = _make_dequant(call.n, call.k, call.group_size)
        self._gemm = _make_gemm(call.m, call.n, call.k)
        self._scale = _make_scale(call.m, call.n)

    def forward(
        self,
        activation: torch.Tensor,
        packed_weight: torch.Tensor,
        weight_group_scale: torch.Tensor,
        weight_channel_scale: torch.Tensor,
        activation_scale: torch.Tensor,
    ) -> torch.Tensor:
        weight = self._dequant(packed_weight, weight_group_scale)
        acc = self._gemm(activation, weight)
        return self._scale(acc, activation_scale, weight_channel_scale)


class AscendW4A8RepackKernel(Kernel, W4A8RepackFwdInterface):
    """The weight order this backend's W4A8 GEMM reads: the row-major packing itself.

    Unpacking reads each row's nibbles in place, so the repack only has to hand back a
    tensor of the contracted shape and dtype.
    """

    devices = frozenset({"npu"})

    def __init__(self, call: W4A8RepackCall):
        super().__init__(device_index=None)

    def forward(self, packed_weight: torch.Tensor) -> torch.Tensor:
        return packed_weight.contiguous().clone()


tileops.backend._registry.register_kernel_type(
    "GEMMW4A8FwdOp", "ascend", AscendGEMMW4A8Kernel
)
tileops.backend._registry.register_kernel_type(
    "GEMMW4A8FwdOp", "ascend_repack", AscendW4A8RepackKernel
)
