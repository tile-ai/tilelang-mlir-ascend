"""W4A8 NT GEMM with exact E4M3FN decoding and FP16 operand carriers.

Adapted from TileOPs PR #82, examples/deepseek_v4/example_fp8_gemm_kernel.py,
examples/gemm/matmul.py and testing/npuir/indexing_ops/test_slice_gather_exp.py.
The public tensors retain their FP8/packed INT4 types. Only the internal Cube
operands use FP16: all finite E4M3FN values are exactly representable in FP16.
This is not a native FP8 throughput implementation. Set Expert mode before
importing TileLang. Execution never falls back to a PyTorch matrix multiply.
"""

import os
from functools import lru_cache

import torch

FP8 = torch.float8_e4m3fn
BM, BN, BK = 32, 128, 128
BLOCKS = 32


@lru_cache(maxsize=1)
def _host_tables():
    """Only constant encodings, never user data, are evaluated on the host.

    The 4096-entry table covers every FP8 scale bit pattern and signed INT4
    nibble, including negative scales, subnormals, overflow and NaN. PyTorch's
    nearest-even E4M3FN conversion matches the upstream workload reference.
    """
    decode = (
        torch.arange(256, dtype=torch.int16, device="cpu")
        .to(torch.uint8)
        .view(FP8)
        .float()
    )
    nibble = torch.arange(16, dtype=torch.int32, device="cpu")
    signed = torch.where(nibble < 8, nibble, nibble - 16).float()
    rounded = (decode[:, None] * signed[None, :]).to(FP8).to(torch.float16)
    return decode.to(torch.float16).reshape(1, 256), rounded.reshape(1, 4096)


@lru_cache(maxsize=8)
def _device_tables(device_index):
    device = torch.device("npu", device_index)
    return tuple(table.to(device) for table in _host_tables())


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


@lru_cache(maxsize=64)
def _make_decode(m, k, device_index):
    """FP8 storage bytes -> exact FP16 values, one 128-element vector per task."""
    import tilelang.language as T

    tasks = m * (k // BK)

    @T.prim_func
    def main(
        A: T.Tensor((m, k), "uint8"),
        Table: T.Tensor((1, 256), "float16"),
        Out: T.Tensor((m, k), "float16"),
    ):
        with T.Kernel(BLOCKS, is_npu=True) as (pid, _):
            table = T.alloc_ub((1, 256), "float16")
            raw = T.alloc_ub((1, BK), "uint8")
            wide = T.alloc_ub((1, BK), "float16")
            indices = T.alloc_ub((1, BK), "int32")
            byte_mask = T.alloc_ub((1, BK), "int32")
            result = T.alloc_ub((1, BK), "float16")
            T.copy(Table, table)
            T.vbrc(255, byte_mask)
            for work in T.serial(T.ceildiv(tasks, BLOCKS)):
                task = work * BLOCKS + pid
                with T.If(task < tasks), T.Then():
                    row = task // (k // BK)
                    col = task % (k // BK) * BK
                    T.copy(A[row : row + 1, col : col + BK], raw)
                    T.vcast(raw, wide)
                    T.vcast(wide, indices)
                    # NPUIR lowers byte buffers to signless i8. Recover the
                    # unsigned encoding regardless of the widening convention.
                    T.vand(indices, byte_mask, indices)
                    T.gather(table, result, indices)
                    T.copy(result, Out[row : row + 1, col : col + BK])

    return _compile(main)


@lru_cache(maxsize=64)
def _make_dequant(n, k, device_index):
    """Decode both nibbles and round q*scale to E4M3FN through one lookup."""
    import tilelang.language as T

    groups = k // BK
    tasks = n * groups

    @T.prim_func
    def main(
        W: T.Tensor((n, k // 2), "uint8"),
        GS: T.Tensor((n, groups), "uint8"),
        Table: T.Tensor((1, 4096), "float16"),
        Out: T.Tensor((n, k), "float16"),
    ):
        with T.Kernel(BLOCKS, is_npu=True) as (pid, _):
            table = T.alloc_ub((1, 4096), "float16")
            raw = T.alloc_ub((1, BK // 2), "uint8")
            wide = T.alloc_ub((1, BK // 2), "float16")
            packed = T.alloc_ub((1, BK // 2), "int32")
            low = T.alloc_ub((1, BK // 2), "int32")
            high = T.alloc_ub((1, BK // 2), "int32")
            mask = T.alloc_ub((1, BK // 2), "int32")
            byte_mask = T.alloc_ub((1, BK // 2), "int32")
            shift = T.alloc_ub((1, BK // 2), "int32")
            even = T.alloc_ub((1, BK // 2), "float16")
            odd = T.alloc_ub((1, BK // 2), "float16")
            result = T.alloc_ub((1, BK), "float16")
            T.copy(Table, table)
            T.vbrc(15, mask)
            T.vbrc(255, byte_mask)
            T.vbrc(4, shift)
            for work in T.serial(T.ceildiv(tasks, BLOCKS)):
                task = work * BLOCKS + pid
                with T.If(task < tasks), T.Then():
                    row = task // groups
                    group = task % groups
                    # Mask after promotion: signed scalar extension must not
                    # turn FP8 sign-bit encodings into negative table indices.
                    scale_base = (T.cast(GS[row, group], "int32") & 255) * 16
                    T.copy(W[row : row + 1, group * 64 : group * 64 + 64], raw)
                    T.vcast(raw, wide)
                    T.vcast(wide, packed)
                    T.vand(packed, byte_mask, packed)
                    T.vand(packed, mask, low)
                    T.vshr(packed, shift, high)
                    T.vadd(low, scale_base, low)
                    T.vadd(high, scale_base, high)
                    T.gather(table, even, low)
                    T.gather(table, odd, high)
                    T.interleave(even, odd, result)
                    T.copy(result, Out[row : row + 1, group * BK : (group + 1) * BK])

    return _compile(main)


@lru_cache(maxsize=64)
def _make_gemm(m, n, k, device_index):
    """FP16 carriers, FP32 accumulation; the last M tile is explicitly bounded."""
    import tilelang.language as T

    n_tiles = n // BN
    tasks = ((m + BM - 1) // BM) * n_tiles

    @T.prim_func
    def main(
        A: T.Tensor((m, k), "float16"),
        B: T.Tensor((n, k), "float16"),
        Out: T.Tensor((m, n), "float32"),
    ):
        with T.Kernel(BLOCKS, is_npu=True) as (pid, _):
            with T.Scope("Cube"):
                a_l1 = T.alloc_L1((BM, BK), "float16")
                b_l1 = T.alloc_L1((BN, BK), "float16")
                acc = T.alloc_L0C((BM, BN), "float32")
                for work in T.serial(T.ceildiv(tasks, BLOCKS)):
                    task = work * BLOCKS + pid
                    with T.If(task < tasks), T.Then():
                        row = task // n_tiles * BM
                        col = task % n_tiles * BN
                        rows = T.min(BM, m - row)
                        for step in T.serial(k // BK):
                            start = step * BK
                            T.copy(
                                A[row : row + rows, start : start + BK],
                                a_l1[:rows, :BK],
                            )
                            T.copy(B[col : col + BN, start : start + BK], b_l1)
                            T.gemm(
                                a_l1,
                                b_l1,
                                acc,
                                size=[rows, BK, BN],
                                initC=(step == 0),
                                b_transpose=True,
                            )
                        T.store_fixpipe(
                            acc, Out[row, col], size=[rows, BN], enable_nz2nd=True
                        )

    return _compile(main)


@lru_cache(maxsize=64)
def _make_scale(m, n, device_index):
    """Keep token-first FP32 multiplication order and round to BF16 only once."""
    import tilelang.language as T

    tasks = m * (n // BN)

    @T.prim_func
    def main(
        X: T.Tensor((m, n), "float32"),
        Token: T.Tensor((m,), "float32"),
        Channel: T.Tensor((1, n), "float32"),
        Out: T.Tensor((m, n), "bfloat16"),
    ):
        with T.Kernel(BLOCKS, is_npu=True) as (pid, _):
            values = T.alloc_ub((1, BN), "float32")
            channel = T.alloc_ub((1, BN), "float32")
            result = T.alloc_ub((1, BN), "bfloat16")
            for work in T.serial(T.ceildiv(tasks, BLOCKS)):
                task = work * BLOCKS + pid
                with T.If(task < tasks), T.Then():
                    row = task // (n // BN)
                    col = task % (n // BN) * BN
                    token = Token[row]
                    T.copy(X[row : row + 1, col : col + BN], values)
                    T.copy(Channel[:, col : col + BN], channel)
                    T.vmul(values, token, values)
                    T.vmul(values, channel, values)
                    T.vcast(values, result, round_mode="rint")
                    T.copy(result, Out[row : row + 1, col : col + BN])

    return _compile(main)


def _validate(a, w, gs, channel, token):
    expected = (
        ("activation", a, FP8, 2),
        ("packed_weight", w, torch.uint8, 2),
        ("weight_group_scale", gs, FP8, 2),
        ("weight_channel_scale", channel, torch.float32, 1),
        ("activation_scale", token, torch.float32, 1),
    )
    for name, value, dtype, rank in expected:
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != dtype
            or value.ndim != rank
        ):
            raise ValueError(f"{name} must be a rank-{rank} {dtype} tensor")
    m, k = a.shape
    n = w.shape[0]
    if m <= 0 or n <= 0 or k <= 0 or n % BN or k % BK:
        raise ValueError(
            "M must be positive; N and K must be positive multiples of 128"
        )
    if w.shape != (n, k // 2) or gs.shape != (n, k // BK):
        raise ValueError(
            "packed_weight must be [N,K/2] and weight_group_scale [N,K/128]"
        )
    if channel.shape != (n,) or token.shape != (m,):
        raise ValueError("weight_channel_scale must be [N] and activation_scale [M]")
    if max(m * k, n * k, m * n) >= 2**31:
        raise ValueError("tensor element counts must fit int32 indexing")
    for name, value, _, _ in expected:
        if value.device != a.device or not value.is_contiguous():
            raise ValueError(f"{name} must be contiguous on {a.device}")
    return m, n, k


class AscendW4A8RepackKernel:
    """The NPUIR backend consumes the original row-major nibble order."""

    def __call__(self, packed_weight):
        if (
            not isinstance(packed_weight, torch.Tensor)
            or packed_weight.dtype != torch.uint8
            or packed_weight.ndim != 2
        ):
            raise ValueError("repack expects a rank-2 uint8 packed_weight")
        n, packed_k = packed_weight.shape
        if n <= 0 or n % 16 or packed_k <= 0 or packed_k % 64:
            raise ValueError(
                "repack requires positive N divisible by 16 and K divisible by 128"
            )
        return packed_weight.contiguous().clone()

    forward = __call__


class GEMMW4A8FwdOp:
    """FP8 A @ round_fp8(INT4 W * group_scale).T, then token/channel scaling.

    Inputs: A[M,K] E4M3FN, W[N,K/2] uint8 (even K in the low nibble),
    group_scale[N,K/128] E4M3FN, channel[N] and token[M] FP32.
    Output: BF16[M,N]. M>0; N,K are positive multiples of 128.
    All inputs must be contiguous on the same NPU. Inputs are never modified;
    repeated calls read fresh weights/scales. Repack returns an independent copy.
    """

    def __init__(self, group_size=128):
        if type(group_size) is not int or group_size != 128:
            raise ValueError("only group_size=128 is supported")
        self.group_size = group_size

    def repack(self, packed_weight):
        return AscendW4A8RepackKernel()(packed_weight)

    def __call__(
        self,
        activation,
        packed_weight,
        weight_group_scale,
        weight_channel_scale,
        activation_scale,
    ):
        m, n, k = _validate(
            activation,
            packed_weight,
            weight_group_scale,
            weight_channel_scale,
            activation_scale,
        )
        if activation.device.type != "npu":
            raise RuntimeError("W4A8 GEMM requires an Ascend NPU; no CPU fallback")
        if os.environ.get("TILELANG_ASCEND_MODE", "Expert") != "Expert":
            raise RuntimeError(
                "set TILELANG_ASCEND_MODE=Expert before importing TileLang"
            )
        device = activation.device
        with torch.npu.device(device):
            decode, dequant = _device_tables(device.index)
            a = torch.empty((m, k), dtype=torch.float16, device=device)
            w = torch.empty((n, k), dtype=torch.float16, device=device)
            acc = torch.empty((m, n), dtype=torch.float32, device=device)
            out = torch.empty((m, n), dtype=torch.bfloat16, device=device)
            _make_decode(m, k, device.index)(activation.view(torch.uint8), decode, a)
            _make_dequant(n, k, device.index)(
                packed_weight, weight_group_scale.view(torch.uint8), dequant, w
            )
            _make_gemm(m, n, k, device.index)(a, w, acc)
            _make_scale(m, n, device.index)(
                acc, activation_scale, weight_channel_scale.view(1, n), out
            )
        return out

    forward = __call__


AscendGEMMW4A8Kernel = GEMMW4A8FwdOp
