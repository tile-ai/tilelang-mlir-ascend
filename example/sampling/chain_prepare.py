"""Philox4x32-10 and exact accepted-prefix preparation in NPUIR."""

from functools import lru_cache

import tilelang.language as T

from .radix_reduce import compile_kernel

RNG_TILE = 256


@T.macro
def _mulhilo(a, hi, lo, multiplier: int):
    """Split the product into 16-bit limbs to avoid signed int64 overflow."""
    temp = T.alloc_ub((RNG_TILE,), "int64")
    shift = T.alloc_ub((1,), "int64")
    T.vand(a, 65535, lo)
    T.vmul(lo, multiplier, lo)
    T.vbrc(T.int64(16), shift)
    T.vshr(a, shift, hi)
    T.vmul(hi, multiplier, hi)
    T.vand(hi, 65535, temp)
    T.vmul(temp, 65536, temp)
    T.vadd(lo, temp, lo)
    T.vshr(hi, shift, hi)
    T.vbrc(T.int64(32), shift)
    T.vshr(lo, shift, temp)
    T.vadd(hi, temp, hi)
    T.vand(hi, 4294967295, hi)
    T.vand(lo, 4294967295, lo)


@lru_cache(maxsize=128)
def build_uniforms(batch, draws):
    total = batch * draws
    tiles = (total + RNG_TILE - 1) // RNG_TILE
    blocks = min(tiles, 32)

    @T.prim_func
    def main(
        Seed: T.Tensor((1,), "int64"),
        Offset: T.Tensor((1,), "int64"),
        Uniform: T.Tensor((total,), "float32"),
    ):
        with T.Kernel(blocks, is_npu=True) as (pid, _):
            c0 = T.alloc_ub((RNG_TILE,), "int64")
            c1 = T.alloc_ub((RNG_TILE,), "int64")
            c2 = T.alloc_ub((RNG_TILE,), "int64")
            c3 = T.alloc_ub((RNG_TILE,), "int64")
            k0 = T.alloc_ub((RNG_TILE,), "int64")
            k1 = T.alloc_ub((RNG_TILE,), "int64")
            hi0 = T.alloc_ub((RNG_TILE,), "int64")
            lo0 = T.alloc_ub((RNG_TILE,), "int64")
            hi1 = T.alloc_ub((RNG_TILE,), "int64")
            lo1 = T.alloc_ub((RNG_TILE,), "int64")
            shift = T.alloc_ub((1,), "int64")
            uniform = T.alloc_ub((RNG_TILE,), "float32")
            parameter = T.alloc_ub((1,), "int64")
            for work in T.serial((tiles + blocks - 1) // blocks):
                tile = work * blocks + pid
                with T.If(tile < tiles), T.Then():
                    start = tile * RNG_TILE
                    count = T.min(RNG_TILE, total - start)
                    for j in T.serial(RNG_TILE):
                        c0[j] = T.cast((start + j) % draws, "int64")
                        c1[j] = T.cast((start + j) // draws, "int64")
                    T.copy(Offset, parameter)
                    T.vbrc(parameter, c2)
                    T.vbrc(T.int64(32), shift)
                    T.vshr(c2, shift, c3)
                    T.vand(c2, 4294967295, c2)
                    T.vand(c3, 4294967295, c3)
                    T.copy(Seed, parameter)
                    T.vbrc(parameter, k0)
                    T.vshr(k0, shift, k1)
                    T.vand(k0, 4294967295, k0)
                    T.vand(k1, 4294967295, k1)
                    for round_id in T.serial(10):
                        with T.If(round_id > 0), T.Then():
                            T.vadd(k0, 2654435769, k0)
                            T.vand(k0, 4294967295, k0)
                            T.vadd(k1, 3144134277, k1)
                            T.vand(k1, 4294967295, k1)
                        _mulhilo(c0, hi0, lo0, 3528531795)
                        _mulhilo(c2, hi1, lo1, 3449720151)
                        T.vxor(hi1, c1, c0)
                        T.vxor(c0, k0, c0)
                        T.vxor(hi0, c3, c2)
                        T.vxor(c2, k1, c2)
                        T.copy(lo1, c1)
                        T.copy(lo0, c3)
                    T.vbrc(T.int64(8), shift)
                    T.vshr(c0, shift, c0)
                    T.vcast(c0, uniform, round_mode="rint")
                    T.vmul(uniform, 2.0**-24, uniform)
                    T.copy(uniform[:count], Uniform[start : start + count])

    return compile_kernel(main)


@lru_cache(maxsize=128)
def build_acceptance(batch, drafts, vocab):
    blocks = min(batch, 32)

    @T.prim_func
    def main(
        Draft: T.Tensor((batch, drafts, vocab), "float32"),
        Ids: T.Tensor((batch, drafts), "int32"),
        Target: T.Tensor((batch, drafts + 1, vocab), "float32"),
        Uniform: T.Tensor((batch, drafts + 1), "float32"),
        Accepted: T.Tensor((batch,), "int32"),
    ):
        with T.Kernel(blocks, is_npu=True) as (pid, _):
            accepted = T.alloc_ub((1,), "int32")
            for work in T.serial((batch + blocks - 1) // blocks):
                row = work * blocks + pid
                with T.If(row < batch), T.Then():
                    accepted[0] = drafts
                    for j in T.serial(drafts):
                        token = Ids[row, j]
                        with (
                            T.If(
                                (j < accepted[0])
                                & (
                                    Uniform[row, j] * Draft[row, j, token]
                                    >= Target[row, j, token]
                                )
                            ),
                            T.Then(),
                        ):
                            accepted[0] = j
                    Accepted[row] = accepted[0]

    return compile_kernel(main)
