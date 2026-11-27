"""Bounded-UB NPUIR radix selection shared by top-k and nucleus filtering."""

import os
from functools import lru_cache

import tilelang
import tilelang.language as T

TILE = 512


def compile_kernel(main):
    if os.environ.get("TILELANG_ASCEND_MODE", "Expert") != "Developer":
        raise RuntimeError(
            "sampling requires TILELANG_ASCEND_MODE=Developer before importing tilelang"
        )
    return tilelang.compile(
        main,
        target="npuir",
        pass_configs={
            tilelang.PassConfigKey.NPUIR_ENABLE_AUTO_MULTI_BUFFER: True,
            tilelang.PassConfigKey.NPUIR_DISABLE_HIVM_AUTO_INJECT_SYNC: False,
        },
    )


def storage_info(dtype):
    # Signed storage type, sign bit, absolute infinity, negative-infinity bits.
    return {
        "float16": ("int16", 32768, 31744, -1024),
        "bfloat16": ("int16", 32768, 32640, -128),
        "float32": ("int32", 2147483648, 2139095040, -8388608),
    }[dtype]


@lru_cache(maxsize=128)
def build_cutoff(batch, vocab, key_dtype, bits, weighted=False):
    """Largest key whose inclusive upper tail holds at least the requested count/mass.

    NaNs sort above +inf for top-k. Nucleus keys are nonnegative float32
    probability storage words, so probability ties (not merely logit ties) agree.
    The traversal and reduction grouping do not depend on batch/launch size.
    """
    chunks = (vocab + TILE - 1) // TILE
    blocks = min(batch, 32)
    limit_dtype = "float32" if weighted else "int32"
    weight_width = vocab if weighted else 1

    @T.prim_func
    def main(
        Keys: T.Tensor((batch, vocab), key_dtype),
        Weights: T.Tensor((batch, weight_width), "float32"),
        Limit: T.Tensor((batch,), limit_dtype),
        Cutoff: T.Tensor((batch,), "int64"),
    ):
        with T.Kernel(blocks, is_npu=True) as (pid, _):
            keys = T.alloc_ub((TILE,), key_dtype)
            pivot = T.alloc_ub((TILE,), key_dtype)
            weights = T.alloc_ub((TILE,), "float32")
            zero = T.alloc_ub((TILE,), "float32")
            selected = T.alloc_ub((TILE,), "float32")
            above = T.alloc_ub((TILE,), "bool")
            reduced = T.alloc_ub((1,), "float32")
            total = T.alloc_ub((1,), "float32")
            cutoff = T.alloc_ub((1,), "int64")
            step = T.alloc_ub((1,), "int64")
            T.vbrc(T.float32(0.0), zero)
            for work in T.serial((batch + blocks - 1) // blocks):
                row = work * blocks + pid
                with T.If(row < batch), T.Then():
                    cutoff[0] = T.int64(0)
                    step[0] = T.int64(2 ** (bits - 1))
                    for _bit in T.serial(bits):
                        total[0] = T.float32(0)
                        T.vbrc(T.cast(cutoff[0] + step[0], key_dtype), pivot)
                        for chunk in T.serial(chunks):
                            start = chunk * TILE
                            count = T.min(TILE, vocab - start)
                            T.vbrc(T.cast(-1, key_dtype), keys)
                            T.copy(Keys[row, start : start + count], keys[:count])
                            T.vcmp(keys, pivot, above, "ge")
                            if weighted:
                                T.vbrc(T.float32(0.0), weights)
                                T.copy(
                                    Weights[row, start : start + count], weights[:count]
                                )
                            else:
                                T.vbrc(T.float32(1.0), weights)
                            T.vselect(above, weights, zero, selected)
                            T.reduce(
                                selected,
                                reduced,
                                dims=[0],
                                reduce_mode="sum",
                                clear=True,
                            )
                            total[0] = total[0] + reduced[0]
                        with T.If(total[0] >= T.cast(Limit[row], "float32")), T.Then():
                            cutoff[0] = cutoff[0] + step[0]
                        step[0] = step[0] // T.int64(2)
                    Cutoff[row] = cutoff[0]

    return compile_kernel(main)
