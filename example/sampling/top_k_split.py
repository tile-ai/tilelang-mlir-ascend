"""Split-row ordered keys and exact top-k masking, including NaNs and signed zero."""

from functools import lru_cache

import tilelang.language as T

from .radix_reduce import TILE, compile_kernel, storage_info


@lru_cache(maxsize=128)
def build_keys(batch, vocab, dtype):
    storage, sign, infinity, _ = storage_info(dtype)
    chunks = (vocab + TILE - 1) // TILE
    tasks = batch * chunks
    blocks = min(tasks, 32)

    @T.prim_func
    def main(
        Raw: T.Tensor((batch, vocab), storage), Keys: T.Tensor((batch, vocab), "int64")
    ):
        with T.Kernel(blocks, is_npu=True) as (pid, _):
            raw = T.alloc_ub((TILE,), storage)
            keys = T.alloc_ub((TILE,), "int64")
            absolute = T.alloc_ub((1,), "int64")
            for work in T.serial((tasks + blocks - 1) // blocks):
                task = work * blocks + pid
                with T.If(task < tasks), T.Then():
                    row = task // chunks
                    start = task % chunks * TILE
                    count = T.min(TILE, vocab - start)
                    T.copy(Raw[row, start : start + count], raw[:count])
                    for j in T.serial(count):
                        absolute[0] = T.cast(raw[j], "int64")
                        with T.If(raw[j] < 0), T.Then():
                            absolute[0] = absolute[0] + T.int64(sign)
                        keys[j] = T.int64(sign) + absolute[0]
                        with T.If(raw[j] < 0), T.Then():
                            keys[j] = T.int64(sign - 1) - absolute[0]
                        # Both signed zeros must have one rank. All NaNs sort first.
                        with T.If(absolute[0] == 0), T.Then():
                            keys[j] = T.int64(sign)
                        with T.If(absolute[0] > T.int64(infinity)), T.Then():
                            keys[j] = T.int64(2 * sign - 1)
                    T.copy(keys[:count], Keys[row, start : start + count])

    return compile_kernel(main)


@lru_cache(maxsize=128)
def build_top_k_write(batch, vocab, dtype):
    storage, sign, _, negative_inf = storage_info(dtype)
    chunks = (vocab + TILE - 1) // TILE
    tasks = batch * chunks
    blocks = min(tasks, 32)

    @T.prim_func
    def main(
        Raw: T.Tensor((batch, vocab), storage),
        Keys: T.Tensor((batch, vocab), "int64"),
        K: T.Tensor((batch,), "int32"),
        Cutoff: T.Tensor((batch,), "int64"),
        Out: T.Tensor((batch, vocab), storage),
    ):
        with T.Kernel(blocks, is_npu=True) as (pid, _):
            raw = T.alloc_ub((TILE,), storage)
            keys = T.alloc_ub((TILE,), "int64")
            threshold = T.alloc_ub((TILE,), "int64")
            cutoff_scalar = T.alloc_ub((1,), "int64")
            removed = T.alloc_ub((TILE,), storage)
            out = T.alloc_ub((TILE,), storage)
            keep = T.alloc_ub((TILE,), "bool")
            T.vbrc(T.cast(negative_inf, storage), removed)
            for work in T.serial((tasks + blocks - 1) // blocks):
                task = work * blocks + pid
                with T.If(task < tasks), T.Then():
                    row = task // chunks
                    start = task % chunks * TILE
                    count = T.min(TILE, vocab - start)
                    T.vbrc(T.cast(0, storage), raw)
                    T.vbrc(T.int64(-1), keys)
                    T.copy(Raw[row, start : start + count], raw[:count])
                    T.copy(Keys[row, start : start + count], keys[:count])
                    T.copy(Cutoff[row : row + 1], cutoff_scalar)
                    T.vbrc(cutoff_scalar, threshold)
                    T.vcmp(keys, threshold, keep, "ge")
                    T.vselect(keep, raw, removed, out)
                    T.vbrc(T.int64(2 * sign - 1), threshold)
                    T.vcmp(keys, threshold, keep, "lt")
                    T.vselect(keep, out, removed, out)
                    with T.If(K[row] >= vocab), T.Then():
                        T.copy(raw, out)
                    T.copy(out[:count], Out[row, start : start + count])

    return compile_kernel(main)
