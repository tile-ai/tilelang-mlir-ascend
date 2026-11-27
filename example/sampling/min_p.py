"""Row maxima with explicit NaN propagation and bit-preserving MinP selection."""

from functools import lru_cache

import tilelang.language as T

from .radix_reduce import TILE, compile_kernel, storage_info


@lru_cache(maxsize=128)
def build_maximum(batch, vocab, dtype):
    blocks = min(batch, 32)
    chunks = (vocab + TILE - 1) // TILE

    @T.prim_func
    def main(
        X: T.Tensor((batch, vocab), dtype), Maximum: T.Tensor((batch,), "float32")
    ):
        with T.Kernel(blocks, is_npu=True) as (pid, _):
            raw = T.alloc_ub((TILE,), dtype)
            values = T.alloc_ub((TILE,), "float32")
            ninf = T.alloc_ub((TILE,), "float32")
            zeros = T.alloc_ub((TILE,), "float32")
            ones = T.alloc_ub((TILE,), "float32")
            flags = T.alloc_ub((TILE,), "float32")
            nan = T.alloc_ub((TILE,), "bool")
            reduced = T.alloc_ub((1,), "float32")
            maximum = T.alloc_ub((1,), "float32")
            has_nan = T.alloc_ub((1,), "int32")
            T.vbrc(-T.infinity("float32"), ninf)
            T.vbrc(T.float32(0.0), zeros)
            T.vbrc(T.float32(1.0), ones)
            for work in T.serial((batch + blocks - 1) // blocks):
                row = work * blocks + pid
                with T.If(row < batch), T.Then():
                    maximum[0] = -T.infinity("float32")
                    has_nan[0] = 0
                    for chunk in T.serial(chunks):
                        start = chunk * TILE
                        count = T.min(TILE, vocab - start)
                        T.vbrc(-T.infinity(dtype), raw)
                        T.copy(X[row, start : start + count], raw[:count])
                        T.vcast(raw, values, round_mode="rint")
                        T.vcmp(values, values, nan, "ne")
                        T.vselect(nan, ones, zeros, flags)
                        T.reduce(
                            flags, reduced, dims=[0], reduce_mode="sum", clear=True
                        )
                        with T.If(reduced[0] > 0), T.Then():
                            has_nan[0] = 1
                        T.vselect(nan, ninf, values, values)
                        T.reduce(
                            values, reduced, dims=[0], reduce_mode="max", clear=True
                        )
                        maximum[0] = T.max(maximum[0], reduced[0])
                    Maximum[row] = maximum[0]
                    with T.If(has_nan[0] != 0), T.Then():
                        Maximum[row] = T.float32(float("nan"))

    return compile_kernel(main)


@lru_cache(maxsize=128)
def build_min_p_write(batch, vocab, dtype):
    storage, _, _, negative_inf = storage_info(dtype)
    chunks = (vocab + TILE - 1) // TILE
    tasks = batch * chunks
    blocks = min(tasks, 32)

    @T.prim_func
    def main(
        X: T.Tensor((batch, vocab), dtype),
        Raw: T.Tensor((batch, vocab), storage),
        P: T.Tensor((batch,), "float32"),
        Maximum: T.Tensor((batch,), "float32"),
        Out: T.Tensor((batch, vocab), storage),
    ):
        with T.Kernel(blocks, is_npu=True) as (pid, _):
            x = T.alloc_ub((TILE,), dtype)
            values = T.alloc_ub((TILE,), "float32")
            threshold = T.alloc_ub((TILE,), "float32")
            largest = T.alloc_ub((TILE,), "float32")
            parameter = T.alloc_ub((1,), "float32")
            raw = T.alloc_ub((TILE,), storage)
            removed = T.alloc_ub((TILE,), storage)
            out = T.alloc_ub((TILE,), storage)
            mask = T.alloc_ub((TILE,), "bool")
            T.vbrc(T.cast(negative_inf, storage), removed)
            for work in T.serial((tasks + blocks - 1) // blocks):
                task = work * blocks + pid
                with T.If(task < tasks), T.Then():
                    row = task // chunks
                    start = task % chunks * TILE
                    count = T.min(TILE, vocab - start)
                    T.vbrc(T.cast(0, dtype), x)
                    T.vbrc(T.cast(0, storage), raw)
                    T.copy(X[row, start : start + count], x[:count])
                    T.copy(Raw[row, start : start + count], raw[:count])
                    T.vcast(x, values, round_mode="rint")
                    T.copy(P[row : row + 1], parameter)
                    T.vbrc(parameter, threshold)
                    T.vln(threshold, threshold)
                    T.copy(Maximum[row : row + 1], parameter)
                    T.vbrc(parameter, largest)
                    T.vadd(largest, threshold, threshold)
                    T.vcmp(values, threshold, mask, "lt")
                    T.vselect(mask, removed, raw, out)
                    T.copy(out[:count], Out[row, start : start + count])

    return compile_kernel(main)
