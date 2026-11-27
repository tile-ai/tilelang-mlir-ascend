"""Float32 softmax and nucleus selection; TopKTopP composes the two filters."""

from functools import lru_cache

import tilelang.language as T

from .radix_reduce import TILE, compile_kernel, storage_info


@lru_cache(maxsize=128)
def build_probabilities(batch, vocab, dtype):
    blocks = min(batch, 32)
    chunks = (vocab + TILE - 1) // TILE

    @T.prim_func
    def main(
        X: T.Tensor((batch, vocab), dtype),
        Maximum: T.Tensor((batch,), "float32"),
        Probabilities: T.Tensor((batch, vocab), "float32"),
    ):
        with T.Kernel(blocks, is_npu=True) as (pid, _):
            raw = T.alloc_ub((TILE,), dtype)
            values = T.alloc_ub((TILE,), "float32")
            largest = T.alloc_ub((TILE,), "float32")
            denominator = T.alloc_ub((TILE,), "float32")
            reduced = T.alloc_ub((1,), "float32")
            mass = T.alloc_ub((1,), "float32")
            maximum = T.alloc_ub((1,), "float32")
            for work in T.serial((batch + blocks - 1) // blocks):
                row = work * blocks + pid
                with T.If(row < batch), T.Then():
                    mass[0] = T.float32(0)
                    T.copy(Maximum[row : row + 1], maximum)
                    T.vbrc(maximum, largest)
                    for chunk in T.serial(chunks):
                        start = chunk * TILE
                        count = T.min(TILE, vocab - start)
                        T.vbrc(-T.infinity(dtype), raw)
                        T.copy(X[row, start : start + count], raw[:count])
                        T.vcast(raw, values, round_mode="rint")
                        T.vsub(values, largest, values)
                        T.vexp(values, values)
                        T.reduce(
                            values, reduced, dims=[0], reduce_mode="sum", clear=True
                        )
                        mass[0] = mass[0] + reduced[0]
                        T.copy(
                            values[:count], Probabilities[row, start : start + count]
                        )
                    T.vbrc(mass[0], denominator)
                    for chunk in T.serial(chunks):
                        start = chunk * TILE
                        count = T.min(TILE, vocab - start)
                        T.vbrc(T.float32(0.0), values)
                        T.copy(
                            Probabilities[row, start : start + count], values[:count]
                        )
                        T.vdiv(values, denominator, values)
                        T.copy(
                            values[:count], Probabilities[row, start : start + count]
                        )

    return compile_kernel(main)


@lru_cache(maxsize=128)
def build_top_p_write(batch, vocab, dtype):
    storage, _, _, negative_inf = storage_info(dtype)
    chunks = (vocab + TILE - 1) // TILE
    tasks = batch * chunks
    blocks = min(tasks, 32)

    @T.prim_func
    def main(
        Raw: T.Tensor((batch, vocab), storage),
        ProbabilityBits: T.Tensor((batch, vocab), "int32"),
        P: T.Tensor((batch,), "float32"),
        Maximum: T.Tensor((batch,), "float32"),
        Cutoff: T.Tensor((batch,), "int64"),
        Out: T.Tensor((batch, vocab), storage),
    ):
        with T.Kernel(blocks, is_npu=True) as (pid, _):
            raw = T.alloc_ub((TILE,), storage)
            keys = T.alloc_ub((TILE,), "int32")
            threshold = T.alloc_ub((TILE,), "int32")
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
                    T.vbrc(T.cast(0, "int32"), keys)
                    T.copy(Raw[row, start : start + count], raw[:count])
                    T.copy(ProbabilityBits[row, start : start + count], keys[:count])
                    T.vbrc(T.cast(Cutoff[row], "int32"), threshold)
                    T.vcmp(keys, threshold, keep, "ge")
                    T.vselect(keep, raw, removed, out)
                    # The endpoint contract excludes underflowed zero probabilities.
                    with T.If(P[row] == 1.0), T.Then():
                        T.vbrc(T.cast(0, "int32"), threshold)
                        T.vcmp(keys, threshold, keep, "gt")
                        T.vselect(keep, raw, removed, out)
                    with T.If(P[row] == 0.0), T.Then():
                        T.copy(removed, out)
                    # NaN softmax rows compare false to every p and pass through.
                    with (
                        T.If(
                            (Maximum[row] != Maximum[row])
                            | (Maximum[row] == T.infinity("float32"))
                            | (Maximum[row] == -T.infinity("float32"))
                        ),
                        T.Then(),
                    ):
                        T.copy(raw, out)
                    T.copy(out[:count], Out[row, start : start + count])

    return compile_kernel(main)
