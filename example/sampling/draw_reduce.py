"""Residual weights and split-row sums for categorical draws."""

from functools import lru_cache

import tilelang.language as T

from .radix_reduce import TILE, compile_kernel


@lru_cache(maxsize=128)
def build_parts(batch, drafts, vocab, chain):
    chunks = (vocab + TILE - 1) // TILE
    tasks = batch * chunks
    blocks = min(tasks, 32)
    draft_width = drafts if chain else 1

    @T.prim_func
    def main(
        Draft: T.Tensor((batch, draft_width, vocab), "float32"),
        Target: T.Tensor((batch, drafts + 1, vocab), "float32"),
        Accepted: T.Tensor((batch,), "int32"),
        Weights: T.Tensor((batch, vocab), "float32"),
        Parts: T.Tensor((batch, chunks), "float32"),
    ):
        with T.Kernel(blocks, is_npu=True) as (pid, _):
            weights = T.alloc_ub((TILE,), "float32")
            draft = T.alloc_ub((TILE,), "float32")
            zero = T.alloc_ub((TILE,), "float32")
            reduced = T.alloc_ub((1,), "float32")
            T.vbrc(T.float32(0.0), zero)
            for work in T.serial((tasks + blocks - 1) // blocks):
                task = work * blocks + pid
                with T.If(task < tasks), T.Then():
                    row = task // chunks
                    chunk = task % chunks
                    start = chunk * TILE
                    count = T.min(TILE, vocab - start)
                    T.vbrc(T.float32(0.0), weights)
                    T.copy(
                        Target[row, Accepted[row], start : start + count],
                        weights[:count],
                    )
                    if chain:
                        with T.If(Accepted[row] < drafts), T.Then():
                            T.vbrc(T.float32(0.0), draft)
                            T.copy(
                                Draft[row, Accepted[row], start : start + count],
                                draft[:count],
                            )
                            T.vsub(weights, draft, weights)
                            T.vmax(weights, zero, weights)
                    T.reduce(weights, reduced, dims=[0], reduce_mode="sum", clear=True)
                    Parts[row, chunk] = reduced[0]
                    T.copy(weights[:count], Weights[row, start : start + count])

    return compile_kernel(main)
