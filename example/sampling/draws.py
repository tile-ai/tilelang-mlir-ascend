"""Inverse-CDF draw and speculative output assembly with positive-support fallback."""

from functools import lru_cache

import tilelang.language as T

from .radix_reduce import TILE, compile_kernel


@lru_cache(maxsize=128)
def build_draw(batch, drafts, vocab):
    chunks = (vocab + TILE - 1) // TILE
    blocks = min(batch, 32)
    id_width = max(drafts, 1)

    @T.prim_func
    def main(
        Weights: T.Tensor((batch, vocab), "float32"),
        Parts: T.Tensor((batch, chunks), "float32"),
        Uniform: T.Tensor((batch, drafts + 1), "float32"),
        Ids: T.Tensor((batch, id_width), "int32"),
        Accepted: T.Tensor((batch,), "int32"),
        Tokens: T.Tensor((batch, drafts + 1), "int32"),
    ):
        with T.Kernel(blocks, is_npu=True) as (pid, _):
            weights = T.alloc_ub((TILE,), "float32")
            cdf = T.alloc_ub((TILE,), "float32")
            threshold = T.alloc_ub((TILE,), "float32")
            zero = T.alloc_ub((TILE,), "float32")
            indices = T.alloc_ub((TILE,), "int32")
            candidates = T.alloc_ub((TILE,), "int32")
            sentinel = T.alloc_ub((TILE,), "int32")
            negative = T.alloc_ub((TILE,), "int32")
            above = T.alloc_ub((TILE,), "bool")
            positive = T.alloc_ub((TILE,), "bool")
            chosen = T.alloc_ub((1,), "int32")
            last_token = T.alloc_ub((1,), "int32")
            total = T.alloc_ub((1,), "float32")
            point = T.alloc_ub((1,), "float32")
            chunk_id = T.alloc_ub((1,), "int32")
            last_chunk = T.alloc_ub((1,), "int32")
            T.vbrc(T.float32(0.0), zero)
            T.vbrc(T.cast(TILE, "int32"), sentinel)
            T.vbrc(T.cast(-1, "int32"), negative)
            for j in T.serial(TILE):
                indices[j] = j
            for work in T.serial((batch + blocks - 1) // blocks):
                row = work * blocks + pid
                with T.If(row < batch), T.Then():
                    total[0] = T.float32(0)
                    for chunk in T.serial(chunks):
                        total[0] = total[0] + Parts[row, chunk]
                    point[0] = Uniform[row, drafts] * total[0]
                    chunk_id[0] = -1
                    last_chunk[0] = 0
                    for chunk in T.serial(chunks):
                        with T.If(Parts[row, chunk] > 0.0), T.Then():
                            last_chunk[0] = chunk
                            with T.If(chunk_id[0] < 0), T.Then():
                                with T.If(point[0] < Parts[row, chunk]), T.Then():
                                    chunk_id[0] = chunk
                                with T.If(chunk_id[0] < 0), T.Then():
                                    point[0] = point[0] - Parts[row, chunk]
                    # Rounding can exhaust the total, especially for u near one.
                    # Fall back to the last positive chunk, never a trailing zero.
                    with T.If(chunk_id[0] < 0), T.Then():
                        chunk_id[0] = last_chunk[0]
                        point[0] = T.infinity("float32")
                    start = chunk_id[0] * TILE
                    count = T.min(TILE, vocab - start)
                    T.vbrc(T.float32(0.0), weights)
                    T.copy(Weights[row, start : start + count], weights[:count])
                    T.cumsum(weights, cdf, dim=0, reverse=False)
                    T.vbrc(point[0], threshold)
                    T.vcmp(cdf, threshold, above, "gt")
                    T.vcmp(weights, zero, positive, "gt")
                    T.vselect(above, indices, sentinel, candidates)
                    T.vselect(positive, candidates, sentinel, candidates)
                    T.reduce(
                        candidates, chosen, dims=[0], reduce_mode="min", clear=True
                    )
                    T.vselect(positive, indices, negative, candidates)
                    T.reduce(
                        candidates, last_token, dims=[0], reduce_mode="max", clear=True
                    )
                    with T.If(chosen[0] == TILE), T.Then():
                        chosen[0] = last_token[0]
                    for j in T.serial(drafts + 1):
                        Tokens[row, j] = -1
                        with T.If(j < Accepted[row]), T.Then():
                            Tokens[row, j] = Ids[row, j]
                        with T.If(j == Accepted[row]), T.Then():
                            Tokens[row, j] = start + chosen[0]

    return compile_kernel(main)
