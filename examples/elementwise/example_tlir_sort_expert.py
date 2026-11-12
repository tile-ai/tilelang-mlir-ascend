# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Dump TIR -> TLIR for the expert-mode vector sort kernel.

Usage:
  TILELANG_TLIR=1 TILELANG_ASCEND_MODE=expert python example_tlir_sort_expert.py
"""

import os

import tilelang
import tilelang.language as T

os.environ.setdefault("TILELANG_ASCEND_MODE", "expert")
os.environ["TILELANG_TLIR"] = "1"

N = 1024


def vec_sort_1d(N, block_N, dtype="float32", idx_dtype="int32"):
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((N), dtype),
        VAL: T.Tensor((N), dtype),
        IDX: T.Tensor((N), idx_dtype),
        shape: T.int32,
    ):
        with T.Kernel(n_num, is_npu=True) as (cid, _):
            A_VEC = T.alloc_ub((block_N), dtype)
            VAL_VEC = T.alloc_ub((block_N), dtype)
            IDX_VEC = T.alloc_ub((block_N), idx_dtype)
            t0 = cid * block_N
            t0 = shape - t0
            tail_size = T.min(block_N, t0)
            T.copy(A[cid * block_N : cid * block_N + tail_size], A_VEC[0:tail_size])
            T.npuir_sort(A_VEC, VAL_VEC, IDX_VEC, descending=False, sort_axis=-1)
            T.copy(VAL_VEC[0:tail_size], VAL[cid * block_N : cid * block_N + tail_size])
            T.copy(IDX_VEC[0:tail_size], IDX[cid * block_N : cid * block_N + tail_size])

    return main


if __name__ == "__main__":
    tilelang.cache.clear_cache()
    func = vec_sort_1d(N, 1024)
    mlir = tilelang.engine.lower(func, target="npuir")
    print(mlir)
    print(
        "# Lower TLIR -> HIVM with:\n"
        "#   tilelangir-opt --tilelangir-convert-tl-to-hivm <this.mlir>"
    )
