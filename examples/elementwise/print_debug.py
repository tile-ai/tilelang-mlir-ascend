# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Dump TIR -> TLIR for scalar + buffer print debug kernels."""

import os
import tilelang
import tilelang.language as T

os.environ.setdefault("TILELANG_ASCEND_MODE", "expert")
os.environ["TILELANG_TLIR"] = "1"

N = 1024


def print_debug_1d(N, block_N, dtype="float32"):
    n_num = N // block_N

    @T.prim_func
    def main(A: T.Tensor((N), dtype), shape: T.int32):
        with T.Kernel(n_num, is_npu=True) as (cid, _):
            A_VEC = T.alloc_ub((block_N), dtype)
            T.copy(A[cid * block_N : cid * block_N + block_N], A_VEC)
            T.print(A_VEC, "A_VEC")  # buffer print
            T.print(cid, "cid")  # scalar print

    return main


if __name__ == "__main__":
    tilelang.cache.clear_cache()
    func = print_debug_1d(N, 1024)
    print(tilelang.engine.lower(func, target="npuir"))
