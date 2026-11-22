# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
#
# Dump TIR -> TLIR for a kernel using T.block_barrier and T.subblock_barrier.
# NOTE: uses T.npuir_add instead of T.vrelu since vrelu is not yet merged
# into this branch's base.

import os
import tilelang
import tilelang.language as T

os.environ.setdefault("TILELANG_ASCEND_MODE", "expert")
os.environ["TILELANG_TLIR"] = "1"

N = 1024


def sync_block_1d(N, block_N, dtype="float32"):
    n_num = N // block_N

    @T.prim_func
    def main(A: T.Tensor((N), dtype), B: T.Tensor((N), dtype), C: T.Tensor((N), dtype)):
        with T.Kernel(n_num, is_npu=True) as (cid, _):
            A_VEC = T.alloc_ub((block_N), dtype)
            B_VEC = T.alloc_ub((block_N), dtype)
            C_VEC = T.alloc_ub((block_N), dtype)
            T.copy(A[cid * block_N : cid * block_N + block_N], A_VEC)
            T.copy(B[cid * block_N : cid * block_N + block_N], B_VEC)
            with T.rs("PIPE_MTE2"):
                T.block_barrier(0)
                T.vadd(A_VEC, B_VEC, C_VEC)
                T.subblock_barrier(1)
            T.copy(C_VEC, C[cid * block_N : cid * block_N + block_N])

    return main


if __name__ == "__main__":
    tilelang.cache.clear_cache()
    func = sync_block_1d(N, 1024)
    print(tilelang.engine.lower(func, target="npuir"))
