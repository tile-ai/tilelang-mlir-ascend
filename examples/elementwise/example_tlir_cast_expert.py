# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Dump TIR -> TLIR for the expert-mode elementwise cast kernel.

Usage:
  TILELANG_TLIR=1 TILELANG_ASCEND_MODE=expert python example_tlir_cast_expert.py
"""

import os

import tilelang
import tilelang.language as T

os.environ.setdefault("TILELANG_ASCEND_MODE", "expert")
os.environ["TILELANG_TLIR"] = "1"

N = 1024


def vec_cast_1d(N, block_N, src_dtype="float32", dst_dtype="float16"):
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((N), src_dtype),
        C: T.Tensor((N), dst_dtype),
        shape: T.int32,
    ):
        with T.Kernel(n_num, is_npu=True) as (cid, _):
            A_VEC = T.alloc_ub((block_N), src_dtype)
            C_VEC = T.alloc_ub((block_N), dst_dtype)
            t0 = cid * block_N
            t0 = shape - t0
            tail_size = T.min(block_N, t0)
            T.copy(A[cid * block_N : cid * block_N + tail_size], A_VEC[0:tail_size])
            T.npuir_cast(A_VEC, C_VEC, round_mode="rint")
            T.copy(C_VEC[0:tail_size], C[cid * block_N : cid * block_N + tail_size])

    return main


if __name__ == "__main__":
    tilelang.cache.clear_cache()
    func = vec_cast_1d(N, 1024)
    mlir = tilelang.engine.lower(func, target="npuir")
    print(mlir)
    print(
        "# Lower TLIR -> HIVM with:\n"
        "#   tilelangir-opt --tilelangir-convert-tl-to-hivm <this.mlir>"
    )
