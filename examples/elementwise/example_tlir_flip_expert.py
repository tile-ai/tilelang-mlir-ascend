# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Dump TIR -> TLIR for the expert-mode elementwise flip kernel.

Usage:
  TILELANG_TLIR=1 TILELANG_ASCEND_MODE=expert python example_tlir_flip_expert.py
"""

import os

import tilelang
import tilelang.language as T

os.environ.setdefault("TILELANG_ASCEND_MODE", "expert")
os.environ["TILELANG_TLIR"] = "1"

BLOCK_M = 128
BLOCK_N = 256


def vec_flip_2d(block_M, block_N, dtype="float16"):

    @T.prim_func
    def main(
        A: T.Tensor((block_M, block_N), dtype),
        C: T.Tensor((block_M, block_N), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            A_VEC = T.alloc_ub((block_M, block_N), dtype)
            C_VEC = T.alloc_ub((block_M, block_N), dtype)
            T.copy(A, A_VEC)
            T.npuir_flip(A_VEC, C_VEC, 1)
            T.copy(C_VEC, C)

    return main


if __name__ == "__main__":
    tilelang.cache.clear_cache()
    func = vec_flip_2d(BLOCK_M, BLOCK_N)
    mlir = tilelang.engine.lower(func, target="npuir")
    print(mlir)
    print(
        "# Lower TLIR -> HIVM with:\n"
        "#   tilelangir-opt --tilelangir-convert-tl-to-hivm <this.mlir>"
    )
