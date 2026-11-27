# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Dump TIR -> TLIR for the expert-mode reduce-sum kernel.

Usage:
  TILELANG_TLIR=1 TILELANG_ASCEND_MODE=expert python example_tlir_reduce_expert.py
"""

import os

import tilelang
import tilelang.language as T

os.environ.setdefault("TILELANG_ASCEND_MODE", "expert")
os.environ["TILELANG_TLIR"] = "1"

M = 4
N = 8


def reduce_sum_2d(M, N, dtype="float32"):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        C: T.Tensor((M, 1), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            A_VEC = T.alloc_ub((M, N), dtype)
            C_VEC = T.alloc_ub((M, 1), dtype)
            T.copy(A, A_VEC)
            T.npuir_reduce(A_VEC, C_VEC, dims=[1], reduce_mode="sum")
            T.copy(C_VEC, C)

    return main


if __name__ == "__main__":
    tilelang.cache.clear_cache()
    func = reduce_sum_2d(M, N)
    mlir = tilelang.engine.lower(func, target="npuir")
    print(mlir)
    print(
        "# Lower TLIR -> HIVM with:\n"
        "#   tilelangir-opt --tilelangir-convert-tl-to-hivm <this.mlir>"
    )
