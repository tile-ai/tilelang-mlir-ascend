# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Dump TIR -> TLIR for the expert-mode transpose kernel.

Usage:
  TILELANG_TLIR=1 TILELANG_ASCEND_MODE=expert python example_tlir_transpose_expert.py
"""

import os

import tilelang
import tilelang.language as T

os.environ.setdefault("TILELANG_ASCEND_MODE", "expert")
os.environ["TILELANG_TLIR"] = "1"

M, N = 32, 8


def vec_transpose(M, N, dtype="float32"):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        C: T.Tensor((N, M), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            A_VEC = T.alloc_ub((M, N), dtype)
            C_VEC = T.alloc_ub((N, M), dtype)
            T.copy(A, A_VEC)
            T.npuir_transpose(
                A_VEC[:M, :N],
                C_VEC[:N, :M],
                [1, 0],
            )
            T.copy(C_VEC, C)

    return main


if __name__ == "__main__":
    tilelang.cache.clear_cache()
    func = vec_transpose(M, N)
    mlir = tilelang.engine.lower(func, target="npuir")
    print(mlir)
    print(
        "# Lower TLIR -> HIVM with:\n"
        "#   tilelangir-opt --tilelangir-convert-tl-to-hivm <this.mlir>"
    )
