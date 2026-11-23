# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Dump TIR -> TLIR for the expert-mode reshape kernel.

Usage:
  TILELANG_TLIR=1 TILELANG_ASCEND_MODE=expert python example_tlir_reshape_expert.py
"""

import os

import tilelang
import tilelang.language as T

os.environ.setdefault("TILELANG_ASCEND_MODE", "expert")
os.environ["TILELANG_TLIR"] = "1"

M, N = 8, 16


def reshape_expert(M, N, dtype="float16"):

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((N, M), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            a = T.alloc_ub((M, N), dtype)
            b = T.alloc_ub((N, M), dtype)
            T.copy(A, a)
            T.npuir_reshape(a, b)
            T.copy(b, B)

    return main


if __name__ == "__main__":
    tilelang.cache.clear_cache()
    func = reshape_expert(M, N)
    mlir = tilelang.engine.lower(func, target="npuir")
    print(mlir)
    print(
        "# Lower TLIR -> HIVM with:\n"
        "# tilelangir-opt --tilelangir-convert-tl-to-hivm <this.mlir>"
    )
