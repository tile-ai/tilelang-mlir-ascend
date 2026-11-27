# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Dump TIR -> TLIR for the expert-mode elementwise verf (erf) kernel.

Usage:
  TILELANG_TLIR=1 TILELANG_ASCEND_MODE=expert python example_tlir_verf_expert.py
"""

import os

import tilelang
import tilelang.language as T

os.environ.setdefault("TILELANG_ASCEND_MODE", "expert")
os.environ["TILELANG_TLIR"] = "1"

N = 1024


def vec_verf_1d(N, dtype="float32"):

    @T.prim_func
    def main(
        A: T.Tensor((N), dtype),
        C: T.Tensor((N), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            A_VEC = T.alloc_ub((N), dtype)
            C_VEC = T.alloc_ub((N), dtype)
            T.copy(A, A_VEC)
            T.verf(A_VEC, C_VEC)
            T.copy(C_VEC, C)

    return main


if __name__ == "__main__":
    tilelang.cache.clear_cache()
    func = vec_verf_1d(N)
    mlir = tilelang.engine.lower(func, target="npuir")
    print(mlir)
    print(
        "# Lower TLIR -> HIVM with:\n"
        "#   tilelangir-opt --tilelangir-convert-tl-to-hivm <this.mlir>"
    )
