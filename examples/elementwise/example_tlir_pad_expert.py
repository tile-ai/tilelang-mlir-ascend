# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Dump TIR -> TLIR for the expert-mode pad kernel.

Usage:
  TILELANG_TLIR=1 TILELANG_ASCEND_MODE=expert python example_tlir_pad_expert.py
"""

import os

import tilelang
import tilelang.language as T

os.environ.setdefault("TILELANG_ASCEND_MODE", "expert")
os.environ["TILELANG_TLIR"] = "1"

SRC_N = 60
DST_N = 64


def vec_pad_1d(src_n, dst_n, dtype="float32"):
    low = [0]
    high = [dst_n - src_n]

    @T.prim_func
    def main(
        A: T.Tensor((src_n), dtype),
        C: T.Tensor((dst_n), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            A_VEC = T.alloc_ub((src_n), dtype)
            C_VEC = T.alloc_ub((dst_n), dtype)
            T.copy(A[0:src_n], A_VEC[0:src_n])
            T.npuir_pad(A_VEC, C_VEC, 0.0, low, high)
            T.copy(C_VEC[0:dst_n], C[0:dst_n])

    return main


if __name__ == "__main__":
    tilelang.cache.clear_cache()
    func = vec_pad_1d(SRC_N, DST_N)
    mlir = tilelang.engine.lower(func, target="npuir")
    print(mlir)
    print(
        "# Lower TLIR -> HIVM with:\n"
        "#   tilelangir-opt --tilelangir-convert-tl-to-hivm <this.mlir>"
    )
