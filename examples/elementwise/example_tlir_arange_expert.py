# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Dump TIR -> TLIR for the expert-mode arange kernel.

Usage:
  TILELANG_TLIR=1 TILELANG_ASCEND_MODE=expert python example_tlir_arange_expert.py
"""

import os

import tilelang
import tilelang.language as T

os.environ.setdefault("TILELANG_ASCEND_MODE", "expert")
os.environ["TILELANG_TLIR"] = "1"

M, N = 8, 8
BLOCK_M, BLOCK_N = 8, 8


def arange_2d(M, N, block_M, block_N, dtype="float32"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            A_VEC = T.alloc_ub((block_M, block_N), dtype)
            bx = (cid // n_num) * block_M
            by = (cid % n_num) * block_N
            T.npuir_arange(A_VEC, [bx, by], bx)
            T.copy(A_VEC, A[bx:bx + block_M, by:by + block_N])

    return main


if __name__ == "__main__":
    tilelang.cache.clear_cache()
    func = arange_2d(M, N, BLOCK_M, BLOCK_N)
    mlir = tilelang.engine.lower(func, target="npuir")
    print(mlir)
    print(
        "# Lower TLIR -> HIVM with:\\n"
        "# tilelangir-opt --tilelangir-convert-tl-to-hivm <this.mlir>"
    )
