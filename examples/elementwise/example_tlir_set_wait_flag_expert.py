# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Dump TIR -> TLIR for an expert-mode elementwise add kernel that uses
explicit tl.set_flag / tl.wait_flag cross-pipe synchronization.

Usage:
  TILELANG_TLIR=1 TILELANG_ASCEND_MODE=expert python example_tlir_set_wait_flag_expert.py
"""

import os

import tilelang
import tilelang.language as T

os.environ.setdefault("TILELANG_ASCEND_MODE", "expert")
os.environ["TILELANG_TLIR"] = "1"

N = 1024


def vec_add_1d_manual_sync(N, block_N, dtype="float32"):
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((N), dtype),
        B: T.Tensor((N), dtype),
        C: T.Tensor((N), dtype),
        shape: T.int32,
    ):
        with T.Kernel(n_num, is_npu=True) as (cid, _):
            A_VEC = T.alloc_ub((block_N), dtype)
            B_VEC = T.alloc_ub((block_N), dtype)
            C_VEC = T.alloc_ub((block_N), dtype)
            t0 = cid * block_N
            t0 = shape - t0
            tail_size = T.min(block_N, t0)
            # NOTE: this expert-mode TLIR path now lowers both tl.set_flag
            # and tl.wait_flag, so this is a complete, runnable sync pair.
            with T.rs("PIPE_MTE2"):
                T.copy(A[cid * block_N : cid * block_N + tail_size], A_VEC[0:tail_size])
                T.copy(B[cid * block_N : cid * block_N + tail_size], B_VEC[0:tail_size])
                T.set_flag("PIPE_V", 0)
            with T.rs("PIPE_V"):
                T.wait_flag("PIPE_MTE2", 0)
                T.npuir_add(A_VEC, B_VEC, C_VEC)
                T.set_flag("PIPE_MTE3", 0)
            with T.rs("PIPE_MTE3"):
                T.wait_flag("PIPE_V", 0)
                T.copy(C_VEC[0:tail_size], C[cid * block_N : cid * block_N + tail_size])

    return main


if __name__ == "__main__":
    tilelang.cache.clear_cache()
    func = vec_add_1d_manual_sync(N, 1024)
    mlir = tilelang.engine.lower(func, target="npuir")
    print(mlir)
    print(
        "# Lower TLIR -> HIVM with:\n"
        "#   tilelangir-opt --tilelangir-convert-tl-to-hivm <this.mlir>"
    )
