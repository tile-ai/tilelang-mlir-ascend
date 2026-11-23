# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Run the full TIR -> TLIR -> HIVM pipeline for the set_flag example in one shot.

Usage:
  python run_tlir_set_flag_pipeline.py
"""

import os
import shutil
import subprocess
import sys

os.environ.setdefault("TILELANG_ASCEND_MODE", "expert")
os.environ["TILELANG_TLIR"] = "1"

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402

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
            # NOTE: this expert-mode TLIR path currently lowers tl.set_flag
            # only (no tl.wait_flag yet), so this is illustrating/testing the
            # new op, not a runnable sync pair.
            with T.rs("PIPE_MTE2"):
                T.copy(A[cid * block_N : cid * block_N + tail_size], A_VEC[0:tail_size])
                T.copy(B[cid * block_N : cid * block_N + tail_size], B_VEC[0:tail_size])
                T.set_flag("PIPE_V", 0)
            with T.rs("PIPE_V"):
                T.npuir_add(A_VEC, B_VEC, C_VEC)
                T.set_flag("PIPE_MTE3", 0)
            with T.rs("PIPE_MTE3"):
                T.copy(C_VEC[0:tail_size], C[cid * block_N : cid * block_N + tail_size])

    return main


def find_tilelangir_opt():
    candidates = [
        os.environ.get("TILELANGIR_OPT"),
        shutil.which("tilelangir-opt"),
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "build",
            "tilelangir",
            "tools",
            "tilelangir-opt",
            "tilelangir-opt",
        ),
    ]
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return os.path.abspath(c)
    raise RuntimeError(
        "Could not find tilelangir-opt. Build it first with:\n"
        "  cmake --build build --target tilelangir-opt\n"
        "or set TILELANGIR_OPT=/path/to/tilelangir-opt"
    )


def main():
    tilelang.cache.clear_cache()
    func = vec_add_1d_manual_sync(N, 1024)

    tlir_text = tilelang.engine.lower(func, target="npuir")
    print("=== TLIR ===")
    print(tlir_text)

    opt = find_tilelangir_opt()
    result = subprocess.run(
        [opt, "--tilelangir-convert-tl-to-hivm", "-"],
        input=str(tlir_text),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError("tilelangir-opt failed")

    print("=== HIVM ===")
    print(result.stdout)


if __name__ == "__main__":
    main()
