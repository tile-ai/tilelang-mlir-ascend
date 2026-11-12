# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Run the full TIR -> TLIR -> HIVM pipeline for the reshape example in one shot.

Usage:
  python run_tlir_reshape_pipeline.py
"""

import os
import shutil
import subprocess
import sys

os.environ.setdefault("TILELANG_ASCEND_MODE", "expert")
os.environ["TILELANG_TLIR"] = "1"

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402

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


def find_tilelangir_opt():
    candidates = [
        os.environ.get("TILELANGIR_OPT"),
        shutil.which("tilelangir-opt"),
        os.path.join(
            os.path.dirname(__file__),
            "..", "..", "build", "tilelangir", "tools",
            "tilelangir-opt", "tilelangir-opt",
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
    func = reshape_expert(M, N)

    # Step 1: TIR -> TLIR
    tlir_text = tilelang.engine.lower(func, target="npuir")
    print("=== TLIR ===")
    print(tlir_text)

    # Step 2: TLIR -> HIVM, via the real tilelangir-opt binary
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
