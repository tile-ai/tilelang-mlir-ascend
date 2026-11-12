# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Run the full TIR -> TLIR -> HIVM pipeline for the arange example in one shot.

Usage:
  python run_tlir_arange_pipeline.py
"""

import os
import shutil
import subprocess
import sys

os.environ.setdefault("TILELANG_ASCEND_MODE", "expert")
os.environ["TILELANG_TLIR"] = "1"

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402

M, N = 8, 8


def arange_expert(M, N, dtype="float32"):

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            a = T.alloc_ub((M, N), dtype)
            T.npuir_arange(a, offset=0, strides=(0, 0))
            T.copy(a, A)

    return main


def find_tilelangir_opt():
    candidates = [
        os.environ.get("TILELANGIR_OPT"),
        shutil.which("tilelangir-opt"),
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
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
    func = arange_expert(M, N)

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
