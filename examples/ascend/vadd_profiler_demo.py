"""Warm up, benchmark, and profile a TileLang VADD kernel on Ascend NPU.

Run from the repository root after configuring CANN, OpenTileAS, and CCEC:

    python examples/ascend/vadd_profiler_demo.py \
        --numel 1048576 \
        --warmup 10 \
        --benchmark-iters 100 \
        --profile-iters 5 \
        --output ./log/vadd_profiler_comparison
"""

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.language as T


VECTOR_LENGTH = 64
PREFERRED_TILE_ELEMS = 8192


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def vector_add(n: int):
    """Build VADD for an internally vector-aligned one-dimensional buffer."""
    tile_elems = math.gcd(n, PREFERRED_TILE_ELEMS)
    num_blocks = n // tile_elems
    dtype = "float32"

    @T.prim_func
    def main(
        a: T.Buffer((n,), dtype),
        b: T.Buffer((n,), dtype),
        out: T.Buffer((n,), dtype),
    ):
        with T.Kernel(num_blocks) as bx:
            a_shared = T.alloc_shared((tile_elems,), dtype)
            b_shared = T.alloc_shared((tile_elems,), dtype)
            out_shared = T.alloc_shared((tile_elems,), dtype)
            offset = bx * tile_elems

            T.copy(a[offset : offset + tile_elems], a_shared)
            T.copy(b[offset : offset + tile_elems], b_shared)

            with T.SimdVF():
                for chunk in range(tile_elems // VECTOR_LENGTH):
                    begin = chunk * VECTOR_LENGTH
                    end = begin + VECTOR_LENGTH
                    a_frag = T.alloc_frag((VECTOR_LENGTH,), dtype)
                    b_frag = T.alloc_frag((VECTOR_LENGTH,), dtype)
                    out_frag = T.alloc_frag((VECTOR_LENGTH,), dtype)

                    T.copy(a_shared[begin:end], a_frag)
                    T.copy(b_shared[begin:end], b_frag)
                    T.vadd(a_frag, b_frag, out_frag)
                    T.copy(out_frag, out_shared[begin:end])

            T.copy(out_shared, out[offset : offset + tile_elems])

    return main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Warm up, benchmark, and profile a TileLang Ascend VADD kernel."
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--numel", type=int, default=1 << 20)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--benchmark-iters", type=int, default=100)
    parser.add_argument("--profile-iters", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./log/vadd_profiler_comparison"),
    )
    return parser.parse_args()


def set_profile_enabled(enabled: bool) -> None:
    os.environ["TILELANG_NPU_PROFILE"] = "1" if enabled else "0"


def main() -> None:
    args = parse_args()
    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")

    logical_numel = args.numel
    aligned_numel = align_up(logical_numel, VECTOR_LENGTH)
    output_dir = args.output.expanduser().resolve()

    a_cpu = torch.zeros(aligned_numel, dtype=torch.float32)
    b_cpu = torch.zeros(aligned_numel, dtype=torch.float32)
    a_cpu[:logical_numel] = torch.randn(logical_numel, dtype=torch.float32)
    b_cpu[:logical_numel] = torch.randn(logical_numel, dtype=torch.float32)

    a = a_cpu.to(device)
    b = b_cpu.to(device)
    reference = a_cpu[:logical_numel] + b_cpu[:logical_numel]

    print("[1/3] Compiling and warming up with profiler disabled")
    set_profile_enabled(False)
    tilelang_add = tilelang.compile(
        vector_add(aligned_numel),
        target="tile",
        out_idx=[2],
    )

    for _ in range(args.warmup):
        output = tilelang_add(a, b)
    torch.npu.synchronize()
    torch.testing.assert_close(output[:logical_numel].cpu(), reference)

    print("[2/3] Measuring the steady-state path with profiler disabled")
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(args.benchmark_iters):
        output = tilelang_add(a, b)
    torch.npu.synchronize()
    average_us = (time.perf_counter() - start) * 1e6 / args.benchmark_iters

    print("[3/3] Collecting the multi-iteration profiler Timeline")
    os.environ["TILELANG_NPU_PROFILE_DIR"] = str(output_dir)
    os.environ["TILELANG_NPU_PROFILE_SKIP_FIRST"] = "0"
    os.environ["TILELANG_NPU_PROFILE_WARMUP"] = "0"
    os.environ["TILELANG_NPU_PROFILE_ACTIVE"] = str(args.profile_iters)
    os.environ["TILELANG_NPU_PROFILE_ANALYSE"] = "1"
    set_profile_enabled(True)

    for _ in range(args.profile_iters):
        output = tilelang_add(a, b)
    torch.npu.synchronize()

    torch.testing.assert_close(output[:logical_numel].cpu(), reference)
    print("correctness: passed")
    print(f"logical numel: {logical_numel}")
    print(f"aligned numel: {aligned_numel}")
    print(f"warmup iterations: {args.warmup}")
    print(f"benchmark iterations: {args.benchmark_iters}")
    print(f"unprofiled average: {average_us:.3f} us")
    print(f"profile iterations: {args.profile_iters}")
    print(f"profile output: {output_dir}")


if __name__ == "__main__":
    main()
