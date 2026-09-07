"""Profile comparable fp16, non-causal Flash Attention cases with msprof.

This runner intentionally performs no reference computation, printing, or
correctness assertion after compilation.  It supports both the repository's
``flash_attn_npuir.py`` example and the migrated GQA factory supplied by the
user.  The latter is loaded by path so this helper does not duplicate it.

Examples:
  python examples/flash_attention/benchmark_fa_msprof.py --impl baseline --seq-len 512
  python examples/flash_attention/benchmark_fa_msprof.py --impl gqa \
      --gqa-module examples/multi_head_attention/_gqa_prefill_fwd_kernel.py \
      --seq-len 512
"""

import argparse
import importlib.util
import os
from pathlib import Path

# Keep both implementations on the same NPU lowering path.  This must happen
# before either implementation imports tilelang.
os.environ["TILELANG_ASCEND_MODE"] = "Developer"

import torch
import torch_npu  # noqa: F401  # Register the NPU device backend.


CASE_LENGTHS = (512, 1024, 2048, 4096)


def load_gqa_module(module_path: Path):
    spec = importlib.util.spec_from_file_location("gqa_prefill_benchmark", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load GQA module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "_gqa_prefill_fwd_kernel"):
        raise AttributeError(
            f"{module_path} does not export _gqa_prefill_fwd_kernel"
        )
    return module


def make_baseline_kernel(args):
    # This file lives beside flash_attn_npuir.py, so this import works when the
    # runner is invoked from the repository root as documented below.
    from flash_attn_npuir import flash_attn_kernel

    kernel = flash_attn_kernel(
        "float16",
        "float32",
        args.seq_len,
        args.dim,
        args.baseline_block_m,
        args.baseline_block_n,
        args.baseline_block_k,
    )
    q = torch.randn((args.seq_len, args.dim), dtype=torch.float16).npu()
    k = torch.randn((args.seq_len, args.dim), dtype=torch.float16).npu()
    v = torch.randn((args.seq_len, args.dim), dtype=torch.float16).npu()
    workspaces = make_baseline_workspace(args)
    return lambda: kernel(q, k, v, *workspaces)


def make_baseline_workspace(args):
    """Allocate exactly the workspaces required by flash_attn_npuir.py."""
    num_blocks = (args.seq_len - 1) // args.baseline_block_n + 1
    dtype = torch.float16
    return (
        torch.empty((args.seq_len, args.seq_len), dtype=dtype).npu(),
        torch.empty((args.seq_len, args.seq_len), dtype=dtype).npu(),
        torch.empty(
            (args.seq_len, args.dim * num_blocks), dtype=dtype
        ).npu(),
    )


def make_gqa_kernel(args):
    module_path = Path(args.gqa_module).resolve()
    if not module_path.is_file():
        raise FileNotFoundError(f"GQA source file not found: {module_path}")

    module = load_gqa_module(module_path)
    # (64, 64, 1) is the migrated factory's wrapper-default tuple.  For the
    # requested cases it activates its built-in tuned dispatch.
    factory = module._gqa_prefill_fwd_kernel(
        1, 1, 1, args.seq_len, args.seq_len, args.dim,
        False, None, 0.0, "float16",
    )
    kernel = factory(args.gqa_block_m, args.gqa_block_n, args.gqa_num_stages)
    q = torch.randn((1, args.seq_len, 1, args.dim), dtype=torch.float16).npu()
    k = torch.randn((1, args.seq_len, 1, args.dim), dtype=torch.float16).npu()
    v = torch.randn((1, args.seq_len, 1, args.dim), dtype=torch.float16).npu()
    return lambda: kernel(q, k, v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=("baseline", "gqa"), required=True)
    parser.add_argument("--seq-len", type=int, choices=CASE_LENGTHS, required=True)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument(
        "--iterations", type=int, default=20,
        help="Kernel invocations after compilation; must exceed msprof warm-up + launch-count.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gqa-module", type=str)
    parser.add_argument("--gqa-block-m", type=int, default=64)
    parser.add_argument("--gqa-block-n", type=int, default=64)
    parser.add_argument("--gqa-num-stages", type=int, default=1)
    parser.add_argument("--baseline-block-m", type=int, default=96)
    parser.add_argument("--baseline-block-n", type=int, default=256)
    parser.add_argument("--baseline-block-k", type=int, default=128)
    args = parser.parse_args()

    if args.impl == "gqa" and not args.gqa_module:
        parser.error("--gqa-module is required when --impl=gqa")
    if args.iterations < 1:
        parser.error("--iterations must be positive")

    torch.manual_seed(args.seed)
    invoke = make_baseline_kernel(args) if args.impl == "baseline" else make_gqa_kernel(args)

    # Compilation and allocations above are deliberately outside this loop.
    # msprof's --warm-up/--launch-count selects the samples from these calls.
    for _ in range(args.iterations):
        invoke()
    torch.npu.synchronize()


if __name__ == "__main__":
    main()
