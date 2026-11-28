# W4A8 GEMM on NPUIR

Standalone adaptation of [TileOPs PR #82](https://github.com/yyttt6/TileOPs/pull/82),
commit `be2d2fe1085c9bb832bc8215150cfe909e035058`, based on target `main`
at `a1b56d2453ed4a84cf0ccceea3022cc48533e0b5`.

The source PR changes `src/tileops/_kernels/ascend/gemm/w4a8.py` and its package
exports. Their local counterparts are `w4a8.py` and `__init__.py`. The standalone
`GEMMW4A8FwdOp` replaces TileOPs registration and call-spec dependencies.
`test_w4a8.py` and `manifest_cases.json` add executable regression coverage.

## Contract

```python
from example.gemm_w4a8 import GEMMW4A8FwdOp

op = GEMMW4A8FwdOp(group_size=128)
weight = op.repack(packed_weight)
output = op(activation, weight, weight_group_scale, weight_channel_scale, activation_scale)
```

| Tensor | Shape | Dtype |
| --- | --- | --- |
| activation | `[M,K]` | `torch.float8_e4m3fn` |
| packed_weight | `[N,K/2]` | `torch.uint8` |
| weight_group_scale | `[N,K/128]` | `torch.float8_e4m3fn` |
| weight_channel_scale | `[N]` | `torch.float32` |
| activation_scale | `[M]` | `torch.float32` |
| output | `[M,N]` | `torch.bfloat16` |

M must be positive; N and K must be positive multiples of 128, matching the
source Ascend kernel's refusal contract. Only group size 128 is implemented.
All forward inputs must be contiguous on the same NPU. Repack also accepts
noncontiguous weights and returns an independent contiguous copy, preserving
the source Ascend row-major layout: even K is the low nibble, odd K the high
nibble, both signed two's-complement INT4 in `[-8,7]`.

For each weight, multiply its signed INT4 value by its group scale, then round
to E4M3FN **before** the contraction. Accumulate the product in FP32, multiply
by the token scale and then the channel scale in FP32, and finally round to BF16.
Inputs are preserved and dequantized weights are recomputed on every call.

## Backend adaptation

This revision's NPUIR type conversion and GEMM path do not lower E4M3FN:
see `docs/Tilelang.language/线性代数操作/T.gemm.md`,
`docs/Tilelang.language/数据类型转换操作/T.vcast.md` and
`src/target/codegen_npuir_api.cc::DTypetoMLIRType`.
The limitation is recorded as CG-2026-0018 in the capability-gap registry.

Following `examples/deepseek_v4/example_fp8_gemm_kernel.py`, the Cube operands
use FP16 carriers, which exactly represent every finite FP8 value. Four NPUIR
launches decode activations, unpack/scale/round weights, multiply with FP32
accumulation, and apply the output scales. The Vector stages use `vcast`,
`vand`, `vshr`, `vadd`, `gather`, `interleave` and `vmul`; the Cube stage uses
Expert-mode L1/L0C buffers and bounded M tails.

Two small constant lookup tables are constructed once on CPU and cached per
NPU device. No input tensors are read back to the host. The weight table covers
all 256 scale encodings and 16 nibble values, using the installed PyTorch's
E4M3FN conversion just like the upstream reference. Finite results use
nearest-even rounding; overflow saturation follows that PyTorch version.
NaN payloads and the sign of zero are not specified by the source contract.

There is no PyTorch matrix-multiply fallback. This implementation does **not**
retain the native Ascend 950 FP8 throughput or the source's tuned scheduler.
Additional scratch is `2*M*K + 2*N*K + 4*M*N` bytes, excluding the BF16 output
and the 8.5 KiB constant tables. Kernel caches include the NPU device index;
workspaces are allocated per call. NPU performance has not been measured.

## Validation

Use an installed NPUIR build and matching CANN/torch_npu environment. Select
Expert mode before importing TileLang:

```bash
export TILELANG_ASCEND_MODE=Expert
python -m pytest example/gemm_w4a8/test_w4a8.py -q

# Large model shapes; requires sufficient host and device memory.
RUN_W4A8_MANIFEST=1 python -m pytest example/gemm_w4a8/test_w4a8.py -q
```

On the development host (CPU PyTorch 2.15.0): 376 tests pass, including all
256 FP8 encodings with all 16 signed nibbles and metadata checks for the 98
source manifest shapes. The 112 device tests are skipped: 14 small regressions
and 98 opt-in manifest cases. Compiler/runtime failures on an available NPU
are not suppressed or converted into skips.

An additional local AST execution check passes 13 cases for the actual four
kernel bodies, including multi-wave scheduling, M tails, packed-byte ordering,
all scale encodings and input preservation. These CPU checks do not validate
NPUIR lowering, NPU execution, or the source PR's performance claims. Those
remain unverified on this host.
