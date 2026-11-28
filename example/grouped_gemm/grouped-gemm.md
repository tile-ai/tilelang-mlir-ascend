# NPUIR Grouped GEMM

This is a standalone adaptation of [TileOPs PR #21](https://github.com/yyttt6/TileOPs/pull/21),
at `a09896da34fa2fdc37553f061cc71e476b71dfc4`, for this repository's NPUIR backend.
It covers all six source manifest shapes and seven dtype cases, including packed
ragged routes and empty groups. It does not require the TileOPs registry.

## Interface

`GroupedGemmFwdOp(transpose_a=False, transpose_b=True)(a, b, batch_sizes, batch_offsets)`
returns a new tensor in the operand dtype. Both operands must be fp16 or bf16;
the accumulation is fp32. All four inputs must be contiguous on the same NPU.

| Layout | A | B | Output |
| --- | --- | --- | --- |
| NT (default) | `[M,K]` | `[G,N,K]` | `[M,N]` |
| NN | `[M,K]` | `[G,K,N]` | `[M,N]` |
| TN | `[M,N]` | `[M,K]` | `[G,N,K]` |

`batch_sizes[G]` is nonnegative int32 and sums to M. `batch_offsets[G]` is its
exclusive prefix sum, also int32. As in the source interface, these value
invariants are the caller's responsibility; the implementation does not read
routing metadata back to the host. Groups need no row padding or aligned starts.
TT is rejected. N and K must be positive; M may be zero.

## Run

Use an installed build of this repository and its matching CANN/BishengIR,
PyTorch and torch_npu. Set Expert mode before importing TileLang:

```bash
export TILELANG_ASCEND_MODE=Expert
python -m pytest example/grouped_gemm/test_grouped.py -q
```

From the repository root:

```python
import torch
import torch_npu
from example.grouped_gemm import GroupedGemmFwdOp

sizes = torch.tensor([1, 0, 17, 32], device="npu", dtype=torch.int32)
offsets = sizes.cumsum(0, dtype=torch.int32) - sizes
a = torch.randn(50, 128, device="npu", dtype=torch.bfloat16)
b = torch.randn(4, 64, 128, device="npu", dtype=torch.bfloat16)
out = GroupedGemmFwdOp()(a, b, sizes, offsets)
```

## Implementation

The Expert-mode Cube pattern follows `examples/gemm/matmul.py` and the dynamic
tail copy tests in `testing/npuir/memory_ops`. `T.copy` stages bounded slices
into L1, `T.gemm` accumulates into fp32 L0C, and `T.store_fixpipe` writes only
the live rows and columns while converting to the operand dtype.

NT/NN use a small device kernel to build exclusive tile counts, followed by a
group-major flat tile walk. The launch boundary orders the prefix workspace;
there is no host synchronization or per-group host launch. TN has one output
tile space per group and reads each group's packed contraction bounds on device.
Every TN tile starts with a 16-deep zero MAD, so empty groups store zeros even
after a previous tile has accumulated a nonzero result.

The port uses conservative `64 x 128 x 64` tiles and compiler-injected pipeline
synchronization. It supports small/tail shapes in addition to the manifest
shapes; it does not reproduce the source's hand-tuned 950 L0 staging schedule.
NPU lowering, execution and performance have not been validated on this host.

## Source files and validation provenance

| Source PR file | Local counterpart |
| --- | --- |
| `src/tileops/kernels/ascend/gemm/__init__.py` | `__init__.py` |
| `src/tileops/kernels/ascend/gemm/grouped.py` | `grouped.py` |
| `docs/ascend950/grouped-gemm.md` | `grouped-gemm.md` |
| `docs/ascend950/grouped-gemm-provenance.json` | `grouped-gemm-provenance.json` |

`test_grouped.py` adds independent per-group references and executable tests for
the seven source dtype cases, ragged/tail boundaries, empty groups, rerouting,
storage offsets and input preservation. Missing hardware skips device tests;
build/runtime failures are not hidden. CPU contract tests run without TileLang.

The JSON file retains the upstream report under `upstream_report`, separately
from the port's validation status. Its latency numbers and seven successful
device cases describe the **original Ascend 950 implementation**, not this port.
No port performance measurements are available.
