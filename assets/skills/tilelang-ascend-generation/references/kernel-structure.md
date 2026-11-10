# Structure mixed Cube/Vector kernels

Use this reference when writing or adapting attention and similar mixed kernels.
The pattern comes from `examples/ascend/flash_attention/core.py`, which
illustrates stage structure, VF selection and a pipelined KV caller loop.
For a new workload, leave eligible per-tile versions compiler-managed and
inspect its current lowered storage. Do not assume the example's exact tile
sizes, VF latencies or core partition fit the adaptation. Lowering does not
establish the new kernel's device correctness or performance.

## Separate stages from orchestration

Keep the kernel builder separate from its tensor wrapper, reference and CLI.
In the builder, capture tiling options as compile-time scalars before defining
`T.macro` stages. Let the PrimFunc own allocations, layout/version annotations
and work traversal. Express a KV iteration as a short composition of stages:

```text
gather -> value transpose -> QK -> softmax -> pack P -> PV -> accumulate O
```

Initialization, final normalization/cast and LSE stores are separate stages.
Macro boundaries should expose meaningful producer/consumer buffers, so a
stage's memory layout, precision and core/PIPE role are easy to inspect.
Avoid extracting arbitrary blocks whose large parameter lists hide those roles.
Macros organize source; they do not themselves force scheduler tasks or barriers.
Use `T.Task` only when its single-PIPE/single-core contract is appropriate.

## Schedule the caller's tile loop

Stage macros alone do not establish a pipeline. Put `T.Pipelined` around the
loop calling `kv_step`; macro expansion exposes its operations to scheduling.
Distinguish the recurrent `m`, `l` and output accumulator from each tile's K/V,
scores, probabilities, rescale factors and partial output. Assess each scratch
buffer's complete write/read epoch, including layout padding and aliases.

Follow [the automatic tiled-pipeline procedure](tiled-pipeline.md)
for the stage bound, memory budget, version selection and generated-code checks.
Do not copy `core.py`'s fixed 2/3-version counts or measured VF latencies into a
wider head dimension without checking its different storage and dependencies.

## Choose a VF mode for each stage

| Work | Starting choice | Reason |
| --- | --- | --- |
| Contiguous memory transfers | `T.copy` | Express the supported DMA path and layout explicitly |
| Indexed, guarded GM reads | Isolated SIMT gather or supported indexed DMA | Keep irregular address computation separate from dense math |
| Dense UB initialization, rescaling, normalization and casts | SIMD MicroAPI | Contiguous vector loads/stores and broadcast scalars |
| Row softmax on known SIMD tiles | SIMD max/sum and exponential intrinsics | Register reduction without a thread collective |
| UB permutation or NZ packing | SIMD permutation/gather/scatter | Work with the actual physical layout consumed by Cube |
| Matrix products | Cube GEMM | Follow the selected L1/L0 transpose and capacity contracts |

SIMT remains useful for irregular indexing, atomics, RNG or a thread-parallel
algorithm. Do not choose it for every `T.Parallel` loop merely because that is
the easiest transcription from a GPU kernel. Conversely, do not force SIMD
onto unsupported GM accesses. Assess indexed DMA, SIMT gather and SIMD UB
packing independently, then measure the resulting kernel on the target.

## Preserve layout and numerical contracts

- Distinguish logical extents from physical pitches and vector load footprints.
  A 16/32-column softmax may use a 64-lane physical score row. Initialize padding
  and restrict predicates/reductions to the logical lanes.
- For sparse softmax, exclude invalid entries before both loads and reduction.
  Handle an all-masked first tile, an empty tile after valid data and a wholly
  empty row without `-inf - -inf`. Preserve output-zero and LSE conventions.
- Direct compact-NZ probability packing can remove an intermediate ND buffer
  and conversion. Recheck BF16 lane packing, 16-element block strides, padding
  rows, masks and copy extents when changing tile dimensions. Do not copy the
  dense attention kernel's hardcoded 128-column packing unchanged.
- An output BF16 view can alias normalized FP32 UB storage, as in `core.py`.
  Finish normalization before casting; keep the cast traversal from overwriting
  unread FP32 data, and retain correct alias/dependency tracking.
- L1-input GEMM can perform internal L0 K tiling. Do not copy a full wide
  operand into L0 without checking physical capacity. Follow the chosen
  GEMM lowering rather than mechanically cloning the dense example's buffers.
- Keep online-softmax state loop-carried. Choose temporary buffer versions and
  AIV partitioning from their ownership/capacity contracts. Never copy measured
  VF latency annotations from a different shape or instruction sequence.

## Verify the adaptation

Inspect generated code for the intended VF modes, guarded indexed reads,
compact-NZ copies, casts, allocation footprints and synchronization. Source
lowering proves neither vendor compilation nor device correctness. Test both
O and LSE on hardware, including both NZ column blocks, tails, causal/decode
offsets, masked tiles, group/head mapping and repeated scratch reuse. Report
device tests skipped without hardware and keep performance claims separate.
