# Establish an automatic tiled pipeline

For repeated mixed-kernel tile work, establish the automatic pipeline before
experimenting with `T.Stage`. See `examples/ascend/flash_attention/core.py`:
its KV caller loop uses `T.Pipelined`, while `kv_step` composes stage macros.
Macro expansion makes the internal copies, GEMMs and VFs visible to scheduling;
separate pipeline loops around each macro do not express the whole tile epoch.

1. **Classify storage.** Identify invariant inputs, recurrent state and per-tile
   scratch. In online attention, `m`, `l` and the accumulated output carry values
   between KV iterations; K/V tiles, scores, probabilities, alpha and partial
   outputs are candidates for independent epochs. Check every access and alias
   before deciding eligibility. A recurrence in softmax does not require all
   transfers and scratch storage to be single-version.
2. **Budget physical storage.** For each UB/L1/L0 domain, use physical element
   counts and dtype bytes, multiplied by candidate version counts. Include
   layout padding, alignment, lowering-created conversion/GEMM scratch and
   target/runtime reservations. Count alias storage once. Start with a
   conservative sum; credit storage reuse only after verifying its lifetime
   constraints in the lowered allocation. If the candidate exceeds capacity,
   adjust tile sizes, lifetimes, layout or version/stage bounds and lower again.
3. **Express a candidate.** Use an explicit `num_stages` budget on the caller
   loop and initially let the compiler choose eligible scratch versions. The
   following is a scheduling sketch, not a complete runnable kernel:

   ```python
   T.annotate_buffer_versions({m_ub: 1, l_ub: 1, o_ub: 1})
   # Per-tile scratch is left eligible for compiler-managed versions.
   for tile in T.Pipelined(N_TILES, num_stages=2, annotations={"enable_offset": True}):
       kv_step(tile, ...)  # Expanded tile producers and consumers.
   ```

   Here 2 is a candidate search bound, not a prescribed physical count for
   every buffer. Use `enable_offset=True` when seeking cross-iteration stage
   shifts, such as interleaved QK/PV work; it is distinct from buffer eligibility.
   Fixed counts are appropriate when justified by an inspected schedule or
   storage contract. A bare scratch annotation of `1` opts out; `(1, "auto")`
   still keeps eligibility with a fixed single slot. Check the version modes in
   the target checkout's `.agents/skills/tilelang-ascend/references/synchronization.md`.
4. **Check what was generated.** Inspect task order, cross-iteration offsets,
   prologue/epilogue guards, synchronization, actual version counts and physical
   allocations. Check each version's initialization and entire read footprint:
   a score row may load 64 physical lanes after only 16/32 were copied. An
   ordinary initialization VF outside the KV owner loop cannot simply be
   assumed to initialize every scratch version. Move needed initialization
   into each epoch, use a supported independent broadcast fill, or justify
   retaining that storage as single-version. Follow the same checkout's
   ownership rules.

Automatic scheduling being enabled does not demonstrate a multibuffer pipeline:
check for both a suitable loop stage budget and eligible storage. If every
temporary is fixed to `1`, or the repeated tile loop remains serial, record why
and identify any unresolved pipeline work. A correctness-first serial baseline
is useful for debugging, but should be reported as that baseline. Without
hardware, lower and inspect feasible candidates and report unmeasured device
behavior; choose performance winners only from device measurements.
