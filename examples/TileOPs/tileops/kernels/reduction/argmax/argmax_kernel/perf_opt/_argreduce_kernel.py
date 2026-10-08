# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Argmax/argmin first-occurrence reduction kernel (NPU, Developer mode).

ROUND-6 VERSION (``_argreduce_kernel_r6``): current best (rounds 1-4) plus
two later vectorization fixes -- the r5 vectorized int64 epilogue
(``T.reshape`` + ``T.vcast``) and the r6 in-kernel idx construction
(1-D ``T.arange`` -> ``T.reshape`` -> ``T.vbrc``, replacing the round-4 FIX
GM idx input). Call contract back to ``main(x) -> out`` for the resident
path (no host-side idx tensor injection).

Migrated from the GPU TileOPs ``_argreduce_kernel`` (ArgmaxFwdOp) to
``target="npuir"``.

Semantics (DESIGN.md section 0.1, frozen):
    y[i] = min{ j in [0, N) : x[i, j] == ext_i(x) },  ext = max (argmax) / min (argmin)

    First-occurrence tie-break, int64 output, matching ``torch.argmax`` /
    ``torch.argmin`` exactly on the (M, N) -> (M,) contract.

NPU redesign (DESIGN.md sections 0.6 / 1.4, frozen):
    R1: vectorized first-occurrence: ``first = reduce_min_j(ite(x_j == m, j, BIG))``
        with ``BIG = 2**30`` (fp32-exact sentinel; replaces the GPU serial
        scan + loop_break).
    R2: persistent one-dimensional core split: ``num_kernels = min(num_row_blocks,
        vector_cores)`` with an in-core ``T.serial`` grid-stride loop and an
        if-guard for out-of-range row blocks (static bounds, Vector cores x2).
    R3: raw-N contract: kernel receives the original (M, N) (no host F.pad);
        non-divisible tiles handled by a static-width tail tile (no kernel-side
        masking).
    R4: block_m / tile_n chosen from a probe-calibrated UB budget table
        (DESIGN.md section 4.5); resident path for N * B/elem * bm <= 64KB,
        tiled online path (forced bm=1) otherwise.
    R5: vectorized int64 epilogue on every path: ``T.reshape`` +
        ``T.vcast(..., round_mode="rint")`` replaces the generic
        ``T.Parallel`` cast loops, which stayed scalar because
        NpuLoopVectorize rejects rank-reducing Cast (npu_loop_vectorize.cc:
        785-794; see CG-2026-0020). Applied to resident (first), tiled
        (running_idx), and nsplit merge (first). The narrow path needed no
        change: its ``first_flat[i]`` load is already rank-1 (1->1) and
        NpuLoopVectorize auto-vectorizes it (verified in final IR:
        ``npuir_cast`` region op -> ``hivm.hir.vcast``). Value-identical to
        the old fptosi trunc on the integer-valued domain.
    R6: in-kernel idx construction: ``T.arange(idx_src (N,), [1], 0)`` ->
        ``T.reshape(idx_src, idx_row (1,N))`` -> ``T.vbrc(idx_row, idx_j)``
        (bm==1: reshape directly into idx_j). Replaces the GM idx input of
        the 2026-09-30 FIX. Probe evidence (2026-09-30, /tmp/opencode/
        idx_probe): the old r4b 2-D form (``T.arange(idx_row (1,N), [0,1],
        0)`` -> vbrc) lowers to ``varange strides[%c0, %c1]`` (dim0
        stride-0) and races under padded-stride layouts N%16 in [9,12] with
        bm>=2 (block-first-row tail lanes go stale); the 1-D form lowers to
        ``varange strides[%c1]`` + metadata-only ``tensor.expand_shape`` and
        passed the identical sensitive detector (0 fail / 315 launches)
        plus a 21-case random battery while the 2-D control failed 23
        (shape, column) combos. msprof kernel-only vs the GM-idx version:
        (128,300) bf16 5.02 -> 3.12us (-37.9%), (2048,256) fp16 5.34 ->
        4.74us (-11.2%), hidden-state ~-1%.

Known toolchain constraints enforced here (DESIGN.md section 9.1, probe-backed):
    C-1 reduce dst dtype MUST equal src dtype (mixed dtype silently corrupts).
    C-2 bm==1 (.,1)-indexed operand in a Parallel condition miscompiles
        (f16 -> i1 broadcast) -> bm==1 uses ``T.vbrc`` into a same-shape buffer.
    C-3 unused fragment alloc is NOT dead-code-eliminated -> bm==1 / bm>=2 and
        bf16 / non-bf16 each get a dedicated prim_func (no unconditional alloc).
        The TVM tracer also does not track buffers allocated inside a
        conditional block (variant allocs must live in dedicated prim_funcs).
    C-4 shared multi-consumer buffers race under auto-multi-buffer when the
        double-buffer budget overflows -> all compute reads fragments; shared
        is GM staging only.
    C-5 multitile + bm>1 compiler SIGSEGV -> tiled path forces bm=1.
    C-6 ``T.vcast`` f32->f32 is non-identity -> fp32 path uses same-dtype copy.
    C-8 if_then_else condition referencing a serial loop var segfaults -> the
        candidate condition only references buffer elements; tile base index is
        plain arithmetic in the value position.
    C-9 2-D stride-0 ``varange`` (``T.arange`` into a (1,N) region with
        strides [0,1]) feeding a VCOPY broadcast races on padded-stride
        layouts (N%16 in [9,12], bm>=2) -> the resident idx construction
        uses the 1-D arange + reshape form (C-9 does NOT affect the 1-D
        ``varange strides[%c1]`` + ``tensor.expand_shape`` lowering; probe
        evidence in the R6 note).
    C-10 same-shape ``T.vbrc`` (no size-1 dim) emits an empty broadcast_dims
        array -> MLIR verify fail -> the bm==1 idx path reshapes (N,) ->
        (1,N) directly instead of a (1,N)->(1,N) vbrc.
"""

import os

# Developer mode: compiler manages UB (alloc_shared) + fragment + auto sync.
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  (registers the "npu" device)
from tilelang.utils.npu_utils import NPUUtils

# Programming mode actually validated by this Stage 3 implementation.
# Must stay consistent with the TILELANG_ASCEND_MODE env above.
ASCEND_MODE = "Developer"

# fp32-exact sentinel (2**30 is a power of two, exactly representable in fp32,
# and larger than any valid index j < N <= 2**24).
BIG = 2**30

# Probe-calibrated race-safe manual UB budget per block (bytes); x2 for
# auto-multi-buffer double-buffering fits the 192KB UB (DESIGN.md section 4.5).
UB_MANUAL_BUDGET = 65536

# Column alignment boundary for the tiled path (DESIGN.md section 5.2).
TILE_ALIGNMENT = 256

# Launch-item threshold above which the extended block_m ladder engages
# (DESIGN.md section 5.2; REVIEW blocking-1 fix).
_LAUNCH_GATE = 96

_SUPPORTED_DTYPES = ("float16", "float32", "bfloat16")
_KINDS = ("argmax", "argmin")

_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


# ---------------------------------------------------------------------------
# Factory-period config helpers (DESIGN.md section 5.2 ladder)
# ---------------------------------------------------------------------------
def _ceildiv(a, b):
    return (a + b - 1) // b


def _B_elem(dtype, bm_class):
    """Fragment byte budget per element (DESIGN.md section 4.5 B/elem table)."""
    table = {
        "float16": {"multi": 8, "bm1": 10},
        "float32": {"multi": 12, "bm1": 16},
        "bfloat16": {"multi": 10, "bm1": 14},
    }
    return table[dtype][bm_class]


def _ub_slab_units(elem_bytes, dtype_slabs=1, fp32_slabs=0):
    """Large UB buffer inventory in elem-sized slab units (inlined _primitives)."""
    total_bytes = dtype_slabs * elem_bytes + fp32_slabs * 4
    return (total_bytes + elem_bytes - 1) // elem_bytes


def _tiled_slab(dtype, elem_bytes):
    """Tiled-path slab count per DESIGN.md section 5.2 (fp16 5 / bf16 7 / fp32 4)."""
    if dtype == "float16":
        # x_ub(2B) + x_work(2B) + ext_brc(2B) + cand(4B)
        return _ub_slab_units(elem_bytes, dtype_slabs=3, fp32_slabs=1)
    if dtype == "bfloat16":
        # x_ub(2B) + x_work(4B) + ext_brc(4B) + cand(4B)
        return _ub_slab_units(elem_bytes, dtype_slabs=1, fp32_slabs=3)
    # float32: x_ub(4B) + x_work(4B) + ext_brc(4B) + cand(4B)
    return _ub_slab_units(elem_bytes, dtype_slabs=4, fp32_slabs=0)


def _pick_tile_n(N, dtype, budget=UB_MANUAL_BUDGET, alignment=TILE_ALIGNMENT, margin=0.9):
    """Tiled-path tile_n: prefer a 256-aligned divisor with >=10% budget margin,
    fall back to the budget cap + static tail tile (DESIGN.md section 5.2)."""
    elem_bytes = 2 if dtype in ("float16", "bfloat16") else 4
    slab = _tiled_slab(dtype, elem_bytes)
    if slab * 1 * elem_bytes * N <= budget:
        return N  # fits entirely (not expected on the tiled path)
    max_cols = budget // (slab * 1 * elem_bytes)
    tile_n_cap = (max_cols // alignment) * alignment
    best = 0
    for candidate in range(tile_n_cap, 0, -alignment):
        if N % candidate == 0 and slab * elem_bytes * candidate <= int(budget * margin):
            best = candidate
            break
    if best > 0:
        return best
    return tile_n_cap


def _select_config(M, N, dtype):
    """Factory-period ladder (DESIGN.md section 5.2).

    Returns ``{"block_m", "path", "tile_n"}``. ``path`` is "resident" or
    "tiled". The basic ladder uses ``B_bm1`` for p=1 (REVIEW N1 fix: bm=1
    allocates the extra broadcast buffer) and ``B_multi`` for p>=2.
    """
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"unsupported dtype {dtype!r}; expected one of {sorted(_SUPPORTED_DTYPES)}"
        )
    if N <= 0:
        raise ValueError(f"reduction dim N must be positive, got N={N}")
    if N > 2**24:
        raise ValueError(
            f"N={N} exceeds 2**24; fp32 cannot exactly represent the index "
            f"sentinel arithmetic (E1/E3 equivalence domain, DESIGN.md section 5.2)"
        )

    B_multi = _B_elem(dtype, "multi")
    B_bm1 = _B_elem(dtype, "bm1")

    # Basic ladder {1,2,4,8} (p=1 uses B_bm1 to account for ext_brc).
    candidates = [
        p for p in (1, 2, 4, 8) if p * N * (B_bm1 if p == 1 else B_multi) <= UB_MANUAL_BUDGET
    ]
    block_m = max(candidates) if candidates else None

    # Extended ladder (launch-item driven) for small-N/large-M workloads.
    if block_m is not None and _ceildiv(M, block_m) > _LAUNCH_GATE:
        # Narrow-N (N < TILE_ALIGNMENT) workloads inflate more under
        # auto-multi-buffer than the wide-N 2x the basic ladder was calibrated
        # against (DESIGN.md section 9.2-R10: (2048,4) 64KB manual -> 256KB,
        # ~4x). Halve the manual budget for narrow N so the real UB footprint
        # stays within 192KB; wide-N workloads do not engage the extended
        # ladder meaningfully (their basic-ladder bm is already budget-capped).
        # ROUND-1-A: the chain form carries 4 extra full-width buffers
        # (ext_brc/cmp/idx_j/sent_v vs the fused form's cand-only); on the
        # narrow-N (N < 256) shapes the auto-multi-buffer pass measured a
        # ~3.06x inflation (L0-6b bm=1024@N=4: 68KB manual -> 208KB required
        # vs 192KB UB). Divide the narrow-N extended-ladder budget by 3 so the
        # ladder lands at bm=512 (34KB manual -> ~105KB actual).
        ext_budget = UB_MANUAL_BUDGET if N >= TILE_ALIGNMENT else UB_MANUAL_BUDGET // 3
        ext = [p for p in (16, 32, 64, 128, 256, 512, 1024, 2048) if p * N * B_multi <= ext_budget]
        block_m = max(ext + [block_m])

    if block_m is None:
        return {"block_m": 1, "path": "tiled", "tile_n": _pick_tile_n(N, dtype)}
    # Never allocate more rows than exist: block_m > M leaves uninitialized
    # trailing rows in the UB staging buffer; for narrow N the vectorizer's
    # small-N row handling can then corrupt the valid row (garbage rows).
    block_m = min(block_m, M)
    # ROUND-2-A: narrow-N wide-view deinterleave path takes precedence over
    # the plain resident path when the shape fits.
    ncfg = _select_narrow(M, N, dtype)
    if ncfg is not None:
        return {
            "block_m": ncfg["bw"] * ncfg["G"],
            "path": "narrow",
            "tile_n": None,
            "G": ncfg["G"],
            "bw": ncfg["bw"],
        }
    # Narrow-N correctness guard (adoption re-verification probe, 2026-09-28):
    # for N < TILE_ALIGNMENT the chain-form resident with block_m > 1 and the
    # tiled path (both prim_func variants) silently corrupt a data-dependent
    # row subset -- value corruption, not tie-break differences (e.g.
    # (8192,2) tiled fp16: 10-23/8192 wrong, kernel returns the smaller
    # element; (2048,16) tiled emits garbage indices ~-1.08e9; (2048,30)
    # tiled bf16: 412/2048; chain resident (2048,16) bm=128: 261/2048).
    # resident with block_m=1 is exact on this domain (324-combo battery:
    # N in {2,3,4,5,8,16,30,100,255} x M in {1,8,2048,8192} x
    # {fp16,bf16,fp32} x seeds {0,1,42}, 0 mismatches). Route every narrow-N
    # shape that does not fit the deinterleave path to resident bm=1.
    # N >= TILE_ALIGNMENT keeps the chain form (probe-clean there; carries
    # the tuned hidden-state workload). N == 1 remains unsupported (compile
    # error in every path; the Stage 3 baseline holds the i1 workaround).
    if N < TILE_ALIGNMENT:
        return {"block_m": 1, "path": "resident", "tile_n": None}
    return {"block_m": block_m, "path": "resident", "tile_n": None}


# ---------------------------------------------------------------------------
# Golden (PyTorch CPU reference, independent of the NPU algorithm)
# ---------------------------------------------------------------------------
def golden_argreduce(x: torch.Tensor, op_kind: str = "argmax") -> torch.Tensor:
    """PyTorch reference: ``torch.argmax`` / ``torch.argmin`` first-occurrence.

    Runs on CPU. Does NOT reproduce the NPU algorithm (no mask candidate /
    reduce_min / online recurrence), guaranteeing validation independence
    (DESIGN.md section 8.1).
    """
    if op_kind == "argmax":
        return torch.argmax(x, dim=-1)
    return torch.argmin(x, dim=-1)


# ---------------------------------------------------------------------------
# STAGE-4 FINAL (merged current best, rounds 1-4):
#   - resident chain (r1a) with (1,N) arange + first-axis vbrc broadcast
#     (r4b: kills the (bm,N) arange scalar materialization floor);
#   - narrow-N wide-view deinterleave path (r2a, 3d-class workloads);
#   - C6 N-split partial+merge dispatch for lm-head-class shapes
#     (r3e/r3f: DESIGN.md 1.6.4 adjudication FLIPPED -- fp16 0.461x /
#     bf16 0.386x of the single-kernel main-select, both < the
#     pre-registered 0.8x threshold; per-row (1,tn) partial + transpose
#     merge, tn rule = largest single-wave nchunk <= 48).
# ---------------------------------------------------------------------------
# Resident path (path S, whole row resident in UB)
# ---------------------------------------------------------------------------
def _build_resident(M, N, op_kind, dtype, work_dtype, vector_cores):
    """ROUND-1-A (chain): resident path with the vector-op candidate chain.

    Single change vs baseline: the fused ``T.Parallel + if_then_else``
    candidate loop (measured 0.95 scalar ratio on hidden-state) is replaced by
    the DESIGN.md section 1.6.3-#5 chain fallback:
        vbrc(ext) -> vcmp("eq") -> vselect(idx_j, sent_v) -> reduce_min
    - ext_brc is materialized via T.vbrc for ALL bm (no (bm,1) broadcast
      operand inside any Parallel condition; C-2 moot, bm=1/bm>=2 unified).
    - idx_j (column indices) is built in-kernel via the r6 1-D arange chain:
      ``T.arange(idx_src (N,), [1], 0)`` -> ``T.reshape`` to (1,N) ->
      ``T.vbrc`` to (bm,N); bm==1 reshapes (N,) into idx_j directly (C-10:
      same-shape vbrc fails MLIR verify). This replaces the 2026-09-30 GM
      idx input (MTE2 copy). Probe evidence: the r4b 2-D form
      (``T.arange`` into (1,N) with strides [0,1]) lowers to a stride-0
      ``varange`` whose tail writes race the VCOPY broadcast on
      padded-stride layouts (N%16 in [9,12], bm>=2; C-9); the 1-D form
      lowers to ``varange strides[%c1]`` + metadata-only
      ``tensor.expand_shape`` and is race-free on the identical sensitive
      detector (0 fail / 315 launches + 21-case random battery vs 23
      failing (shape, col) combos for the 2-D control). msprof vs the GM
      idx version: (128,300) bf16 -37.9%, (2048,256) fp16 -11.2%,
      hidden-state ~-1%.
    - x_frag staging dropped: compute (reduce_max/vcmp) reads x_ub directly
      (vcmp-on-shared proven by T.vselect.md section 2.4 example); keeps the
      chain buffer set within the x2 auto-multi-buffer envelope at bm=2.
    Two prim_funcs (bf16 / non-bf16); C-3 respected (ext_brc always used).
    r5 epilogue: the int64 output cast is vectorized via
    ``T.reshape(first, first_flat)`` + ``T.vcast(first_flat, out_ub,
    round_mode="rint")``. The original ``for i in T.Parallel(block_m):
    out_ub[i] = T.cast(first[i, 0], "int64")`` stayed scalar because
    NpuLoopVectorize deliberately rejects rank-reducing Cast (load
    ``first[i, 0]`` rank-2 vs store ``out_ub[i]`` rank-1,
    npu_loop_vectorize.cc HandleUnaryExpression guard) and the npuir
    codegen prints generic TIR loops verbatim as scalar scf.for. rint vs the
    old fptosi trunc is value-identical: ``first`` holds exact small integers
    (valid indices j < N <= 2**24 or the BIG sentinel, all fp32-exact).
    """
    is_bf16 = dtype == "bfloat16"

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(block_m):
        num_logical = _ceildiv(M, block_m)
        num_kernels = min(num_logical, vector_cores)
        num_local_tasks = _ceildiv(num_logical, num_kernels)

        if is_bf16:

            @T.prim_func
            def main(
                x: T.Tensor((M, N), dtype),
                out: T.Tensor((M,), "int64"),
            ):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    x_ub = T.alloc_shared((block_m, N), dtype)
                    x_work = T.alloc_fragment((block_m, N), "float32")
                    row_ext = T.alloc_fragment((block_m, 1), "float32")
                    ext_brc = T.alloc_fragment((block_m, N), "float32")
                    cmp_eq = T.alloc_fragment((block_m, N), "bool")
                    idx_src = T.alloc_shared((N,), "float32")
                    idx_row = T.alloc_shared((1, N), "float32")
                    idx_j = T.alloc_shared((block_m, N), "float32")
                    sent_v = T.alloc_fragment((block_m, N), "float32")
                    cand = T.alloc_fragment((block_m, N), "float32")
                    first = T.alloc_fragment((block_m, 1), "float32")
                    first_flat = T.alloc_fragment((block_m,), "float32")
                    out_ub = T.alloc_shared((block_m,), "int64")

                    # Task-invariant operands hoisted out of the task loop.
                    # r6: 1-D arange -> reshape -> vbrc (in-kernel idx). The
                    # 1-D varange lowers to a contiguous fill; expand_shape is
                    # metadata-only. NOT the r4b 2-D form (T.arange into (1,N)
                    # with strides [0,1]) -- that lowers to a stride-0 varange
                    # that races the VCOPY broadcast on padded-stride layouts
                    # (N%16 in [9,12], bm>=2; C-9). bm==1: reshape (N,) ->
                    # (1,N) directly into idx_j (C-10 same-shape vbrc trap).
                    T.arange(idx_src, [1], 0)
                    if block_m >= 2:
                        T.reshape(idx_src, idx_row)
                        T.vbrc(idx_row, idx_j)
                    else:
                        T.reshape(idx_src, idx_j)
                    T.vbrc(T.float32(BIG), sent_v)

                    for s in T.serial(num_local_tasks):
                        block_id = s * num_kernels + cid
                        if block_id < num_logical:
                            off_m = block_id * block_m
                            real_m = T.min(block_m, M - off_m)
                            T.copy(x[off_m : off_m + real_m, 0:N], x_ub[0:real_m, 0:N])
                            T.vcast(x_ub, x_work, round_mode="rint")
                            if op_kind == "argmax":
                                T.reduce_max(x_work, row_ext, dim=1)
                            else:
                                T.reduce_min(x_work, row_ext, dim=1)
                            T.vbrc(row_ext, ext_brc)
                            T.vcmp(x_work, ext_brc, cmp_eq, "eq")
                            T.vselect(cmp_eq, idx_j, sent_v, cand)
                            T.reduce_min(cand, first, dim=1)
                            # r5 epilogue: vectorized first-occurrence ->
                            # int64. The old ``T.Parallel`` cast loop lowered
                            # to a scalar scf.for (NpuLoopVectorize rejects
                            # rank-reducing Cast: first[i,0] rank-2 store
                            # rank-1; npu_loop_vectorize.cc:785-794).
                            T.reshape(first, first_flat)
                            T.vcast(first_flat, out_ub, round_mode="rint")
                            T.copy(out_ub[0:real_m], out[off_m : off_m + real_m])

        else:

            @T.prim_func
            def main(
                x: T.Tensor((M, N), dtype),
                out: T.Tensor((M,), "int64"),
            ):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    x_ub = T.alloc_shared((block_m, N), dtype)
                    row_ext = T.alloc_fragment((block_m, 1), dtype)
                    ext_brc = T.alloc_fragment((block_m, N), dtype)
                    cmp_eq = T.alloc_fragment((block_m, N), "bool")
                    idx_src = T.alloc_shared((N,), "float32")
                    idx_row = T.alloc_shared((1, N), "float32")
                    idx_j = T.alloc_shared((block_m, N), "float32")
                    sent_v = T.alloc_fragment((block_m, N), "float32")
                    cand = T.alloc_fragment((block_m, N), "float32")
                    first = T.alloc_fragment((block_m, 1), "float32")
                    first_flat = T.alloc_fragment((block_m,), "float32")
                    out_ub = T.alloc_shared((block_m,), "int64")

                    # Task-invariant operands hoisted out of the task loop.
                    # r6: same 1-D arange -> reshape -> vbrc form as the bf16
                    # branch (C-9 / C-10 notes there).
                    T.arange(idx_src, [1], 0)
                    if block_m >= 2:
                        T.reshape(idx_src, idx_row)
                        T.vbrc(idx_row, idx_j)
                    else:
                        T.reshape(idx_src, idx_j)
                    T.vbrc(T.float32(BIG), sent_v)

                    for s in T.serial(num_local_tasks):
                        block_id = s * num_kernels + cid
                        if block_id < num_logical:
                            off_m = block_id * block_m
                            real_m = T.min(block_m, M - off_m)
                            T.copy(x[off_m : off_m + real_m, 0:N], x_ub[0:real_m, 0:N])
                            if op_kind == "argmax":
                                T.reduce_max(x_ub, row_ext, dim=1)
                            else:
                                T.reduce_min(x_ub, row_ext, dim=1)
                            T.vbrc(row_ext, ext_brc)
                            T.vcmp(x_ub, ext_brc, cmp_eq, "eq")
                            T.vselect(cmp_eq, idx_j, sent_v, cand)
                            T.reduce_min(cand, first, dim=1)
                            # r5 epilogue: vectorized first-occurrence ->
                            # int64 (see bf16 variant note above).
                            T.reshape(first, first_flat)
                            T.vcast(first_flat, out_ub, round_mode="rint")
                            T.copy(out_ub[0:real_m], out[off_m : off_m + real_m])

        return main

    return _func


# ---------------------------------------------------------------------------
# Narrow-N wide-view path (path W, ROUND-2-A): deinterleave phase decomposition
# ---------------------------------------------------------------------------
def _narrow_G(M, N, elem_bytes):
    """Wide-row grouping factor: largest power of two G <= 256 with
    M % G == 0 and N * G * elem_bytes >= 512 (wide GM copy rows)."""
    g = 1
    for k in range(1, 9):  # 2..256
        cand = 1 << k
        if M % cand == 0 and cand <= 256 and N * cand * elem_bytes >= 512:
            g = cand
    # also allow G smaller than the 512B threshold if no divisor qualifies
    # (fallback to the regular resident path happens in _select_config).
    return g


def _select_narrow(M, N, dtype):
    """Narrow-path config: {"G", "bw"} or None when the shape does not fit.

    N is restricted to 4: the phase extraction uses a two-level
    channel_nums=2 deinterleave tree (multi-dst deinterleave with
    channel_nums>=3 crashes bishengir-compile with SIGABRT -- probe
    probe_deint_bisect.py; the c2 form is probe-verified).
    """
    if N != 4:
        return None
    # float32 excluded (adoption re-verification, 2026-09-28): the
    # deinterleave form corrupts a row subset for fp32 at N=4 ((2048,4)
    # bm=512: 483/2048 wrong in the TileOPs suite; fp16/bf16 bm=1024 are
    # clean). fp32 shapes fall through to the narrow-N resident-bm=1
    # guard (324-combo battery clean, includes (2048,4) fp32).
    if dtype == "float32":
        return None
    elem_bytes = 2 if dtype in ("float16", "bfloat16") else 4
    if N * elem_bytes > 16:
        return None
    G = _narrow_G(M, N, elem_bytes)
    if G < 2 or N * G * elem_bytes < 512:
        return None
    # block = bw wide rows; manual budget ~13B/elem (fp16) incl. out staging.
    bpc = 13 if dtype != "float32" else 17
    for bw in (8, 4, 2):
        if bw * G * N * bpc + (N + 1) * G * 4 <= 65536:
            return {"G": G, "bw": bw}
    return None


def _build_narrow(M, N, op_kind, dtype, work_dtype, vector_cores, G, bw):
    """ROUND-2-A narrow-N kernel: wide-row GM view + deinterleave phases.

    Root cause being fixed (bisect probes V0/V1): a (bm, N) narrow shared
    staging buffer feeding compute triggers auto-multi-buffer restructuring
    that re-reads the block from GM 16x, and narrow (bm, N) vector ops
    decompose per row. Structure:
      - kernel param declared as the wide-row view (M//G, N*G) over the SAME
        contiguous memory (runtime passes the (M, N) tensor; shape not
        validated, probe-verified roundtrip);
      - block copies (bw, N*G) wide rows (no GM inflation);
      - T.deinterleave(channel_nums=N) -> N phase buffers (bw, G) with
        phase_n(r, q) = x[G*r + q, n];
      - row max = vmax tree over phases; first-occurrence = vselect(cmp_n,
        const_n, BIG) then vmin tree (per-(r,q) over the N phases);
      - reshape (bw, G) -> (bm,) + int64 epilogue.
    """
    Mr = M // G
    Nr = N * G
    bm = bw * G
    num_logical = _ceildiv(M, bm)
    num_kernels = min(num_logical, vector_cores)
    num_local_tasks = _ceildiv(num_logical, num_kernels)
    is_fp16 = dtype == "float16"

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(block_m):
        # block_m is derived from bw*G; the argument is accepted for
        # interface uniformity and must equal bm.

        if is_fp16:

            @T.prim_func
            def main(x: T.Tensor((Mr, Nr), dtype), out: T.Tensor((M,), "int64")):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    w = T.alloc_shared((bw, Nr), dtype)
                    e01 = T.alloc_fragment((bw, 2 * G), dtype)
                    o01 = T.alloc_fragment((bw, 2 * G), dtype)
                    p0 = T.alloc_fragment((bw, G), dtype)
                    p1 = T.alloc_fragment((bw, G), dtype)
                    p2 = T.alloc_fragment((bw, G), dtype)
                    p3 = T.alloc_fragment((bw, G), dtype)
                    row_max = T.alloc_fragment((bw, G), dtype)
                    t01 = T.alloc_fragment((bw, G), dtype)
                    t23 = T.alloc_fragment((bw, G), dtype)
                    cmp0 = T.alloc_fragment((bw, G), "bool")
                    cmp1 = T.alloc_fragment((bw, G), "bool")
                    cmp2 = T.alloc_fragment((bw, G), "bool")
                    cmp3 = T.alloc_fragment((bw, G), "bool")
                    c0 = T.alloc_fragment((bw, G), "float32")
                    c1 = T.alloc_fragment((bw, G), "float32")
                    c2 = T.alloc_fragment((bw, G), "float32")
                    c3 = T.alloc_fragment((bw, G), "float32")
                    k01 = T.alloc_fragment((bw, G), "float32")
                    k23 = T.alloc_fragment((bw, G), "float32")
                    first = T.alloc_fragment((bw, G), "float32")
                    first_flat = T.alloc_fragment((bm,), "float32")
                    const0 = T.alloc_fragment((bw, G), "float32")
                    const1 = T.alloc_fragment((bw, G), "float32")
                    const2 = T.alloc_fragment((bw, G), "float32")
                    const3 = T.alloc_fragment((bw, G), "float32")
                    big_v = T.alloc_fragment((bw, G), "float32")
                    out_ub = T.alloc_shared((bm,), "int64")

                    # task-invariant constants (hoisted)
                    T.vbrc(T.float32(0), const0)
                    T.vbrc(T.float32(1), const1)
                    T.vbrc(T.float32(2), const2)
                    T.vbrc(T.float32(3), const3)
                    T.vbrc(T.float32(BIG), big_v)

                    for s in T.serial(num_local_tasks):
                        block_id = s * num_kernels + cid
                        if block_id < num_logical:
                            off_w = block_id * bw
                            off_m = block_id * bm
                            real_w = T.min(bw, Mr - off_w)
                            real_m = T.min(bm, M - off_m)
                            T.copy(x[off_w : off_w + real_w, 0:Nr], w[0:real_w, 0:Nr])
                            # two-level c2 deinterleave tree (channel_nums=4
                            # multi-dst form crashes the compiler; see
                            # _select_narrow docstring)
                            T.deinterleave(w, e01, o01, channel_nums=2)
                            T.deinterleave(e01, p0, p2, channel_nums=2)
                            T.deinterleave(o01, p1, p3, channel_nums=2)
                            if op_kind == "argmax":
                                T.vmax(p0, p1, t01)
                                T.vmax(p2, p3, t23)
                                T.vmax(t01, t23, row_max)
                            else:
                                T.vmin(p0, p1, t01)
                                T.vmin(p2, p3, t23)
                                T.vmin(t01, t23, row_max)
                            T.vcmp(p0, row_max, cmp0, "eq")
                            T.vcmp(p1, row_max, cmp1, "eq")
                            T.vcmp(p2, row_max, cmp2, "eq")
                            T.vcmp(p3, row_max, cmp3, "eq")
                            T.vselect(cmp0, const0, big_v, c0)
                            T.vselect(cmp1, const1, big_v, c1)
                            T.vselect(cmp2, const2, big_v, c2)
                            T.vselect(cmp3, const3, big_v, c3)
                            T.vmin(c0, c1, k01)
                            T.vmin(c2, c3, k23)
                            T.vmin(k01, k23, first)
                            T.reshape(first, first_flat)
                            for i in T.Parallel(bm):
                                out_ub[i] = T.cast(first_flat[i], "int64")
                            T.copy(out_ub[0:real_m], out[off_m : off_m + real_m])

        else:

            @T.prim_func
            def main(x: T.Tensor((Mr, Nr), dtype), out: T.Tensor((M,), "int64")):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    w = T.alloc_shared((bw, Nr), dtype)
                    wf = T.alloc_fragment((bw, Nr), "float32")
                    e01 = T.alloc_fragment((bw, 2 * G), "float32")
                    o01 = T.alloc_fragment((bw, 2 * G), "float32")
                    p0 = T.alloc_fragment((bw, G), "float32")
                    p1 = T.alloc_fragment((bw, G), "float32")
                    p2 = T.alloc_fragment((bw, G), "float32")
                    p3 = T.alloc_fragment((bw, G), "float32")
                    row_max = T.alloc_fragment((bw, G), "float32")
                    t01 = T.alloc_fragment((bw, G), "float32")
                    t23 = T.alloc_fragment((bw, G), "float32")
                    cmp0 = T.alloc_fragment((bw, G), "bool")
                    cmp1 = T.alloc_fragment((bw, G), "bool")
                    cmp2 = T.alloc_fragment((bw, G), "bool")
                    cmp3 = T.alloc_fragment((bw, G), "bool")
                    c0 = T.alloc_fragment((bw, G), "float32")
                    c1 = T.alloc_fragment((bw, G), "float32")
                    c2 = T.alloc_fragment((bw, G), "float32")
                    c3 = T.alloc_fragment((bw, G), "float32")
                    k01 = T.alloc_fragment((bw, G), "float32")
                    k23 = T.alloc_fragment((bw, G), "float32")
                    first = T.alloc_fragment((bw, G), "float32")
                    first_flat = T.alloc_fragment((bm,), "float32")
                    const0 = T.alloc_fragment((bw, G), "float32")
                    const1 = T.alloc_fragment((bw, G), "float32")
                    const2 = T.alloc_fragment((bw, G), "float32")
                    const3 = T.alloc_fragment((bw, G), "float32")
                    big_v = T.alloc_fragment((bw, G), "float32")
                    out_ub = T.alloc_shared((bm,), "int64")

                    T.vbrc(T.float32(0), const0)
                    T.vbrc(T.float32(1), const1)
                    T.vbrc(T.float32(2), const2)
                    T.vbrc(T.float32(3), const3)
                    T.vbrc(T.float32(BIG), big_v)

                    for s in T.serial(num_local_tasks):
                        block_id = s * num_kernels + cid
                        if block_id < num_logical:
                            off_w = block_id * bw
                            off_m = block_id * bm
                            real_w = T.min(bw, Mr - off_w)
                            real_m = T.min(bm, M - off_m)
                            T.copy(x[off_w : off_w + real_w, 0:Nr], w[0:real_w, 0:Nr])
                            T.vcast(w, wf, round_mode="rint")
                            # two-level c2 deinterleave tree (see fp16 variant)
                            T.deinterleave(wf, e01, o01, channel_nums=2)
                            T.deinterleave(e01, p0, p2, channel_nums=2)
                            T.deinterleave(o01, p1, p3, channel_nums=2)
                            if op_kind == "argmax":
                                T.vmax(p0, p1, t01)
                                T.vmax(p2, p3, t23)
                                T.vmax(t01, t23, row_max)
                            else:
                                T.vmin(p0, p1, t01)
                                T.vmin(p2, p3, t23)
                                T.vmin(t01, t23, row_max)
                            T.vcmp(p0, row_max, cmp0, "eq")
                            T.vcmp(p1, row_max, cmp1, "eq")
                            T.vcmp(p2, row_max, cmp2, "eq")
                            T.vcmp(p3, row_max, cmp3, "eq")
                            T.vselect(cmp0, const0, big_v, c0)
                            T.vselect(cmp1, const1, big_v, c1)
                            T.vselect(cmp2, const2, big_v, c2)
                            T.vselect(cmp3, const3, big_v, c3)
                            T.vmin(c0, c1, k01)
                            T.vmin(c2, c3, k23)
                            T.vmin(k01, k23, first)
                            T.reshape(first, first_flat)
                            for i in T.Parallel(bm):
                                out_ub[i] = T.cast(first_flat[i], "int64")
                            T.copy(out_ub[0:real_m], out[off_m : off_m + real_m])

        return main

    return _func


# ---------------------------------------------------------------------------
# Tiled path (path T, online running-(extreme, first-index), bm=1 forced)
# ---------------------------------------------------------------------------
def _build_tiled(M, N, op_kind, dtype, work_dtype, vector_cores, tile_n):
    """Build the tiled online kernel (bm=1 forced by C-5).

    Two prim_funcs (bf16 / non-bf16). Tile-0 initializes the running state
    directly; full tiles and the static tail tile use the strict-greater
    (argmax) / strict-less (argmin) update rule for first-occurrence tie-break.
    r5 epilogue: the int64 output cast is vectorized via
    ``T.reshape(running_idx, running_flat)`` + ``T.vcast`` (same rank-reduction
    scalarization as the resident path; at the C-5-forced bm=1 it degenerated
    to one scalar extract/fptosi/insert per task). The bm=1
    ``chunk_global`` / ``tail_global`` scalar add stores are left as-is (one
    element each, not the cast pattern).
    """
    is_bf16 = dtype == "bfloat16"
    num_full = N // tile_n
    tail = N - num_full * tile_n
    # Static tail-tile buffers are allocated unconditionally with a min width
    # of 1 (the TVM tracer does not track buffers allocated in a conditional
    # block and referenced later); the tail *processing* is guarded by
    # `if tail > 0` below. The extra (block_m, 1) buffers are negligible.
    tail_w = tail if tail > 0 else 1

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(block_m):
        num_logical = _ceildiv(M, block_m)
        num_kernels = min(num_logical, vector_cores)
        num_local_tasks = _ceildiv(num_logical, num_kernels)

        if is_bf16:

            @T.prim_func
            def main(x: T.Tensor((M, N), dtype), out: T.Tensor((M,), "int64")):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    x_ub = T.alloc_shared((block_m, tile_n), dtype)
                    x_work = T.alloc_fragment((block_m, tile_n), "float32")
                    chunk_ext = T.alloc_fragment((block_m, 1), "float32")
                    ext_brc = T.alloc_fragment((block_m, tile_n), "float32")
                    cand = T.alloc_fragment((block_m, tile_n), "float32")
                    chunk_first = T.alloc_fragment((block_m, 1), "float32")
                    chunk_global = T.alloc_fragment((block_m, 1), "float32")
                    cond = T.alloc_fragment((block_m, 1), "bool")
                    running_ext = T.alloc_fragment((block_m, 1), "float32")
                    running_idx = T.alloc_fragment((block_m, 1), "float32")
                    new_ext = T.alloc_fragment((block_m, 1), "float32")
                    new_idx = T.alloc_fragment((block_m, 1), "float32")
                    running_flat = T.alloc_fragment((block_m,), "float32")
                    out_ub = T.alloc_shared((block_m,), "int64")

                    x_tail = T.alloc_shared((block_m, tail_w), dtype)
                    xw_tail = T.alloc_fragment((block_m, tail_w), "float32")
                    ext_tail = T.alloc_fragment((block_m, tail_w), "float32")
                    cand_tail = T.alloc_fragment((block_m, tail_w), "float32")
                    tail_ext = T.alloc_fragment((block_m, 1), "float32")
                    tail_first = T.alloc_fragment((block_m, 1), "float32")
                    tail_global = T.alloc_fragment((block_m, 1), "float32")
                    cond_tail = T.alloc_fragment((block_m, 1), "bool")

                    for s in T.serial(num_local_tasks):
                        block_id = s * num_kernels + cid
                        if block_id < num_logical:
                            # Tile 0 initializes the running state directly.
                            T.copy(x[block_id : block_id + block_m, 0:tile_n], x_ub)
                            T.vcast(x_ub, x_work, round_mode="rint")
                            if op_kind == "argmax":
                                T.reduce_max(x_work, running_ext, dim=1)
                            else:
                                T.reduce_min(x_work, running_ext, dim=1)
                            T.vbrc(running_ext, ext_brc)
                            for i, j in T.Parallel(block_m, tile_n):
                                cand[i, j] = T.if_then_else(
                                    x_work[i, j] == ext_brc[i, j],
                                    T.cast(j, "float32"),
                                    T.float32(BIG),
                                )
                            T.reduce_min(cand, running_idx, dim=1)

                            # Full tiles 1..num_full-1.
                            if num_full > 1:
                                for t in T.serial(num_full - 1):
                                    T.copy(
                                        x[
                                            block_id : block_id + block_m,
                                            (t + 1) * tile_n : (t + 2) * tile_n,
                                        ],
                                        x_ub,
                                    )
                                    T.vcast(x_ub, x_work, round_mode="rint")
                                    if op_kind == "argmax":
                                        T.reduce_max(x_work, chunk_ext, dim=1)
                                    else:
                                        T.reduce_min(x_work, chunk_ext, dim=1)
                                    T.vbrc(chunk_ext, ext_brc)
                                    for i, j in T.Parallel(block_m, tile_n):
                                        cand[i, j] = T.if_then_else(
                                            x_work[i, j] == ext_brc[i, j],
                                            T.cast(j, "float32"),
                                            T.float32(BIG),
                                        )
                                    T.reduce_min(cand, chunk_first, dim=1)
                                    for i in T.Parallel(block_m):
                                        chunk_global[i, 0] = (
                                            T.cast((t + 1) * tile_n, "float32") + chunk_first[i, 0]
                                        )
                                    if op_kind == "argmax":
                                        T.vcmp(chunk_ext, running_ext, cond, "gt")
                                    else:
                                        T.vcmp(chunk_ext, running_ext, cond, "lt")
                                    # no-alias update: in-place T.vselect(cond, A, B, B)
                                    # miscompiles the loop-carried running state on
                                    # multi-iteration serial tile loops (probe gap);
                                    # select into scratch then copy back instead.
                                    T.vselect(cond, chunk_ext, running_ext, new_ext)
                                    T.vselect(cond, chunk_global, running_idx, new_idx)
                                    T.copy(new_ext, running_ext)
                                    T.copy(new_idx, running_idx)

                            # Static tail tile.
                            if tail > 0:
                                T.copy(
                                    x[block_id : block_id + block_m, num_full * tile_n : N],
                                    x_tail,
                                )
                                T.vcast(x_tail, xw_tail, round_mode="rint")
                                if op_kind == "argmax":
                                    T.reduce_max(xw_tail, tail_ext, dim=1)
                                else:
                                    T.reduce_min(xw_tail, tail_ext, dim=1)
                                T.vbrc(tail_ext, ext_tail)
                                for i, j in T.Parallel(block_m, tail):
                                    cand_tail[i, j] = T.if_then_else(
                                        xw_tail[i, j] == ext_tail[i, j],
                                        T.cast(j, "float32"),
                                        T.float32(BIG),
                                    )
                                T.reduce_min(cand_tail, tail_first, dim=1)
                                for i in T.Parallel(block_m):
                                    tail_global[i, 0] = (
                                        T.cast(num_full * tile_n, "float32") + tail_first[i, 0]
                                    )
                                if op_kind == "argmax":
                                    T.vcmp(tail_ext, running_ext, cond_tail, "gt")
                                else:
                                    T.vcmp(tail_ext, running_ext, cond_tail, "lt")
                                T.vselect(cond_tail, tail_ext, running_ext, new_ext)
                                T.vselect(cond_tail, tail_global, running_idx, new_idx)
                                T.copy(new_ext, running_ext)
                                T.copy(new_idx, running_idx)

                            # r5 epilogue (see _build_resident): the
                            # ``T.Parallel`` cast loop on running_idx[i,0]
                            # (rank-2 load, rank-1 store) stayed scalar
                            # (NpuLoopVectorize rank-reduction guard).
                            T.reshape(running_idx, running_flat)
                            T.vcast(running_flat, out_ub, round_mode="rint")
                            T.copy(out_ub, out[block_id : block_id + block_m])

        else:

            @T.prim_func
            def main(x: T.Tensor((M, N), dtype), out: T.Tensor((M,), "int64")):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    x_ub = T.alloc_shared((block_m, tile_n), dtype)
                    x_work = T.alloc_fragment((block_m, tile_n), work_dtype)
                    chunk_ext = T.alloc_fragment((block_m, 1), work_dtype)
                    ext_brc = T.alloc_fragment((block_m, tile_n), work_dtype)
                    cand = T.alloc_fragment((block_m, tile_n), "float32")
                    chunk_first = T.alloc_fragment((block_m, 1), "float32")
                    chunk_global = T.alloc_fragment((block_m, 1), "float32")
                    cond = T.alloc_fragment((block_m, 1), "bool")
                    running_ext = T.alloc_fragment((block_m, 1), work_dtype)
                    running_idx = T.alloc_fragment((block_m, 1), "float32")
                    new_ext = T.alloc_fragment((block_m, 1), work_dtype)
                    new_idx = T.alloc_fragment((block_m, 1), "float32")
                    running_flat = T.alloc_fragment((block_m,), "float32")
                    out_ub = T.alloc_shared((block_m,), "int64")

                    x_tail = T.alloc_shared((block_m, tail_w), dtype)
                    xw_tail = T.alloc_fragment((block_m, tail_w), work_dtype)
                    ext_tail = T.alloc_fragment((block_m, tail_w), work_dtype)
                    cand_tail = T.alloc_fragment((block_m, tail_w), "float32")
                    tail_ext = T.alloc_fragment((block_m, 1), work_dtype)
                    tail_first = T.alloc_fragment((block_m, 1), "float32")
                    tail_global = T.alloc_fragment((block_m, 1), "float32")
                    cond_tail = T.alloc_fragment((block_m, 1), "bool")

                    for s in T.serial(num_local_tasks):
                        block_id = s * num_kernels + cid
                        if block_id < num_logical:
                            # Tile 0 initializes the running state directly.
                            T.copy(x[block_id : block_id + block_m, 0:tile_n], x_ub)
                            T.copy(x_ub, x_work)
                            if op_kind == "argmax":
                                T.reduce_max(x_work, running_ext, dim=1)
                            else:
                                T.reduce_min(x_work, running_ext, dim=1)
                            T.vbrc(running_ext, ext_brc)
                            for i, j in T.Parallel(block_m, tile_n):
                                cand[i, j] = T.if_then_else(
                                    x_work[i, j] == ext_brc[i, j],
                                    T.cast(j, "float32"),
                                    T.float32(BIG),
                                )
                            T.reduce_min(cand, running_idx, dim=1)

                            # Full tiles 1..num_full-1.
                            if num_full > 1:
                                for t in T.serial(num_full - 1):
                                    T.copy(
                                        x[
                                            block_id : block_id + block_m,
                                            (t + 1) * tile_n : (t + 2) * tile_n,
                                        ],
                                        x_ub,
                                    )
                                    T.copy(x_ub, x_work)
                                    if op_kind == "argmax":
                                        T.reduce_max(x_work, chunk_ext, dim=1)
                                    else:
                                        T.reduce_min(x_work, chunk_ext, dim=1)
                                    T.vbrc(chunk_ext, ext_brc)
                                    for i, j in T.Parallel(block_m, tile_n):
                                        cand[i, j] = T.if_then_else(
                                            x_work[i, j] == ext_brc[i, j],
                                            T.cast(j, "float32"),
                                            T.float32(BIG),
                                        )
                                    T.reduce_min(cand, chunk_first, dim=1)
                                    for i in T.Parallel(block_m):
                                        chunk_global[i, 0] = (
                                            T.cast((t + 1) * tile_n, "float32") + chunk_first[i, 0]
                                        )
                                    if op_kind == "argmax":
                                        T.vcmp(chunk_ext, running_ext, cond, "gt")
                                    else:
                                        T.vcmp(chunk_ext, running_ext, cond, "lt")
                                    # no-alias update: in-place T.vselect(cond, A, B, B)
                                    # miscompiles the loop-carried running state on
                                    # multi-iteration serial tile loops (probe gap);
                                    # select into scratch then copy back instead.
                                    T.vselect(cond, chunk_ext, running_ext, new_ext)
                                    T.vselect(cond, chunk_global, running_idx, new_idx)
                                    T.copy(new_ext, running_ext)
                                    T.copy(new_idx, running_idx)

                            # Static tail tile.
                            if tail > 0:
                                T.copy(
                                    x[block_id : block_id + block_m, num_full * tile_n : N],
                                    x_tail,
                                )
                                T.copy(x_tail, xw_tail)
                                if op_kind == "argmax":
                                    T.reduce_max(xw_tail, tail_ext, dim=1)
                                else:
                                    T.reduce_min(xw_tail, tail_ext, dim=1)
                                T.vbrc(tail_ext, ext_tail)
                                for i, j in T.Parallel(block_m, tail):
                                    cand_tail[i, j] = T.if_then_else(
                                        xw_tail[i, j] == ext_tail[i, j],
                                        T.cast(j, "float32"),
                                        T.float32(BIG),
                                    )
                                T.reduce_min(cand_tail, tail_first, dim=1)
                                for i in T.Parallel(block_m):
                                    tail_global[i, 0] = (
                                        T.cast(num_full * tile_n, "float32") + tail_first[i, 0]
                                    )
                                if op_kind == "argmax":
                                    T.vcmp(tail_ext, running_ext, cond_tail, "gt")
                                else:
                                    T.vcmp(tail_ext, running_ext, cond_tail, "lt")
                                T.vselect(cond_tail, tail_ext, running_ext, new_ext)
                                T.vselect(cond_tail, tail_global, running_idx, new_idx)
                                T.copy(new_ext, running_ext)
                                T.copy(new_idx, running_idx)

                            # r5 epilogue (see _build_resident): the
                            # ``T.Parallel`` cast loop on running_idx[i,0]
                            # (rank-2 load, rank-1 store) stayed scalar
                            # (NpuLoopVectorize rank-reduction guard).
                            T.reshape(running_idx, running_flat)
                            T.vcast(running_flat, out_ub, round_mode="rint")
                            T.copy(out_ub, out[block_id : block_id + block_m])

        return main

    return _func


# ---------------------------------------------------------------------------
# Factory (harness contract: _argreduce_kernel(M, N, op_kind, dtype) -> _func(block_m))
# ---------------------------------------------------------------------------
def _build_nsplit_dispatch(M, N, op_kind, dtype, work_dtype, vector_cores, tn, nchunk):
    """C6 composition: factory contract preserved, dual launch inside.

    Returns ``_func(block_m) -> main(x) -> out``; main allocates the
    (nchunk*M,) flat workspace per call (torch.empty, host-side) and
    launches partial then merge (prev-task round9 form: harness bench +
    golden PASS proven; the wrapper still sees a single callable).
    """
    partial = _build_nsplit_partial(M, N, op_kind, dtype, work_dtype, vector_cores, tn, nchunk)(M)
    merge = _build_nsplit_merge(M, N, op_kind, dtype, work_dtype, tn, nchunk)()
    ws_elems = nchunk * M
    val_torch_dtype = _DTYPE_MAP[work_dtype]

    def _func(block_m):
        # block_m accepted for interface uniformity (nsplit uses bm = M).

        def main(x):
            ws_val = torch.empty(ws_elems, dtype=val_torch_dtype, device=x.device)
            ws_idx = torch.empty(ws_elems, dtype=torch.float32, device=x.device)
            partial(x, ws_val, ws_idx)
            return merge(ws_val, ws_idx)

        return main

    return _func


def _argreduce_kernel(M, N, op_kind, dtype):
    """Build the argmax/argmin NPU kernel factory.

    Mirrors the GPU source structure ``_argreduce_kernel(M, N, op_kind, dtype)
    -> _func(block_m) -> main``. K9: the ``threads`` backend parameter is
    removed (NPU has no CUDA threads concept); the backend config is the
    UB-budget-driven block_m / tile_n (DESIGN.md sections 5.2 / 5.5).
    """
    if op_kind not in _KINDS:
        raise ValueError(f"unsupported op_kind {op_kind!r}; expected one of {sorted(_KINDS)}")
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"unsupported dtype {dtype!r}; expected one of {sorted(_SUPPORTED_DTYPES)}"
        )
    if M <= 0:
        raise ValueError(f"M must be positive, got M={M}")

    cfg = _select_config(M, N, dtype)
    work_dtype = "float32" if dtype == "bfloat16" else dtype
    vector_cores = NPUUtils.get().get_aicore_num() * 2

    # ROUND-3-F: C6 N-split dispatch (DESIGN.md 1.6.4 adjudication flipped:
    # fp16 11.92us vs single 25.90us = 0.461x; bf16 11.83us vs 30.69us =
    # 0.386x; both < the 0.8x pre-registered threshold).
    ncfg = _nsplit_config(M, N, dtype, vector_cores)
    if ncfg is not None:
        _f = _build_nsplit_dispatch(
            M,
            N,
            op_kind,
            dtype,
            work_dtype,
            vector_cores,
            ncfg["tn"],
            ncfg["nchunk"],
        )
        # msprof anchor: dual-launch dispatch -- anchor the dominant
        # partial kernel (Stage 4: partial ~8.4us + merge ~3.3us). The
        # wrapper lifts this to the Op level (tier-2/3 resolution in
        # tileops.benchmark.msprof).
        _f.msprof_kernel_name = "argreduce_partial"
        return _f

    if cfg["path"] == "narrow":
        _f = _build_narrow(
            M,
            N,
            op_kind,
            dtype,
            work_dtype,
            vector_cores,
            cfg["G"],
            cfg["bw"],
        )
        _f.msprof_kernel_name = "main"
        return _f

    if cfg["path"] == "resident":
        # r6: the resident kernel builds its column indices in-kernel (1-D
        # arange -> reshape -> vbrc; see _build_resident) -- no GM idx input,
        # no host-side injection wrapper. The call API stays
        # ``_argreduce_kernel(...)(block_m)(x)`` for the TileOPs wrapper, the
        # L0 suite and the bench, and build_case's resident branch is a plain
        # single-input kernel again.
        _f = _build_resident(M, N, op_kind, dtype, work_dtype, vector_cores)
        _f.msprof_kernel_name = "main"
        return _f
    _f = _build_tiled(M, N, op_kind, dtype, work_dtype, vector_cores, cfg["tile_n"])
    _f.msprof_kernel_name = "main"
    return _f


# ---------------------------------------------------------------------------
# Precision comparison helper
# ---------------------------------------------------------------------------
def _run_case(M, N, op_kind, dtype_str, tag, expect_block_m=None, expect_tile_n=None):
    torch_dtype = _DTYPE_MAP[dtype_str]
    cfg = _select_config(M, N, dtype_str)
    block_m = cfg["block_m"]

    if expect_block_m is not None and block_m != expect_block_m:
        raise AssertionError(
            f"[{tag}] block_m={block_m} != expected {expect_block_m} "
            f"(shape=({M},{N}) dtype={dtype_str})"
        )
    if expect_tile_n is not None and cfg["tile_n"] != expect_tile_n:
        raise AssertionError(
            f"[{tag}] tile_n={cfg['tile_n']} != expected {expect_tile_n} "
            f"(shape=({M},{N}) dtype={dtype_str})"
        )

    x = torch.randn(M, N, dtype=torch_dtype, device="npu")
    kernel = _argreduce_kernel(M, N, op_kind, dtype_str)(block_m)
    y = kernel(x)
    ref = golden_argreduce(x.cpu(), op_kind)

    assert y.dtype == torch.int64, f"[{tag}] output dtype {y.dtype} != int64"
    assert torch.equal(y.cpu(), ref), (
        f"[{tag}] mismatch shape=({M},{N}) dtype={dtype_str} op={op_kind}: "
        f"got={y.cpu().tolist()} want={ref.tolist()}"
    )

    ncfg = _nsplit_config(M, N, dtype_str, NPUUtils.get().get_aicore_num() * 2)
    eff = f"nsplit(tn={ncfg['tn']},nchunk={ncfg['nchunk']})" if ncfg else cfg["path"]
    extra = f" tile_n={cfg['tile_n']}" if cfg["tile_n"] is not None else ""
    print(
        f"[{tag}] PASS: shape=({M},{N}) dtype={dtype_str} op={op_kind} "
        f"path={eff} block_m={block_m}{extra}"
    )


# ---------------------------------------------------------------------------
# L0: gate tests (must pass) -- DESIGN.md section 8.2
# ---------------------------------------------------------------------------
def run_L0():
    cases = [
        # (M, N, dtype, op_kind, expect_block_m, expect_tile_n)
        (32, 256, "float16", "argmax", 8, None),  # L0-1a smoke fp16
        (32, 256, "float32", "argmax", 8, None),  # L0-1b smoke fp32
        (32, 256, "bfloat16", "argmax", 8, None),  # L0-1c smoke bf16
        (4, 102400, "float16", "argmax", 1, 5120),  # L0-2 lm-head fp16 tiled
        (4, 102400, "bfloat16", "argmax", 1, 4096),  # L0-3 lm-head bf16 tiled
        (2048, 4096, "float16", "argmax", 2, None),  # L0-4 hidden-state fp16
        (2048, 4096, "bfloat16", "argmax", 1, None),  # L0-5a hidden-state bf16
        (2048, 4096, "float32", "argmax", 1, None),  # L0-5b hidden-state fp32
        (4096, 4, "float16", "argmax", 1024, None),  # L0-6a 3d quick regression (narrow deint path)
        (
            524288,
            4,
            "float16",
            "argmax",
            1024,
            None,
        ),  # L0-6b 3d manifest true shape (narrow deint path)
        (128, 300, "float16", "argmax", 8, None),  # L0-7a N unaligned
        (128, 300, "bfloat16", "argmax", 8, None),  # L0-7b N unaligned bf16
        (129, 512, "float16", "argmax", 8, None),  # L0-7c M tail (129 % 8 = 1)
        (1, 512, "float16", "argmax", 1, None),  # L0-8a single row resident
        (1, 102400, "float16", "argmax", 1, 5120),  # L0-8b single row tiled
        # L0-10 argmin mirror
        (32, 256, "float16", "argmin", 8, None),
        (32, 256, "float32", "argmin", 8, None),
        (32, 256, "bfloat16", "argmin", 8, None),
        (4, 102400, "float16", "argmin", 1, 5120),
    ]
    for M, N, dtype_str, op_kind, bm, tn in cases:
        _run_case(M, N, op_kind, dtype_str, "L0", expect_block_m=bm, expect_tile_n=tn)

    # L0-9 constructed edge rows (tie / all -inf / all +inf / +-inf mix).
    _run_edge_cases("float16", "argmax", "L0")
    _run_edge_cases("float32", "argmin", "L0")
    print(f"[L0] ALL PASS: {len(cases)} shape cases + edge cases")


def _run_edge_cases(dtype_str, op_kind, tag):
    torch_dtype = _DTYPE_MAP[dtype_str]
    N = 300
    rows = []
    # Row 0: duplicate max at col 5 and col 200 -> 5 (first-occurrence tie).
    r = torch.randn(N) * 0.1
    r[5] = 7.0
    r[200] = 7.0
    rows.append(r)
    # Row 1: all -inf -> 0.
    rows.append(torch.full((N,), float("-inf")))
    # Row 2: all +inf -> 0.
    rows.append(torch.full((N,), float("inf")))
    # Row 3: +inf among -inf (argmax) / -inf among +inf (argmin) -> 10.
    r = torch.full((N,), float("-inf") if op_kind == "argmax" else float("inf"))
    r[10] = float("inf") if op_kind == "argmax" else float("-inf")
    rows.append(r)

    x = torch.stack(rows).to(dtype=torch_dtype, device="npu")
    M = x.shape[0]
    cfg = _select_config(M, N, dtype_str)
    kernel = _argreduce_kernel(M, N, op_kind, dtype_str)(cfg["block_m"])
    y = kernel(x)
    ref = golden_argreduce(x.cpu(), op_kind)
    assert y.dtype == torch.int64, f"[{tag}] edge output dtype {y.dtype} != int64"
    assert torch.equal(y.cpu(), ref), (
        f"[{tag}] edge mismatch op={op_kind} dtype={dtype_str}: "
        f"got={y.cpu().tolist()} want={ref.tolist()}"
    )
    print(f"[{tag}] PASS: edge rows M={M} N={N} dtype={dtype_str} op={op_kind}")


# ---------------------------------------------------------------------------
# L1: functional coverage (must pass)
# ---------------------------------------------------------------------------
def run_L1():
    cases = [
        (128, 512, "float32", "argmax", 8, None),
        (128, 512, "bfloat16", "argmax", 8, None),
        (130, 256, "float16", "argmax", 8, None),  # M tail 130 % 8 = 2
        (64, 1024, "float16", "argmax", 8, None),
        (256, 4096, "float16", "argmax", 2, None),  # persistent multi-wave
        (3, 300, "bfloat16", "argmin", 3, None),  # small M + argmin (bm capped to M)
        (8, 32768, "float16", "argmax", 1, 4096),  # tiled fp16 multi-tile
        (128, 16384, "float32", "argmax", 1, 2048),  # tiled fp32 multi-tile
        (4, 8192, "bfloat16", "argmin", 1, 4096),  # tiled bf16 argmin
    ]
    for M, N, dtype_str, op_kind, bm, tn in cases:
        _run_case(M, N, op_kind, dtype_str, "L1", expect_block_m=bm, expect_tile_n=tn)
    print(f"[L1] ALL PASS: {len(cases)} cases")


# ---------------------------------------------------------------------------
# L2: boundary / known-divergence (warn only, non-blocking) -- DESIGN.md 8.2
# ---------------------------------------------------------------------------
def run_L2():
    cases = [
        (1, 1, "float16", "argmax", 1, None),  # minimal
        (1, 2, "float16", "argmax", 1, None),  # N=2
        (2, 1, "float16", "argmax", 2, None),  # N=1
        (9, 256, "float16", "argmax", 8, None),  # M = block_m + 1
        (4, 7000, "float16", "argmax", 1, 6400),  # tiled static tail (7000 % 6400 = 600)
    ]
    for M, N, dtype_str, op_kind, bm, tn in cases:
        try:
            _run_case(M, N, op_kind, dtype_str, "L2", expect_block_m=bm, expect_tile_n=tn)
        except Exception as e:  # noqa: BLE001
            print(f"[L2] WARN (record only): shape=({M},{N}) {dtype_str} {op_kind}: {e}")

    # NaN row: NPU reduce propagates NaN -> sentinel 2**30; torch returns first
    # NaN index. Documented divergence (DESIGN.md section 9.2-R1) -> record only.
    try:
        x = torch.randn(4, 256, dtype=torch.float16, device="npu")
        x[0, 100] = float("nan")
        cfg = _select_config(4, 256, "float16")
        y = _argreduce_kernel(4, 256, "argmax", "float16")(cfg["block_m"])(x)
        ref = golden_argreduce(x.cpu(), "argmax")
        print(f"[L2] NaN record: kernel={y.cpu().tolist()} torch={ref.tolist()}")
    except Exception as e:  # noqa: BLE001
        print(f"[L2] WARN (record only): NaN row: {e}")

    # N > 2^24 factory assertion (raises, does not silently lose precision).
    try:
        _select_config(1, 2**24 + 1, "float16")
        print("[L2] WARN: N>2^24 did not raise (unexpected)")
    except (ValueError, AssertionError) as e:
        print(f"[L2] N>2^24 raise OK: {type(e).__name__}: {e}")
    except Exception as e:  # noqa: BLE001
        print(f"[L2] WARN (record only): N>2^24: {e}")

    # M=0 boundary (factory raises; Op layer short-circuits in harness).
    try:
        _argreduce_kernel(0, 256, "argmax", "float16")
        print("[L2] WARN: M=0 did not raise (unexpected)")
    except ValueError as e:
        print(f"[L2] M=0 raise OK: {e}")
    except Exception as e:  # noqa: BLE001
        print(f"[L2] WARN (record only): M=0: {e}")


# ---------------------------------------------------------------------------
# Boundary: extreme values (warn only, non-blocking)
# ---------------------------------------------------------------------------
def run_boundary():
    # Mixed +-0.0 rows: IEEE-equal match vs torch-CPU golden (record only,
    # DESIGN.md section 9.2-R2 notes device-kernel divergence).
    try:
        x = torch.zeros(4, 256, dtype=torch.float32, device="npu")
        x[1, 3] = 1.0
        x[2, 7] = -0.0
        cfg = _select_config(4, 256, "float32")
        y = _argreduce_kernel(4, 256, "argmax", "float32")(cfg["block_m"])(x)
        ref = golden_argreduce(x.cpu(), "argmax")
        print(f"[Boundary] +-0.0 record: kernel={y.cpu().tolist()} torch={ref.tolist()}")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): +-0.0: {e}")

    # fp32 subnormal extreme: verify first-occurrence still exact.
    try:
        x = torch.zeros(8, 256, dtype=torch.float32, device="npu")
        sub = torch.tensor(1e-40, dtype=torch.float32)  # fp32 subnormal
        x[0, 50] = sub
        cfg = _select_config(8, 256, "float32")
        y = _argreduce_kernel(8, 256, "argmax", "float32")(cfg["block_m"])(x)
        ref = golden_argreduce(x.cpu(), "argmax")
        assert torch.equal(y.cpu(), ref), f"subnormal mismatch {y.cpu()} vs {ref}"
        print("[Boundary] PASS: fp32 subnormal extreme")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): fp32 subnormal: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "all", "nsplit"])
    args, _ = parser.parse_known_args()
    if args.level == "nsplit":
        run_nsplit_L0()
    elif args.level == "L0":
        run_L0()
    else:
        run_L0()
        run_L1()
        run_L2()
        run_boundary()
    print("\033[92mAll check passed!\033[0m")


# ---------------------------------------------------------------------------
# ROUND-3-A: C6 N-split partial+merge (DESIGN.md section 1.6.4 A/B adjudication)
# ---------------------------------------------------------------------------
# Dispatch predicate (DESIGN.md 1.6.4-(1)): M <= 16, ceildiv(M, 1) < 48
# vector cores, N >= 32768 (lm-head class; the single-kernel tiled path is
# forced to bm=1 by trap C-5, leaving M/48 cores utilized).
_NSPLIT_MAX_M = 16
_NSPLIT_MIN_N = 32768
# Single-block merge working-set cap (prev-task round9 form: M*nchunk elems).
_NSPLIT_MERGE_ELEMS = 6144

# Chain-form manual bytes per element over the (M, tn) working set:
# x_ub + ext_brc + cmp(bool) + idx_j + sent_v + cand (+ fp32 x_work for bf16).
_NSPLIT_CHAIN_B = {"float16": 17, "bfloat16": 23, "float32": 21}
# Empirically stable chain footprint ceiling: the hidden-state fp16 chain
# (2, 4096) allocates 17B/elem * 8192 elems = 139264B manual and is stable
# (r1a/r2a L0 + profiles); reused as the manual-bytes ceiling.
_NSPLIT_MANUAL_CAP = 139264


def _nsplit_config(M, N, dtype, vector_cores):
    """C6 dispatch predicate + (tn, nchunk) derivation (DESIGN.md 1.6.4).

    Returns ``{"tn", "nchunk"}`` or None when the shape should stay on the
    single-kernel path. Predicate (DESIGN.md 1.6.4-(1)): M <= 16,
    ceildiv(M, 1) < vector cores, N >= 32768 (lm-head class; the tiled
    single kernel is forced to bm=1 by trap C-5 and leaves M/48 cores
    utilized).

    tn rule (r3e sweep, fp16 tn in {1280,2048,2560,5120} -> partial
    11.94/11.32/8.58/11.34us): pick the largest nchunk = N/tn that stays
    single-wave (nchunk <= vector_cores) -- tn ascending scan over
    256-multiples, first divisor hit wins (fp16 (4,102400): tn=2560,
    nchunk=40). Manual budget = per-row buffers reused across the M rows:
    tn * B_row(dtype) (+ tiny (M,) staging), B_row = x_r + w_r + e_r + c_r.
    """
    B_row = {"float16": 10, "bfloat16": 14, "float32": 16}.get(dtype)
    if B_row is None:
        return None
    if M > _NSPLIT_MAX_M or N < _NSPLIT_MIN_N or vector_cores <= M:
        return None
    if N % 256 != 0:
        return None
    max_nc = min(vector_cores, N // 256, _NSPLIT_MERGE_ELEMS // M)
    if max_nc < 2:
        return None
    tn = 256
    while tn * 2 <= N:
        if N % tn == 0:
            nc = N // tn
            if 2 <= nc <= max_nc and tn * B_row <= _NSPLIT_MANUAL_CAP:
                return {"tn": tn, "nchunk": nc}
        tn += 256
    return None


def _build_nsplit_partial(M, N, op_kind, dtype, work_dtype, vector_cores, tn, nchunk):
    """ROUND-3-E partial kernel (per-row form): all-(1,tn) tiled-path shapes.

    Fix vs r3d: the (M, tn) 2D strided column-block GM read (4 row segments
    at 204KB stride, per chunk) runs at ~1 GB/s/core effective (r3d floor
    ~14us regardless of tn; gm_to_ub_bw 0.8-1.1 GB/s vs the tiled path's
    7.6 GB/s on contiguous (1, 5120) tiles). Restructured to per-row
    (1, tn) contiguous reads + per-row (1, tn) compute -- every op shape is
    the tiled-path-proven form (copy/vbrc/fused-ite/reduce on (1, tn)),
    unrolled over the M rows at trace time (M <= 16 by the dispatch
    predicate). ws writes stay flat contiguous (chunk-major, r3d form).
    """
    num_kernels = min(nchunk, vector_cores)
    num_local_tasks = _ceildiv(nchunk, num_kernels)
    is_bf16 = dtype == "bfloat16"

    @tilelang.jit(target="npuir")  # out_idx=None: side-effect kernel (ws out)
    def _partial(block_m):
        # block_m accepted for interface uniformity; must equal M.

        if is_bf16:

            @T.prim_func
            def argreduce_partial(
                x: T.Tensor((M, N), dtype),
                ws_val: T.Tensor((nchunk * M,), work_dtype),
                ws_idx: T.Tensor((nchunk * M,), "float32"),
            ):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    x_r = T.alloc_shared((1, tn), dtype)
                    w_r = T.alloc_fragment((1, tn), "float32")
                    m_r = T.alloc_fragment((1, 1), "float32")
                    e_r = T.alloc_fragment((1, tn), "float32")
                    c_r = T.alloc_fragment((1, tn), "float32")
                    f_r = T.alloc_fragment((1, 1), "float32")
                    ext_out = T.alloc_shared((M,), work_dtype)
                    gidx_out = T.alloc_shared((M,), "float32")

                    for s in T.serial(num_local_tasks):
                        c = cid + s * num_kernels
                        if c < nchunk:
                            for r in range(M):
                                T.copy(x[r : r + 1, c * tn : (c + 1) * tn], x_r)
                                T.vcast(x_r, w_r, round_mode="rint")
                                if op_kind == "argmax":
                                    T.reduce_max(w_r, m_r, dim=1)
                                else:
                                    T.reduce_min(w_r, m_r, dim=1)
                                T.vbrc(m_r, e_r)
                                for i, j in T.Parallel(1, tn):
                                    c_r[i, j] = T.if_then_else(
                                        w_r[i, j] == e_r[i, j],
                                        T.cast(j, "float32"),
                                        T.float32(BIG),
                                    )
                                T.reduce_min(c_r, f_r, dim=1)
                                ext_out[r] = m_r[0, 0]
                                gidx_out[r] = T.cast(c * tn, "float32") + f_r[0, 0]
                            T.copy(ext_out[0:M], ws_val[c * M : (c + 1) * M])
                            T.copy(gidx_out[0:M], ws_idx[c * M : (c + 1) * M])

        else:

            @T.prim_func
            def argreduce_partial(
                x: T.Tensor((M, N), dtype),
                ws_val: T.Tensor((nchunk * M,), dtype),
                ws_idx: T.Tensor((nchunk * M,), "float32"),
            ):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    x_r = T.alloc_shared((1, tn), dtype)
                    w_r = T.alloc_fragment((1, tn), dtype)
                    m_r = T.alloc_fragment((1, 1), dtype)
                    e_r = T.alloc_fragment((1, tn), dtype)
                    c_r = T.alloc_fragment((1, tn), "float32")
                    f_r = T.alloc_fragment((1, 1), "float32")
                    ext_out = T.alloc_shared((M,), dtype)
                    gidx_out = T.alloc_shared((M,), "float32")

                    for s in T.serial(num_local_tasks):
                        c = cid + s * num_kernels
                        if c < nchunk:
                            for r in range(M):
                                T.copy(x[r : r + 1, c * tn : (c + 1) * tn], x_r)
                                T.copy(x_r, w_r)
                                if op_kind == "argmax":
                                    T.reduce_max(w_r, m_r, dim=1)
                                else:
                                    T.reduce_min(w_r, m_r, dim=1)
                                T.vbrc(m_r, e_r)
                                for i, j in T.Parallel(1, tn):
                                    c_r[i, j] = T.if_then_else(
                                        w_r[i, j] == e_r[i, j],
                                        T.cast(j, "float32"),
                                        T.float32(BIG),
                                    )
                                T.reduce_min(c_r, f_r, dim=1)
                                ext_out[r] = m_r[0, 0]
                                gidx_out[r] = T.cast(c * tn, "float32") + f_r[0, 0]
                            T.copy(ext_out[0:M], ws_val[c * M : (c + 1) * M])
                            T.copy(gidx_out[0:M], ws_idx[c * M : (c + 1) * M])

        return argreduce_partial

    return _partial


def _build_nsplit_merge(M, N, op_kind, dtype, work_dtype, tn, nchunk):
    """ROUND-3-E merge kernel (transpose form): back to the fast dims=1 chain.

    Fix vs r3d: dims=0 reduction on the long-skinny (nchunk, M) view lowers
    to per-column serial walks (r3d merge: 8.8us at (80,4), 33.4us at
    (400,4)). This merge stages the chunk-major ws through shared, applies
    the documented 2-axis T.transpose ((nchunk, M) -> (M, nchunk),
    PL-1.1-transpose-chain family), then runs the proven (M, nchunk)
    dims=1 tie-break chain (r3a merge: 1.46us).
    """

    @tilelang.jit(out_idx=[2], target="npuir")
    def _merge():

        @T.prim_func
        def argreduce_merge(
            ws_val: T.Tensor((nchunk, M), work_dtype),
            ws_idx: T.Tensor((nchunk, M), "float32"),
            out: T.Tensor((M,), "int64"),
        ):
            with T.Kernel(1, is_npu=True) as (cid, _):
                val_t = T.alloc_shared((nchunk, M), work_dtype)
                val_s = T.alloc_shared((M, nchunk), work_dtype)
                idx_t = T.alloc_shared((nchunk, M), "float32")
                idx_s = T.alloc_shared((M, nchunk), "float32")
                val_f = T.alloc_fragment((M, nchunk), work_dtype)
                idx_f = T.alloc_fragment((M, nchunk), "float32")
                g = T.alloc_fragment((M, 1), work_dtype)
                g_brc = T.alloc_fragment((M, nchunk), work_dtype)
                cmp_eq = T.alloc_fragment((M, nchunk), "bool")
                sent_v = T.alloc_fragment((M, nchunk), "float32")
                cand = T.alloc_fragment((M, nchunk), "float32")
                first = T.alloc_fragment((M, 1), "float32")
                first_flat = T.alloc_fragment((M,), "float32")
                out_ub = T.alloc_shared((M,), "int64")

                T.vbrc(T.float32(BIG), sent_v)
                T.copy(ws_val, val_t)
                T.transpose(val_t, val_s, permutation=[1, 0])
                T.copy(ws_idx, idx_t)
                T.transpose(idx_t, idx_s, permutation=[1, 0])
                T.copy(val_s, val_f)
                T.copy(idx_s, idx_f)
                if op_kind == "argmax":
                    T.reduce_max(val_f, g, dim=1)
                else:
                    T.reduce_min(val_f, g, dim=1)
                T.vbrc(g, g_brc)
                T.vcmp(val_f, g_brc, cmp_eq, "eq")
                T.vselect(cmp_eq, idx_f, sent_v, cand)
                T.reduce_min(cand, first, dim=1)
                # r5 epilogue (see _build_resident): vectorized cast; the
                # ``T.Parallel`` loop on first[i,0] (rank-2 -> rank-1) stayed
                # scalar (NpuLoopVectorize rank-reduction guard).
                T.reshape(first, first_flat)
                T.vcast(first_flat, out_ub, round_mode="rint")
                T.copy(out_ub[0:M], out[0:M])

        return argreduce_merge

    return _merge


def build_nsplit(M, N, op_kind, dtype, tn=None):
    """C6 composition entry for run_bench --mode nsplit (round-3 A/B).

    Returns ``f(x, ws_val, ws_idx) -> out`` launching partial then merge;
    the runner allocates and reuses the (M, nchunk) x2 workspace.
    """
    vector_cores = NPUUtils.get().get_aicore_num() * 2
    work_dtype = "float32" if dtype == "bfloat16" else dtype
    if tn is None:
        cfg = _nsplit_config(M, N, dtype, vector_cores)
        if cfg is None:
            raise ValueError(f"shape ({M},{N}) {dtype} not eligible for nsplit")
        tn = cfg["tn"]
    if N % tn != 0:
        raise ValueError(f"tn={tn} must divide N={N} (zero-tail chunking)")
    nchunk = N // tn
    partial = _build_nsplit_partial(M, N, op_kind, dtype, work_dtype, vector_cores, tn, nchunk)(M)
    merge = _build_nsplit_merge(M, N, op_kind, dtype, work_dtype, tn, nchunk)()

    def launch(x, ws_val, ws_idx):
        partial(x, ws_val, ws_idx)
        return merge(ws_val, ws_idx)

    return launch


def run_nsplit_L0():
    """C6 precision gate: constructed tie/edge rows across chunk boundaries.

    Covers the round8-op9 failure class: cross-chunk first-occurrence
    tie-break (duplicate extremes in different chunks, adjacent-chunk tie,
    extreme only in a late chunk), all-equal rows, all-(+-inf) rows, M=1,
    and randn sweeps per dtype/kind.
    """
    vector_cores = NPUUtils.get().get_aicore_num() * 2
    n_cases = 0
    for dtype_str in ("float16", "bfloat16"):
        dt = _DTYPE_MAP[dtype_str]
        ws_dt = torch.float32 if dtype_str == "bfloat16" else dt
        for op_kind in ("argmax", "argmin"):
            cfg = _nsplit_config(4, 102400, dtype_str, vector_cores)
            assert cfg is not None, f"nsplit config missing for {dtype_str}"
            tn, nchunk = cfg["tn"], cfg["nchunk"]
            launch = build_nsplit(4, 102400, op_kind, dtype_str, tn)
            ws_val = torch.empty(4, nchunk, dtype=ws_dt, device="npu")
            ws_idx = torch.empty(4, nchunk, dtype=torch.float32, device="npu")

            v = 7.0 if op_kind == "argmax" else -7.0
            v2 = 14.0 if op_kind == "argmax" else -14.0
            x = torch.zeros(4, 102400, dtype=dt, device="npu")
            x[0, 5] = v
            x[0, 102400 - tn] = v  # duplicate extreme, late chunk -> 5
            x[1, 102400 - tn] = v2  # extreme only in late chunk
            x[3, tn - 1] = v  # adjacent-chunk tie -> tn-1
            x[3, tn] = v
            y = launch(x, ws_val, ws_idx)
            ref = golden_argreduce(x.cpu(), op_kind)
            assert y.dtype == torch.int64 and torch.equal(y.cpu(), ref), (
                f"[nsplit-L0] tie mismatch {dtype_str}/{op_kind}: "
                f"got={y.cpu().tolist()} want={ref.tolist()}"
            )
            n_cases += 1

            # all-(extreme-unit) rows: all -inf for argmax / all +inf for argmin
            fill = float("-inf") if op_kind == "argmax" else float("inf")
            x2 = torch.full((2, 102400), fill, dtype=dt, device="npu")
            ws_val2 = torch.empty(2, nchunk, dtype=ws_dt, device="npu")
            ws_idx2 = torch.empty(2, nchunk, dtype=torch.float32, device="npu")
            y2 = launch(x2, ws_val2, ws_idx2) if False else None
            launch1 = build_nsplit(2, 102400, op_kind, dtype_str, tn)
            y2 = launch1(x2, ws_val2, ws_idx2)
            ref2 = golden_argreduce(x2.cpu(), op_kind)
            assert torch.equal(y2.cpu(), ref2), (
                f"[nsplit-L0] all-{fill} mismatch {dtype_str}/{op_kind}"
            )
            n_cases += 1

            # M=1 with a cross-chunk tie. NOTE: the original test used
            # ``40 * tn`` which equals N (out of bounds) under the r3e tn
            # rule (fp16 tn=2560, nchunk=40); use an in-bounds last-chunk
            # index instead (fixed here only; the perf_opt original still
            # carries the rot).
            x3 = torch.zeros(1, 102400, dtype=dt, device="npu")
            x3[0, 3 * tn + 7] = v
            x3[0, (nchunk - 1) * tn + 7] = v
            ws_val3 = torch.empty(1, nchunk, dtype=ws_dt, device="npu")
            ws_idx3 = torch.empty(1, nchunk, dtype=torch.float32, device="npu")
            launch2 = build_nsplit(1, 102400, op_kind, dtype_str, tn)
            y3 = launch2(x3, ws_val3, ws_idx3)
            ref3 = golden_argreduce(x3.cpu(), op_kind)
            assert torch.equal(y3.cpu(), ref3), (
                f"[nsplit-L0] M=1 tie mismatch {dtype_str}/{op_kind}"
            )
            n_cases += 1

            # randn sweeps (3 seeds).
            for seed in range(3):
                torch.manual_seed(seed)
                xr = torch.randn(4, 102400, dtype=dt, device="npu")
                yr = launch(xr, ws_val, ws_idx)
                rr = golden_argreduce(xr.cpu(), op_kind)
                assert torch.equal(yr.cpu(), rr), (
                    f"[nsplit-L0] randn seed={seed} mismatch {dtype_str}/{op_kind}"
                )
                n_cases += 1
            print(f"[nsplit-L0] PASS: {dtype_str}/{op_kind} tn={tn} nchunk={nchunk}")
    print(f"[nsplit-L0] ALL PASS: {n_cases} cases")


# ---------------------------------------------------------------------------
# Stage-4 bench helper: uniform entry for perf_opt/run_bench.py
# ---------------------------------------------------------------------------
def build_case(M, N, op_kind, dtype, block_m=None, tile_n=None):
    """Build a launch closure for one workload (Stage 4 bench entry)."""
    cfg = _select_config(M, N, dtype)
    if block_m is None:
        block_m = cfg["block_m"]
    vector_cores = NPUUtils.get().get_aicore_num() * 2
    ncfg = _nsplit_config(M, N, dtype, vector_cores)
    if ncfg is not None:
        return _build_nsplit_dispatch(
            M,
            N,
            op_kind,
            dtype,
            "float32" if dtype == "bfloat16" else dtype,
            vector_cores,
            ncfg["tn"],
            ncfg["nchunk"],
        )(block_m)
    if cfg["path"] == "narrow":
        kernel = _build_narrow(
            M,
            N,
            op_kind,
            dtype,
            "float32" if dtype == "bfloat16" else dtype,
            NPUUtils.get().get_aicore_num() * 2,
            cfg["G"],
            cfg["bw"],
        )(cfg["bw"] * cfg["G"])
    elif cfg["path"] == "resident":
        kernel = _build_resident(
            M,
            N,
            op_kind,
            dtype,
            "float32" if dtype == "bfloat16" else dtype,
            NPUUtils.get().get_aicore_num() * 2,
        )(block_m)
    else:
        tn = tile_n if tile_n is not None else cfg["tile_n"]
        kernel = _build_tiled(
            M,
            N,
            op_kind,
            dtype,
            "float32" if dtype == "bfloat16" else dtype,
            NPUUtils.get().get_aicore_num() * 2,
            tn,
        )(block_m)
    return kernel


if __name__ == "__main__":
    main()
