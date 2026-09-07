# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Tensor-weight lerp forward kernel (NPU, Developer mode).

Migrated from the GPU repo's ``_make_lerp_tensor_kernel`` (TileOPs,
Pattern-B factory closure) to TileLang ``target="npuir"``.

Semantics (frozen by DESIGN.md section 0.1):
    out[i] = a[i] + w[i] * (b[i] - a[i])
on three pre-broadcast, flattened, contiguous ``(N,)`` tensors of the
same dtype (float16 / bfloat16 / float32); output ``(N,)`` same dtype.

NPU redesign (DESIGN.md section 0.6):
    R1: one-dim ``T.Kernel`` + v-prefix vector chain (vsub -> vmul ->
        vadd, in place) replaces the CUDA threads x npt two-level
        parallel; fp32 computes in the native dtype domain.
    R2: bfloat16 has no Vector arithmetic (T.vadd/vsub/vmul dtype
        matrix), so bf16 goes through ``T.vcast(rint)`` -> fp32 compute
        -> ``T.vcast(rint)`` back (lossless upcast, single final
        rounding). Extended to fp16 by the attempt-2 precision fix
        (see Implementation Notes below).
    R3: persistent core split: ``num_kernels = min(num_logical,
        vector_cores)`` with an in-core ``T.serial`` grid-stride loop
        and an if-guard for out-of-range tiles (static bounds).

Implementation Notes (attempt-2 precision fix, 2026-09-07):
    Deviation from DESIGN.md section 0.6 R1: R1 words the fp16 path
    as "computation stays in the native dtype domain", and attempt 1
    implemented that faithfully (three-step fp16 arithmetic with
    per-op fp16 rounding). Attempt 1 then failed the fp16 precision
    gate while bf16 was bit-exact (0.0) and fp32 all-pass: the frozen
    golden (section 8.1, ``torch.lerp`` CPU, torch 2.9.0+cpu) was
    measured to compute fp16 through fp32 opmath + a single
    round-back to fp16, so the two rounding paths differ by ~2-3 ulp
    fp16 and ~0.125% of elements exceed the 1e-3 tolerance at large
    N -- a statistical property of the rounding paths, independent
    of shape / block_size / core count, not an implementation
    defect. Fix: extend the R2 fp32 transit to fp16 --
    ``vcast(rint)`` up to fp32 -> fp32 three-step chain
    (vsub/vmul/vadd) -> ``vcast(rint)`` back. Equivalence basis: the
    golden itself evaluates lerp in fp32 opmath, so the transit path
    reproduces the golden's own computation domain; measured
    max_diff vs golden = 4.883e-04 (1 ulp fp16) < atol 1e-3, zero
    violations. Cost: 4 extra vcast per tile (same instruction shape
    as the bf16 path); the fp16 UB budget becomes 20 B/elem
    (4 x fp16 buffers + 3 x fp32 transit), so the fp16 block_size
    cap drops from 24576 to 9830, kept conservatively at 8192
    (same guard form as the REVIEW.md dim-2 erratum for bf16).

Factory contract (wrapper E1/E2, see
examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor.py):
    _make_lerp_tensor_kernel(N, dtype) -> kernel(block_size) -> main
    where ``main(a, b, w)`` returns ``out`` (``out_idx=[3]``).

Run the embedded hierarchical tests:
    python _make_lerp_tensor_kernel.py --level L0
    python _make_lerp_tensor_kernel.py --level all

Stage 4 tuning note (perf_opt version, 2026-09-07):
    The only adopted change vs the Stage 3 baseline is the fp32 default
    block_size (2048 -> 8192; see _DEFAULT_BLOCK). Kernel structure,
    dispatch paths, UB budget and the factory contract (E1/E2) are
    byte-identical to the baseline. Measured (msprof op Task Duration,
    median-of-15; interleaved A/B/A/B multi-run protocol, Ascend910B2C):
    elementwise-16m/fp32 163.65 -> 158.59 us (-3.1% pooled median over
    3+3 independent runs; single-run first estimate -5.2% was inflated
    by run-level bimodality); smoke-1m/fp32 flat within noise (~11.1
    us, no regression); all fp16/bf16 workloads unchanged (already at
    the copy-only floor: the 7-pass fp32-transit vector chain is fully
    hidden under MTE2 by auto multi-buffer at >= 16M -- probe_copy
    delta <= 1.1 us). Rejected candidates are documented in
    perf_opt/opt_log.md (static tail tie, 3-buffer fp32 no gain,
    contiguous chunk +3.2% at 256M, block_size scans for fp16/bf16,
    T.Pipelined / Expert manual pipeline per mish precedent).
"""

import os

# Developer mode: single-stage copy -> compute -> copy per tile, no
# manual sync points (DESIGN.md section 7).
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse
import time

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  (registers the "npu" device)
from tilelang.utils.npu_utils import NPUUtils

__all__ = ["_make_lerp_tensor_kernel", "golden_lerp_tensor"]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Unified Buffer capacity per Vector core (docs/开发指南.md: 192 KB).
_UB_CAPACITY_BYTES = 196608

# Max block_size per dtype from the UB budget (DESIGN.md section 4.5 +
# REVIEW.md dim-2 erratum + attempt-2 precision fix): 4 live UB buffers
# of the input dtype plus, for the fp32-transit dtypes (fp16/bf16, see
# module Implementation Notes), the fp32 transit buffers. bf16 is
# budgeted conservatively at 4 x fp32 transit (24 B/elem -> 8192,
# REVIEW.md dim-2 erratum). fp16 transits since the attempt-2 fix:
# 4 x fp16 + 3 x fp32 = 20 B/elem -> 9830 hard bound, kept at the
# conservative 8192 cap (same erratum form). Both implementations
# reuse b32 as the t chain, so both guards are conservative.
_UB_MAX_BLOCK = {
    "float32": _UB_CAPACITY_BYTES // (4 * 4),  # 12288
    "float16": 8192,  # fp32 transit (attempt-2): 196608 // 20 = 9830
    "bfloat16": _UB_CAPACITY_BYTES // (4 * 2 + 4 * 4),  # 8192 (erratum)
}

# Default tile sizes. fp16/bf16 keep the DESIGN.md section 5.2 value
# (GPU threads(512) * npt(8) = 4096; the Stage 4 scan showed 2048/6144/
# 8192 are all equal or worse). float32 is raised 2048 -> 8192 by the
# Stage 4 tuning (round 1, confirmed by the round-4 interleaved A/B
# protocol): the 32KB MTE2 burst improves the load-path efficiency of
# the 4-buffer fp32 layout -- elementwise-16m/fp32 pooled median
# 163.65 -> 158.59 us (-3.1% over 3+3 independent runs; 1M flat within
# noise, no regression); 12288 is blocked by auto-multi-buffer UB
# inflation (24 B/elem budget x 1.7x inflation > 192KB).
_DEFAULT_BLOCK = {
    "float32": 8192,
    "float16": 4096,
    "bfloat16": 4096,
}

# Precision tolerances (DESIGN.md section 8.2, identical to the source
# repo test_lerp_tensor.py ``_lerp_tol``).
_TOLERANCE = {
    "float16": (torch.float16, 1e-3, 1e-3),
    "bfloat16": (torch.bfloat16, 1e-2, 1e-2),
    "float32": (torch.float32, 1e-6, 1e-6),
}


# ---------------------------------------------------------------------------
# Golden (PyTorch CPU reference implementation)
# ---------------------------------------------------------------------------
def golden_lerp_tensor(a, b, w):
    """Tensor-weight lerp reference: ``out = a + w * (b - a)``.

    Same-dtype ``torch.lerp`` (Tensor-weight overload), matching the
    source-repo test benchmark (examples/TileOPs/tests/ops/
    test_lerp_tensor.py: ``ref = torch.lerp(a, b, w)``). Independent of
    the NPU algorithm (no vcast / fp32 transit / persistent structure).
    Inputs are flattened ``(N,)`` same-dtype tensors; output ``(N,)``.
    Runs on CPU and does not require an NPU device.
    """
    return torch.lerp(a.detach().cpu(), b.detach().cpu(), w.detach().cpu())


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
def _make_lerp_tensor_kernel(N, dtype):
    """Build the Tensor-weight lerp kernel factory (E1: ``(N, dtype)``).

    Returns a JIT callable taking a single ``block_size`` (E2), which
    compiles and returns ``main``; ``main(a, b, w)`` returns ``out``
    (``out_idx=[3]``, source-operator contract).
    """
    if N <= 0:
        # DESIGN.md section 9.3: N == 0 never reaches the kernel (the Op
        # layer intercepts); reject defensively with a clear message.
        raise ValueError(f"N must be a positive int, got N={N}")
    if dtype not in _UB_MAX_BLOCK:
        raise ValueError(f"unsupported dtype {dtype!r}; expected one of {sorted(_UB_MAX_BLOCK)}")

    # Physical Vector cores (DESIGN.md section 5.5: AI cores x 2 for a
    # pure-Vector op; measured 24 -> 48 on this device). Re-queried per
    # factory call so deployments on other devices adapt automatically.
    vector_cores = NPUUtils.get().get_aicore_num() * 2

    # fp32 transit for both half-precision dtypes: bf16 has no Vector
    # arithmetic (R2); fp16 transits to match the golden's fp32 opmath
    # rounding path (attempt-2 precision fix, see module Implementation
    # Notes).
    use_fp32_transit = dtype in ("float16", "bfloat16")
    max_block = _UB_MAX_BLOCK[dtype]
    default_block = _DEFAULT_BLOCK[dtype]

    @tilelang.jit(out_idx=[3], target="npuir")
    def kernel(block_size=default_block):
        # Host-side UB budget guard (DESIGN.md section 9.2; REVIEW.md
        # dim-2 erratum: the bf16 cap is 8192, not 24576). ValueError is
        # used instead of a bare assert so the guard survives `python -O`.
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if block_size > max_block:
            raise ValueError(
                f"block_size={block_size} exceeds the UB budget for "
                f"{dtype} (max {max_block} elements, "
                f"{_UB_CAPACITY_BYTES} B UB capacity)"
            )

        # Persistent core split (DESIGN.md section 5.5). All quantities
        # are compile-time constants: N and block_size are Python ints
        # and T.ceildiv/min fold to IntImm at trace time.
        num_logical = T.ceildiv(N, block_size)  # logical tile count
        num_kernels = min(num_logical, vector_cores)  # <= physical cores
        num_local_tasks = T.ceildiv(num_logical, num_kernels)  # static bound

        # Compile-time dtype dispatch (DESIGN.md section 3.3 note: the
        # fp32-transit branch is selected inside the factory, mirroring
        # the source's single-template multi-dtype specialization).
        # Keeping the branch outside the traced body also avoids the
        # TVM-script parser's if-block variable-table scoping
        # limitation (pattern-library C11).
        if use_fp32_transit:

            @T.prim_func
            def main(
                a: T.Tensor((N,), dtype),
                b: T.Tensor((N,), dtype),
                w: T.Tensor((N,), dtype),
                out: T.Tensor((N,), dtype),
            ):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    # UB buffers (DESIGN.md section 4.3). b32 doubles as
                    # the in-place t chain.
                    a_ub = T.alloc_ub((block_size,), dtype)
                    b_ub = T.alloc_ub((block_size,), dtype)
                    w_ub = T.alloc_ub((block_size,), dtype)
                    out_ub = T.alloc_ub((block_size,), dtype)
                    # R2 fp32 transit buffers (bf16 has no v-add/sub/mul;
                    # fp16 transits to match the golden's fp32 opmath
                    # rounding path, see module Implementation Notes).
                    a32 = T.alloc_ub((block_size,), "float32")
                    b32 = T.alloc_ub((block_size,), "float32")
                    w32 = T.alloc_ub((block_size,), "float32")

                    # Persistent in-core serial task loop (static bound,
                    # grid-stride mapping, if-guard masks out-of-range
                    # tiles).
                    for i in T.serial(num_local_tasks):
                        block_id = i * num_kernels + cid
                        if block_id < num_logical:
                            t0 = block_id * block_size
                            # Tail handling (DESIGN.md section 6.4): last
                            # tile may be shorter; dst slices are always
                            # zero-origin (UB alignment pattern).
                            tail = T.min(block_size, N - t0)

                            # Load GM -> UB (same-dtype copies only).
                            T.copy(a[t0 : t0 + tail], a_ub[0:tail])
                            T.copy(b[t0 : t0 + tail], b_ub[0:tail])
                            T.copy(w[t0 : t0 + tail], w_ub[0:tail])

                            # R2: upcast to fp32 (rint, lossless since
                            # fp16/bf16 are subsets of fp32), compute
                            # the three-step chain in fp32, round back
                            # with a single rint (round-to-nearest-even,
                            # matching the golden's fp32-opmath +
                            # single-round-back path).
                            T.vcast(a_ub, a32, round_mode="rint")
                            T.vcast(b_ub, b32, round_mode="rint")
                            T.vcast(w_ub, w32, round_mode="rint")
                            T.vsub(b32, a32, b32)  # t = b - a (in place)
                            T.vmul(w32, b32, b32)  # t = w * t
                            T.vadd(a32, b32, b32)  # t = a + t
                            T.vcast(b32, out_ub, round_mode="rint")

                            # Store UB -> GM. The v-ops above ran on the
                            # full block_size buffer, so stale [tail:]
                            # lanes hold garbage; only [0:tail] is copied
                            # out (official vec_add_1d.py pattern).
                            T.copy(out_ub[0:tail], out[t0 : t0 + tail])

        else:

            @T.prim_func
            def main(
                a: T.Tensor((N,), dtype),
                b: T.Tensor((N,), dtype),
                w: T.Tensor((N,), dtype),
                out: T.Tensor((N,), dtype),
            ):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    # UB buffers (DESIGN.md section 4.3). b_ub doubles as
                    # the in-place t chain.
                    a_ub = T.alloc_ub((block_size,), dtype)
                    b_ub = T.alloc_ub((block_size,), dtype)
                    w_ub = T.alloc_ub((block_size,), dtype)
                    out_ub = T.alloc_ub((block_size,), dtype)

                    # Persistent in-core serial task loop (static bound,
                    # grid-stride mapping, if-guard masks out-of-range
                    # tiles).
                    for i in T.serial(num_local_tasks):
                        block_id = i * num_kernels + cid
                        if block_id < num_logical:
                            t0 = block_id * block_size
                            # Tail handling (DESIGN.md section 6.4): last
                            # tile may be shorter; dst slices are always
                            # zero-origin (UB alignment pattern).
                            tail = T.min(block_size, N - t0)

                            # Load GM -> UB (same-dtype copies only).
                            T.copy(a[t0 : t0 + tail], a_ub[0:tail])
                            T.copy(b[t0 : t0 + tail], b_ub[0:tail])
                            T.copy(w[t0 : t0 + tail], w_ub[0:tail])

                            # R1: fp32 computes natively in the input
                            # dtype domain (source rounding path;
                            # bit-identical to the golden).
                            T.vsub(b_ub, a_ub, b_ub)  # t = b - a (in place)
                            T.vmul(w_ub, b_ub, b_ub)  # t = w * t
                            T.vadd(a_ub, b_ub, out_ub)  # out = a + t

                            # Store UB -> GM (stale [tail:] lanes never
                            # land in GM; official vec_add_1d.py pattern).
                            T.copy(out_ub[0:tail], out[t0 : t0 + tail])

        return main

    return kernel


# ---------------------------------------------------------------------------
# Precision comparing func + hierarchical testing
# ---------------------------------------------------------------------------

_FACTORY_CACHE = {}


def _get_factory(N, dtype_str):
    """Cache factory instances (mirrors the wrapper's one-factory-per-op
    usage; the jit wrapper then caches compiled mains per block_size)."""
    key = (N, dtype_str)
    if key not in _FACTORY_CACHE:
        _FACTORY_CACHE[key] = _make_lerp_tensor_kernel(N, dtype_str)
    return _FACTORY_CACHE[key]


# Per-dtype worst max_diff across the whole run (for the final summary).
_STATS_MAX_DIFF = {}


def _compare(out, ref, atol, rtol, equal_nan=False):
    """Return (max_diff, violation_count) between out and ref.

    NaN/Inf aware: positions where both are NaN count as equal when
    equal_nan is set; equal infinities count as equal.
    """
    eq = out == ref
    if equal_nan:
        eq = eq | (torch.isnan(out) & torch.isnan(ref))
    diff = torch.where(
        eq,
        torch.zeros((), dtype=torch.float32),
        (out.float() - ref.float()).abs(),
    )
    max_diff = diff.max().item()
    violations = int((diff > (atol + rtol * ref.float().abs())).sum().item())
    return max_diff, violations


def _run_case(
    dtype_str,
    N=None,
    block_size=None,
    tag="L0",
    a=None,
    b=None,
    w=None,
    equal_nan=False,
):
    """Run one kernel case and compare against the golden.

    Tensors a/b/w may be pre-built (endpoint / special-value cases); by
    default randn(a, b) + rand(w) per DESIGN.md section 8.3 L0-1.
    Raises AssertionError on tolerance violation.
    """
    torch_dtype, atol, rtol = _TOLERANCE[dtype_str]
    if block_size is None:
        block_size = _DEFAULT_BLOCK[dtype_str]
    if a is None:
        # CPU-generate then cast+move (mish.py pattern; avoids NPU RNG
        # dtype quirks and keeps data generation device-independent).
        a = torch.randn(N, dtype=torch.float32).to(torch_dtype).npu()
        b = torch.randn(N, dtype=torch.float32).to(torch_dtype).npu()
        w = torch.rand(N, dtype=torch.float32).to(torch_dtype).npu()
    else:
        assert b is not None and w is not None, "need all three inputs"
        N = a.shape[0]
        assert a.dtype == torch_dtype, f"input dtype {a.dtype} != expected {torch_dtype}"

    kernel = _get_factory(N, dtype_str)(block_size)
    out = kernel(a, b, w)
    ref = golden_lerp_tensor(a, b, w)
    out_cpu = out.cpu()

    max_diff, violations = _compare(out_cpu, ref, atol, rtol, equal_nan=equal_nan)
    if max_diff == max_diff:  # NaN guard (all-equal cases give 0.0)
        best = _STATS_MAX_DIFF.get(dtype_str, 0.0)
        if max_diff > best:
            _STATS_MAX_DIFF[dtype_str] = max_diff

    try:
        torch.testing.assert_close(out_cpu, ref, rtol=rtol, atol=atol, equal_nan=equal_nan)
    except AssertionError:
        print(
            f"[{tag}] FAIL: N={N} dtype={dtype_str} block_size={block_size} "
            f"max_diff={max_diff:.3e} violations={violations}/{N}"
        )
        raise
    print(f"[{tag}] PASS: N={N} dtype={dtype_str} block_size={block_size} max_diff={max_diff:.3e}")
    return max_diff, violations


def _try_case(failures, case_id, **kwargs):
    """Run _run_case, collecting (not re-raising) per-case failures.

    AssertionError is treated as a precision failure; any other
    exception is recorded as a runtime error (both block the gate).
    """
    try:
        _run_case(**kwargs)
    except AssertionError as exc:
        failures.append((case_id, exc))
        first_line = str(exc).splitlines()[0] if str(exc) else repr(exc)
        print(f"[collected] {case_id} FAILED: {first_line}")
        return False
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} RUNTIME-ERROR: {exc!r}")
        return False
    return True


def _run_L0_endpoints(failures):
    """L0-5: w=0 / w=1 / extrapolation / +-inf & NaN propagation.

    Note on the golden's branch formula: torch.lerp computes
    ``zabs(w) < 0.5 ? a + w*(b-a) : b - (b-a)*(1-w)``, which diverges
    from the source kernel's plain IEEE formula exactly at
    (+-inf operand with w >= 0.5) corners (e.g. a=+inf, b finite,
    w=0.7: golden=+inf while plain-IEEE gives NaN). The inf cases below
    therefore pair +-inf operands with w < 0.5 (where torch.lerp also
    takes the plain path) or use inf-vs-inf pairs (NaN in both
    formulas), keeping the golden a valid reference for the DESIGN.md
    section 0.1 IEEE semantics.
    """
    for dtype_str in ("float16", "bfloat16", "float32"):
        torch_dtype = _TOLERANCE[dtype_str][0]
        N = 4096

        # B023-safe closure: bind the loop variable via a default arg
        # (the closure is consumed within this iteration only).
        def _to_npu(t, _dt=torch_dtype):
            return t.to(_dt).npu()

        # Random base data reused across the finite sub-cases.
        a = torch.randn(N, dtype=torch.float32)
        b = torch.randn(N, dtype=torch.float32)

        # (a) w = 0 -> out == a exactly.
        _try_case(
            failures,
            f"L0-5/{dtype_str}/w=0",
            dtype_str=dtype_str,
            tag="L0-5",
            a=_to_npu(a),
            b=_to_npu(b),
            w=_to_npu(torch.zeros(N, dtype=torch.float32)),
        )
        # (b) w = 1 -> out == b (within tolerance).
        _try_case(
            failures,
            f"L0-5/{dtype_str}/w=1",
            dtype_str=dtype_str,
            tag="L0-5",
            a=_to_npu(a),
            b=_to_npu(b),
            w=_to_npu(torch.ones(N, dtype=torch.float32)),
        )
        # (c) extrapolation w in [-1, 2) (no clamping, source semantics).
        _try_case(
            failures,
            f"L0-5/{dtype_str}/extrap",
            dtype_str=dtype_str,
            tag="L0-5",
            a=_to_npu(a),
            b=_to_npu(b),
            w=_to_npu(torch.rand(N, dtype=torch.float32) * 3.0 - 1.0),
        )
        # (d) a/b with +-inf and NaN (w kept < 0.5, see note above;
        #     inf-vs-inf pairs give NaN in both formulas).
        a2 = torch.randn(N, dtype=torch.float32)
        b2 = torch.randn(N, dtype=torch.float32)
        w2 = torch.rand(N, dtype=torch.float32) * 0.49  # strictly < 0.5
        a2[0::8] = float("inf")
        a2[1::8] = float("-inf")
        a2[2::8] = float("nan")
        b2[3::8] = float("inf")
        b2[4::8] = float("-inf")
        b2[5::8] = float("nan")
        a2[6::8] = float("inf")
        b2[6::8] = float("-inf")
        a2[7::8] = float("-inf")
        b2[7::8] = float("inf")
        _try_case(
            failures,
            f"L0-5/{dtype_str}/inf-nan",
            dtype_str=dtype_str,
            tag="L0-5",
            a=_to_npu(a2),
            b=_to_npu(b2),
            w=_to_npu(w2),
            equal_nan=True,
        )
        # (e) w with NaN / +-inf (finite a/b; formula-safe at any w).
        a3 = torch.randn(N, dtype=torch.float32)
        b3 = torch.randn(N, dtype=torch.float32)
        w3 = torch.rand(N, dtype=torch.float32)
        w3[0::8] = float("nan")
        w3[1::8] = float("inf")
        w3[2::8] = float("-inf")
        _try_case(
            failures,
            f"L0-5/{dtype_str}/w-special",
            dtype_str=dtype_str,
            tag="L0-5",
            a=_to_npu(a3),
            b=_to_npu(b3),
            w=_to_npu(w3),
            equal_nan=True,
        )


def _run_L0_persistent(failures):
    """L0-6: persistent core-count switch boundary.

    num_logical in {47, 48, 49} (fp16 block 4096): 47 -> fewer kernels
    than cores, 48 -> exact fit, 49 -> masking + uneven tasks active.
    """
    bs = 4096
    for num_logical, N in ((47, 47 * bs), (48, 48 * bs), (49, 48 * bs + 1)):
        _try_case(
            failures,
            f"L0-6/fp16/num_logical={num_logical}",
            N=N,
            dtype_str="float16",
            block_size=bs,
            tag="L0-6",
        )
    # bf16 at the masking boundary (49) to cover the vcast path under
    # the if-guard + serial loop.
    _try_case(
        failures,
        "L0-6/bf16/num_logical=49",
        N=48 * bs + 1,
        dtype_str="bfloat16",
        block_size=bs,
        tag="L0-6",
    )


def run_L0():
    """L0 gate suite (DESIGN.md section 8.3, six items).

    Returns the list of (case_id, exception) failures.
    """
    failures = []

    # L0-1: smoke, all dtypes, N=2^20, randn(a,b) + rand(w).
    for dtype_str in ("float16", "bfloat16", "float32"):
        _try_case(
            failures,
            f"L0-1/{dtype_str}",
            dtype_str=dtype_str,
            N=2**20,
            tag="L0-1",
        )

    # L0-2: large-scale spot check, N=2^24, fp16 + bf16.
    for dtype_str in ("float16", "bfloat16"):
        _try_case(
            failures,
            f"L0-2/{dtype_str}",
            dtype_str=dtype_str,
            N=2**24,
            tag="L0-2",
        )

    # L0-3: non-divisible tails (single partial tile / tail==1 /
    # non-32B-aligned tail), fp16 + fp32.
    for dtype_str in ("float16", "float32"):
        for N in (1000, 4097, 5000):
            _try_case(
                failures,
                f"L0-3/{dtype_str}/N={N}",
                dtype_str=dtype_str,
                N=N,
                tag="L0-3",
            )

    # L0-4: tiny shapes (single non-full tile / num_logical=1 degenerate).
    _try_case(failures, "L0-4/fp16/N=1", dtype_str="float16", N=1, tag="L0-4")
    _try_case(failures, "L0-4/fp16/N=7", dtype_str="float16", N=7, tag="L0-4")
    _try_case(failures, "L0-4/fp16/N=32", dtype_str="float16", N=32, tag="L0-4")
    _try_case(failures, "L0-4/bf16/N=7", dtype_str="bfloat16", N=7, tag="L0-4")
    _try_case(failures, "L0-4/fp32/N=32", dtype_str="float32", N=32, tag="L0-4")

    # L0-5: endpoints & special-value propagation.
    _run_L0_endpoints(failures)

    # L0-6: persistent core-count switch boundary.
    _run_L0_persistent(failures)

    if failures:
        print(f"[L0] {len(failures)} failing case(s): " + ", ".join(cid for cid, _ in failures))
    else:
        print("[L0] ALL PASS (six items)")
    return failures


def run_L1():
    """L1: full functional coverage (dtype x shape x block_size).

    Returns the list of (case_id, exception) failures.
    """
    failures = []

    # dtype x {exact multiple, tail==1-ish, non-32B-aligned tail}.
    for dtype_str in ("float16", "bfloat16", "float32"):
        for N in (8192, 1000, 4097):
            _try_case(
                failures,
                f"L1/{dtype_str}/N={N}",
                dtype_str=dtype_str,
                N=N,
                tag="L1",
            )

    # Non-default block sizes (wrapper config sweep; fp16/bf16 8192
    # sit exactly at the UB caps -- REVIEW.md dim-2 erratum for bf16,
    # attempt-2 fix 20 B/elem budget for fp16).
    _try_case(
        failures,
        "L1/fp16/N=2^20/bs=2048",
        dtype_str="float16",
        N=2**20,
        block_size=2048,
        tag="L1",
    )
    _try_case(
        failures,
        "L1/fp32/N=2^20/bs=4096",
        dtype_str="float32",
        N=2**20,
        block_size=4096,
        tag="L1",
    )
    _try_case(
        failures,
        "L1/fp16/N=2^20/bs=8192",
        dtype_str="float16",
        N=2**20,
        block_size=8192,
        tag="L1",
    )
    _try_case(
        failures,
        "L1/bf16/N=2^20/bs=8192",
        dtype_str="bfloat16",
        N=2**20,
        block_size=8192,
        tag="L1",
    )
    # Stage 4 tuned default for fp32 (block_size=8192) -- explicit L1
    # coverage, mirroring the fp16/bf16 bs=8192 cases above.
    _try_case(
        failures,
        "L1/fp32/N=2^20/bs=8192",
        dtype_str="float32",
        N=2**20,
        block_size=8192,
        tag="L1",
    )

    # Wrapper E2 contract: one factory callable serving two block_sizes
    # (the jit wrapper caches the compiled main per block_size).
    try:
        dtype_str = "float16"
        N = 8192
        torch_dtype, atol, rtol = _TOLERANCE[dtype_str]
        fn = _get_factory(N, dtype_str)
        a = torch.randn(N, dtype=torch.float32).to(torch_dtype).npu()
        b = torch.randn(N, dtype=torch.float32).to(torch_dtype).npu()
        w = torch.rand(N, dtype=torch.float32).to(torch_dtype).npu()
        ref = golden_lerp_tensor(a, b, w)
        for bs in (4096, 2048):
            out = fn(bs)(a, b, w).cpu()
            torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
        print("[L1] PASS: E2 factory(block_size) dual-config reuse")
    except AssertionError as exc:
        failures.append(("L1/E2-reuse", exc))
        print(f"[collected] L1/E2-reuse FAILED: {exc}")
    except Exception as exc:  # noqa: BLE001
        failures.append(("L1/E2-reuse", exc))
        print(f"[collected] L1/E2-reuse RUNTIME-ERROR: {exc!r}")

    # UB budget guard (DESIGN.md section 9.2 + REVIEW.md dim-2 erratum
    # + attempt-2 fp16 20 B/elem transit budget).
    try:
        _make_lerp_tensor_kernel(4096, "bfloat16")(8193)
    except ValueError as exc:
        print(f"[L1] PASS: UB guard rejects bf16 block_size=8193 ({exc})")
    else:
        failures.append(("L1/UB-guard-bf16", AssertionError("bf16 8193 not rejected")))
    try:
        _make_lerp_tensor_kernel(4096, "float16")(8193)
    except ValueError as exc:
        print(f"[L1] PASS: UB guard rejects fp16 block_size=8193 ({exc})")
    else:
        failures.append(("L1/UB-guard-fp16", AssertionError("fp16 8193 not rejected")))
    try:
        _make_lerp_tensor_kernel(4096, "float32")(12289)
    except ValueError as exc:
        print(f"[L1] PASS: UB guard rejects fp32 block_size=12289 ({exc})")
    else:
        failures.append(("L1/UB-guard-fp32", AssertionError("fp32 12289 not rejected")))

    if failures:
        print(f"[L1] {len(failures)} failing case(s): " + ", ".join(cid for cid, _ in failures))
    else:
        print("[L1] ALL PASS")
    return failures


def run_L2():
    """L2: extreme sizes (warn only, non-blocking)."""
    for N, dtype_str in (
        (2, "float16"),
        (3, "float32"),
        (2**22, "bfloat16"),
        (2**25, "float16"),
    ):
        try:
            _run_case(dtype_str=dtype_str, N=N, tag="L2")
        except Exception as e:  # noqa: BLE001
            print(f"[L2] WARN (Record without blocking): N={N} {dtype_str}: {e}")


def run_boundary():
    """Boundary: special-value inputs (warn only, non-blocking)."""
    N = 4096

    # zeros with w=0.5 midpoint (exact zeros in, zeros out).
    try:
        _run_case(
            dtype_str="float16",
            tag="Boundary-zeros",
            a=torch.zeros(N, dtype=torch.float16).npu(),
            b=torch.zeros(N, dtype=torch.float16).npu(),
            w=torch.full((N,), 0.5, dtype=torch.float16).npu(),
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): zeros: {e}")

    # a == b (out == a regardless of w).
    try:
        a = torch.randn(N, dtype=torch.float32).to(torch.float32).npu()
        _run_case(
            dtype_str="float32",
            tag="Boundary-a-eq-b",
            a=a,
            b=a.clone(),
            w=torch.rand(N, dtype=torch.float32).npu(),
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): a==b: {e}")

    # Subnormal fp16 magnitudes (~1e-6 scale; atol dominates).
    try:
        _run_case(
            dtype_str="float16",
            tag="Boundary-subnormal",
            a=(torch.randn(N, dtype=torch.float32) * 1e-6).to(torch.float16).npu(),
            b=(torch.randn(N, dtype=torch.float32) * 1e-6).to(torch.float16).npu(),
            w=torch.rand(N, dtype=torch.float32).to(torch.float16).npu(),
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): subnormal: {e}")

    # Large fp16 magnitudes (x100 scale; documents the residual <=1 ulp
    # gap between the kernel's fp32 plain formula and the golden's fp32
    # branch formula at cancellation corners, within the 1e-3 rtol).
    try:
        _run_case(
            dtype_str="float16",
            tag="Boundary-large",
            a=(torch.randn(N, dtype=torch.float32) * 100.0).to(torch.float16).npu(),
            b=(torch.randn(N, dtype=torch.float32) * 100.0).to(torch.float16).npu(),
            w=torch.rand(N, dtype=torch.float32).to(torch.float16).npu(),
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): large: {e}")

    # Informational (no golden assert): torch.lerp's branch formula
    # diverges from the source's plain IEEE formula exactly at
    # (+-inf operand, w >= 0.5) corners. Verify the kernel's inf-corner
    # IEEE propagation there (per-op fp16 simulation on CPU; NaN/Inf
    # corners are rounding-path invariant, and the one finite pair in
    # the probe uses exact tie arithmetic where both paths agree --
    # the authoritative finite-value comparison is vs the golden in
    # L0/L1, see module Implementation Notes).
    try:
        inf = float("inf")
        n = 8
        a = torch.tensor(
            [inf, inf, -inf, 1.0, 1.0, inf, 2.0, -inf],
            dtype=torch.float16,
        ).npu()
        b = torch.tensor(
            [1.0, 1.0, 1.0, inf, inf, -inf, 3.0, 4.0],
            dtype=torch.float16,
        ).npu()
        w = torch.full((n,), 0.7, dtype=torch.float16).npu()
        out = _get_factory(n, "float16")(4096)(a, b, w).cpu()
        # Plain per-op fp16 simulation (source semantics, section 0.1).
        a_c, b_c, w_c = a.cpu(), b.cpu(), w.cpu()
        t = b_c - a_c
        t = w_c * t
        src_sim = a_c + t
        golden = torch.lerp(a_c, b_c, w_c)
        same = bool(((out == src_sim) | (out.isnan() & src_sim.isnan())).all().item())
        print(
            f"[Boundary] INFO: inf-corner w>=0.5 source-semantics "
            f"match={same} kernel={out.tolist()} source_sim={src_sim.tolist()} "
            f"golden_branch={golden.tolist()}"
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): inf-corner info: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "all"])
    args, _ = parser.parse_known_args()

    failures = []

    t0 = time.time()
    failures += run_L0()
    print(f"[L0] elapsed {time.time() - t0:.1f}s")

    if args.level == "all":
        t1 = time.time()
        failures += run_L1()
        print(f"[L1] elapsed {time.time() - t1:.1f}s")
        t2 = time.time()
        run_L2()
        print(f"[L2] elapsed {time.time() - t2:.1f}s")
        t3 = time.time()
        run_boundary()
        print(f"[Boundary] elapsed {time.time() - t3:.1f}s")

    print(
        "Summary max_diff per dtype: "
        + ", ".join(f"{k}={v:.3e}" for k, v in sorted(_STATS_MAX_DIFF.items()))
    )
    if failures:
        raise AssertionError(
            f"{len(failures)} case(s) failed: " + "; ".join(cid for cid, _ in failures)
        )
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    main()
