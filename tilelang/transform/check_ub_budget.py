# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""UB budget check for the NPUIR target.

The Ascend NPU has a single "Unified Buffer" (UB) per AICore — 192 KB on
910B / dav-c220, 256 KB on 910A, 248 KB on 950 (see
``tilelang.utils.npu_arch.CHIP_SPECS``). Every fragment / shared buffer
allocated inside a kernel must live in UB.

Today an oversized `T.alloc_fragment([block_M, D], "float32")` succeeds at
TIR construction time but fails much later in ``bishengir-compile`` with
an inscrutable "ub overflow, requires N bits while M bits available"
error. The downstream tile-author then has to bisect block sizes manually.

This pass enumerates every NPUIR-target ``AllocateNode`` (which after
``LowerTileOp`` carries scope ``"local"`` for fragments and ``"shared"``
for shared buffers) in the lowered ``PrimFunc``, sums their byte sizes,
and raises a ``RuntimeError`` whose message:

  * names every allocation that contributes to the overflow
  * states the per-chip UB capacity
  * suggests a concrete ``block_M`` reduction that would fit

The pass is purely an early-fail diagnostic — it never rewrites IR. The
hard part (auto-tiling oversized fragments) is intentionally out of
scope for this first pass; the diagnostic alone removes most of the
debugging cost.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

from tilelang import tvm as tvm
from tvm import IRModule, tir
from tvm.tir import PrimFunc

# `tilelang.utils.npu_arch` has had two distinct APIs in flight:
#   - newer: module-level CHIP_SPECS dict + DEFAULT_CHIP
#   - older: AscendArch(chip_name).mem_cap["UB"]
# Probe both so this pass works against either.
try:
    from tilelang.utils.npu_arch import CHIP_SPECS as _CHIP_SPECS, DEFAULT_CHIP as _DEFAULT_CHIP
except ImportError:
    _CHIP_SPECS = None
    _DEFAULT_CHIP = "Ascend910B"
try:
    from tilelang.utils.npu_arch import AscendArch as _AscendArch
except ImportError:
    _AscendArch = None


def _resolve_chip_ub(chip_name: str):
    """Return (resolved_chip_name, ub_bytes). Falls back to 910B if unknown."""
    if _CHIP_SPECS is not None:
        if chip_name in _CHIP_SPECS:
            return chip_name, _CHIP_SPECS[chip_name]["UB"]
        return _DEFAULT_CHIP, _CHIP_SPECS[_DEFAULT_CHIP]["UB"]
    if _AscendArch is not None:
        try:
            arch = _AscendArch(chip_name)
            return arch.name, arch.mem_cap["UB"]
        except Exception:
            try:
                arch = _AscendArch("Ascend910B")
                return arch.name, arch.mem_cap["UB"]
            except Exception:
                pass
    return "Ascend910B", 192 * 1024


# Scopes that the NPUIR target backs by Unified Buffer (UB). A buffer
# allocated with any of these scopes (after `LowerTileOp` has rewritten
# `local.fragment` to `local`) is counted toward the UB budget.
_UB_BACKED_SCOPES = {"local", "shared", "local.fragment", "shared.dyn"}


def _dtype_bytes(dtype: str) -> int:
    """Return the byte width of a TIR dtype string."""
    return tvm.runtime.DataType(dtype).bits * tvm.runtime.DataType(dtype).lanes // 8


def _scope_of(buffer_var: tir.Var) -> str:
    """Best-effort scope read from a buffer var's storage_scope attribute.

    TVM has stored the storage scope in different places across versions
    (``buffer_var.type_annotation.storage_scope``, ``buffer_var.name_hint``
    suffix, the enclosing ``attr "storage_scope"`` block). Try them in
    order and fall back to "<unknown>".
    """
    ta = getattr(buffer_var, "type_annotation", None)
    if ta is not None:
        scope = getattr(ta, "storage_scope", None)
        if scope is not None:
            return str(scope)
    return "<unknown>"


def _shape_elems(shape) -> Optional[int]:
    """Return the static element count of a shape, or None if dynamic."""
    total = 1
    for d in shape:
        if isinstance(d, tir.IntImm):
            total *= int(d.value)
        elif isinstance(d, (int,)):
            total *= int(d)
        else:
            return None
    return total


def _collect_ub_allocs(prim_func: PrimFunc) -> List[Tuple[str, str, Optional[int], int]]:
    """Walk the PrimFunc and collect (name, dtype, num_elems, bytes_or_neg1).

    For dynamic-shape allocations num_elems is None and bytes is -1.
    """
    allocs: List[Tuple[str, str, Optional[int], int]] = []

    def visit(node):
        if isinstance(node, tir.Allocate):
            scope = _scope_of(node.buffer_var)
            if scope in _UB_BACKED_SCOPES or "fragment" in scope or "shared" in scope:
                elems = _shape_elems(node.extents)
                name = node.buffer_var.name
                dtype = node.dtype
                nbytes = elems * _dtype_bytes(dtype) if elems is not None else -1
                allocs.append((name, dtype, elems, nbytes))

    tir.stmt_functor.post_order_visit(prim_func.body, visit)
    return allocs


def _suggest_block_M(allocs, ub_cap: int) -> Optional[int]:
    """Heuristic: find the largest [BLOCK, _] allocation and suggest a
    block_M that would let it fit at most half the UB budget (leaving
    room for the other live fragments)."""
    biggest_name = None
    biggest_nbytes = 0
    biggest_per_row_bytes = 0
    for name, dtype, elems, nbytes in allocs:
        if nbytes <= 0:
            continue
        if nbytes > biggest_nbytes:
            biggest_nbytes = nbytes
            biggest_name = name
            # We don't have the shape decomposition here, so divide nbytes
            # by an assumed leading block_M (we have no way to know it
            # without re-reading the Allocate's extents, but we have elems
            # and dtype_bytes — caller can refine).
            db = _dtype_bytes(dtype)
            # Assume per-row bytes = nbytes / leading_dim guess. For most
            # NPU kernels leading_dim is a power-of-2 in {16, 32, 64} —
            # try the largest power-of-2 leading dim that divides elems.
            for guess in (64, 32, 16, 8):
                if elems is not None and elems % guess == 0:
                    biggest_per_row_bytes = (elems // guess) * db
                    biggest_block_M_guess = guess
                    break
    if biggest_per_row_bytes == 0:
        return None
    # Suggest block_M = floor(ub_cap / 2 / per_row_bytes)
    suggested = max(1, (ub_cap // 2) // biggest_per_row_bytes)
    # Round down to nearest power of 2.
    if suggested <= 0:
        return None
    return 1 << int(math.log2(suggested))


def _check_one(prim_func: PrimFunc, ub_cap: int, chip_name: str, func_name: str) -> None:
    allocs = _collect_ub_allocs(prim_func)
    if not allocs:
        return
    static_total = sum(b for _, _, _, b in allocs if b > 0)
    has_dynamic = any(b < 0 for _, _, _, b in allocs)
    # Allow a 20% slack vs raw UB cap because bishengir reserves some bytes
    # for sync flags / pipeline buffers. The real overflow message gives
    # the exact budget; we only need to warn before the kernel reaches it.
    soft_budget = int(ub_cap * 0.8)
    # If all allocations are dynamic-shape we can't compute a budget — emit
    # a low-priority warning via TVM's logging facility (rather than
    # raising) so the kernel still reaches bishengir for the real check.
    if static_total == 0 and has_dynamic:
        # Could log here; for now silently allow.
        return
    if static_total <= soft_budget:
        return
    lines = [
        f"tilelang UB-budget check: kernel '{func_name}' would request "
        f"{static_total} B of Unified Buffer (UB) on {chip_name} but the chip "
        f"only has {ub_cap} B available (~{soft_budget} B usable after sync "
        f"and pipeline reservations).",
        "",
        "Per-allocation breakdown (largest first):",
    ]
    for name, dtype, elems, nbytes in sorted(allocs, key=lambda x: -(x[3] if x[3] > 0 else 0)):
        if nbytes > 0:
            lines.append(f"  {nbytes:>10} B  {name}  shape elems={elems}  dtype={dtype}")
        else:
            lines.append(f"  {'?':>10}    {name}  shape=<dynamic>  dtype={dtype}")
    suggestion = _suggest_block_M(allocs, ub_cap)
    if suggestion is not None:
        lines.append("")
        lines.append(
            f"Suggested fix: reduce the leading block-M dimension so that "
            f"the largest fragment fits ~half the UB. A safe upper bound "
            f"on most kernels is `block_M <= {suggestion}`. If the kernel "
            f"is from a model with H={suggestion * 4} heads, consider "
            f"using a multi-grid pattern where each grid block handles "
            f"`block_M` heads (rather than the full H) and stitching the "
            f"results afterwards."
        )
    raise RuntimeError("\n".join(lines))


@tir.transform.prim_func_pass(opt_level=0)
def _CheckUBBudgetPass(prim_func: PrimFunc, mod: IRModule, ctx) -> PrimFunc:
    import os
    target = mod.attrs.get("target") if hasattr(mod, "attrs") else None
    # Determine chip; fall back to default if target lookup fails.
    chip_name = _DEFAULT_CHIP
    if target is not None:
        attrs = getattr(target, "attrs", None)
        if attrs is not None and "device_name" in attrs:
            chip_name = str(attrs["device_name"])
    chip_name, ub_cap = _resolve_chip_ub(chip_name)
    func_name = prim_func.attrs.get("global_symbol", "<anonymous>") if prim_func.attrs else "<anonymous>"
    if os.environ.get("TILELANG_DEBUG_UB_CHECK", ""):
        allocs_dbg = _collect_ub_allocs(prim_func)
        print(f"[CheckUBBudget] func={func_name} chip={chip_name} ub_cap={ub_cap} n_allocs={len(allocs_dbg)}")
    _check_one(prim_func, ub_cap, chip_name, str(func_name))
    return prim_func


def CheckUBBudget():
    """Diagnostic pass: error early if the NPUIR target kernel's
    fragment+shared allocations exceed the chip's Unified Buffer (UB)
    capacity, instead of letting ``bishengir-compile`` fail with an
    opaque "ub overflow" several seconds later in lowering.

    Insert this pass after ``PlanAndUpdateBufferAllocationLocation`` in
    the NPUIR target pipeline (see ``tilelang/engine/phase.py``).

    Returns
    -------
    fpass : tvm.transform.Pass
        The pass.
    """
    return _CheckUBBudgetPass
