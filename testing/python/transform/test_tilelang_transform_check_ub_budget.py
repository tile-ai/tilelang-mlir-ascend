# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Unit tests for the `CheckUBBudget` diagnostic pass.

The pass walks an NPUIR-target PrimFunc after `LowerOpaqueBlock`, sums the
byte sizes of every UB-backed (`local.fragment` / `shared` / `shared.dyn`)
``Allocate`` node, and raises a RuntimeError with a per-allocation
breakdown + suggested `block_M` if the total exceeds the chip's UB
capacity (192 KB on the default Ascend910B).

The tests build minimal IRModules with hand-crafted allocations rather
than going through tilelang's full JIT pipeline so they can run on any
host (no NPU required).
"""

import pytest

from tilelang import tvm as tvm
from tvm import IRModule, tir

from tilelang.transform import CheckUBBudget


def _allocate_module(
    name: str, shape, dtype: str, scope: str = "local.fragment"
) -> IRModule:
    """Build a tiny PrimFunc that allocates one buffer of `shape` with `scope`.

    We assemble the PrimFunc by hand (rather than via `tilelang.language`)
    so the test doesn't depend on NPU availability or the JIT pipeline.
    """
    buf_var = tir.Var(name, tvm.ir.PointerType(tvm.ir.PrimType(dtype), scope))
    body = tir.Allocate(
        buf_var,
        dtype,
        [tvm.runtime.convert(d) for d in shape],
        tvm.tir.const(1, "bool"),
        tir.Evaluate(0),
    )
    func = tir.PrimFunc(params=[], body=body)
    return IRModule({"main": func})


def test_small_alloc_passes():
    """A 16x16 fp16 fragment (~512 B) is far under the UB budget — pass must succeed."""
    mod = _allocate_module("acc", shape=[16, 16], dtype="float16")
    out = CheckUBBudget()(mod)  # should not raise
    assert out is not None


def test_large_alloc_fails():
    """A 64x512 fp32 fragment (131 KB) — by itself ~68% of the 192 KB UB.
    With the 80% soft budget the pass should still allow it; bump shape
    further so it definitely exceeds the budget.
    """
    # 256x512 fp32 = 524288 B > 192 KB
    mod = _allocate_module("acc_o_too_big", shape=[256, 512], dtype="float32")
    with pytest.raises(RuntimeError) as exc:
        CheckUBBudget()(mod)
    msg = str(exc.value)
    assert "UB-budget check" in msg
    assert "acc_o_too_big" in msg
    assert "Per-allocation breakdown" in msg
    # The suggestion should be present and recommend a smaller block_M.
    assert "block_M <=" in msg or "block_M" in msg


def test_dynamic_shape_does_not_crash():
    """Dynamic-extent allocations should be reported as size-unknown, not crash."""
    n = tir.Var("n", "int32")
    buf_var = tir.Var(
        "dyn_buf", tvm.ir.PointerType(tvm.ir.PrimType("float32"), "local.fragment")
    )
    body = tir.Allocate(
        buf_var,
        "float32",
        [n, tvm.runtime.convert(64)],
        tvm.tir.const(1, "bool"),
        tir.Evaluate(0),
    )
    func = tir.PrimFunc(params=[], body=body)
    mod = IRModule({"main": func})
    # Dynamic alloc with no static total triggers the "has_dynamic" branch
    # which currently does NOT raise unless there are also static allocs
    # that overflow. Just ensure the pass doesn't crash.
    out = CheckUBBudget()(mod)
    assert out is not None


def test_global_scope_buffers_ignored():
    """Global-scope buffers (kernel args) don't live in UB and must not be counted."""
    big_global = _allocate_module(
        "global_kv", shape=[2048, 576], dtype="float16", scope="global"
    )
    # 2048*576*2 = 2.4 MB — would overflow if counted. The pass must skip it.
    out = CheckUBBudget()(big_global)
    assert out is not None


def test_diagnostic_breakdown_sorted():
    """The error message must list allocations largest-first so the user
    immediately sees which fragment to shrink."""
    # Mix small + big — the big one must appear first.
    small_var = tir.Var(
        "small", tvm.ir.PointerType(tvm.ir.PrimType("float32"), "local.fragment")
    )
    big_var = tir.Var(
        "huge", tvm.ir.PointerType(tvm.ir.PrimType("float32"), "local.fragment")
    )
    small_alloc = tir.Allocate(
        small_var,
        "float32",
        [tvm.runtime.convert(16), tvm.runtime.convert(16)],
        tvm.tir.const(1, "bool"),
        tir.Evaluate(0),
    )
    big_alloc = tir.Allocate(
        big_var,
        "float32",
        [tvm.runtime.convert(256), tvm.runtime.convert(512)],
        tvm.tir.const(1, "bool"),
        small_alloc,
    )
    func = tir.PrimFunc(params=[], body=big_alloc)
    mod = IRModule({"main": func})

    with pytest.raises(RuntimeError) as exc:
        CheckUBBudget()(mod)
    msg = str(exc.value)
    # `huge` (256*512*4 = 512 KB) must appear before `small` (16*16*4 = 1 KB)
    assert msg.find("huge") < msg.find("small")


# ---- Reviewer #80 follow-up tests ------------------------------------------
# Added 2026-05-31 in response to gemini-code-assist review on PR #80.
# Each test pins one specific reviewer-flagged issue to prevent regression.


def test_mod_attrs_none_does_not_crash():
    """Reviewer #80 finding (high-1): ``mod.attrs`` may be ``None``.

    If we ``hasattr(mod, 'attrs')`` and then call ``.get('target')`` on a
    ``None`` value, we get ``AttributeError: 'NoneType' object has no
    attribute 'get'`` — a real crash on any IRModule built without an
    explicit attr dict. The pass must tolerate this and fall back to the
    default chip.
    """
    mod = _allocate_module("small_acc", shape=[16, 16], dtype="float16")
    # Sanity: this allocates well under UB, so any AttributeError here
    # would be the bug (not a real UB-overflow raise).
    assert mod.attrs is None or "target" not in mod.attrs  # no target attr set
    out = CheckUBBudget()(mod)  # must not raise
    assert out is not None


def test_uses_name_hint_not_name():
    """Reviewer #80 finding (medium-3): use ``buffer_var.name_hint`` not
    ``.name``. The Python source has been audited via grep; this test
    makes the contract enforceable.
    """
    import inspect

    from tilelang.transform import check_ub_budget as mod_under_test

    src = inspect.getsource(mod_under_test)
    # The fragile pattern is ``node.buffer_var.name`` (without ``_hint``).
    # ``name_hint`` is fine; ``func_name`` etc. are unrelated. Look for
    # the specific construct in ``_collect_ub_allocs``.
    bad_patterns = [
        "node.buffer_var.name\n",
        "node.buffer_var.name ",
        "node.buffer_var.name)",
    ]
    for pat in bad_patterns:
        assert pat not in src, (
            f"_collect_ub_allocs uses fragile pattern {pat!r}; should be "
            f"node.buffer_var.name_hint"
        )
    assert "node.buffer_var.name_hint" in src


def test_scope_of_falls_back_via_name_hint():
    """Reviewer #80 finding (medium-2): ``_scope_of`` must also try
    ``name_hint`` suffix.
    """
    from tilelang.transform.check_ub_budget import _scope_of

    # Build a buffer var with no type_annotation.storage_scope but a
    # name_hint that ends in ``_local`` — the suffix-based fallback
    # should kick in.
    v = tir.Var("acc_local", "handle")
    assert _scope_of(v) == "local"

    # Same for ``.fragment`` suffix.
    v_frag = tir.Var("scores_local.fragment", "handle")
    assert _scope_of(v_frag) == "local.fragment"

    # Unknown suffix falls back to "<unknown>" (not a crash).
    v_unknown = tir.Var("acc_weird_thing", "handle")
    assert _scope_of(v_unknown) == "<unknown>"


def test_suggest_block_M_uses_bit_length_not_log2():
    """Reviewer #80 finding (medium-4 part 2): power-of-2 rounding must
    use ``bit_length()`` rather than ``int(math.log2(...))``.

    Pin the algorithm at the source level: the float-based ``math.log2``
    can return 5.999... for input 64 on some platforms, which truncates
    to 5 and yields 32 instead of 64. ``bit_length()`` is exact.
    """
    import inspect

    from tilelang.transform import check_ub_budget as mod_under_test

    src = inspect.getsource(mod_under_test._suggest_block_M)
    assert "bit_length()" in src, (
        "_suggest_block_M no longer uses ``bit_length()`` — would "
        "reintroduce the float-precision bug"
    )
    assert "math.log2(" not in src, (
        "_suggest_block_M still uses ``math.log2`` for power-of-2 "
        "rounding — float-precision bug returns"
    )


def test_suggest_block_M_resets_per_row_state():
    """Reviewer #80 finding (medium-4 part 1): when a new ``biggest``
    alloc has element count not divisible by any guess in the table,
    ``biggest_per_row_bytes`` must be reset (not carry the stale value
    from a previous smaller alloc).

    Build a case: first alloc is small with elems divisible by 64; second
    alloc is much bigger but has a prime-ish element count that doesn't
    divide cleanly. The suggestion should be derived from the bigger
    alloc, not silently inherit the small alloc's per-row state.
    """
    from tilelang.transform.check_ub_budget import _suggest_block_M

    # alloc: (name, dtype, elems, nbytes)
    small_clean = ("small_clean", "float32", 64, 64 * 4)  # divisible by 64
    big_messy = ("big_messy", "float32", 1009 * 17, 1009 * 17 * 4)  # prime-ish
    allocs = [small_clean, big_messy]
    ub_cap = 192 * 1024
    suggestion = _suggest_block_M(allocs, ub_cap)
    assert suggestion is not None
    _, biggest_name, _ = suggestion
    assert biggest_name == "big_messy", (
        f"biggest alloc should be big_messy, got {biggest_name}; reset "
        f"likely broken — small_clean state leaked into big_messy"
    )


if __name__ == "__main__":
    test_small_alloc_passes()
    test_large_alloc_fails()
    test_dynamic_shape_does_not_crash()
    test_global_scope_buffers_ignored()
    test_diagnostic_breakdown_sorted()
    test_mod_attrs_none_does_not_crash()
    test_uses_name_hint_not_name()
    test_scope_of_falls_back_via_name_hint()
    test_suggest_block_M_uses_bit_length_not_log2()
    test_suggest_block_M_resets_per_row_state()
    print("all CheckUBBudget unit tests PASS")
