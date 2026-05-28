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
from tvm.script import tir as T

from tilelang.transform import CheckUBBudget


def _allocate_module(name: str, shape, dtype: str, scope: str = "local.fragment") -> IRModule:
    """Build a tiny PrimFunc that allocates one buffer of `shape` with `scope`.

    We assemble the PrimFunc by hand (rather than via `tilelang.language`)
    so the test doesn't depend on NPU availability or the JIT pipeline.
    """
    elem_offset = tir.Var("elem_offset", "int32")  # unused but required by Buffer
    buf_var = tir.Var(name, tvm.ir.PointerType(tvm.ir.PrimType(dtype), scope))
    body = tir.Allocate(buf_var, dtype, [tvm.runtime.convert(d) for d in shape],
                        tvm.tir.const(1, "bool"), tir.Evaluate(0))
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
    buf_var = tir.Var("dyn_buf",
                      tvm.ir.PointerType(tvm.ir.PrimType("float32"), "local.fragment"))
    body = tir.Allocate(buf_var, "float32", [n, tvm.runtime.convert(64)],
                        tvm.tir.const(1, "bool"), tir.Evaluate(0))
    func = tir.PrimFunc(params=[], body=body)
    mod = IRModule({"main": func})
    # Dynamic alloc with no static total triggers the "has_dynamic" branch
    # which currently does NOT raise unless there are also static allocs
    # that overflow. Just ensure the pass doesn't crash.
    out = CheckUBBudget()(mod)
    assert out is not None


def test_global_scope_buffers_ignored():
    """Global-scope buffers (kernel args) don't live in UB and must not be counted."""
    big_global = _allocate_module("global_kv", shape=[2048, 576], dtype="float16",
                                  scope="global")
    # 2048*576*2 = 2.4 MB — would overflow if counted. The pass must skip it.
    out = CheckUBBudget()(big_global)
    assert out is not None


def test_diagnostic_breakdown_sorted():
    """The error message must list allocations largest-first so the user
    immediately sees which fragment to shrink."""
    # Mix small + big — the big one must appear first.
    small_var = tir.Var("small",
                        tvm.ir.PointerType(tvm.ir.PrimType("float32"), "local.fragment"))
    big_var = tir.Var("huge",
                      tvm.ir.PointerType(tvm.ir.PrimType("float32"), "local.fragment"))
    small_alloc = tir.Allocate(small_var, "float32",
                               [tvm.runtime.convert(16), tvm.runtime.convert(16)],
                               tvm.tir.const(1, "bool"), tir.Evaluate(0))
    big_alloc = tir.Allocate(big_var, "float32",
                             [tvm.runtime.convert(256), tvm.runtime.convert(512)],
                             tvm.tir.const(1, "bool"), small_alloc)
    func = tir.PrimFunc(params=[], body=big_alloc)
    mod = IRModule({"main": func})

    with pytest.raises(RuntimeError) as exc:
        CheckUBBudget()(mod)
    msg = str(exc.value)
    # `huge` (256*512*4 = 512 KB) must appear before `small` (16*16*4 = 1 KB)
    assert msg.find("huge") < msg.find("small")


if __name__ == "__main__":
    test_small_alloc_passes()
    test_large_alloc_fails()
    test_dynamic_shape_does_not_crash()
    test_global_scope_buffers_ignored()
    test_diagnostic_breakdown_sorted()
    print("all CheckUBBudget unit tests PASS")
