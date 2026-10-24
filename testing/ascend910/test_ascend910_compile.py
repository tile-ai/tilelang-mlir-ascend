"""Compile-only JIT path for A2."""
from __future__ import annotations

from pathlib import Path

import pytest

import tilelang
import tilelang.language as T


@T.prim_func
def _copy(A: T.Tensor((64,), "float32"), B: T.Tensor((64,), "float32")):
    with T.Kernel(1) as (cid, _):
        s = T.alloc_ub((64,), "float32")
        T.copy(A[cid * 64 : cid * 64 + 64], s)
        T.copy(s, B[cid * 64 : cid * 64 + 64])


def test_compile_only():
    kernel = tilelang.compile(_copy, target="ascend910")
    src = kernel.get_kernel_source()
    assert "hivm.hir.set_ffts_base_addr" in src
    assert "func.func" in src
    # Execution is covered by test_ascend910_runtime_exec.py; here we only
    # assert that the compile-only kernel exposes its generated source.


def test_compile_artifacts():
    kernel = tilelang.compile(_copy, target="ascend910")
    obj, launcher = kernel.adapter.compile_artifacts()
    assert Path(obj).is_file()
    assert Path(launcher).is_file()
    assert obj.endswith("kernel.o")
    assert launcher.endswith(".so")
    # A second call must reuse the content-addressed cache.
    obj2, launcher2 = kernel.adapter.compile_artifacts()
    assert obj2 == obj
    assert launcher2 == launcher
