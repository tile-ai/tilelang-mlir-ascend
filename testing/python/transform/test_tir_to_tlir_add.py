# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""
Test for the TIR -> TLIR import pass (tir_to_tlir.cc / CodeGenTIRToTLIR),
registered as the "target.build.tilelang_tlir" TVM build target.

This is the FYP's TIR -> TLIR import slice (see fyp-tilelang-mlir-dialect-
proposal.md section 5.1 / 6.1: "FileCheck tests: one positive test per TLIR
op"). Structural/text assertions here play that role at the Python level,
since the importer is exercised through TVM's build-target registration
rather than through tilelangir-opt's FileCheck harness.

v1 scope: only the elementwise add benchmark kernel (whole-buffer regions,
no ABI padding) -- see tir_to_tlir.h for the full list of deliberate
restrictions.
"""

from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tilelang.engine.lower import LowerAndLegalize, OptimizeForTarget


def _lower_to_tlir(func):
    """Runs the real TileLang lowering pipeline (LowerAndLegalize +
    OptimizeForTarget -- the same phases tilelang.engine.lower uses) and
    then calls our tir_to_tlir importer directly, bypassing the downstream
    Bishengir post-processing pipeline (which only understands hivm/
    standard dialects, not our new `tl` dialect)."""
    target = tvm.target.Target("npuir")
    mod = tvm.IRModule({func.attrs["global_symbol"]: func})
    mod = LowerAndLegalize(mod, target)
    mod = OptimizeForTarget(mod, target)

    tlir_builder = tvm.get_global_func("target.build.tilelang_tlir")
    codegen_mod = tlir_builder(mod, target)
    return codegen_mod.get_source()


def _vec_add_kernel(N=1024, block_N=1024, dtype="float32"):

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
        shape: T.int32,
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            A_VEC = T.alloc_ub((block_N,), dtype)
            B_VEC = T.alloc_ub((block_N,), dtype)
            C_VEC = T.alloc_ub((block_N,), dtype)
            T.copy(A[0:block_N], A_VEC[0:block_N])
            T.copy(B[0:block_N], B_VEC[0:block_N])
            T.vadd(A_VEC, B_VEC, C_VEC)
            T.copy(C_VEC[0:block_N], C[0:block_N])

    return main


def test_tir_to_tlir_add_op_present():
    """tl.add is emitted for the T.npuir_add / T.vadd call, matching the
    proposal's benchmark kernel #1 (elementwise add)."""
    tlir = _lower_to_tlir(_vec_add_kernel())
    assert "tl.add" in tlir
    assert "ins(" in tlir and "outs(" in tlir


def test_tir_to_tlir_alloc_op_present():
    """Each of the 3 T.alloc_ub buffers (A_VEC/B_VEC/C_VEC) becomes a
    tl.alloc."""
    tlir = _lower_to_tlir(_vec_add_kernel())
    assert tlir.count("tl.alloc") == 3


def test_tir_to_tlir_copy_op_present():
    """4 T.copy calls (A->A_VEC, B->B_VEC, C_VEC->C, plus the implicit
    ones the lowering pipeline may introduce) all become tl.copy."""
    tlir = _lower_to_tlir(_vec_add_kernel())
    assert tlir.count("tl.copy") >= 3


def test_tir_to_tlir_output_parses_with_tl_dialect():
    """Sanity check the output is well-formed module/func text (not an
    error string, not truncated) -- a lightweight substitute for
    tilelangir-opt's real parser, since we call the importer directly
    rather than round-tripping through the tl dialect's own tool here."""
    tlir = _lower_to_tlir(_vec_add_kernel())
    assert tlir.strip().startswith("module")
    assert "func.func @main" in tlir
    assert tlir.strip().endswith("}")


if __name__ == "__main__":
    tilelang.testing.main()
