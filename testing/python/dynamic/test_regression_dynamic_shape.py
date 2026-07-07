"""Regression test: verify existing bare-var dynamic shapes still work after the fix."""

import sys

sys.path.insert(0, "testing/npuir")
from testcommon import gen_tensor, assert_close
import tilelang
import tilelang.language as T
import torch_npu  # noqa: F401


def test_jit_bare_var_dynamic():
    """Test the @tilelang.jit decorator path with bare-var dynamic shapes.

    This is the pattern used by all existing npuir dynamic-shape kernels
    (e.g. test_2d_grid_gemm_dyn_exp.py, test_copy_shape_dynamic.py).
    """

    @tilelang.jit(target="npuir")
    def make_kernel(dtype: str):
        M = T.symbolic("M")

        @T.prim_func
        def kernel(
            A: T.Tensor((M,), dtype),
            B: T.Tensor((M,), dtype),
        ):
            with T.Kernel(M, is_npu=True) as (cid, _):
                A_BUF = T.alloc_ub((1,), dtype)
                T.copy(A[cid : cid + 1], A_BUF[0:1])
                T.copy(A_BUF[0:1], B[cid : cid + 1])

        return kernel

    kernel = make_kernel(dtype="float16")
    v1 = gen_tensor((8,), "float16", kind="randn")
    v2 = gen_tensor((8,), "float16", kind="zeros")
    kernel(v1, v2)
    assert_close(v2.cpu(), v1.cpu(), dtype="float16", rtol=1e-2, atol=1e-2)
    print("[PASS] test_jit_bare_var_dynamic")


def test_compile_bare_var_dynamic():
    """Test tilelang.compile path with bare-var dynamic shapes."""
    M = T.symbolic("M")

    @T.prim_func
    def kernel(
        A: T.Tensor((M,), "float16"),
        B: T.Tensor((M,), "float16"),
        shape_M: T.int32,
    ):
        with T.Kernel(M, is_npu=True) as (cid, _):
            A_BUF = T.alloc_ub((1,), "float16")
            T.copy(A[cid : cid + 1], A_BUF[0:1])
            T.copy(A_BUF[0:1], B[cid : cid + 1])

    compiled = tilelang.compile(kernel, target="npuir")
    v1 = gen_tensor((8,), "float16", kind="randn")
    v2 = gen_tensor((8,), "float16", kind="zeros")
    compiled(v1, v2, 8)
    assert_close(v2.cpu(), v1.cpu(), dtype="float16", rtol=1e-2, atol=1e-2)
    print("[PASS] test_compile_bare_var_dynamic")


if __name__ == "__main__":
    test_jit_bare_var_dynamic()
    test_compile_bare_var_dynamic()
    print("\nAll regression tests passed!")
