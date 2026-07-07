"""End-to-end integration test: compile and run a kernel with composite dynamic shape
([batch_size + 1]) on NPU, verifying the fix for the _process_dynamic_symbolic limitation."""

import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor


def make_copy_kernel_with_composite_shape(block_N, dtype):
    """A kernel whose seq_map_r2c tensor has shape [raw_batch_size + 1].

    This mirrors the correct_h0 pattern where raw_batch_size only appears
    inside a composite (Add) expression in the tensor shape.
    The grid uses raw_batch_size, which must be resolved via affine inverse.
    """
    raw_batch_size = T.dynamic("raw_batch_size")

    @T.prim_func
    def kernel(
        seq_map_r2c: T.Tensor([raw_batch_size + 1], dtype),
        out: T.Tensor([raw_batch_size + 1], dtype),
    ):
        with T.Kernel(raw_batch_size, is_npu=True) as (cid, _):
            A_BUF = T.alloc_ub((1,), dtype)
            T.copy(seq_map_r2c[cid : cid + 1], A_BUF[0:1])
            T.copy(A_BUF[0:1], out[cid : cid + 1])

    return kernel


def make_kernel_with_direct_and_composite(block_N, dtype):
    """A kernel where raw_batch_size appears both as a bare var (in A's shape)
    and inside a composite expression (in seq_map_r2c's shape).

    This tests the 'prefer direct binding' logic.
    """
    raw_batch_size = T.dynamic("raw_batch_size")

    @T.prim_func
    def kernel(
        A: T.Tensor([raw_batch_size], dtype),
        seq_map_r2c: T.Tensor([raw_batch_size + 1], dtype),
        out: T.Tensor([raw_batch_size], dtype),
    ):
        with T.Kernel(raw_batch_size, is_npu=True) as (cid, _):
            A_BUF = T.alloc_ub((1,), dtype)
            T.copy(A[cid : cid + 1], A_BUF[0:1])
            T.copy(A_BUF[0:1], out[cid : cid + 1])

    return kernel


def test_composite_shape_e2e():
    """Compile and run a kernel with [raw_batch_size + 1] shape on NPU."""
    dtype = "float16"
    block_N = 1

    raw_batch_size_val = 8  # actual raw_batch_size
    seq_map_len = raw_batch_size_val + 1  # = 9

    kernel = make_copy_kernel_with_composite_shape(block_N, dtype)
    compiled = tilelang.compile(kernel, target="npuir")

    seq_map_r2c = gen_tensor((seq_map_len,), dtype, kind="randn")
    out = gen_tensor((seq_map_len,), dtype, kind="zeros")

    compiled(seq_map_r2c, out)

    # Grid iterates over raw_batch_size (=8), so only first 8 elements are copied.
    # The last element (index 8) is not touched by the kernel.
    assert_close(
        out[:raw_batch_size_val].cpu(),
        seq_map_r2c[:raw_batch_size_val].cpu(),
        dtype=dtype,
        rtol=1e-2,
        atol=1e-2,
    )
    print("[PASS] test_composite_shape_e2e")


def test_direct_and_composite_e2e():
    """Compile and run a kernel where the same var appears in both direct
    and composite shape positions."""
    dtype = "float16"
    block_N = 1

    raw_batch_size_val = 8

    kernel = make_kernel_with_direct_and_composite(block_N, dtype)
    compiled = tilelang.compile(kernel, target="npuir")

    A = gen_tensor((raw_batch_size_val,), dtype, kind="randn")
    seq_map_r2c = gen_tensor((raw_batch_size_val + 1,), dtype, kind="randn")
    out = gen_tensor((raw_batch_size_val,), dtype, kind="zeros")

    compiled(A, seq_map_r2c, out)

    assert_close(out.cpu(), A.cpu(), dtype=dtype, rtol=1e-2, atol=1e-2)
    print("[PASS] test_direct_and_composite_e2e")


if __name__ == "__main__":
    test_composite_shape_e2e()
    test_direct_and_composite_e2e()
    print("\nAll integration tests passed!")
