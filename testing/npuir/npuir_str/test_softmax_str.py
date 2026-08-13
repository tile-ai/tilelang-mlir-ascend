"""Numerically stable row-wise softmax for the Ascend NPUIR target."""

import os
import torch
import tilelang
import tilelang.language as T


@tilelang.jit(target="npuir")
def softmax_2d_func(M, N, block_m, dtype="float32"):
    """Return a kernel that computes softmax over dimension 1.

    Each NPU core handles ``block_m`` complete rows.  Keeping a tile's rows
    together permits the vector reduction results of shape ``(block_m, 1)``
    to broadcast over its ``(block_m, N)`` operands.
    """
    if block_m <= 0:
        raise ValueError(f"block_m must be positive, got {block_m}")
    if M % block_m:
        raise ValueError(
            "M must be divisible by block_m: vector copies do not mask a "
            f"partial final row tile (M={M}, block_m={block_m})"
        )

    num_tiles = M // block_m

    @T.prim_func
    def softmax_kernel(
        x: T.Tensor((M, N), dtype),
        y: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(num_tiles, is_npu=True) as (pid_m, _):
            x_ub = T.alloc_ub((block_m, N), dtype)
            x_f32 = T.alloc_ub((block_m, N), "float32")
            row_max = T.alloc_ub((block_m, 1), "float32")
            exp_ub = T.alloc_ub((block_m, N), "float32")
            row_sum = T.alloc_ub((block_m, 1), "float32")
            y_ub = T.alloc_ub((block_m, N), dtype)

            T.copy(x[pid_m * block_m, 0], x_ub)
            T.vcast(x_ub, x_f32, round_mode="rint")

            # softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
            T.reduce_max(x_f32, row_max, dim=1)
            T.vsub(x_f32, row_max, exp_ub)
            T.vexp(exp_ub, exp_ub)
            T.reduce(exp_ub, row_sum, dims=[1], reduce_mode="sum")
            T.vdiv(exp_ub, row_sum, x_f32)

            T.vcast(x_f32, y_ub, round_mode="rint")
            T.copy(y_ub, y[pid_m * block_m, 0])

    return softmax_kernel


def test_softmax_2d_func():
    M, N, block_m = 32, 64, 8
    torch.manual_seed(42)

    x = torch.randn((M, N), dtype=torch.float32, device="npu")
    y = torch.empty_like(x)
    expected = torch.softmax(x.float(), dim=1).to(torch.float32)

    kernel = softmax_2d_func(M, N, block_m, dtype="float32")
    kernel(x, y)

    torch.testing.assert_close(
        y.cpu().float(), expected.cpu().float(), rtol=5e-1, atol=5e-1
    )
    print("test_softmax_single passed for dtype float32")


@tilelang.jit(target="npuir")
def softmax_2d_tail_func(M, N, block_m, fixed_n, dtype="float32"):
    """Static-shape softmax with a partial final row tile."""
    if block_m <= 0:
        raise ValueError(f"block_m must be positive, got {block_m}")
    if fixed_n < N:
        raise ValueError(f"N ({N}) must not exceed fixed_n ({fixed_n})")

    @T.prim_func
    def softmax_2d_tail_kernel(
        x: T.Tensor((M, N), dtype),
        y: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_m), is_npu=True) as (pid_m, _):
            tail_m = T.min(block_m, M - pid_m * block_m)
            x_ub = T.alloc_ub((block_m, fixed_n), dtype)
            x_f32 = T.alloc_ub((block_m, fixed_n), "float32")
            row_max = T.alloc_ub((block_m, 1), "float32")
            exp_ub = T.alloc_ub((block_m, fixed_n), "float32")
            row_sum = T.alloc_ub((block_m, 1), "float32")
            y_ub = T.alloc_ub((block_m, fixed_n), dtype)

            # Partial copy in, full fixed-size softmax, partial copy out.
            T.vbrc(-T.infinity(dtype), x_ub)
            T.copy(
                x[pid_m * block_m, 0],
                x_ub[0, 0],
                size=[tail_m, N],
            )
            T.vcast(x_ub, x_f32, round_mode="rint")
            T.reduce_max(x_f32, row_max, dim=1)
            T.vsub(x_f32, row_max, exp_ub)
            T.vexp(exp_ub, exp_ub)
            T.reduce(exp_ub, row_sum, dims=[1], reduce_mode="sum")
            T.vdiv(exp_ub, row_sum, x_f32)
            T.vcast(x_f32, y_ub, round_mode="rint")
            T.copy(
                y_ub[0, 0],
                y[pid_m * block_m, 0],
                size=[tail_m, N],
            )

    return softmax_2d_tail_kernel


def test_softmax_2d_tail_func():
    M, N, block_m, fixed_n = 19, 47, 8, 64
    torch.manual_seed(42)

    x = torch.randn((M, N), dtype=torch.float32, device="npu")
    y = torch.empty_like(x)
    expected = torch.softmax(x.float(), dim=1).to(torch.float32)

    kernel = softmax_2d_tail_func(M, N, block_m, fixed_n, dtype="float32")
    kernel(x, y)

    torch.testing.assert_close(
        y.cpu().float(), expected.cpu().float(), rtol=5e-1, atol=5e-1
    )
    print("test_softmax_2d_tail_func passed for dtype float32")


@tilelang.jit(target="npuir")
def softmax_dyn_func(block_M, fixed_N, dtype="float32"):

    M = T.symbolic("M")
    N = T.symbolic("N")

    @T.prim_func
    def softmax_dyn_kernel(
        x: T.Tensor((M, N), dtype),
        y: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M), is_npu=True) as (pid_m, _):
            valid_n = T.min(N, fixed_N)
            x_ub = T.alloc_ub((block_M, fixed_N), dtype)
            x_f32 = T.alloc_ub((block_M, fixed_N), "float32")
            row_max = T.alloc_ub((block_M, 1), "float32")
            exp_ub = T.alloc_ub((block_M, fixed_N), "float32")
            row_sum = T.alloc_ub((block_M, 1), "float32")
            y_ub = T.alloc_ub((block_M, fixed_N), dtype)

            # Mask unused fixed_N columns with -inf so they contribute zero
            # to exp and do not affect the row-wise maximum or sum.
            T.vbrc(-T.infinity(dtype), x_ub)
            T.copy(
                x[pid_m * block_M, 0],
                x_ub[0, 0],
                size=[block_M, valid_n],
            )
            T.vcast(x_ub, x_f32, round_mode="rint")

            # softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
            T.reduce_max(x_f32, row_max, dim=1)
            T.vsub(x_f32, row_max, exp_ub)
            T.vexp(exp_ub, exp_ub)
            T.reduce(exp_ub, row_sum, dims=[1], reduce_mode="sum")
            T.vdiv(exp_ub, row_sum, x_f32)

            T.vcast(x_f32, y_ub, round_mode="rint")
            T.copy(
                y_ub[0, 0],
                y[pid_m * block_M, 0],
                size=[block_M, valid_n],
            )

    return softmax_dyn_kernel


def test_softmax_dyn_func():
    M = 64
    block_M = 8
    fixed_N = 64
    N = 47
    assert fixed_N >= N, f"N ({N}) must not exceed fixed_N ({fixed_N})"
    torch.manual_seed(42)

    x = torch.randn((M, N), dtype=torch.float32, device="npu")
    y = torch.empty_like(x)
    expected = torch.softmax(x.float(), dim=1).to(torch.float32)
    kernel = softmax_dyn_func(block_M, fixed_N, dtype="float32")
    kernel(x, y)
    torch.testing.assert_close(
        y.cpu().float(), expected.cpu().float(), rtol=5e-1, atol=5e-1
    )
    print("test_softmax_dyn_func passed for dtype float32")


if __name__ == "__main__":
    os.environ["USE_NPUIR_STR"] = "1"
    tilelang.cache.clear_cache()
    test_softmax_2d_func()
    test_softmax_2d_tail_func()
    test_softmax_dyn_func()
