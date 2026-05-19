import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T


pytestmark = [
    pytest.mark.op("parallel_min_max"),
    pytest.mark.mode("Developer"),
]

CASES_1D = [
    pytest.param(128, 32, id="minmax_128_32"),
    pytest.param(256, 64, id="minmax_256_64"),
]

CASES_2D = [
    pytest.param(8, 16, id="loop_var_max_8x16"),
    pytest.param(16, 32, id="loop_var_max_16x32"),
]


def kernel_parallel_min_from_load(numel, block):
    @T.prim_func
    def main(
        x: T.Tensor((numel,), "float32"),
        y: T.Tensor((numel,), "float32"),
        out: T.Tensor((numel,), "float32"),
    ):
        with T.Kernel(T.ceildiv(numel, block), is_npu=True) as (bx, _):
            x_shared = T.alloc_shared((block,), "float32")
            y_shared = T.alloc_shared((block,), "float32")
            out_shared = T.alloc_shared((block,), "float32")

            T.copy(x[bx * block : (bx + 1) * block], x_shared)
            T.copy(y[bx * block : (bx + 1) * block], y_shared)
            for i in T.Parallel(block):
                out_shared[i] = T.min(x_shared[i], y_shared[i])
            T.copy(out_shared, out[bx * block : (bx + 1) * block])

    return main


def kernel_parallel_max_from_load(numel, block):
    @T.prim_func
    def main(
        x: T.Tensor((numel,), "float32"),
        y: T.Tensor((numel,), "float32"),
        out: T.Tensor((numel,), "float32"),
    ):
        with T.Kernel(T.ceildiv(numel, block), is_npu=True) as (bx, _):
            x_shared = T.alloc_shared((block,), "float32")
            y_shared = T.alloc_shared((block,), "float32")
            out_shared = T.alloc_shared((block,), "float32")

            T.copy(x[bx * block : (bx + 1) * block], x_shared)
            T.copy(y[bx * block : (bx + 1) * block], y_shared)
            for i in T.Parallel(block):
                out_shared[i] = T.max(x_shared[i], y_shared[i])
            T.copy(out_shared, out[bx * block : (bx + 1) * block])

    return main


def kernel_parallel_min_scalar_lhs(numel, block, limit):
    @T.prim_func
    def main(x: T.Tensor((numel,), "float32"), out: T.Tensor((numel,), "float32")):
        with T.Kernel(T.ceildiv(numel, block), is_npu=True) as (bx, _):
            x_shared = T.alloc_shared((block,), "float32")
            out_shared = T.alloc_shared((block,), "float32")

            T.copy(x[bx * block : (bx + 1) * block], x_shared)
            for i in T.Parallel(block):
                out_shared[i] = T.min(T.float32(limit), x_shared[i])
            T.copy(out_shared, out[bx * block : (bx + 1) * block])

    return main


def kernel_parallel_max_loop_vars(block_m, block_n):
    @T.prim_func
    def main(out: T.Tensor((block_m, block_n), "int32")):
        with T.Kernel(1, is_npu=True):
            out_shared = T.alloc_shared((block_m, block_n), "int32")

            for i, j in T.Parallel(block_m, block_n):
                out_shared[i, j] = T.max(i, j)
            T.copy(out_shared, out)

    return main


@pytest.mark.parametrize("numel, block", CASES_1D)
def test_parallel_min_from_load(numel, block):
    kernel = tilelang.compile(
        kernel_parallel_min_from_load(numel, block), target="npuir"
    )
    x = torch.randn((numel,), dtype=torch.float32, device="npu")
    y = torch.randn((numel,), dtype=torch.float32, device="npu")
    out = torch.zeros((numel,), dtype=torch.float32, device="npu")
    ref = torch.minimum(x, y)

    kernel(x, y, out)

    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("numel, block", CASES_1D)
def test_parallel_max_from_load(numel, block):
    kernel = tilelang.compile(
        kernel_parallel_max_from_load(numel, block), target="npuir"
    )
    x = torch.randn((numel,), dtype=torch.float32, device="npu")
    y = torch.randn((numel,), dtype=torch.float32, device="npu")
    out = torch.zeros((numel,), dtype=torch.float32, device="npu")
    ref = torch.maximum(x, y)

    kernel(x, y, out)

    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("numel, block", CASES_1D)
def test_parallel_min_scalar_lhs(numel, block):
    limit = 0.25
    kernel = tilelang.compile(
        kernel_parallel_min_scalar_lhs(numel, block, limit), target="npuir"
    )
    x = torch.randn((numel,), dtype=torch.float32, device="npu")
    out = torch.zeros((numel,), dtype=torch.float32, device="npu")
    ref = torch.minimum(torch.full_like(x, limit), x)

    kernel(x, out)

    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("block_m, block_n", CASES_2D)
def test_parallel_max_loop_vars(block_m, block_n):
    kernel = tilelang.compile(
        kernel_parallel_max_loop_vars(block_m, block_n), target="npuir"
    )
    out = torch.zeros((block_m, block_n), dtype=torch.int32, device="npu")
    row = torch.arange(block_m, dtype=torch.int32, device="npu").unsqueeze(1)
    col = torch.arange(block_n, dtype=torch.int32, device="npu").unsqueeze(0)
    ref = torch.maximum(row, col)

    kernel(out)

    torch.testing.assert_close(out, ref, rtol=0, atol=0)
