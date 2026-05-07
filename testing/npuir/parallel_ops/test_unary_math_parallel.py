import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T


pytestmark = [
    pytest.mark.op("parallel_unary_math"),
    pytest.mark.mode("Developer"),
]

CASES = [
    pytest.param(128, 32, id="unary_math_128_32"),
    pytest.param(256, 64, id="unary_math_256_64"),
]


def kernel_parallel_sqrt(numel, block):
    @T.prim_func
    def main(x: T.Tensor((numel,), "float32"), out: T.Tensor((numel,), "float32")):
        with T.Kernel(T.ceildiv(numel, block), is_npu=True) as (bx, _):
            x_shared = T.alloc_shared((block,), "float32")
            out_shared = T.alloc_shared((block,), "float32")

            T.copy(x[bx * block : (bx + 1) * block], x_shared)
            for i in T.Parallel(block):
                out_shared[i] = T.sqrt(x_shared[i])
            T.copy(out_shared, out[bx * block : (bx + 1) * block])

    return main


def kernel_parallel_rsqrt(numel, block):
    @T.prim_func
    def main(x: T.Tensor((numel,), "float32"), out: T.Tensor((numel,), "float32")):
        with T.Kernel(T.ceildiv(numel, block), is_npu=True) as (bx, _):
            x_shared = T.alloc_shared((block,), "float32")
            out_shared = T.alloc_shared((block,), "float32")

            T.copy(x[bx * block : (bx + 1) * block], x_shared)
            for i in T.Parallel(block):
                out_shared[i] = T.rsqrt(x_shared[i])
            T.copy(out_shared, out[bx * block : (bx + 1) * block])

    return main


def kernel_parallel_log(numel, block):
    @T.prim_func
    def main(x: T.Tensor((numel,), "float32"), out: T.Tensor((numel,), "float32")):
        with T.Kernel(T.ceildiv(numel, block), is_npu=True) as (bx, _):
            x_shared = T.alloc_shared((block,), "float32")
            out_shared = T.alloc_shared((block,), "float32")

            T.copy(x[bx * block : (bx + 1) * block], x_shared)
            for i in T.Parallel(block):
                out_shared[i] = T.log(x_shared[i])
            T.copy(out_shared, out[bx * block : (bx + 1) * block])

    return main


def kernel_parallel_nested_unary(numel, block):
    @T.prim_func
    def main(x: T.Tensor((numel,), "float32"), out: T.Tensor((numel,), "float32")):
        with T.Kernel(T.ceildiv(numel, block), is_npu=True) as (bx, _):
            x_shared = T.alloc_shared((block,), "float32")
            out_shared = T.alloc_shared((block,), "float32")

            T.copy(x[bx * block : (bx + 1) * block], x_shared)
            for i in T.Parallel(block):
                out_shared[i] = T.sqrt(T.log(x_shared[i])) + T.rsqrt(x_shared[i])
            T.copy(out_shared, out[bx * block : (bx + 1) * block])

    return main


@pytest.mark.parametrize("numel, block", CASES)
def test_parallel_sqrt(numel, block):
    kernel = tilelang.compile(kernel_parallel_sqrt(numel, block), target="npuir")
    x = torch.rand((numel,), dtype=torch.float32, device="npu") + 0.25
    out = torch.zeros((numel,), dtype=torch.float32, device="npu")
    ref = torch.sqrt(x)

    kernel(x, out)

    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("numel, block", CASES)
def test_parallel_rsqrt(numel, block):
    kernel = tilelang.compile(kernel_parallel_rsqrt(numel, block), target="npuir")
    x = torch.rand((numel,), dtype=torch.float32, device="npu") + 0.25
    out = torch.zeros((numel,), dtype=torch.float32, device="npu")
    ref = torch.rsqrt(x)

    kernel(x, out)

    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("numel, block", CASES)
def test_parallel_log(numel, block):
    kernel = tilelang.compile(kernel_parallel_log(numel, block), target="npuir")
    x = torch.rand((numel,), dtype=torch.float32, device="npu") + 1.0
    out = torch.zeros((numel,), dtype=torch.float32, device="npu")
    ref = torch.log(x)

    kernel(x, out)

    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("numel, block", CASES)
def test_parallel_nested_unary(numel, block):
    kernel = tilelang.compile(
        kernel_parallel_nested_unary(numel, block), target="npuir"
    )
    x = torch.rand((numel,), dtype=torch.float32, device="npu") + 1.0
    out = torch.zeros((numel,), dtype=torch.float32, device="npu")
    ref = torch.sqrt(torch.log(x)) + torch.rsqrt(x)

    kernel(x, out)

    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
