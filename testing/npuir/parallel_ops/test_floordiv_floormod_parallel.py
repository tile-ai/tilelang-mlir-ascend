import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T


pytestmark = [
    pytest.mark.op("parallel_floordiv_floormod"),
    pytest.mark.mode("Developer"),
]

CASES = [
    pytest.param(128, 32, id="parallel_div_mod_128_32"),
    pytest.param(256, 64, id="parallel_div_mod_256_64"),
]


def kernel_parallel_floordiv_from_load(numel, block, divisor):
    @T.prim_func
    def main(x: T.Tensor((numel,), "int64"), y: T.Tensor((numel,), "int64")):
        with T.Kernel(T.ceildiv(numel, block), is_npu=True) as (bx, _):
            x_shared = T.alloc_shared((block,), "int64")
            y_shared = T.alloc_shared((block,), "int64")
            T.copy(x[bx * block : (bx + 1) * block], x_shared)
            for i in T.Parallel(block):
                y_shared[i] = x_shared[i] // divisor
            T.copy(y_shared, y[bx * block : (bx + 1) * block])

    return main


def kernel_parallel_floormod_from_load(numel, block, divisor):
    @T.prim_func
    def main(x: T.Tensor((numel,), "int64"), y: T.Tensor((numel,), "int64")):
        with T.Kernel(T.ceildiv(numel, block), is_npu=True) as (bx, _):
            x_shared = T.alloc_shared((block,), "int64")
            y_shared = T.alloc_shared((block,), "int64")
            T.copy(x[bx * block : (bx + 1) * block], x_shared)
            for i in T.Parallel(block):
                y_shared[i] = x_shared[i] % divisor
            T.copy(y_shared, y[bx * block : (bx + 1) * block])

    return main


def kernel_parallel_floordiv_from_loop_var(numel, block, divisor):
    @T.prim_func
    def main(y: T.Tensor((numel,), "int64")):
        with T.Kernel(T.ceildiv(numel, block), is_npu=True) as (bx, _):
            y_shared = T.alloc_shared((block,), "int64")
            for i in T.Parallel(block):
                y_shared[i] = i // divisor
            T.copy(y_shared, y[bx * block : (bx + 1) * block])

    return main


def kernel_parallel_floormod_from_loop_var(numel, block, divisor):
    @T.prim_func
    def main(y: T.Tensor((numel,), "int64")):
        with T.Kernel(T.ceildiv(numel, block), is_npu=True) as (bx, _):
            y_shared = T.alloc_shared((block,), "int64")
            for i in T.Parallel(block):
                y_shared[i] = i % divisor
            T.copy(y_shared, y[bx * block : (bx + 1) * block])

    return main


@pytest.mark.parametrize("numel, block", CASES)
def test_parallel_floordiv_from_load(numel, block):
    divisor = 4
    kernel = tilelang.compile(
        kernel_parallel_floordiv_from_load(numel, block, divisor), target="npuir"
    )
    x = torch.arange(numel, dtype=torch.int64, device="npu")
    y = torch.zeros((numel,), dtype=torch.int64, device="npu")
    ref = x // divisor

    kernel(x, y)

    torch.testing.assert_close(y, ref, rtol=0, atol=0)


@pytest.mark.parametrize("numel, block", CASES)
def test_parallel_floormod_from_load(numel, block):
    divisor = 8
    kernel = tilelang.compile(
        kernel_parallel_floormod_from_load(numel, block, divisor), target="npuir"
    )
    x = torch.arange(numel, dtype=torch.int64, device="npu")
    y = torch.zeros((numel,), dtype=torch.int64, device="npu")
    ref = x % divisor

    kernel(x, y)

    torch.testing.assert_close(y, ref, rtol=0, atol=0)


def kernel_parallel_floordiv_invariant_offset(numel, block, divisor):
    @T.prim_func
    def main(
        base: T.Tensor((1,), "int64"),
        x: T.Tensor((numel + block,), "int64"),
        y: T.Tensor((numel,), "int64"),
    ):
        with T.Kernel(T.ceildiv(numel, block), is_npu=True) as (bx, _):
            y_shared = T.alloc_shared((block,), "int64")
            for i in T.Parallel(block):
                y_shared[i] = x[base[0] // divisor + bx * block + i]
            T.copy(y_shared, y[bx * block : (bx + 1) * block])

    return main


def kernel_parallel_floormod_invariant_offset(numel, block, divisor):
    @T.prim_func
    def main(
        base: T.Tensor((1,), "int64"),
        x: T.Tensor((numel + block,), "int64"),
        y: T.Tensor((numel,), "int64"),
    ):
        with T.Kernel(T.ceildiv(numel, block), is_npu=True) as (bx, _):
            y_shared = T.alloc_shared((block,), "int64")
            for i in T.Parallel(block):
                y_shared[i] = x[base[0] % divisor + bx * block + i]
            T.copy(y_shared, y[bx * block : (bx + 1) * block])

    return main


@pytest.mark.parametrize("numel, block", CASES)
def test_parallel_floordiv_invariant_offset(numel, block):
    divisor = 4
    kernel = tilelang.compile(
        kernel_parallel_floordiv_invariant_offset(numel, block, divisor), target="npuir"
    )
    base = torch.tensor([16], dtype=torch.int64, device="npu")
    x = torch.arange(numel + block, dtype=torch.int64, device="npu")
    y = torch.zeros((numel,), dtype=torch.int64, device="npu")
    offset = int((base.cpu() // divisor).item())
    ref = x[offset : offset + numel]

    kernel(base, x, y)

    torch.testing.assert_close(y, ref, rtol=0, atol=0)


@pytest.mark.parametrize("numel, block", CASES)
def test_parallel_floormod_invariant_offset(numel, block):
    divisor = 8
    kernel = tilelang.compile(
        kernel_parallel_floormod_invariant_offset(numel, block, divisor), target="npuir"
    )
    base = torch.tensor([19], dtype=torch.int64, device="npu")
    x = torch.arange(numel + block, dtype=torch.int64, device="npu")
    y = torch.zeros((numel,), dtype=torch.int64, device="npu")
    offset = int((base.cpu() % divisor).item())
    ref = x[offset : offset + numel]

    kernel(base, x, y)

    torch.testing.assert_close(y, ref, rtol=0, atol=0)


@pytest.mark.parametrize("numel, block", CASES)
def test_parallel_floordiv_from_loop_var(numel, block):
    divisor = 4
    kernel = tilelang.compile(
        kernel_parallel_floordiv_from_loop_var(numel, block, divisor), target="npuir"
    )
    y = torch.zeros((numel,), dtype=torch.int64, device="npu")
    pattern = torch.arange(block, dtype=torch.int64, device="npu") // divisor
    ref = pattern.repeat(numel // block)

    kernel(y)

    torch.testing.assert_close(y, ref, rtol=0, atol=0)


@pytest.mark.parametrize("numel, block", CASES)
def test_parallel_floormod_from_loop_var(numel, block):
    divisor = 8
    kernel = tilelang.compile(
        kernel_parallel_floormod_from_loop_var(numel, block, divisor), target="npuir"
    )
    y = torch.zeros((numel,), dtype=torch.int64, device="npu")
    pattern = torch.arange(block, dtype=torch.int64, device="npu") % divisor
    ref = pattern.repeat(numel // block)

    kernel(y)

    torch.testing.assert_close(y, ref, rtol=0, atol=0)
