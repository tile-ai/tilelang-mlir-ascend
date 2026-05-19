import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T


pytestmark = [
    pytest.mark.op("parallel_cast"),
    pytest.mark.mode("Developer"),
]

DIRECT_CASES = [
    pytest.param(128, 32, id="cast_direct_128_32"),
    pytest.param(256, 64, id="cast_direct_256_64"),
]

EXPR_CASES = [
    pytest.param(256, 8, 4, id="cast_expr_256_8_4"),
    pytest.param(512, 16, 8, id="cast_expr_512_16_8"),
]

RANK_MISMATCH_CASES = [
    pytest.param(2, 1024, 4, id="cast_rank_mismatch_2x1024_4"),
]


def kernel_parallel_cast_direct(numel, block):
    @T.prim_func
    def main(x: T.Tensor((numel,), "float32"), y: T.Tensor((numel,), "float16")):
        with T.Kernel(T.ceildiv(numel, block), is_npu=True) as (bx, _):
            x_shared = T.alloc_shared((block,), "float32")
            y_shared = T.alloc_shared((block,), "float16")
            T.copy(x[bx * block : (bx + 1) * block], x_shared)
            for i in T.Parallel(block):
                y_shared[i] = T.cast(x_shared[i], "float16")
            T.copy(y_shared, y[bx * block : (bx + 1) * block])

    return main


def kernel_parallel_cast_expr(numel, threads, npt):
    block_size = threads * npt

    @T.prim_func
    def main(x: T.Tensor((numel,), "float32"), y: T.Tensor((numel,), "float32")):
        with T.Kernel(T.ceildiv(numel, block_size), threads=threads) as bx:
            for i, j in T.Parallel(threads, npt):
                idx = (bx * threads + i) * npt + j
                y[idx] = T.cast(T.cast(x[idx], "float16"), "float32") + T.float32(1.0)

    return main


def kernel_parallel_cast_rank_mismatch(heads, width, block):
    @T.prim_func
    def main(
        go_smem: T.Tensor((heads, width), "float16"),
        out: T.Tensor((block,), "float32"),
    ):
        with T.Kernel(1, is_npu=True):
            go_x_local = T.alloc_shared((block,), "float32")
            head_id = 0
            tile_id = 0
            i_sub = 0
            sub_warp_id = 0
            lane_id = 0
            for i_k_split in T.Parallel(block):
                go_x_local[i_k_split] = T.Cast(
                    "float32",
                    go_smem[
                        head_id,
                        tile_id * 1024
                        + i_sub * 256
                        + sub_warp_id * 128
                        + lane_id * 4
                        + i_k_split,
                    ],
                )
            T.copy(go_x_local, out)

    return main


@pytest.mark.parametrize("numel, block", DIRECT_CASES)
def test_parallel_cast_direct(numel, block):
    kernel = tilelang.compile(kernel_parallel_cast_direct(numel, block), target="npuir")
    x = torch.randn((numel,), dtype=torch.float32, device="npu")
    y = torch.zeros((numel,), dtype=torch.float16, device="npu")
    ref = x.to(torch.float16)

    kernel(x, y)

    torch.testing.assert_close(y, ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("heads, width, block", RANK_MISMATCH_CASES)
def test_parallel_cast_rank_mismatch(heads, width, block):
    kernel = tilelang.compile(
        kernel_parallel_cast_rank_mismatch(heads, width, block), target="npuir"
    )
    go_smem = torch.randn((heads, width), dtype=torch.float16, device="npu")
    out = torch.zeros((block,), dtype=torch.float32, device="npu")
    ref = go_smem[0, :block].to(torch.float32)

    kernel(go_smem, out)

    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("numel, threads, npt", EXPR_CASES)
def test_parallel_cast_expr(numel, threads, npt):
    kernel = tilelang.compile(
        kernel_parallel_cast_expr(numel, threads, npt), target="npuir"
    )
    x = torch.randn((numel,), dtype=torch.float32, device="npu")
    y = torch.zeros((numel,), dtype=torch.float32, device="npu")
    ref = x.to(torch.float16).to(torch.float32) + 1.0

    kernel(x, y)

    torch.testing.assert_close(y, ref, rtol=1e-3, atol=1e-3)
