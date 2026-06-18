# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor

pytestmark = [
    pytest.mark.op("copy"),
    pytest.mark.mode("Developer"),
]

DTYPES = ["float16", "float32"]
# Keep case lists extensible for future copy-shape regressions.
COPY_SHAPE_2D_CASES = [(256, 1024, 32, 32)]
COPY_SHAPE_3D_CASES = [(256, 1024, 32, 32)]
COPY_SHAPE_SINGLETON_CASES = [(256, 32)]
COPY_SHAPE_DYNAMIC_TILE_CASES = [(77, 88, 32, 32, 64, 64)]


def copy_shape_1d_2d(M, N, block_M, block_N, dtype):
    @T.prim_func
    def copy_shape_dev_1d_2d(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M) * T.ceildiv(N, block_N), is_npu=True) as (cid, _):
            blockx = cid // T.ceildiv(N, block_N)
            blocky = cid % T.ceildiv(N, block_N)
            by = blocky * block_N

            A_BUF = T.alloc_shared((block_N), dtype)

            for i in T.serial(block_M):
                bx = blockx * block_M + i
                T.copy(A[bx, by:by + block_N], A_BUF)
                T.copy(A_BUF, B[bx, by:by + block_N])

    return copy_shape_dev_1d_2d


def copy_shape_2d_3d(M, N, block_M, block_N, dtype):
    @T.prim_func
    def copy_shape_dev_2d_3d(
        A: T.Tensor((1, M, N), dtype),
        B: T.Tensor((1, M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M) * T.ceildiv(N, block_N), is_npu=True) as (cid, _):
            blockx = cid // T.ceildiv(N, block_N)
            blocky = cid % T.ceildiv(N, block_N)
            by = blocky * block_N

            A_BUF = T.alloc_shared((1, block_N), dtype)

            for i in T.serial(block_M):
                bx = blockx * block_M + i
                T.copy(A[0, bx, by:by + block_N], A_BUF)
                T.copy(A_BUF, B[0, bx, by:by + block_N])

    return copy_shape_dev_2d_3d


def copy_shape_1d_2d_trailing_singleton(M, block_M, dtype):
    @T.prim_func
    def copy_shape_dev_trailing_singleton(
        A: T.Tensor((M,), dtype),
        B: T.Tensor((M,), dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M), is_npu=True) as (bx, _):
            UB = T.alloc_shared((block_M, 1), dtype)
            T.copy(A[bx * block_M], UB)
            T.copy(UB, B[bx * block_M])

    return copy_shape_dev_trailing_singleton


def copy_shape_dynamic_2d_tile(M, N, block_M, block_N, dtype):
    @T.prim_func
    def copy_shape_dev_dynamic_2d_tile(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        bx: T.int32,
        by: T.int32,
        remain_m: T.int32,
        remain_n: T.int32,
    ):
        with T.Kernel(1, is_npu=True):
            UB = T.alloc_shared((block_M, block_N), dtype)
            T.copy(A[bx:bx + remain_m, by:by + remain_n], UB)
            T.copy(UB, B[bx:bx + remain_m, by:by + remain_n])

    return copy_shape_dev_dynamic_2d_tile


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("M, N, block_M, block_N", COPY_SHAPE_2D_CASES)
def test_copy_shape_1d_2d_dev(dtype, M, N, block_M, block_N):
    v1 = gen_tensor((M, N), dtype, kind="randn")
    v2 = gen_tensor((M, N), dtype, kind="zeros")
    v_ref = v1.clone()

    func = copy_shape_1d_2d(M, N, block_M=block_M, block_N=block_N, dtype=dtype)
    compiled_kernel = tilelang.compile(func, target="npuir")
    compiled_kernel(v1, v2)

    assert_close(v2.cpu(), v_ref.cpu(), dtype=dtype, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("M, N, block_M, block_N", COPY_SHAPE_3D_CASES)
def test_copy_shape_2d_3d_dev(dtype, M, N, block_M, block_N):
    func = copy_shape_2d_3d(M, N, block_M, block_N, dtype)
    compiled_kernel = tilelang.compile(func, target="npuir")

    v1 = gen_tensor((1, M, N), dtype, kind="randn")
    v2 = gen_tensor((1, M, N), dtype, kind="randn")
    v_ref = v1.clone()
    compiled_kernel(v1, v2)

    assert_close(v2.cpu(), v_ref.cpu(), dtype=dtype, rtol=1e-2, atol=1e-2)

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("M, block_M", COPY_SHAPE_SINGLETON_CASES)
def test_copy_shape_1d_2d_trailing_singleton_dev(dtype, M, block_M):
    func = copy_shape_1d_2d_trailing_singleton(M, block_M, dtype)
    compiled_kernel = tilelang.compile(func, target="npuir")

    v1 = gen_tensor((M,), dtype, kind="randn")
    v2 = gen_tensor((M,), dtype, kind="zeros")
    v_ref = v1.clone()
    compiled_kernel(v1, v2)

    assert_close(v2.cpu(), v_ref.cpu(), dtype=dtype, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("M, N, block_M, block_N, bx, by", COPY_SHAPE_DYNAMIC_TILE_CASES)
def test_copy_shape_dynamic_2d_tile_dev(dtype, M, N, block_M, block_N, bx, by):
    func = copy_shape_dynamic_2d_tile(M, N, block_M, block_N, dtype)
    compiled_kernel = tilelang.compile(func, target="npuir")

    remain_m = M - bx
    remain_n = N - by
    assert remain_m <= block_M, f"remain_m({remain_m}) must be <= block_M({block_M})"
    assert remain_n <= block_N, f"remain_n({remain_n}) must be <= block_N({block_N})"
    v1 = gen_tensor((M, N), dtype, kind="randn")
    v2 = gen_tensor((M, N), dtype, kind="zeros")
    v_ref = torch.zeros_like(v2)
    v_ref[bx:bx + remain_m, by:by + remain_n] = v1[bx:bx + remain_m, by:by + remain_n]
    compiled_kernel(v1, v2, bx, by, remain_m, remain_n)

    assert_close(v2.cpu(), v_ref.cpu(), dtype=dtype, rtol=1e-2, atol=1e-2)
