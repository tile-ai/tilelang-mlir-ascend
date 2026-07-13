# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import os
import filecmp

import pytest
import tilelang
import tilelang.language as T

tilelang.cache.clear_cache()

M = 128
N = 64
BLOCK_M = 16
BLOCK_N = 32
BLOCK_SIZE = 8


def vec_deinterleave_all_channels(M, N, block_M, block_N, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, 2 * N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            A_VEC = T.alloc_ub((block_M, block_N), dtype)
            B_VEC = T.alloc_ub((block_M, block_N), dtype)
            C_VEC = T.alloc_ub((block_M, 2 * block_N), dtype)
            for i in T.serial(T.ceildiv(m_num * n_num, BLOCK_SIZE)):
                block_id = i * BLOCK_SIZE + cid
                bx = (block_id // n_num) * block_M
                by = (block_id % n_num) * block_N
                T.copy(C[bx, 2 * by], C_VEC)
                T.npuir_deinterleave(C_VEC, A_VEC, B_VEC, index_mode="ALL_CHANNELS")
                T.copy(A_VEC, A[bx, by])
                T.copy(B_VEC, B[bx, by])

    return main


def vec_deinterleave_channel_0(M, N, block_M, block_N, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, 2 * N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            A_VEC = T.alloc_ub((block_M, block_N), dtype)
            C_VEC = T.alloc_ub((block_M, 2 * block_N), dtype)
            for i in T.serial(T.ceildiv(m_num * n_num, BLOCK_SIZE)):
                block_id = i * BLOCK_SIZE + cid
                bx = (block_id // n_num) * block_M
                by = (block_id % n_num) * block_N
                T.copy(C[bx, 2 * by], C_VEC)
                T.npuir_deinterleave(C_VEC, A_VEC, index_mode="CHANNEL_0")
                T.copy(A_VEC, A[bx, by])

    return main


def vec_deinterleave_channel_1(M, N, block_M, block_N, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, 2 * N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            B_VEC = T.alloc_ub((block_M, block_N), dtype)
            C_VEC = T.alloc_ub((block_M, 2 * block_N), dtype)
            for i in T.serial(T.ceildiv(m_num * n_num, BLOCK_SIZE)):
                block_id = i * BLOCK_SIZE + cid
                bx = (block_id // n_num) * block_M
                by = (block_id % n_num) * block_N
                T.copy(C[bx, 2 * by], C_VEC)
                T.npuir_deinterleave(C_VEC, B_VEC, index_mode="CHANNEL_1")
                T.copy(B_VEC, B[bx, by])

    return main


def _compare_mlir(kernel: str, ref_basename: str):
    output_file = "./output/" + ref_basename
    os.makedirs("./output", exist_ok=True)
    with open(output_file, "w") as f:
        f.write(kernel)
    ref_file = "./mlir_files/" + ref_basename
    assert filecmp.cmp(output_file, ref_file, shallow=False), (
        f"'{output_file}' and '{ref_file}' are not identical"
    )


@pytest.mark.parametrize(
    ("builder", "ref_mlir"),
    [
        (vec_deinterleave_all_channels, "vec_deinterleave_all_channels.mlir"),
        (vec_deinterleave_channel_0, "vec_deinterleave_channel_0.mlir"),
        (vec_deinterleave_channel_1, "vec_deinterleave_channel_1.mlir"),
    ],
)
def test_vec_deinterleave(builder, ref_mlir):
    func = builder(M, N, BLOCK_M, BLOCK_N)
    kernel = tilelang.engine.lower(func)
    _compare_mlir(kernel, ref_mlir)


if __name__ == "__main__":
    for builder, ref in [
        (vec_deinterleave_all_channels, "vec_deinterleave_all_channels.mlir"),
        (vec_deinterleave_channel_0, "vec_deinterleave_channel_0.mlir"),
        (vec_deinterleave_channel_1, "vec_deinterleave_channel_1.mlir"),
    ]:
        test_vec_deinterleave(builder, ref)
