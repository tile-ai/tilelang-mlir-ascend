# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import os
import filecmp

import tilelang
import tilelang.language as T

tilelang.cache.clear_cache()

M = 128
N = 128
K = 128


def vec_or(M, N, K, block_M, block_N, dtype="int32"):
    m_num = M // block_M
    n_num = N // block_N

    BLOCK_SIZE = 20

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            bx_ = cid // n_num
            bx = bx_ * block_M
            by_ = cid % n_num
            by = by_ * block_N

            A_VEC = T.alloc_ub((block_M, block_N), dtype)
            B_VEC = T.alloc_ub((block_M, block_N), dtype)
            C_VEC = T.alloc_ub((block_M, block_N), dtype)
            for i in T.serial(T.ceildiv(m_num * n_num, BLOCK_SIZE)):
                block_id_base = i * BLOCK_SIZE
                block_id = block_id_base + cid
                block_id_m = block_id // n_num
                block_id_n = block_id % n_num
                bx = block_id_m * block_M
                by = block_id_n * block_N
                T.copy(A[bx, by], A_VEC)
                T.copy(B[bx, by], B_VEC)
                T.npuir_or(A_VEC, B_VEC, C_VEC)
                T.copy(C_VEC, C[bx, by])

    return main


def test_vec_or():
    func = vec_or(M, N, K, 32, 64)
    kernel = tilelang.engine.lower(func)

    curr_name = os.path.splitext(os.path.basename(__file__))[0][5:] + ".mlir"
    output_file = "./output/" + curr_name
    with open(output_file, "w") as f:
        f.write(kernel)

    ref_file = "./mlir_files/" + curr_name
    are_identical = filecmp.cmp(output_file, ref_file, shallow=False)
    assert are_identical, f"'{output_file}' and '{ref_file}' are not identical"


if __name__ == "__main__":
    test_vec_or()
