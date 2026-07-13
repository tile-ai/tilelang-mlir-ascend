# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import os
import filecmp

import tilelang
import tilelang.language as T

tilelang.cache.clear_cache()

M = 512
N = 512
CLAMP_MIN = -1.0
CLAMP_MAX = 1.0


def vec_clamp(M, N, block_M, block_N, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N
    BLOCK_SIZE = 16

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            A_VEC = T.alloc_ub((block_M, block_N), dtype)
            B_VEC = T.alloc_ub((block_M, block_N), dtype)
            for i in T.serial(T.ceildiv(m_num * n_num, BLOCK_SIZE)):
                block_id = i * BLOCK_SIZE + cid
                if block_id < m_num * n_num:
                    block_id_m = block_id // n_num
                    block_id_n = block_id % n_num
                    bx = block_id_m * block_M
                    by = block_id_n * block_N
                    T.copy(A[bx, by], A_VEC)
                    T.vclamp(A_VEC, B_VEC, CLAMP_MIN, CLAMP_MAX)
                    T.copy(B_VEC, B[bx, by])

    return main


def test_vec_clamp():
    func = vec_clamp(M, N, 128, 256)
    kernel = tilelang.engine.lower(func, target="npuir")

    curr_name = os.path.splitext(os.path.basename(__file__))[0][5:] + ".mlir"
    output_file = "./output/" + curr_name
    os.makedirs("./output", exist_ok=True)
    with open(output_file, "w") as f:
        f.write(kernel)

    ref_file = "./mlir_files/" + curr_name
    are_identical = filecmp.cmp(output_file, ref_file, shallow=False)
    assert are_identical, f"'{output_file}' and '{ref_file}' are not identical"


if __name__ == "__main__":
    test_vec_clamp()
