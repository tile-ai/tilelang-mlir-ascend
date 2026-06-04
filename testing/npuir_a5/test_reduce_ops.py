# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import os

import tilelang
import tilelang.language as T

import torch
import torch_npu

tilelang.cache.clear_cache()

REDUCE_OPS = ["sum", "mean", "min", "max"]
SHAPE_CASES = [(2048, 4096), (64, 32768)]
DTYPES = ["float32"]


def vec_reduce_2d(M, N, op_name="sum", dtype="float32"):

    BLOCK_M = 8
    BLOCK_N = 128

    @T.prim_func
    def main_reduce(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, 1), dtype)
    ):

        with T.Kernel(1, is_npu=True) as (cid, _):

            # input tile
            a = T.alloc_shared((BLOCK_M, BLOCK_N), dtype)

            # tile reduce result
            s_local = T.alloc_shared((BLOCK_M, 1), dtype)

            # final accumulator
            s = T.alloc_shared((BLOCK_M, 1), dtype)

            # M blocking
            for mo in T.serial(0, (M + BLOCK_M - 1) // BLOCK_M):

                remain_m = M - mo * BLOCK_M

                # init accumulator
                if op_name == "sum" or op_name == "mean":

                    for i in T.Parallel(BLOCK_M):
                        if i < remain_m:
                            s[i, 0] = 0

                elif op_name == "min":

                    for i in T.Parallel(BLOCK_M):
                        if i < remain_m:
                            s[i, 0] = 1e38

                elif op_name == "max":

                    for i in T.Parallel(BLOCK_M):
                        if i < remain_m:
                            s[i, 0] = -1e38

                # N blocking
                for no in T.serial(0, (N + BLOCK_N - 1) // BLOCK_N):

                    remain_n = N - no * BLOCK_N

                    # load tile
                    for i, j in T.Parallel(BLOCK_M, BLOCK_N):

                        if i < remain_m and j < remain_n:

                            a[i, j] = A[
                                mo * BLOCK_M + i,
                                no * BLOCK_N + j
                            ]

                        else:

                            if op_name == "min":
                                a[i, j] = 1e38

                            elif op_name == "max":
                                a[i, j] = -1e38

                            else:
                                a[i, j] = 0

                    # reduce tile
                    T.reduce(
                        a,
                        s_local,
                        dims=1,
                        reduce_mode=op_name if op_name != "mean" else "sum",
                        clear=True
                    )

                    # accumulate
                    if op_name == "sum" or op_name == "mean":

                        for i in T.Parallel(BLOCK_M):

                            if i < remain_m:
                                s[i, 0] += s_local[i, 0]

                    elif op_name == "min":

                        for i in T.Parallel(BLOCK_M):

                            if i < remain_m:

                                if s_local[i, 0] < s[i, 0]:
                                    s[i, 0] = s_local[i, 0]

                    elif op_name == "max":

                        for i in T.Parallel(BLOCK_M):

                            if i < remain_m:

                                if s_local[i, 0] > s[i, 0]:
                                    s[i, 0] = s_local[i, 0]

                # mean finalize
                if op_name == "mean":

                    for i in T.Parallel(BLOCK_M):

                        if i < remain_m:
                            B[mo * BLOCK_M + i, 0] = (
                                s[i, 0] / float(N)
                            )

                else:

                    for i in T.Parallel(BLOCK_M):

                        if i < remain_m:
                            B[mo * BLOCK_M + i, 0] = s[i, 0]

    return main_reduce


def compute_reference(A, op_name):

    if op_name == "sum":
        return torch.sum(A, dim=1, keepdim=True)

    elif op_name == "mean":
        return torch.mean(A, dim=1, keepdim=True)

    elif op_name == "min":
        return torch.min(A, dim=1, keepdim=True).values

    elif op_name == "max":
        return torch.max(A, dim=1, keepdim=True).values


def test_vec_reduce():

    torch.npu.set_device(0)

    os.environ["TILELANG_ASCEND_MODE"] = "Developer"

    for M, N in SHAPE_CASES:

        for dtype in DTYPES:

            for op_name in REDUCE_OPS:

                print(
                    f"Testing shape=({M}, {N}), "
                    f"dtype={dtype}, op={op_name}"
                )

                func = vec_reduce_2d(
                    M,
                    N,
                    op_name,
                    dtype
                )

                compiled_kernel = tilelang.compile(
                    func,
                    target="npuir"
                )

                v1 = torch.randn(
                    size=[M, N],
                    dtype=eval("torch." + dtype)
                ).npu()

                v2 = torch.empty(
                    size=[M, 1],
                    dtype=eval("torch." + dtype)
                ).npu()

                v_ref = compute_reference(v1, op_name)

                compiled_kernel(v1, v2)

                torch.testing.assert_close(
                    v_ref,
                    v2,
                    rtol=1e-2,
                    atol=1e-2
                )

                print(
                    f"Shape ({M}, {N}) "
                    f"dtype={dtype} "
                    f"op={op_name} Pass!"
                )

    print("=" * 60)
    print("All reduce ops tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_vec_reduce()