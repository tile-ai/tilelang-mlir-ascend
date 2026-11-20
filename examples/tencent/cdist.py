import argparse
import os

import torch
import tilelang
import tilelang.language as T


def cdist_squared_pytorch(x1, x2):
    diff = x1.to(torch.float32).unsqueeze(1) - x2.to(torch.float32).unsqueeze(0)
    return torch.sum(diff * diff, dim=-1).squeeze(0)


def ref_program(A, B):
    return cdist_squared_pytorch(A, B)


def supply_prog(args):
    dtype = torch.bfloat16
    # NPU
    query_hidden_states = torch.ones(1, 256, dtype=dtype, device="npu")
    c_embs = torch.ones(2000000, 256, dtype=dtype, device="npu")
    return [query_hidden_states, c_embs]


def manual_check_prog(actual, expected):
    actual_tensor = actual[0] if isinstance(actual, (list, tuple)) else actual
    expected_tensor = expected[0] if isinstance(expected, (list, tuple)) else expected
    return torch.allclose(actual_tensor, expected_tensor, rtol=1e-2, atol=1e-2)


def get_configs():
    return [
        {"block_M": block_M, "block_N": block_N}
        for block_M in [16, 32, 64]
        for block_N in [128, 256]
    ]


# NPU
@tilelang.jit(out_idx=[-1], target="npuir")
def distance(
    M,
    N,
    block_N=256,
    block_M=32,
    dtype="bfloat16",
    accum_dtype="float32",
    num_kernels=56,
):
    """Developer 模式 NPU kernel：计算 query 与 codebook 每一行的平方欧氏距离。

    输入:
      A: [1, N]，单条 query 向量
      B: [M, N]，M 条待比较向量
    输出:
      C: [M]，C[m] = sum_j((A[0, j] - B[m, j]) ** 2)
    """

    num_logic_kernels = T.ceildiv(M, block_M)

    @T.prim_func
    def dist(
        A: T.Tensor((1, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M,), accum_dtype),
    ):
        with T.Kernel(num_kernels, is_npu=True) as (kernel_id, _):
            A_shared = T.alloc_shared((1, block_N), dtype)
            B_shared = T.alloc_shared((block_M, block_N), dtype)
            diff = T.alloc_shared((block_M, block_N), dtype)
            diff_sq = T.alloc_shared((block_M, block_N), dtype)
            partial_sum = T.alloc_shared((block_M, 1), dtype)
            total_sum = T.alloc_shared((block_M, 1), dtype)
            total_sum_f32 = T.alloc_shared((block_M, 1), accum_dtype)

            for task_id in T.serial(T.ceildiv(num_logic_kernels, num_kernels)):
                logic_kernel_id = task_id * num_kernels
                logic_kernel_id = logic_kernel_id + kernel_id
                if logic_kernel_id < num_logic_kernels:
                    row_offset = logic_kernel_id * block_M
                    real_m = T.min(block_M, M - row_offset)

                    T.clear(total_sum)

                    for n_tile in T.serial(T.ceildiv(N, block_N)):
                        col_offset = n_tile * block_N
                        real_n = T.min(block_N, N - col_offset)

                        T.clear(A_shared)
                        T.clear(B_shared)

                        T.copy(
                            A[0:1, col_offset : col_offset + real_n],
                            A_shared[0:1, 0:real_n],
                        )
                        T.copy(
                            B[
                                row_offset : row_offset + real_m,
                                col_offset : col_offset + real_n,
                            ],
                            B_shared[0:real_m, 0:real_n],
                        )


                        T.vsub(A_shared, B_shared, diff)
                        T.vmul(diff, diff, diff_sq)

                        T.reduce_sum(diff_sq, partial_sum, dim=1)
                        T.vadd(total_sum, partial_sum, total_sum)

                    T.vcast(total_sum, total_sum_f32, round_mode="rint")
                    T.copy(
                        total_sum_f32[0:real_m, 0],
                        C[row_offset : row_offset + real_m],
                    )

    return dist


def run_test(M=1024, N=256, block_M=32, block_N=256, dtype="bfloat16"):
    os.environ["TILELANG_ASCEND_MODE"] = "Developer"
    torch.npu.set_device(0)
    tilelang.cache.clear_cache()

    torch_dtype = getattr(torch, dtype)
    query = torch.randn((1, N), dtype=torch_dtype, device="npu")
    codebook = torch.randn((M, N), dtype=torch_dtype, device="npu")

    kernel = distance(
        M,
        N,
        block_N=block_N,
        block_M=block_M,
        dtype=dtype,
        accum_dtype="float32",
    )
    output = kernel(query, codebook)
    expected = ref_program(query, codebook)

    print("NPU output:")
    print(output)
    print("Reference:")
    print(expected)
    print(output.shape)
    torch.testing.assert_close(output.cpu(), expected.cpu(), rtol=1e-2, atol=1e-2)
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CDist squared L2 NPU Developer kernel")
    parser.add_argument("--M", type=int, default=1986514)
    parser.add_argument("--N", type=int, default=256)
    parser.add_argument("--block_M", type=int, default=128)
    parser.add_argument("--block_N", type=int, default=256)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "float32", "bfloat16"],
    )
    args = parser.parse_args()
    run_test(args.M, args.N, args.block_M, args.block_N, args.dtype)
