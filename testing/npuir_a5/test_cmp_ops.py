# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import os
import torch
import torch_npu

import tilelang
import tilelang.language as T

os.environ["TILELANG_ASCEND_MODE"] = "Developer"
tilelang.cache.clear_cache()

DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bool": torch.bool,
}


def gen_tensor(shape, dtype, kind="randn", device="npu"):
    torch_dtype = DTYPE_MAP[dtype]
    if kind == "zeros":
        out = torch.zeros(shape, dtype=torch_dtype)
    elif kind == "randn":
        out = torch.randn(shape, dtype=torch_dtype)
    else:
        raise ValueError(f"Unsupported kind: {kind}")
    return out.to(device=device)


def assert_close(actual, expected, dtype=None, rtol=None, atol=None):
    torch.testing.assert_close(
        actual,
        expected,
        rtol=rtol if rtol is not None else 1e-2,
        atol=atol if atol is not None else 1e-2,
        equal_nan=True,
    )


CMP_OPS = ["eq", "ne", "gt", "lt", "ge", "le"]
SHAPE_CASES = [
    (1024, 4096),
    (1024, 10240),
    (1024, 20480),
]
DTYPE_CASES = ["float16", "float32"]


def make_cmp_kernel_2d(M, N, cmp_mode, dtype):
    block_M = 32
    block_N = 256
    grid_M = M // block_M
    grid_N = N // block_N

    @T.prim_func
    def cmp_kernel_2d(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), "int8"),
    ):
        with T.Kernel(grid_M * grid_N, is_npu=True) as (cid, _):
            bx = cid % grid_N
            by = cid // grid_N

            A_local = T.alloc_shared((block_M, block_N), dtype)
            B_local = T.alloc_shared((block_M, block_N), dtype)
            
            C_local = T.alloc_shared((block_M, block_N), "int8")

            offset_M = by * block_M
            offset_N = bx * block_N

            T.copy(A[offset_M : offset_M + block_M, offset_N : offset_N + block_N], A_local)
            T.copy(B[offset_M : offset_M + block_M, offset_N : offset_N + block_N], B_local)

            T.npuir_cmp(A_local, B_local, C_local, cmp_mode)

            T.copy(C_local, C[offset_M : offset_M + block_M, offset_N : offset_N + block_N])

    return cmp_kernel_2d


def compute_reference(A, B, cmp_mode):
    if cmp_mode == "eq":
        return A == B
    elif cmp_mode == "ne":
        return A != B
    elif cmp_mode == "gt":
        return A > B
    elif cmp_mode == "lt":
        return A < B
    elif cmp_mode == "ge":
        return A >= B
    elif cmp_mode == "le":
        return A <= B
    else:
        raise ValueError(f"Unsupported cmp_mode: {cmp_mode}")


def run_cmp_test(M, N, dtype, cmp_mode):
    print(f"Testing cmp op '{cmp_mode}' with shape ({M}, {N}), dtype={dtype}")
    A = gen_tensor((M, N), dtype, kind="randn")
    B = gen_tensor((M, N), dtype, kind="randn")
    C = gen_tensor((M, N), "bool", kind="zeros")

    kernel = make_cmp_kernel_2d(M, N, cmp_mode, dtype)
    compiled = tilelang.compile(kernel, target="npuir")
    compiled(A, B, C)

    ref = compute_reference(A.cpu(), B.cpu(), cmp_mode)
    assert_close(C.cpu(), ref, dtype="bool")
    print(f"  PASSED")


def main():
    torch.npu.set_device(0)

    for M, N in SHAPE_CASES:
        for dtype in DTYPE_CASES:
            for cmp_mode in CMP_OPS:
                run_cmp_test(M, N, dtype, cmp_mode)

    print("=" * 60)
    print("All comparison ops tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()