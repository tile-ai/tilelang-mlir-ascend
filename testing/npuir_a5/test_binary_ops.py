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
    "int32": torch.int32,
    "bool": torch.bool,
}

DEFAULT_TOLERANCE = {
    "float16": (1e-2, 1e-2),
    "float32": (1e-2, 1e-2),
    "int32": (0, 0),
    "bool": (0.0, 0.0),
}


def gen_tensor(shape, dtype, kind="randn", device="npu"):
    torch_dtype = DTYPE_MAP[dtype]
    if kind == "zeros":
        out = torch.zeros(shape, dtype=torch_dtype)
    elif kind == "randn":
        out = torch.randn(shape, dtype=torch_dtype)
    elif kind == "rand":
        out = torch.rand(shape, dtype=torch_dtype)
    elif kind == "randint":
        if dtype == "int32":
            out = torch.randint(-100, 100, shape, dtype=torch_dtype)
        else:
            out = torch.randint(-100, 100, shape, dtype=torch_dtype).to(torch_dtype)
    else:
        raise ValueError(f"Unsupported kind: {kind}")
    return out.to(device=device)


def assert_close(actual, expected, dtype=None, rtol=None, atol=None):
    if dtype is None:
        if actual.dtype == torch.float32:
            dtype = "float32"
        elif actual.dtype == torch.float16:
            dtype = "float16"
        elif actual.dtype == torch.int32:
            dtype = "int32"
        else:
            dtype = "float32"
    default_rtol, default_atol = DEFAULT_TOLERANCE.get(dtype, (1e-2, 1e-2))
    torch.testing.assert_close(
        actual,
        expected,
        rtol=rtol if rtol is not None else default_rtol,
        atol=atol if atol is not None else default_atol,
        equal_nan=True,
    )


BINARY_OPS = ["add", "sub", "mul", "div", "pow", "max", "min"]
FLOORDIV_OP = ["floordiv"]
SHAPE_1D_CASES = [4096]
SHAPE_2D_CASES = [
    (1024, 1024),
    (1024, 4096),
    (1024, 10240),
    (1024, 16384),
    (1024, 20480),
]
DTYPE_CASES = ["float16", "float32"]


def make_binary_kernel_1d(N, op_name, dtype):
    block_N = min(N, 8192)
    grid_N = T.ceildiv(N, block_N)

    @T.prim_func
    def binary_kernel_1d(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(grid_N, is_npu=True) as (bx, _):
            A_local = T.alloc_shared((block_N,), dtype)
            B_local = T.alloc_shared((block_N,), dtype)
            C_local = T.alloc_shared((block_N,), dtype)

            T.copy(A[bx * block_N : bx * block_N + block_N], A_local)
            T.copy(B[bx * block_N : bx * block_N + block_N], B_local)

            if op_name == "add":
                T.vadd(A_local, B_local, C_local)
            elif op_name == "sub":
                T.vsub(A_local, B_local, C_local)
            elif op_name == "mul":
                T.vmul(A_local, B_local, C_local)
            elif op_name == "div":
                T.vdiv(A_local, B_local, C_local)
            elif op_name == "pow":
                T.vpow(A_local, B_local, C_local)
            elif op_name == "max":
                T.vmax(A_local, B_local, C_local)
            elif op_name == "min":
                T.vmin(A_local, B_local, C_local)
            elif op_name == "floordiv":
                T.vfloordiv(A_local, B_local, C_local)

            T.copy(C_local, C[bx * block_N : bx * block_N + block_N])

    return binary_kernel_1d


def make_binary_kernel_2d(M, N, op_name, dtype):
    block_M = min(M, 32)
    block_N = min(N, 256)
    grid_M = T.ceildiv(M, block_M)
    grid_N = T.ceildiv(N, block_N)

    @T.prim_func
    def binary_kernel_2d(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(grid_M * grid_N, is_npu=True) as (cid, _):
            bx = cid % grid_N
            by = cid // grid_N

            A_local = T.alloc_shared((block_M, block_N), dtype)
            B_local = T.alloc_shared((block_M, block_N), dtype)
            C_local = T.alloc_shared((block_M, block_N), dtype)

            T.copy(
                A[by * block_M : by * block_M + block_M, bx * block_N : bx * block_N + block_N],
                A_local,
            )
            T.copy(
                B[by * block_M : by * block_M + block_M, bx * block_N : bx * block_N + block_N],
                B_local,
            )

            if op_name == "add":
                T.vadd(A_local, B_local, C_local)
            elif op_name == "sub":
                T.vsub(A_local, B_local, C_local)
            elif op_name == "mul":
                T.vmul(A_local, B_local, C_local)
            elif op_name == "div":
                T.vdiv(A_local, B_local, C_local)
            elif op_name == "pow":
                T.vpow(A_local, B_local, C_local)
            elif op_name == "max":
                T.vmax(A_local, B_local, C_local)
            elif op_name == "min":
                T.vmin(A_local, B_local, C_local)
            elif op_name == "floordiv":
                T.vfloordiv(A_local, B_local, C_local)

            T.copy(
                C_local,
                C[by * block_M : by * block_M + block_M, bx * block_N : bx * block_N + block_N],
            )

    return binary_kernel_2d


def compute_reference(A, B, op_name):
    if op_name == "add":
        return A + B
    elif op_name == "sub":
        return A - B
    elif op_name == "mul":
        return A * B
    elif op_name == "div":
        return A / B
    elif op_name == "pow":
        return torch.pow(A, B)
    elif op_name == "max":
        return torch.maximum(A, B)
    elif op_name == "min":
        return torch.minimum(A, B)
    elif op_name == "floordiv":
        return torch.div(A, B, rounding_mode='floor')
    else:
        raise ValueError(f"Unsupported op: {op_name}")


def run_binary_test_1d(N, dtype, op_name):
    print(f"Testing binary op '{op_name}' with shape ({N},), dtype={dtype}")
    if op_name == "floordiv":
        A = gen_tensor((N,), dtype, kind="randint")
        B = gen_tensor((N,), dtype, kind="randint")
        B = torch.where(B == 0, torch.ones_like(B), B)
    else:
        A = gen_tensor((N,), dtype, kind="randn")
        B = gen_tensor((N,), dtype, kind="randn")
        if op_name == "div":
            B = torch.where(B == 0, torch.ones_like(B), B)
        elif op_name == "pow":
            B = torch.abs(B) + 1.0

    C = gen_tensor((N,), dtype, kind="zeros")

    kernel = make_binary_kernel_1d(N, op_name, dtype)
    compiled = tilelang.compile(kernel, target="npuir")
    compiled(A, B, C)

    ref = compute_reference(A.cpu(), B.cpu(), op_name)
    assert_close(C.cpu(), ref, dtype=dtype)
    print(f"  PASSED")


def run_binary_test_2d(M, N, dtype, op_name):
    print(f"Testing binary op '{op_name}' with shape ({M}, {N}), dtype={dtype}")
    if op_name == "floordiv":
        A = gen_tensor((M, N), dtype, kind="randint")
        B = gen_tensor((M, N), dtype, kind="randint")
        B = torch.where(B == 0, torch.ones_like(B), B)
    else:
        A = gen_tensor((M, N), dtype, kind="randn")
        B = gen_tensor((M, N), dtype, kind="randn")
        if op_name == "div":
            B = torch.where(B == 0, torch.ones_like(B), B)
        elif op_name == "pow":
            B = torch.abs(B) + 1.0

    C = gen_tensor((M, N), dtype, kind="zeros")

    kernel = make_binary_kernel_2d(M, N, op_name, dtype)
    compiled = tilelang.compile(kernel, target="npuir")
    compiled(A, B, C)

    ref = compute_reference(A.cpu(), B.cpu(), op_name)
    assert_close(C.cpu(), ref, dtype=dtype)
    print(f"  PASSED")


def main():
    torch.npu.set_device(0)

    for N in SHAPE_1D_CASES:
        for dtype in DTYPE_CASES:
            for op_name in BINARY_OPS:
                run_binary_test_1d(N, dtype, op_name)
        for op_name in FLOORDIV_OP:
            run_binary_test_1d(N, "int32", op_name)

    for M, N in SHAPE_2D_CASES:
        for dtype in DTYPE_CASES:
            for op_name in BINARY_OPS:
                run_binary_test_2d(M, N, dtype, op_name)
        for op_name in FLOORDIV_OP:
            run_binary_test_2d(M, N, "int32", op_name)

    print("=" * 60)
    print("All binary ops tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()