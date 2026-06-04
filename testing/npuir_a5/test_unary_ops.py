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
}

DEFAULT_TOLERANCE = {
    "float16": (1e-2, 1e-2),
    "float32": (1e-2, 1e-2),
}


def gen_tensor(shape, dtype, kind="randn", device="npu"):
    torch_dtype = DTYPE_MAP[dtype]
    if kind == "zeros":
        out = torch.zeros(shape, dtype=torch_dtype)
    elif kind == "randn":
        out = torch.randn(shape, dtype=torch_dtype)
    elif kind == "rand":
        out = torch.rand(shape, dtype=torch_dtype)
    else:
        raise ValueError(f"Unsupported kind: {kind}")
    return out.to(device=device)


def assert_close(actual, expected, dtype=None, rtol=None, atol=None):
    if dtype is None:
        dtype = "float16" if actual.dtype == torch.float16 else "float32"
    default_rtol, default_atol = DEFAULT_TOLERANCE.get(dtype, (1e-2, 1e-2))
    torch.testing.assert_close(
        actual,
        expected,
        rtol=rtol if rtol is not None else default_rtol,
        atol=atol if atol is not None else default_atol,
        equal_nan=True,
    )


UNARY_OPS = ["exp", "log", "abs", "log1p", "expm1", "floor"]
SHAPE_CASES = [262144, 1048576, 4000000]
DTYPE_CASES = ["float16", "float32"]


def make_unary_kernel_1d(N, op_name, dtype):
    block_N = 2048
    grid_N = T.ceildiv(N, block_N)

    @T.prim_func
    def unary_kernel_1d(
        A: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
        shape: T.int32,
    ):
        with T.Kernel(grid_N, is_npu=True) as (bx, _):
            A_local = T.alloc_shared((block_N,), dtype)
            C_local = T.alloc_shared((block_N,), dtype)

            offset = bx * block_N
            tail_size = T.max(0, T.min(block_N, shape - offset))

            T.copy(A[offset : offset + tail_size], A_local[0 : tail_size])

            if op_name == "exp":
                T.vexp(A_local[0:block_N], C_local[0:block_N])
            elif op_name == "log":
                T.vln(A_local[0:block_N], C_local[0:block_N])
            elif op_name == "abs":
                T.vabs(A_local[0:block_N], C_local[0:block_N])
            elif op_name == "log1p":
                scalar_one = T.cast(1.0, dtype)
                T.vadd(A_local[0:block_N], scalar_one, C_local[0:block_N])
                T.vln(C_local[0:block_N], C_local[0:block_N])
            elif op_name == "expm1":
                scalar_one = T.cast(1.0, dtype)
                T.vexp(A_local[0:block_N], C_local[0:block_N])
                T.vsub(C_local[0:block_N], scalar_one, C_local[0:block_N])
            elif op_name == "floor":
                T.vfloor(A_local[0:block_N], C_local[0:block_N])

            T.copy(C_local[0 : tail_size], C[offset : offset + tail_size])

    return unary_kernel_1d


def compute_reference(A, op_name):
    if op_name == "exp":
        return torch.exp(A)
    elif op_name == "log":
        return torch.log(A)
    elif op_name == "abs":
        return torch.abs(A)
    elif op_name == "log1p":
        return torch.log1p(A)
    elif op_name == "expm1":
        return torch.expm1(A)
    elif op_name == "floor":
        return torch.floor(A)
    else:
        raise ValueError(f"Unsupported op: {op_name}")


def generate_input_for_op(N, dtype, op_name):
    if op_name == "log" or op_name == "log1p":
        A = gen_tensor((N,), dtype, kind="rand")
        A = A + 1.0
    elif op_name == "exp" or op_name == "expm1":
        A = gen_tensor((N,), dtype, kind="randn")
        A = A * 0.5
    else:
        A = gen_tensor((N,), dtype, kind="randn")
    return A


def run_unary_test(N, dtype, op_name):
    print(f"Testing unary op '{op_name}' with shape ({N},), dtype={dtype}")
    A = generate_input_for_op(N, dtype, op_name)
    C = gen_tensor((N,), dtype, kind="zeros")

    kernel = make_unary_kernel_1d(N, op_name, dtype)
    compiled = tilelang.compile(kernel, target="npuir")
    compiled(A, C, N)

    ref = compute_reference(A.cpu(), op_name)
    assert_close(C.cpu(), ref, dtype=dtype)
    print(f"  PASSED")


def main():
    torch.npu.set_device(0)

    for N in SHAPE_CASES:
        for dtype in DTYPE_CASES:
            for op_name in UNARY_OPS:
                run_unary_test(N, dtype, op_name)

    print("=" * 60)
    print("All unary ops tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()