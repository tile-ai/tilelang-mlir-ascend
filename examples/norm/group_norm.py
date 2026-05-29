import tilelang
import tilelang.language as T
import torch
import os
import torch.nn.functional as F

ALIGNMENT = 256


def _align_up(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


def _group_norm_kernel(M, D, eps, dtype):
    D_padded = _align_up(D, ALIGNMENT)
    pad_count = D_padded - D

    @tilelang.jit(out_idx=[3], target="npuir")
    def _func(block_m):

        @T.prim_func
        def main(
            x: T.Tensor[(M, D_padded), dtype],
            weight: T.Tensor[(D_padded,), dtype],
            bias: T.Tensor[(D_padded,), dtype],
            y: T.Tensor[(M, D_padded), dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), is_npu=True) as (pid_m, _):
                shared_buf = T.alloc_shared((block_m, D_padded), dtype)
                x_local = T.alloc_fragment((block_m, D_padded), dtype)
                x_f32 = T.alloc_fragment((block_m, D_padded), "float32")
                acc = T.alloc_fragment((block_m, 1), "float32")
                mean_val = T.alloc_fragment((block_m, 1), "float32")
                rstd = T.alloc_fragment((block_m, 1), "float32")

                # Load input row block via shared memory
                T.copy(x[pid_m * block_m, 0], shared_buf)
                T.copy(shared_buf, x_local)

                # Cast to fp32 once -- reused across all passes
                for i, j in T.Parallel(block_m, D_padded):
                    x_f32[i, j] = T.cast(x_local[i, j], "float32")

                # --- Mean reduction ---
                T.reduce_sum(x_f32, acc, dim=1)
                for i in T.Parallel(block_m):
                    mean_val[i, 0] = acc[i, 0] / float(D)

                # --- Centered variance reduction ---
                # Rewrite x_f32 in-place with (x - mean)^2.
                # Padded positions (x=0) contribute mean^2; corrected below.
                for i, j in T.Parallel(block_m, D_padded):
                    x_f32[i, j] = (x_f32[i, j] - mean_val[i, 0]) * (
                        x_f32[i, j] - mean_val[i, 0]
                    )

                T.reduce_sum(x_f32, acc, dim=1)
                for i in T.Parallel(block_m):
                    rstd[i, 0] = (
                        acc[i, 0] - float(pad_count) * mean_val[i, 0] * mean_val[i, 0]
                    ) / float(D) + eps
                T.vrsqrt(rstd, rstd)

                # --- Output: y = (x - mean) * rstd * weight + bias ---
                for i, j in T.Parallel(block_m, D_padded):
                    x_local[i, j] = (
                        T.cast(x_local[i, j], "float32") - mean_val[i, 0]
                    ) * rstd[i, 0] * T.cast(weight[j], "float32") + T.cast(
                        bias[j], "float32"
                    )

                # Write output via shared memory
                T.copy(x_local, shared_buf)
                T.copy(shared_buf, y[pid_m * block_m, 0])

        return main

    return _func


def _group_norm_kernel_high_perf(M, D, eps, dtype):
    D_padded = _align_up(D, ALIGNMENT)
    pad_count = D_padded - D

    @tilelang.jit(out_idx=[3], target="npuir")
    def _func_high_perf(block_m, block_n):

        @T.prim_func
        def high_perf(
            x: T.Tensor[(M, D_padded), dtype],
            weight: T.Tensor[(D_padded,), dtype],  # 实际来自 C 通道切片
            bias: T.Tensor[(D_padded,), dtype],
            y: T.Tensor[(M, D_padded), dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m)) as pid_m:
                x_tile = T.alloc_shared((block_m, block_n), "float32")
                w_tile = T.alloc_shared((block_n,), "float32")
                b_tile = T.alloc_shared((block_n,), "float32")
                y_tile = T.alloc_shared((block_m, block_n), "float32")

                acc = T.alloc_fragment((block_m, 1), "float32")
                mean_val = T.alloc_fragment((block_m, 1), "float32")
                var_val = T.alloc_fragment((block_m, 1), "float32")
                rstd = T.alloc_fragment((block_m, 1), "float32")

                # Pass 1: sum over D_padded by tiles
                T.clear(acc)
                for no in T.serial(T.ceildiv(D_padded, block_n)):
                    d_start = no * block_n
                    T.copy(x[pid_m * block_m, d_start], x_tile)
                    T.reduce_sum(x_tile, var_val, dim=1)
                    for i in T.Parallel(block_m):
                        acc[i, 0] += var_val[i, 0]
                for i in T.Parallel(block_m):
                    mean_val[i, 0] = acc[i, 0] / float(D)

                # Pass 2: squared diff sum
                T.clear(acc)
                for no in T.serial(T.ceildiv(D_padded, block_n)):
                    d_start = no * block_n
                    T.copy(x[pid_m * block_m, d_start], x_tile)
                    for i, j in T.Parallel(block_m, block_n):
                        y_tile[i, j] = (x_tile[i, j] - mean_val[i, 0]) * (
                            x_tile[i, j] - mean_val[i, 0]
                        )
                    T.reduce_sum(y_tile, var_val, dim=1)
                    for i in T.Parallel(block_m):
                        acc[i, 0] += var_val[i, 0]
                for i in T.Parallel(block_m):
                    var_val[i, 0] = (
                        acc[i, 0] - float(pad_count) * mean_val[i, 0] * mean_val[i, 0]
                    ) / float(D) + eps
                T.vrsqrt(var_val, rstd)

                # Pass 3: output
                for no in T.serial(T.ceildiv(D_padded, block_n)):
                    d_start = no * block_n
                    T.copy(x[pid_m * block_m, d_start], x_tile)
                    T.copy(weight[d_start], w_tile)
                    T.copy(bias[d_start], b_tile)
                    for i, j in T.Parallel(block_m, block_n):
                        y_tile[i, j] = (x_tile[i, j] - mean_val[i, 0]) * rstd[
                            i, 0
                        ] * w_tile[j] + b_tile[j]
                    T.copy(y_tile, y[pid_m * block_m, d_start])

        return high_perf

    return _func_high_perf


def group_norm_ref(x, weight, bias, g, eps):
    M, N = x.shape
    batch = M // g
    C = g * N
    x_reshape = x.float().reshape(batch, C)
    w_reshape = weight.float().repeat(g)
    b_reshape = bias.float().repeat(g)
    return (
        F.group_norm(
            x_reshape.float(),
            g,
            weight=w_reshape.float(),
            bias=b_reshape.float(),
            eps=eps,
        )
        .reshape(M, N)
        .to(x.dtype)
    )


def run_test(
    M=4096,
    D=4096,
    block_m=64,
    block_n=64,
    eps=1e-5,
    dtype="float16",
    device="npu",
    atol=1e-2,
    rtol=1e-2,
    g=16,
):
    d_padded = _align_up(D, ALIGNMENT)

    torch_dtype = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[dtype]

    x = torch.zeros((M, d_padded), dtype=torch_dtype, device=device)
    x[:, :D] = torch.randn((M, D), dtype=torch_dtype, device=device)
    weight = torch.randn((d_padded,), dtype=torch_dtype, device=device)
    bias = torch.randn((d_padded,), dtype=torch_dtype, device=device)

    y_ref = group_norm_ref(x, weight, bias, g, eps)
    program = _group_norm_kernel_high_perf(M, D, eps, dtype)
    y = program(block_m, block_n)(x, weight, bias)

    torch.testing.assert_close(y.float(), y_ref.float(), atol=atol, rtol=rtol)
    print("\033[32;1mPass!\033[0m")


if __name__ == "__main__":
    os.environ["TILELANG_ASCEND_MODE"] = "Dev"
    run_test()
