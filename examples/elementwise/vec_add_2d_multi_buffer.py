# Copyright (c) Huawei Technologies Co., Ltd. 2025.
"""
This is an example of how to manually implement multi-buffer and software pipelines
using the interfaces provided by the expert mode to achieve ultimate performance.
"""

import argparse
import torch

import tilelang
import tilelang.language as T

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--M", type=int, default=4096, help="")
parser.add_argument("--N", type=int, default=4096, help="")
parser.add_argument("--block_M", type=int, default=128, help="")
parser.add_argument("--block_N", type=int, default=128, help="")


@tilelang.jit(
    out_idx=[-1],
    target="npuir",
    pass_configs={
        # Disable the auto multi buffer to implement it manually.
        tilelang.PassConfigKey.NPUIR_ENABLE_AUTO_MULTI_BUFFER: False,
        # Disable the automatic sync to manually use the sync interface in expert mode.
        tilelang.PassConfigKey.NPUIR_DISABLE_HIVM_AUTO_INJECT_SYNC: True,
    },
)
def vec_add(M, N, block_M, block_N):
    m_num = M // block_M
    n_num = N // block_N
    dtype = "float16"
    num_ph_kernels = 24 * 2
    num_stages = 3
    multi_A = (
        3  # number of buffer slots for matrix A (also serves as pipeline depth for A)
    )
    multi_B = 2  # number of buffer slots for matrix B
    num_lg_kernels = m_num * n_num
    num_ph_kernels = min(num_ph_kernels, num_lg_kernels)

    @T.macro
    def init_flag():
        # Initialize inter‑pipeline flags so that the first `wait_flag` does not block.
        # The flag names follow the convention: "PIPE_XX" is set by the XX pipeline.
        # Here we pre‑set "PIPE_MTE2" flags from MTE3 and V pipelines to satisfy
        # early waits by MTE2 (the load pipeline) later.
        for i in T.serial(multi_A):
            with T.rs("PIPE_MTE3"):
                T.set_flag(
                    "PIPE_MTE2", i
                )  # MTE3 signals MTE2 that A‑buffer slot i is initially free
        for i in T.serial(multi_B):
            with T.rs("PIPE_V"):
                T.set_flag(
                    "PIPE_MTE2", i
                )  # V signals MTE2 that B‑buffer slot i is initially free

    @T.macro
    def clear_flag():
        # Drain all pending flags before kernel exit to avoid deadlock in subsequent launches.
        for i in T.serial(multi_A):
            with T.rs("PIPE_MTE2"):
                T.wait_flag(
                    "PIPE_MTE3", i
                )  # Wait until MTE3 has finished using buffer slot i
        for i in T.serial(multi_B):
            with T.rs("PIPE_MTE2"):
                T.wait_flag(
                    "PIPE_V", i
                )  # Wait until V has finished using B‑buffer slot i

    @T.macro
    def load(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        VEC,
        ph_id,
        task_id,
    ):
        lg_id = task_id * num_ph_kernels + ph_id
        bx = lg_id // n_num
        bx *= block_M
        by = lg_id % n_num
        by *= block_N

        with T.rs("PIPE_MTE2"):
            # Wait until the A‑buffer slot (index = task_id % multi_A) is no longer
            # needed by the store pipeline (MTE3). This guarantees the slot is free
            # for a new load.
            T.wait_flag("PIPE_MTE3", task_id % multi_A)
            T.copy(
                A[bx : bx + block_M, by : by + block_N], VEC[task_id % multi_A, :, :]
            )

            # Similarly, wait until the B‑buffer slot (index = task_id % multi_B)
            # has been consumed by the compute pipeline (V), so it can be reused.
            T.wait_flag("PIPE_V", task_id % multi_B)
            T.copy(
                B[bx : bx + block_M, by : by + block_N],
                VEC[multi_A + task_id % multi_B, :, :],
            )

            # After both A and B are loaded for this task, signal the compute
            # pipeline (V) that input data is ready. The flag index is modulo
            # `num_stages` to match the software pipeline stage.
            T.set_flag("PIPE_V", task_id % num_stages)

    @T.macro
    def compute(
        VEC,
        task_id,
    ):
        with T.rs("PIPE_V"):
            # Wait for the load pipeline (MTE2) to finish loading both A and B
            # for the current pipeline stage. The index matches the flag set in `load`.
            T.wait_flag("PIPE_MTE2", task_id % num_stages)

            T.vadd(
                VEC[task_id % multi_A, :, :],
                VEC[multi_A + task_id % multi_B, :, :],
                VEC[task_id % multi_A, :, :],
            )

            # Signal the load pipeline that the B‑buffer slot just used is now free
            # for the next load operation.
            T.set_flag("PIPE_MTE2", task_id % multi_B)
            # Signal the store pipeline (MTE3) that the result (stored in the A‑buffer
            # slot) is ready to be written back.
            T.set_flag("PIPE_MTE3", task_id % num_stages)

    @T.macro
    def store(
        C: T.Tensor((M, N), dtype),
        VEC,
        ph_id,
        task_id,
    ):
        lg_id = task_id * num_ph_kernels + ph_id
        bx = lg_id // n_num
        bx *= block_M
        by = lg_id % n_num
        by *= block_N

        with T.rs("PIPE_MTE3"):
            # Wait until the compute pipeline (V) has finished producing the result
            # in the A‑buffer slot corresponding to this pipeline stage.
            T.wait_flag("PIPE_V", task_id % num_stages)
            T.copy(
                VEC[task_id % multi_A, :, :], C[bx : bx + block_M, by : by + block_N]
            )

            # Signal the load pipeline that the A‑buffer slot is now free and can
            # be reused for a new load.
            T.set_flag("PIPE_MTE2", task_id % num_stages)

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(num_ph_kernels, is_npu=True) as (ph_id, _):
            # Unified buffer: first multi_A slices for A, next multi_B slices for B.
            VEC = T.alloc_ub((multi_A + multi_B, block_M, block_N), dtype)

            num_local_tasks = T.ceildiv(num_lg_kernels - ph_id, num_ph_kernels)

            init_flag()

            # Software pipeline with overlapping load, compute, and store.
            for task_id in T.serial(num_local_tasks + 2):
                if task_id < num_local_tasks:
                    load(A, B, VEC, ph_id, task_id)
                if task_id > 0 and task_id < num_local_tasks + 1:
                    compute(VEC, task_id - 1)
                if task_id > 1:
                    store(C, VEC, ph_id, task_id - 2)

            clear_flag()

    return main


def run_test(main_args):
    kernel = vec_add(
        main_args.M,
        main_args.N,
        main_args.block_M,
        main_args.block_N,
    )

    torch.manual_seed(88888888)  # set the random seed for torch
    shape = [main_args.M, main_args.N]
    dtype = "float16"

    a = torch.randn(size=shape, dtype=eval("torch." + dtype), device="npu")
    b = torch.randn(size=shape, dtype=eval("torch." + dtype), device="npu")

    ref_output = a + b
    c = kernel(a, b)
    torch.testing.assert_close(c, ref_output, rtol=1e-2, atol=1e-2)
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    torch.npu.set_device(0)
    args = parser.parse_args()
    run_test(args)
