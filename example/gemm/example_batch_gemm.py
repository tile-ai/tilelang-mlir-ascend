# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""BF16 Batch GEMM: A[B,M,K] @ B[B,N,K].T -> FP32."""

import tilelang
import tilelang.ascend.language as T


def batch_gemm(batch, M, N, K):
    batch_m, batch_n, batch_k = 256, 256, 256
    l0_k = 64
    cores = 28
    stages = 2
    prefetch = True
    window = 4
    split_m = 1
    dtype, out_dtype = "bfloat16", "float32"
    uf_partial, uf_final = 2, 3

    tile_m = batch_m // split_m
    m_tiles, n_tiles = M // batch_m, N // batch_n
    k_tiles, l0_tiles = K // batch_k, batch_k // l0_k
    tiles_per_batch = m_tiles * n_tiles
    tiles = batch * tiles_per_batch

    @T.prim_func
    def batch_gemm_bf16(
        A: T.Tensor((batch, M, K), dtype),
        B: T.Tensor((batch, N, K), dtype),
        C: T.Tensor((batch, M, N), out_dtype),
    ):
        with T.Kernel(cores) as cid:
            lhs = T.alloc_l1((batch_m, batch_k), dtype)
            rhs = T.alloc_l1((batch_n, batch_k), dtype)
            lhs_l0 = T.alloc_l0a((tile_m, l0_k), dtype)
            rhs_l0 = T.alloc_l0b((batch_n, l0_k), dtype)
            acc = T.alloc_l0c((tile_m, batch_n), out_dtype)
            T.annotate_buffer_versions({lhs: stages, rhs: stages, lhs_l0: stages, rhs_l0: stages})

            for tile_id in T.Persistent([tiles], cores, cid, num_stages=stages):
                batch_id = tile_id // tiles_per_batch
                tile = tile_id % tiles_per_batch
                group = tile // (n_tiles * window)
                in_window = tile % (n_tiles * window)
                row = (group * window + in_window % window) * batch_m
                col_tile = in_window // window
                col = T.Select(group % 2 == 0, col_tile, n_tiles - 1 - col_tile) * batch_n

                for kt in T.Pipelined(k_tiles, num_stages=stages, annotations={"enable_offset": prefetch}):
                    with T.Stage(0):
                        T.copy(A[batch_id, row, kt * batch_k], lhs)
                        T.copy(B[batch_id, col, kt * batch_k], rhs)
                    with T.Stage(stages - 1):
                        for sub in T.Pipelined(l0_tiles, num_stages=stages):
                            T.copy(rhs[:, sub * l0_k : (sub + 1) * l0_k], rhs_l0)
                            T.copy(lhs[:, sub * l0_k : (sub + 1) * l0_k], lhs_l0)
                            T.gemm(
                                lhs_l0,
                                rhs_l0,
                                acc,
                                transpose_B=True,
                                clear_accum=(kt == 0) & (sub == 0),
                                unit_flag_ctrl=T.Select((kt == k_tiles - 1) & (sub == l0_tiles - 1), uf_final, uf_partial),
                            )

                T.copy(acc, C[batch_id, row, col], unit_flag_ctrl=uf_final)

    return batch_gemm_bf16


def main(compile_only=False):
    import torch

    batch = 28
    M, N, K = 2048, 2048, 2048
    dtype = torch.bfloat16
    kernel = tilelang.compile(batch_gemm(batch, M, N, K), out_idx=-1, target="ascend", execution_backend="tvm_ffi")
    if compile_only:
        print("COMPILE_PASS")
        return
    import torch_npu  # noqa: F401

    rng = torch.Generator().manual_seed(0)
    a = torch.randint(-4, 5, (batch, M, K), dtype=torch.int8, generator=rng).to(dtype) / 8
    b = torch.randint(-4, 5, (batch, N, K), dtype=torch.int8, generator=rng).to(dtype) / 8
    a_npu, b_npu = a.npu(), b.npu()
    actual = kernel(a_npu, b_npu).cpu()
    for i in range(batch):
        expected = a[i].float() @ b[i].float().T
        torch.testing.assert_close(actual[i], expected, rtol=0, atol=0)
    print(f"CORRECTNESS_PASS B={batch}, M={M}, N={N}, K={K}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-only", action="store_true")

    args = parser.parse_args()
    main(compile_only=args.compile_only)
