import math
import torch
import tilelang as tl
import tilelang.language as T
import os

from tilelang.carver.anneal.policy import AnnealTemplate

tl.cache.clear_cache()


# ============================================================
# Phase 1: NPU Tile (not autotuned)
# ============================================================
@tl.jit(target="npuir")
def tile_scheduler_kernel_npu(num_experts: int, max_tiles: int, block_m: int):
    @T.prim_func
    def _sched_main(
        true_sizes: T.Tensor([num_experts], "int32"),
        tile_expert_ids: T.Tensor([max_tiles], "int32"),
        tile_row_offsets: T.Tensor([max_tiles], "int32"),
        total_tiles: T.Tensor([1], "int32"),
    ):
        with T.Kernel(max_tiles, is_npu=True) as (tile_id, _):
            s_cum = T.alloc_fragment([num_experts + 1], "int32")
            s_cum[0] = 0
            for e in T.serial(num_experts):
                s_cum[e + 1] = s_cum[e] + (true_sizes[e] + (block_m - 1)) // block_m
            if tile_id == 0:
                total_tiles[0] = s_cum[num_experts]

            if tile_id < s_cum[num_experts]:
                lo = 0
                hi = num_experts - 1
                log2_up = max(1, math.ceil(math.log2(num_experts + 1)))
                for _ in T.serial(log2_up):
                    mid = (lo + hi) >> 1
                    if s_cum[mid + 1] <= tile_id:
                        lo = mid + 1
                    else:
                        hi = mid
                expert = lo
                offset = (tile_id - s_cum[expert]) * block_m
                tile_expert_ids[tile_id] = expert
                tile_row_offsets[tile_id] = offset
            else:
                tile_expert_ids[tile_id] = -1
                tile_row_offsets[tile_id] = 0

    return _sched_main


# ============================================================
# Phase 2: NPU grouped gemm
# ============================================================

torch.manual_seed(42)
num_experts = 16
numel = 256
N = 256
K = 128
dtype = torch.float16

# def get_config():
#     return [
#         {"block_m":4,"block_n":32,"block_k":32},
#         {"block_m":4,"block_n":32,"block_k":64},
#         {"block_m":8,"block_n":32,"block_k":32},
#         {"block_m":8,"block_n":32,"block_k":64},
#         {"block_m":16,"block_n":32,"block_k":32},
#         {"block_m":16,"block_n":32,"block_k":64},
#     ]


def get_config():
    anneal_template = AnnealTemplate(shape=[numel, N, K], use_template="Matmul")

    hints = anneal_template.get_configs()

    configs = []
    for hint in hints:
        print(hint.kwargs)
        print(hint.value)
        configs.append(
            {
                "block_m": hint.kwargs[0],
                "block_n": hint.kwargs[1],
                "block_k": hint.kwargs[2],
            }
        )
    return configs


def cpu_ref(
    A, B, tile_expert_ids, tile_row_offsets, true_offsets, true_sizes, total_tiles_val
):
    C_ref = torch.zeros((numel, N), dtype=torch.float32)
    start = 0
    for e in range(num_experts):
        size = true_sizes[e].item()
        if size == 0:
            continue
        A_e = A[start : start + size]
        B_e = B[e]
        C_e = A_e @ B_e.T
        C_ref[start : start + size] = C_e
        start += size
    return C_ref.to(dtype).npu()


def supply_prog(params, config):
    torch.manual_seed(0)

    max_tiles = numel // config["block_m"] + num_experts

    routing_idx = torch.randint(0, num_experts, (numel,), device="npu")
    true_sizes = (
        torch.bincount(routing_idx, minlength=num_experts).to(torch.int32).npu()
    )
    true_offsets = torch.cumsum(
        torch.cat([torch.tensor([0], device="npu"), true_sizes[:-1]]), dim=0
    ).to(torch.int32)

    true_sizes_cpu = true_sizes.cpu().numpy()
    tiles_per_expert = [
        (s + config["block_m"] - 1) // config["block_m"] for s in true_sizes_cpu
    ]
    total_tiles_val = sum(tiles_per_expert)

    tile_expert_ids_cpu = []
    tile_row_offsets_cpu = []
    for e, num_tiles in enumerate(tiles_per_expert):
        for tile_idx in range(num_tiles):
            tile_expert_ids_cpu.append(e)
            tile_row_offsets_cpu.append(tile_idx * config["block_m"])
    # 填充到 max_tiles
    tile_expert_ids_cpu += [-1] * (max_tiles - total_tiles_val)
    tile_row_offsets_cpu += [0] * (max_tiles - total_tiles_val)

    tile_expert_ids = torch.tensor(tile_expert_ids_cpu, dtype=torch.int32, device="npu")
    tile_row_offsets = torch.tensor(
        tile_row_offsets_cpu, dtype=torch.int32, device="npu"
    )
    A = torch.randn((numel, K), dtype=dtype, device="npu")
    B = torch.randn((num_experts, N, K), dtype=dtype, device="npu")

    return [
        A,
        B,
        tile_expert_ids,
        tile_row_offsets,
        true_offsets,
        true_sizes,
        total_tiles_val,
    ]


@tl.autotune(
    configs=get_config(),
    ref_prog=cpu_ref,
    supply_prog=supply_prog,
    atol=1e-2,
    rtol=1e-2,
)
@tl.jit(out_idx=[2], target="npuir")
def moe_gemm_kernel_npu(
    numel: int,
    num_experts: int,
    N: int,
    K: int,
    dtype: str,
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    group_size_m: int,
):
    accum_dtype = "float32"
    _k_aligned = K % block_k == 0
    _n_aligned = N % block_n == 0
    _b_copy_ok = _k_aligned and _n_aligned
    _num_pid_n = T.ceildiv(N, block_n)
    max_tiles = numel // block_m + num_experts
    _total_ctas = max_tiles * _num_pid_n
    _num_pid_in_group = group_size_m * _num_pid_n

    @T.prim_func
    def _gemm_main(
        A: T.Tensor([numel, K], dtype),
        B: T.Tensor([num_experts, N, K], dtype),
        C: T.Tensor([numel, N], dtype),
        tile_expert_ids: T.Tensor([max_tiles], "int32"),
        tile_row_offsets: T.Tensor([max_tiles], "int32"),
        true_offsets: T.Tensor([num_experts], "int32"),
        true_sizes: T.Tensor([num_experts], "int32"),
        total_tiles: T.int32,
    ):
        with T.Kernel(_total_ctas, is_npu=True) as (pid, _):
            A_shared = T.alloc_shared([block_m, block_k], dtype)
            B_shared = T.alloc_shared([block_n, block_k], dtype)
            C_local = T.alloc_fragment([block_m, block_n], accum_dtype)

            bx = pid // _num_pid_n
            by = pid % _num_pid_n
            # pid -> (bx, by)
            if group_size_m == 1:
                bx = pid // _num_pid_n
                by = pid % _num_pid_n
            else:
                pid_in_group = pid % _num_pid_in_group
                group_id = pid // _num_pid_in_group
                first_pid_m = group_id * group_size_m
                actual_gsm = T.min(max_tiles - first_pid_m, group_size_m)
                bx = first_pid_m + pid_in_group % actual_gsm
                by = pid_in_group // actual_gsm

            if bx < total_tiles:
                expert_id = tile_expert_ids[bx]
                row_in_expert = tile_row_offsets[bx]
                m_start = true_offsets[expert_id] + row_in_expert
                n_start = by * block_n
                actual_rows = T.min(block_m, true_sizes[expert_id] - row_in_expert)
                actual_cols = T.min(block_n, N - n_start)

                # T.clear(C_local)

                for k in T.Pipelined(T.ceildiv(K, block_k), num_stages=num_stages):
                    k_offset = k * block_k

                    actual_k = T.min(block_k, K - k_offset)

                    T.copy(
                        A[m_start : m_start + block_m, k_offset : k_offset + actual_k],
                        A_shared,
                    )

                    T.copy(
                        B[
                            expert_id,
                            n_start : n_start + actual_cols,
                            k_offset : k_offset + actual_k,
                        ],
                        B_shared,
                    )

                    T.gemm(
                        A_shared, B_shared, C_local, b_transpose=True, initC=(k == 0)
                    )

                T.copy(
                    C_local[:actual_rows, :actual_cols],
                    C[m_start : m_start + actual_rows, n_start : n_start + actual_cols],
                )

    return _gemm_main


# ============================================================
# 测试（不变）
# ============================================================
def test_moe_grouped_gemm_npu():
    if not torch.npu.is_available():
        print("NPU not available, skip test.")
        return

    kernel = moe_gemm_kernel_npu(
        numel=numel,
        num_experts=num_experts,
        N=N,
        K=K,
        dtype="float16",
        num_stages=2,
        group_size_m=1,
    )

    print("Best Config:", kernel.get_tuner_result())
    print("Test Passed!")


if __name__ == "__main__":
    os.environ["TILELANG_ASCEND_MODE"] = "Developer"
    test_moe_grouped_gemm_npu()
