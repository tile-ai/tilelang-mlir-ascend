# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import os
import pytest
import torch
import tilelang
import tilelang.language as T

os.environ["TILELANG_ASCEND_MODE"] = "Developer"

FP16 = "float16"
FP32 = "float32"
INT32 = "int32"


@tilelang.jit(target="npuir")
def _indexer_fwd_kernel(
    heads,
    index_dim,
    block_N=64,
    num_stages=2,
    block_Q=16,
):
    """TileLang Kernel for indexer forward computation."""
    dtype = FP16
    accum_dtype = FP32
    index_dtype = INT32

    seq_len = T.symbolic("seq_len")
    seq_len_kv = T.symbolic("seq_len_kv")
    total_q = T.symbolic("total_q")
    num_q_blocks = T.symbolic("num_q_blocks")

    @T.prim_func
    def indexer_kernel(
        IndexQ: T.Tensor([total_q, index_dim], dtype),
        IndexK: T.Tensor([seq_len_kv, index_dim], dtype),
        Weights: T.Tensor([total_q, 1], accum_dtype),
        Logits: T.Tensor([seq_len, seq_len_kv], accum_dtype),
        CuSeqLenKS: T.Tensor([seq_len], index_dtype),
        CuSeqLenKE: T.Tensor([seq_len], index_dtype),
        Metadata: T.Tensor([num_q_blocks, 2], index_dtype),
    ):
        with T.Kernel(num_q_blocks, is_npu=True) as (cid, _):
            s_start = cid * block_Q

            q_shared = T.alloc_shared((block_Q * heads, index_dim), dtype)
            k_shared = T.alloc_shared((block_N, index_dim), dtype)
            w_shared = T.alloc_shared((block_Q * heads, 1), accum_dtype)

            acc_s = T.alloc_fragment((block_Q * heads, block_N), accum_dtype)
            acc_ub = T.alloc_shared((block_Q * heads, block_N), accum_dtype)
            out_ub = T.alloc_shared((block_Q, block_N), accum_dtype)

            start_block_k = Metadata[cid, 0]
            num_blocks_k = Metadata[cid, 1]

            # Load Query and Weights
            T.copy(
                IndexQ[s_start * heads, 0], q_shared, size=[block_Q * heads, index_dim]
            )
            T.copy(Weights[s_start * heads, 0], w_shared, size=[block_Q * heads, 1])

            for nbn_i in T.Pipelined(num_blocks_k, num_stages=num_stages):
                curr_k_offset = (start_block_k + nbn_i) * block_N

                T.copy(IndexK[curr_k_offset, 0], k_shared, size=[block_N, index_dim])
                T.gemm(q_shared, k_shared, acc_s, initC=True, b_transpose=True)

                T.copy(acc_s, acc_ub)
                T.vrelu(acc_ub, acc_ub)
                T.vmul(acc_ub, w_shared, acc_ub)

                for i in T.serial(block_Q):
                    for j in T.serial(block_N):
                        out_ub[i, j] = 0.0
                    T.reduce_sum(
                        acc_ub[i * heads : (i + 1) * heads, :],
                        out_ub[i : i + 1, :],
                        dim=0,
                        clear=True,
                    )

                for i in T.serial(block_Q):
                    s_idx = s_start + i
                    if s_idx < seq_len:
                        ks_val = CuSeqLenKS[s_idx]
                        ke_val = CuSeqLenKE[s_idx]
                        for j in T.serial(block_N):
                            k_idx = curr_k_offset + j
                            if k_idx < ks_val or k_idx >= ke_val:
                                out_ub[i, j] = -T.infinity(accum_dtype)

                if s_start < seq_len:
                    T.copy(
                        out_ub, Logits[s_start, curr_k_offset], size=[block_Q, block_N]
                    )

    return indexer_kernel


def prepare_metadata(ks, ke, seq_len, block_Q, block_N, device):
    """Compute the Key Block range for each row."""
    num_q_blocks = (seq_len + block_Q - 1) // block_Q
    metadata = torch.zeros((num_q_blocks, 2), dtype=torch.int32, device="cpu")

    ks_cpu = ks.cpu()
    ke_cpu = ke.cpu()

    for i in range(num_q_blocks):
        s_start_idx = i * block_Q
        s_end_idx = min(s_start_idx + block_Q, seq_len)

        curr_ks = ks_cpu[s_start_idx:s_end_idx]
        curr_ke = ke_cpu[s_start_idx:s_end_idx]

        min_k_s = curr_ks.min().item()
        max_k_e = curr_ke.max().item()

        start_k_blk = min_k_s // block_N
        end_k_blk = (max_k_e + block_N - 1) // block_N

        metadata[i, 0] = start_k_blk
        metadata[i, 1] = max(0, end_k_blk - start_k_blk)

    return metadata.to(device), num_q_blocks


def indexer_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    block_Q=16,
    block_N=16,
):
    seq_len, heads, dim = q.shape
    seq_kv = kv.shape[0]
    device = q.device

    assert seq_len % block_Q == 0, (
        f"seq_len ({seq_len}) must be a multiple of block_Q ({block_Q})"
    )
    assert seq_kv % block_N == 0, (
        f"seq_kv ({seq_kv}) must be a multiple of block_N ({block_N})"
    )

    metadata, num_q_blocks = prepare_metadata(ks, ke, seq_len, block_Q, block_N, device)

    kernel = _indexer_fwd_kernel(
        heads=heads, index_dim=dim, block_Q=block_Q, block_N=block_N
    )

    res_npu = torch.full(
        [seq_len, seq_kv], float("-inf"), device=device, dtype=torch.float32
    )
    q_flat = q.view(-1, dim)
    kernel(q_flat, kv, weights, res_npu, ks, ke, metadata)
    return res_npu


TEST_CASES = [
    (1, 4096, 4096, 128),
]


def indexer_torch_ref(q, kv, weights, ks, ke, heads, seq_len, seq_kv):
    """PyTorch reference implementation for indexer forward."""
    q_f = q.float().cpu()
    kv_f = kv.float().cpu()
    w_f = weights.view(seq_len, heads, 1).cpu()

    # [SEQ_LEN, HEADS, DIM] @ [DIM, SEQ_KV] -> [SEQ_LEN, HEADS, SEQ_KV]
    logits = torch.matmul(q_f, kv_f.t())
    res = (torch.relu(logits) * w_f).sum(dim=1)
    seq_idx = torch.arange(seq_kv, device="cpu").unsqueeze(0)
    valid_mask = (seq_idx >= ks.cpu().unsqueeze(1)) & (seq_idx < ke.cpu().unsqueeze(1))
    ref_res = torch.where(
        valid_mask, res, torch.tensor(float("-inf"), dtype=torch.float32)
    )
    return ref_res, valid_mask


class TestIndexFwd:
    def setup_method(self):
        torch.manual_seed(42)

    @pytest.mark.parametrize(
        ("heads", "seq_len", "seq_kv", "dim"),
        [
            pytest.param(*test, id=f"H{test[0]}-S{test[1]}-SKV{test[2]}-D{test[3]}")
            for test in TEST_CASES
        ],
    )
    def test_indexer_fwd_npu(self, heads, seq_len, seq_kv, dim):
        device = "npu"

        q = torch.randn(seq_len, heads, dim, device=device).half()
        kv = torch.randn(seq_kv, dim, device=device).half()
        weights = torch.randn(seq_len * heads, 1, device=device).float()

        # Generate sparse boundaries
        ks = torch.randint(100, 200, (seq_len,), dtype=torch.int32, device=device)
        ke = torch.randint(
            seq_kv - 200, seq_kv - 100, (seq_len,), dtype=torch.int32, device=device
        )

        res_npu = indexer_fwd(q, kv, weights, ks, ke)

        ref_res, valid_mask = indexer_torch_ref(
            q, kv, weights, ks, ke, heads, seq_len, seq_kv
        )

        res_cpu = res_npu.cpu()

        # Compare precision only in valid regions
        diff = torch.abs(res_cpu[valid_mask] - ref_res[valid_mask])
        max_error = diff.max().item() if diff.numel() > 0 else 0.0

        # Check if invalid regions are correctly filled with -inf
        mask_ok = torch.isneginf(res_cpu[~valid_mask]).all().item()

        print(f"Max Absolute Error: {max_error:.6f}")
        print(f"Masking Integrity Check: {'PASSED' if mask_ok else 'FAILED'}")

        torch.testing.assert_close(
            res_cpu[valid_mask], ref_res[valid_mask], rtol=1e-3, atol=1e-3
        )
        assert mask_ok, "Masking error detected: invalid regions not filled with -inf"
