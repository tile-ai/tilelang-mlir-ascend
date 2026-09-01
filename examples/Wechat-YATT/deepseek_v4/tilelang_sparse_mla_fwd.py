# Copyright (c) Huawei Technologies Co., Ltd. 2026.
import os
import torch
import tilelang
import tilelang.language as T

from typing import Optional

FP16 = "float16"
BF16 = "bfloat16"
FP32 = "float32"
INT32 = "int32"
LOG2E = 1.4426950408889634


@tilelang.jit(out_idx=[2, 3], target="npuir")
def sparse_mqa_fwd(
    block_top_k_vec,
    block_top_k_cube,
    block_heads,
    num_heads,
    dim,
    max_top_k=640,
    scale=None,
    dtype=BF16,
    accum_dtype=FP32,
    indices_dtype=INT32,
):
    scale = (1.0 / dim) ** 0.5 if scale is None else scale
    batch_size = T.symbolic("batchSize")
    seq_len = T.symbolic("seqLen")
    seq_len_kv = T.symbolic("seqLenKV")
    top_k = T.symbolic("topK")
    top_k_reserved = T.symbolic("topKReserved")

    @T.prim_func
    def sparseAttn(
        Q: T.Tensor((batch_size, seq_len, num_heads, dim), dtype),
        KV: T.Tensor((batch_size, seq_len_kv, dim), dtype),
        Output: T.Tensor((batch_size, seq_len, num_heads, dim), dtype),
        LSE: T.Tensor((batch_size, seq_len, num_heads), accum_dtype),
        AttnSink: T.Tensor((num_heads, 1), accum_dtype),
        TopKIndices: T.Tensor((batch_size, seq_len, top_k), indices_dtype),
        SparseKVBuffer: T.Tensor((batch_size, seq_len, top_k_reserved, dim), dtype),
        ValidMaskBuffer: T.Tensor((batch_size, seq_len, top_k_reserved), accum_dtype),
        WorkspaceScore: T.Tensor(
            (batch_size, seq_len, block_heads, top_k_reserved), accum_dtype
        ),
    ):
        total_queries = batch_size * seq_len
        with T.Kernel((batch_size * seq_len + 3) // 4, is_npu=True) as (cid, _):
            value_zero = 0

            q_shared = T.alloc_shared((block_heads, dim), dtype)
            kv_gather = T.alloc_shared((block_top_k_vec, dim), dtype)
            kv_shared = T.alloc_shared((block_top_k_cube, dim), dtype)
            o_shared = T.alloc_shared((block_heads, dim), dtype)
            acc_s_cast = T.alloc_shared((block_heads, max_top_k), dtype)
            attn_sink_shared = T.alloc_shared((block_heads, 1), accum_dtype)

            idxs = T.alloc_fragment((block_top_k_vec,), indices_dtype)
            acc_s_block = T.alloc_fragment((block_heads, block_top_k_cube), accum_dtype)
            acc_s = T.alloc_fragment((block_heads, max_top_k), accum_dtype)
            acc_o = T.alloc_fragment((block_heads, dim), accum_dtype)
            scores_max = T.alloc_fragment((block_heads, 1), accum_dtype)
            scores_sum = T.alloc_fragment((block_heads, 1), accum_dtype)
            valid_mask = T.alloc_fragment((1, max_top_k), accum_dtype)
            valid_mask_block = T.alloc_fragment((block_top_k_vec,), accum_dtype)

            for work_id in T.Pipelined(
                4 * T.ceildiv(num_heads, block_heads), num_stages=2
            ):
                query_offset = work_id // T.ceildiv(num_heads, block_heads)
                head_block = work_id % T.ceildiv(num_heads, block_heads)
                query_id = T.min(cid * 4 + query_offset, total_queries - 1)
                by = query_id // seq_len
                bx = query_id % seq_len
                T.vbrc(value_zero, acc_o)
                T.copy(Q[by, bx, head_block * block_heads, 0], q_shared)
                if head_block == 0:
                    for k in T.Pipelined(
                        T.ceildiv(top_k_reserved, block_top_k_vec), num_stages=2
                    ):
                        real_block_top_k = T.min(
                            top_k - k * block_top_k_vec, block_top_k_vec
                        )
                        real_block_top_k = T.max(real_block_top_k, 0)
                        T.copy(TopKIndices[by, bx, k * block_top_k_vec], idxs)
                        T.vbrc(value_zero, valid_mask_block)
                        T.vbrc(value_zero, kv_gather)
                        for i in T.serial(real_block_top_k):
                            cur_idx = idxs[i]
                            if cur_idx != -1:
                                valid_mask_block[i] = 1.0
                                T.copy(
                                    KV[by, cur_idx, 0], kv_gather[i, 0], size=[1, dim]
                                )
                        T.copy(
                            valid_mask_block,
                            ValidMaskBuffer[by, bx, k * block_top_k_vec],
                        )
                        T.copy(
                            kv_gather, SparseKVBuffer[by, bx, k * block_top_k_vec, 0]
                        )

                for k in T.Pipelined(T.ceildiv(top_k, block_top_k_cube), num_stages=2):
                    T.copy(SparseKVBuffer[by, bx, k * block_top_k_cube, 0], kv_shared)
                    T.gemm(
                        q_shared, kv_shared, acc_s_block, initC=True, b_transpose=True
                    )
                    T.copy(acc_s_block, WorkspaceScore[by, bx, 0, k * block_top_k_cube])

                T.copy(
                    WorkspaceScore[by, bx, 0, 0],
                    acc_s,
                    size=[block_heads, top_k_reserved],
                )
                T.vbrc(value_zero, valid_mask)
                T.copy(
                    ValidMaskBuffer[by, bx, 0],
                    valid_mask[0, 0],
                    size=[1, top_k_reserved],
                )
                for i, j in T.Parallel(block_heads, max_top_k):
                    acc_s[i, j] *= scale
                # Mask invalid slots to -inf before softmax (matches gpatch).
                for i, j in T.Parallel(block_heads, max_top_k):
                    acc_s[i, j] = T.if_then_else(
                        valid_mask[0, j] > 0,
                        acc_s[i, j],
                        -T.infinity(accum_dtype),
                    )
                T.reduce_max(acc_s, scores_max, dim=1, size=[block_heads, top_k])
                for i, j in T.Parallel(block_heads, max_top_k):
                    acc_s[i, j] = T.exp(acc_s[i, j] - scores_max[i, 0])
                T.reduce_sum(acc_s, scores_sum, dim=1, size=[block_heads, top_k])
                T.copy(AttnSink[head_block * block_heads, 0], attn_sink_shared)
                for i in T.Parallel(block_heads):
                    scores_sum[i, 0] += T.exp(attn_sink_shared[i, 0] - scores_max[i, 0])
                # lse_log2 = log2(denominator) + scores_max * log2(e), for bwd kernel
                lse_buf = T.alloc_fragment((block_heads, 1), accum_dtype)
                T.copy(scores_sum, lse_buf)
                T.vlog2(lse_buf, lse_buf)
                for i in T.Parallel(block_heads):
                    lse_buf[i, 0] = lse_buf[i, 0] + scores_max[i, 0] * LOG2E
                T.copy(
                    lse_buf,
                    LSE[by, bx, head_block * block_heads],
                    size=[1, block_heads],
                )
                for i, j in T.Parallel(block_heads, max_top_k):
                    acc_s[i, j] /= scores_sum[i, 0]
                T.copy(acc_s, acc_s_cast)

                for k in T.Pipelined(T.ceildiv(top_k, block_top_k_cube), num_stages=2):
                    T.copy(SparseKVBuffer[by, bx, k * block_top_k_cube, 0], kv_shared)
                    T.gemm(
                        acc_s_cast[0, k * block_top_k_cube],
                        kv_shared,
                        acc_o,
                        initC=False,
                        size=[block_heads, block_top_k_cube, dim],
                    )
                T.copy(acc_o, o_shared)
                T.copy(o_shared, Output[by, bx, head_block * block_heads, 0])

    return sparseAttn


def next_divisible_number(num, divisor):
    return num + divisor - (num % divisor) if num % divisor != 0 else num


def sparse_mqa_fwd_interface(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: Optional[float] = None,
):
    """Sparse attention forward with attention sink.

    Returns:
        out: [B, S, H, D]
        lse: [B, S, H] fp32 log2-space LSE (compatible with example_sparse_attn_bwd_kernel)
    """
    batch_size, seq_len, num_heads, dim = q.size()
    top_k = topk_idxs.shape[-1]
    block_vec = 32
    block_cube = 128
    block_heads = 16 if num_heads > 16 else num_heads
    max_top_k = 256
    max_logical_cores = 32768
    queries_per_logical_core = 4
    max_total_queries = max_logical_cores * queries_per_logical_core
    assert batch_size * seq_len <= max_total_queries, (
        f"batch_size * seq_len ({batch_size * seq_len}) must not exceed "
        f"{max_total_queries}"
    )

    assert num_heads % block_heads == 0, (
        f"num_heads ({num_heads}) must be divisible by block_heads ({block_heads})"
    )
    if (
        not hasattr(sparse_mqa_fwd_interface, "kernel")
        or sparse_mqa_fwd_interface.num_heads != num_heads
        or sparse_mqa_fwd_interface.dim != dim
        or sparse_mqa_fwd_interface.block_heads != block_heads
    ):
        os.environ["TILELANG_ASCEND_MODE"] = "Developer"
        bytes_workspace = (
            block_cube * dim * 2
            + block_heads * block_cube * 4 * 2
            + block_heads * dim * 4
        )
        os.environ["TILELANG_ASCEND_WORKSPACE_SIZE"] = str(bytes_workspace * 16)
        sparse_mqa_fwd_interface.kernel = sparse_mqa_fwd(
            block_vec,
            block_cube,
            block_heads,
            num_heads,
            dim,
            max_top_k=max_top_k,
            scale=softmax_scale,
        )
        sparse_mqa_fwd_interface.num_heads = num_heads
        sparse_mqa_fwd_interface.dim = dim
        sparse_mqa_fwd_interface.block_heads = block_heads
    sparse_kv_buffer = torch.empty(
        (batch_size, seq_len, next_divisible_number(top_k, block_cube), dim),
        dtype=q.dtype,
        device=q.device,
    )
    valid_mask_buffer = torch.zeros(
        (batch_size, seq_len, next_divisible_number(top_k, block_cube)),
        dtype=attn_sink.dtype,
        device=attn_sink.device,
    )
    workspace_score = torch.zeros(
        (batch_size, seq_len, block_heads, next_divisible_number(top_k, block_cube)),
        dtype=attn_sink.dtype,
        device=attn_sink.device,
    )
    output, lse = sparse_mqa_fwd_interface.kernel(
        q.to(torch.bfloat16),
        kv.contiguous().to(torch.bfloat16),
        attn_sink,
        topk_idxs,
        sparse_kv_buffer,
        valid_mask_buffer,
        workspace_score,
    )
    return output, lse


def gather_sparse_kv(kv: torch.Tensor, topk_idxs: torch.Tensor):
    """Gather sparse KV rows; invalid indices (-1) are zeroed out."""
    b, s, topk = topk_idxs.shape
    valid_mask = topk_idxs != -1
    safe_idxs = topk_idxs.masked_fill(~valid_mask, 0).long()
    batch_idx = torch.arange(b, device=kv.device).view(b, 1, 1).expand(b, s, topk)
    kv_sparse = kv[batch_idx, safe_idxs, :]
    kv_sparse = kv_sparse * valid_mask.unsqueeze(-1).to(kv_sparse.dtype)
    return kv_sparse, valid_mask


def sparse_attn_torch(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: Optional[float] = None,
):
    """PyTorch reference forward with log2-space LSE (matches bwd kernel convention).

    Returns:
        o: [B, S, H, D] fp32
        lse_log2: [B, S, H] fp32
    """
    q = q.float()
    kv = kv.float()
    if attn_sink.dim() == 2:
        attn_sink = attn_sink.squeeze(-1)
    attn_sink = attn_sink.float()

    _, _, _, dim = q.shape
    if softmax_scale is None:
        softmax_scale = dim**-0.5

    kv_sparse, valid_mask = gather_sparse_kv(kv, topk_idxs)
    scores = torch.einsum("bshd,bskd->bshk", q, kv_sparse) * softmax_scale
    scores = scores.masked_fill(~valid_mask.unsqueeze(2), float("-inf"))

    scores_max = scores.max(dim=-1, keepdim=True).values.clamp(min=-1e30)
    exp_scores = torch.exp(scores - scores_max)
    numerator = torch.einsum("bshk,bskd->bshd", exp_scores, kv_sparse)
    sum_exp = exp_scores.sum(dim=-1)
    sink_term = torch.exp(attn_sink.view(1, 1, -1) - scores_max.squeeze(-1))
    denominator = sum_exp + sink_term
    o = numerator / denominator.unsqueeze(-1)
    lse_log2 = torch.log2(denominator) + scores_max.squeeze(-1) * LOG2E
    return o, lse_log2


def rand_sparse_attn_input(
    batch_size, num_heads, seq_len, seq_len_kv, top_k, dim, seed=88888888, causal=True
):
    """Generate legalized random inputs for Sparse Attention"""
    torch.manual_seed(seed)

    # Generate inputs
    q = torch.randn((batch_size, seq_len, num_heads, dim), dtype=torch.bfloat16).npu()
    kv = torch.randn((batch_size, seq_len_kv, dim), dtype=torch.bfloat16).npu()
    attn_sink = torch.randn((num_heads,), dtype=torch.float32).npu()
    top_k_indices = torch.randint(
        low=0, high=seq_len_kv, size=(batch_size, seq_len, top_k), dtype=torch.int32
    ).npu()

    if causal:
        # Apply causal mask on top_k_indices
        max_len = max(seq_len, top_k)
        causal_mask = torch.tril(torch.ones(max_len, max_len)).to(top_k_indices.device)
        causal_mask = causal_mask[:seq_len, :top_k]
        causal_mask = causal_mask.unsqueeze(dim=0).bool()
        top_k_indices = torch.where(causal_mask, top_k_indices, -1)

    scale = (1.0 / dim) ** 0.5

    return {
        "q": q,
        "kv": kv,
        "attn_sink": attn_sink,
        "topk_idxs": top_k_indices,
        "softmax_scale": scale,
    }


def generate_data():
    return rand_sparse_attn_input(
        batch_size=1,
        num_heads=32,
        seq_len=65536,
        seq_len_kv=4096,
        top_k=128,
        dim=128,
    )


def run_test(inputs):
    output, lse = sparse_mqa_fwd_interface(**inputs)
    ref_out, ref_lse = sparse_attn_torch(**inputs)
    torch.testing.assert_close(ref_out, output.float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(ref_lse, lse.float(), rtol=1e-2, atol=1e-2)
    print("\033[92mAll check passed.\033[0m")


if __name__ == "__main__":
    run_test(generate_data())
