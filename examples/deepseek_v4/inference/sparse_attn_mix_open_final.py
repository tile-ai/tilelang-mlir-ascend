# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import os
from typing import Optional

import torch
import tilelang
import tilelang.language as T

FP16 = "float16"
BF16 = "bfloat16"
FP32 = "float32"
INT32 = "int32"


@tilelang.jit(target="npuir")
def sparse_attn_mix_kernel(
    block_top_k,
    block_heads,
    num_heads,
    dim,
    batch_size,
    seq_len,
    seq_len_kv,
    top_k,
    multibuffer=2,
    scale=None,
    dtype=BF16,
    accum_dtype=FP32,
    indices_dtype=INT32,
):
    # Compile-time tile parameters:
    # - block_top_k: tile size along the sparse KV token dimension (top_k), for example
    #   32 candidate KV tokens per iteration.
    # - block_heads: tile size along the Q num_heads dimension, for example 16 Q heads.
    # - num_heads: number of Q attention heads. This kernel has no KV head dimension,
    #   so its KV layout is closer to MQA/shared-KV.
    # - dim: vector dimension of each head, also the Q/K/V dot-product reduction axis.
    # - batch_size/seq_len/seq_len_kv/top_k: kept as compile-time static shapes like
    #   FA_mix to preserve static reinterpret_cast/subview stride information before
    #   later NPUIR stride alignment.
    # - multibuffer: ping-pong buffer count after the mix pass expands workspaces,
    #   also used as the pipeline stage count.
    # - scale: attention score scale, defaulting to 1/sqrt(dim).
    if scale is None:
        scale = (1.0 / dim) ** 0.5

    assert block_heads % 2 == 0, "mix kernel maps one cube block to two vector sub-blocks"
    assert dim % 2 == 0, "mix V1 gather splits KV dim across two vector sub-blocks"

    shape_q = [batch_size, seq_len, num_heads, dim]
    shape_kv = [batch_size, seq_len_kv, dim]
    shape_o = [batch_size, seq_len, num_heads, dim]
    shape_sink = [num_heads]
    shape_topk = [batch_size, seq_len, top_k]

    # In mix mode one Cube tile covers block_heads heads. Two Vector logical cores
    # use vid to process the first and second block_heads_half heads, forming a
    # CV 1:2 partition.
    block_heads_half = block_heads // 2
    dim_half = dim // 2

    @T.prim_func
    def sparseAttnMix(
        Q: T.Tensor(shape_q, dtype),
        KV: T.Tensor(shape_kv, dtype),
        Output: T.Tensor(shape_o, dtype),
        AttnSink: T.Tensor(shape_sink, accum_dtype),
        TopKIndices: T.Tensor(shape_topk, indices_dtype),
    ):
        # Tensor semantics:
        # - Q[b, s, h, d]: query token s vector for head h.
        # - KV[b, skv, d]: key/value vectors shared by all Q heads; there is no
        #   kv_head dimension here.
        # - TopKIndices[b, s, t]: index of the t-th KV token selected for query s;
        #   -1 marks an invalid entry.
        # - AttnSink[h]: per-Q-head sink term. It only contributes to the softmax
        #   denominator and does not participate in P * V.
        # - Output[b, s, h, d]: output vector for each query token and Q head.
        # cid maps to one query position (batch, seq). For the same cid, vid=0/1
        # splits the Vector-side head sub-blocks while Cube computes full
        # block_heads matrix multiplications.
        with T.Kernel(batch_size * seq_len, is_npu=True) as (cid, vid):
            by = cid // seq_len
            bx = cid % seq_len
            value_zero = 0
            value_min = -T.infinity(accum_dtype)

            # Cube-side local tiles: q_shared/kv_shared feed both QK and PV GEMMs.
            # kv_shared is loaded back from workspace by the first Cube stage for the
            # current top-k tile. The second Cube stage reuses the same L1 data to
            # avoid reloading the same KV tile.
            # EnableMultiBuffer splits C1(s0,s1) and C2(s0,s1) into two stage loops,
            # so kv_shared must be staged. Otherwise C2(s0) may read the KV loaded by
            # C1(s1).
            q_shared = T.alloc_shared((block_heads, dim), dtype)
            kv_shared = T.alloc_shared((block_top_k, dim), dtype, multi_buffer=multibuffer)
            prob_shared = T.alloc_shared((block_heads, block_top_k), dtype)
            scores = T.alloc_fragment((block_heads, block_top_k), accum_dtype)
            scores_cast = T.alloc_shared((block_heads_half, block_top_k), dtype)
            pv_acc = T.alloc_fragment((block_heads, dim), accum_dtype)

            # Vector-side local tiles handle sparse KV gather, masks, online softmax,
            # and output accumulation.
            kv_ub = T.alloc_shared((block_top_k, dim_half), dtype)
            idxs = T.alloc_fragment((block_top_k,), indices_dtype)
            # mask_ub is produced by V1 and consumed by V2. It is a per-stage
            # temporary crossing Vector scopes. Unlike scores_max/sum_exp/acc_o, it
            # is not a cross-k recurrence state, so it needs local multi-buffering.
            mask_ub = T.alloc_shared((1, block_top_k), accum_dtype, multi_buffer=multibuffer)
            scores_ub = T.alloc_shared((block_heads_half, block_top_k), accum_dtype)
            scores_max = T.alloc_shared((block_heads_half, 1), accum_dtype)
            scores_max_prev = T.alloc_shared((block_heads_half, 1), accum_dtype)
            # scores_scale is produced by V2 and consumed by V3 to correct the
            # historical acc_o, matching the correction term in FA_mix.
            scores_scale = T.alloc_shared(
                (block_heads_half, 1), accum_dtype, multi_buffer=multibuffer
            )
            scores_sum = T.alloc_shared((block_heads_half, 1), accum_dtype)
            sum_exp = T.alloc_shared((block_heads_half, 1), accum_dtype)
            acc_o = T.alloc_shared((block_heads_half, dim), accum_dtype)
            acc_o_new = T.alloc_shared((block_heads_half, dim), accum_dtype)
            o_cast = T.alloc_shared((block_heads_half, dim), dtype)

            # Workspaces are GM staging buffers across the Cube/Vector boundary. They
            # also anchor multi-buffer expansion and automatic set/wait synchronization
            # in the mix passes.
            workspace_kv = T.alloc_workspace(
                (2,block_top_k, dim), dtype, multi_buffer=multibuffer
            )
            workspace_score = T.alloc_workspace(
                (2,block_heads, block_top_k), accum_dtype, multi_buffer=multibuffer
            )
            workspace_prob = T.alloc_workspace(
                (2,block_heads, block_top_k), dtype, multi_buffer=multibuffer
            )
            workspace_out = T.alloc_workspace(
                (2,block_heads, dim), accum_dtype, multi_buffer=multibuffer
            )

            # The outer loop iterates over head blocks. This must stay a plain serial
            # loop because the current mix pass treats each T.Pipelined loop as one
            # C/V pipeline region and does not support nested pipelines.
            for n in T.serial(T.ceildiv(num_heads, block_heads)):
                # n selects the current Q head range:
                # [n * block_heads, min((n + 1) * block_heads, num_heads))。
                # The head dimension is not the attention reduction axis; different
                # heads produce independent outputs.
                T.vbrc(value_zero, acc_o)
                T.vbrc(value_zero, sum_exp)
                T.vbrc(value_min, scores_max)
                # After cid is fixed, the Q sequence position is the current bx. This
                # copy loads [block_heads, dim] for the current query token and head
                # block.
                T.copy(Q[by, bx, n * block_heads, 0], q_shared, size=[block_heads, dim])

                # The inner loop tiles sparse attention over top_k. This is the only
                # mix pipeline region, and later passes build multi-buffering and C/V
                # synchronization around it.
                for k in T.Pipelined(T.ceildiv(top_k, block_top_k), num_stages=multibuffer):
                    # k selects the current sparse KV token range:
                    # [k * block_top_k, min((k + 1) * block_top_k, top_k))。
                    # top_k is the sparse KV list length preselected for each query,
                    # not the full seq_len_kv.
                    real_block_top_k = T.min(top_k - k * block_top_k, block_top_k)

                    # Vector stage 1: load this query's top-k indices and gather the
                    # sparse K/V tile from KV. -1 marks an invalid position and maps to
                    # mask=0.
                    # KV has no head dimension, so V1 uses vid to split the dim axis:
                    # two AIVs move the first and second halves of the same top-k KV
                    # tile into different dim ranges of workspace_kv, avoiding duplicate
                    # full-KV copies.
                    dim_offset = vid * dim_half
                    T.vbrc(value_zero, kv_ub)
                    T.vbrc(value_zero, mask_ub)
                    T.copy(
                        TopKIndices[by, bx, k * block_top_k],
                        idxs,
                        size=[real_block_top_k],
                    )
                    for i in T.serial(real_block_top_k):
                        cur_idx = idxs[i]
                        if cur_idx != -1:
                            mask_ub[0, i] = 1.0
                            T.copy(
                                KV[by, cur_idx, dim_offset],
                                kv_ub[i, 0],
                                size=[1, dim_half],
                            )

                    T.copy(
                        kv_ub,
                        workspace_kv[0,0, dim_offset],
                        size=[block_top_k, dim_half],
                    )

                    # Cube stage 1: load KV back from workspace into L1 and compute
                    # scores = Q * K^T, producing block_heads x block_top_k scores.
                    # Shape equivalent:
                    # [block_heads, dim] @ [dim, block_top_k] -> [block_heads, block_top_k]。
                    T.copy(workspace_kv[0,0, 0], kv_shared, size=[block_top_k, dim])
                    T.gemm(
                        q_shared,
                        kv_shared,
                        scores,
                        initC=True,
                        b_transpose=True,
                        size=[block_heads, dim, block_top_k],
                    )
                    T.copy(scores, workspace_score[0,0, 0], size=[block_heads, block_top_k])

                    # Vector stage 2: each vid processes half of the heads. This stage
                    # performs online softmax, maintains historical max/sum, and writes
                    # the current probability tile back to workspace.
                    T.copy(
                        workspace_score[0,vid * block_heads_half, 0],
                        scores_ub,
                        size=[block_heads_half, block_top_k],
                    )

                    # Online softmax state:
                    # scores_max/sum_exp/acc_o keep results from processed top_k tiles.
                    # scores_scale uses the old/new max difference to correct the
                    # historical denominator and output accumulation.
                    T.copy(scores_max, scores_max_prev)
                    T.vmul(scores_ub, scale, scores_ub)
                    T.reduce_max(scores_ub, scores_max, dim=1)
                    for i, j in T.Parallel(block_heads_half, block_top_k):
                        scores_ub[i, j] = T.exp(scores_ub[i, j] - scores_max[i, 0])
                    for i, j in T.Parallel(block_heads_half, block_top_k):
                        scores_ub[i, j] *= mask_ub[0, j]

                    # P crosses into Cube stage 2, so cast and copy it out through MTE3
                    # first. The later reduce_sum/sum_exp updates are Vector-local and
                    # can overlap with the P copy-out.
                    T.vcast(scores_ub, scores_cast, round_mode="rint")
                    # T.reduce_sum(scores_ub, scores_sum, dim=1)
                    T.copy(
                        scores_cast,
                        workspace_prob[0,vid * block_heads_half, 0],
                        size=[block_heads_half, block_top_k],
                    )
                    T.reduce_sum(scores_ub, scores_sum, dim=1)
                    for i in T.Parallel(block_heads_half):
                        scores_scale[i, 0] = T.exp(scores_max_prev[i, 0] - scores_max[i, 0])
                    for i in T.Parallel(block_heads_half):
                        sum_exp[i, 0] = sum_exp[i, 0] * scores_scale[i, 0] + scores_sum[i, 0]

                    # Cube stage 2: read the probabilities written by both Vector
                    # sub-blocks, compute P * V, and write the full block_heads
                    # contribution for this tile to workspace_out.
                    # Shape equivalent:
                    # [block_heads, block_top_k] @ [block_top_k, dim] -> [block_heads, dim]。
                    T.copy(workspace_prob[0,0, 0], prob_shared, size=[block_heads, block_top_k])
                    # kv_shared was loaded back from workspace_kv during QK and is
                    # reused directly in this stage.
                    T.gemm(
                        prob_shared,
                        kv_shared,
                        pv_acc,
                        initC=True,
                        size=[block_heads, block_top_k, dim],
                    )
                    T.copy(pv_acc, workspace_out[0,0, 0], size=[block_heads, dim])

                    # Vector stage 3: load the half-head output tile owned by this vid
                    # and use the online-softmax scale to correct historical
                    # accumulation.
                    T.copy(
                        workspace_out[0,vid * block_heads_half, 0],
                        acc_o_new,
                        size=[block_heads_half, dim],
                    )
                    T.vmul(acc_o, scores_scale, acc_o)
                    T.vadd(acc_o, acc_o_new, acc_o)

                # After all top-k tiles, add the attention sink into the softmax
                # denominator. The sink only changes normalization and produces no
                # value contribution.
                for i in T.Parallel(block_heads_half):
                    head_idx = n * block_heads + vid * block_heads_half + i
                    if head_idx < num_heads:
                        scores_max_prev[i, 0] = AttnSink[head_idx]
                    else:
                        scores_max_prev[i, 0] = value_min
                for i in T.Parallel(block_heads_half):
                    sum_exp[i, 0] += T.exp(scores_max_prev[i, 0] - scores_max[i, 0])
                T.vdiv(acc_o, sum_exp, acc_o)
                T.vcast(acc_o, o_cast, round_mode="rint")
                # Each vid finally writes back its own half-head output.
                real_heads = T.min(
                    block_heads_half,
                    num_heads - n * block_heads - vid * block_heads_half,
                )
                T.copy(
                    o_cast,
                    Output[by, bx, n * block_heads + vid * block_heads_half, 0],
                    size=[real_heads, dim],
                )

    return sparseAttnMix


def sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: Optional[float] = None,
):
    # Python wrapper inputs:
    # - q: [batch_size, seq_len, num_heads, dim], full multi-head Q input.
    # - kv: [batch_size, seq_len_kv, dim], shared KV without a kv_head dimension,
    #   matching an MQA/shared-KV layout.
    # - attn_sink: [num_heads], one sink scalar per Q head.
    # - topk_idxs: [batch_size, seq_len, top_k], sparse KV indices for each query.
    # - softmax_scale: score scale, usually 1/sqrt(dim).
    block = 64
    # block_heads = 64
    multibuffer = 2
    # block is the wrapper alias for kernel block_top_k; block_heads tiles the Q
    # num_heads dimension. With the current configuration, Cube handles one tile of
    # block_heads Q heads x block sparse KV tokens, while the two Vector vids each
    # handle half of the Q heads.
    batch_size, seq_len, num_heads, dim = q.size()
    block_heads = num_heads
    seq_len_kv = kv.size(1)
    top_k = topk_idxs.shape[-1]
    assert kv.size(0) == batch_size and kv.size(2) == dim
    assert topk_idxs.size(0) == batch_size and topk_idxs.size(1) == seq_len
    assert attn_sink.numel() == num_heads
    if (
        not hasattr(sparse_attn, "kernel")
        or sparse_attn.batch_size != batch_size
        or sparse_attn.seq_len != seq_len
        or sparse_attn.seq_len_kv != seq_len_kv
        or sparse_attn.num_heads != num_heads
        or sparse_attn.dim != dim
        or sparse_attn.top_k != top_k
    ):
        os.environ["TILELANG_ASCEND_MODE"] = "Expert"
        # Expert/API lowering maps T.alloc_workspace to memref_ext.alloc_workspace.
        # Later mix passes rely on it to identify Cube/Vector boundaries and
        # multi-buffer workspaces.
        sparse_attn.kernel = sparse_attn_mix_kernel(
            block,
            block_heads,
            num_heads,
            dim,
            batch_size,
            seq_len,
            seq_len_kv,
            top_k,
            multibuffer,
            softmax_scale,
        )
        sparse_attn.batch_size = batch_size
        sparse_attn.seq_len = seq_len
        sparse_attn.seq_len_kv = seq_len_kv
        sparse_attn.num_heads = num_heads
        sparse_attn.dim = dim
        sparse_attn.top_k = top_k

    output = torch.empty((batch_size, seq_len, num_heads, dim), dtype=q.dtype, device=q.device)
    sparse_attn.kernel(q, kv.contiguous(), output, attn_sink, topk_idxs)
    return output


def gather_from_kv(KV, indices):
    b, s1, k = indices.shape
    batch_idx = torch.arange(b, device=KV.device).view(b, 1, 1).expand(-1, s1, k)
    indices_flat = indices.long()
    out = KV[batch_idx, indices_flat, :].squeeze(dim=2)

    mask = (indices != -1).float().unsqueeze(-1)
    out = out * mask

    return out


def softmax_with_sink(x: torch.Tensor, attn_sink: torch.Tensor, head_dim, dim=-1):
    max_vals = torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x - max_vals)
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)

    sink_view_shape = [1] * x.dim()
    sink_view_shape[head_dim if head_dim > 0 else head_dim % x.dim()] = x.shape[head_dim]

    sink_term = torch.exp(attn_sink.view(sink_view_shape) - max_vals)
    adjusted_sum = sum_exp + sink_term

    return exp_x / adjusted_sum


def sparse_attn_torch(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: Optional[float] = None,
):
    base_dtype = torch.bfloat16
    kv_sparse = gather_from_kv(kv, topk_idxs)
    mask_acc_s = torch.where((topk_idxs == -1).unsqueeze(-2), -torch.inf, 0.0)
    mask_acc_s = mask_acc_s.to(device=q.device, dtype=torch.float32)
    ref_output = (
        softmax_with_sink(
            ((q @ kv_sparse.transpose(-2, -1)).to(torch.float32) + mask_acc_s)
            * softmax_scale,
            attn_sink,
            head_dim=-2,
            dim=-1,
        ).to(base_dtype)
        @ kv_sparse
    )

    return ref_output


def rand_sparse_attn_input(
    batch_size, num_heads, seq_len, seq_len_kv, top_k, dim, seed=88888888
):
    base_dtype = torch.bfloat16
    torch.manual_seed(seed)

    q = torch.randn((batch_size, seq_len, num_heads, dim), dtype=base_dtype).npu()
    kv = torch.randn((batch_size, seq_len_kv, dim), dtype=base_dtype).npu()
    attn_sink = torch.randn((num_heads,), dtype=torch.float32).npu()
    top_k_indices = torch.randint(
        low=0,
        high=seq_len_kv,
        size=(batch_size, seq_len, top_k),
        dtype=torch.int32,
    ).npu()

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


def generate_and_save_data(case_id, **kwargs):
    inputs = rand_sparse_attn_input(**kwargs)
    outputs = sparse_attn_torch(**inputs)
    torch.save({"inputs": inputs, "outputs": outputs}, f"case_{case_id}.pt")


def generate_data():
    generate_and_save_data(
        case_id=0,
        batch_size=1,
        num_heads=64,
        seq_len=256,
        seq_len_kv=256,
        top_k=128,
        dim=512,
    )


def run_test():
    data = torch.load("case_0.pt", map_location=torch.device("npu"))
    output = sparse_attn(**data["inputs"])

    torch.testing.assert_close(data["outputs"], output, rtol=1e-2, atol=1e-2)
    print("\033[92mAll check passed.\033[0m")


if __name__ == "__main__":
    generate_data()
    run_test()
