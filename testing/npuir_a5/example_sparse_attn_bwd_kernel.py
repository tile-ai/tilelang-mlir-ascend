# ruff: noqa
# Adapted from miles_plugins/models/glm5/ops/tilelang_sparse_mla_bwd.py for DeepSeek-V4.
# Key differences from GLM-5:
#   - attn_sink: gradient computation for learnable per-head scalar
#   - Single-head KV: kv shape [B, S_kv, D] (no kv_group, no D/D_tail split)
#   - Index shape: [B, S, topk] (no kv_group dim)
#   - Outputs: dQ [B, S, H, D], dKV [B, S_kv, D], dAttnSink [H]
import os
import tilelang
import torch
from tilelang import language as T
from typing import Optional


@tilelang.jit(out_idx=[-1], target="npuir")
def preprocess(
    B,
    S,
    H,
    D,
    block_ND=32,
    num_stages=5,
    dtype="bfloat16",
    accum_dtype="float32",
):
    assert dtype == "bfloat16"
    assert accum_dtype == "float32"
    shape = [B, S, H, D]
    num = T.ceildiv(S, block_ND)

    @T.prim_func
    def preprocess_kernel(
        O: T.Tensor(shape, dtype),
        dO: T.Tensor(shape, dtype),
        Delta: T.Tensor([B, S, H], accum_dtype),
    ):
        with T.Kernel(H * num * B, is_npu=True) as (cid, _):
            bx = cid % H
            by = (cid // H) % num
            bz = cid // (H * num)
            o = T.alloc_fragment([block_ND, block_ND], accum_dtype)
            do = T.alloc_fragment([block_ND, block_ND], accum_dtype)
            delta = T.alloc_fragment([block_ND, 1], accum_dtype)
            acc = T.alloc_fragment([block_ND, block_ND], accum_dtype)
            T.clear(acc)
            for k in T.Pipelined(T.ceildiv(D, block_ND), num_stages=num_stages):
                T.copy(
                    O[
                        bz,
                        by * block_ND : (by + 1) * block_ND,
                        bx,
                        k * block_ND : (k + 1) * block_ND,
                    ],
                    o,
                )
                T.copy(
                    dO[
                        bz,
                        by * block_ND : (by + 1) * block_ND,
                        bx,
                        k * block_ND : (k + 1) * block_ND,
                    ],
                    do,
                )
                for i, j in T.Parallel(block_ND, block_ND):
                    acc[i, j] += o[i, j] * do[i, j]
            T.reduce_sum(acc, delta, 1)
            T.copy(delta[:, 0], Delta[bz, by * block_ND : (by + 1) * block_ND, bx])

    return preprocess_kernel


@tilelang.jit(out_idx=[-1], target="npuir")
def postprocess(
    B,
    S_kv,
    D,
    block_N=64,
    dtype="bfloat16",
    accum_dtype="float32",
):
    assert dtype == "bfloat16"
    assert accum_dtype == "float32"
    dkv_shape = [B, S_kv, D]
    num = T.ceildiv(S_kv, block_N)

    @T.prim_func
    def postprocess_kernel(
        dKV: T.Tensor(dkv_shape, accum_dtype),
        dKV_out: T.Tensor(dkv_shape, dtype),
    ):
        with T.Kernel(num * B, is_npu=True) as (cid, _):
            by = cid // num
            bx = cid % num
            dKV_shared = T.alloc_shared([block_N, D], accum_dtype)
            T.copy(dKV[by, bx * block_N : (bx + 1) * block_N, :], dKV_shared)
            T.copy(dKV_shared, dKV_out[by, bx * block_N : (bx + 1) * block_N, :])

    return postprocess_kernel


def next_power_of_2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


@tilelang.jit(target="npuir")
def bwd(
    B,
    S,
    S_kv,
    H,
    D,
    topk,
    sm_scale=None,
    block_size=16,
    num_stages=0,
    indices_dtype="int32",
    dtype="bfloat16",
    accum_dtype="float32",
):
    assert topk % block_size == 0, (
        f"topk ({topk}) must be divisible by block_size ({block_size})"
    )
    assert dtype == "bfloat16"
    assert accum_dtype == "float32"

    if sm_scale is None:
        sm_scale = D ** (-0.5)
    sm_scale_mul_reciprocal_log2 = sm_scale * 1.44269504  # log2(e)

    q_shape = [B, S, H, D]
    kv_shape = [B, S_kv, D]
    o_shape = [B, S, H, D]
    indices_shape = [B, S, topk]
    delta_shape = [B, S, H]
    lse_shape = [B, S, H]
    attn_sink_shape = [H]

    padded_H = max(next_power_of_2(H), 8)
    block_H = min(32, padded_H)
    assert padded_H % block_H == 0
    NH = padded_H // block_H
    BS = block_size
    NS = tilelang.cdiv(topk, block_size)

    split_store = 2

    @T.prim_func
    def sparse_mqa_bwd_kernel(
        Q: T.Tensor(q_shape, dtype),
        KV: T.Tensor(kv_shape, dtype),
        dO: T.Tensor(o_shape, dtype),
        AttnSink: T.Tensor(attn_sink_shape, accum_dtype),
        Indices: T.Tensor(indices_shape, indices_dtype),
        Lse: T.Tensor(lse_shape, accum_dtype),
        Delta: T.Tensor(delta_shape, accum_dtype),
        dQ: T.Tensor(q_shape, dtype),
        dKV: T.Tensor(kv_shape, accum_dtype),
        dAttnSink: T.Tensor(attn_sink_shape, accum_dtype),
        SparseKVBuffer: T.Tensor([B, S, topk, D], dtype),
    ):
        with T.Kernel(S * B * NH, is_npu=True) as (cid, _):
            s_i = cid % S
            by = (cid // S) % B
            bz = cid // (S * B)
            Q_shared = T.alloc_shared([block_H, D], dtype)
            KV_shared = T.alloc_shared([BS, D], dtype)
            dO_shared = T.alloc_shared([block_H, D], dtype)
            mask = T.alloc_fragment([BS], "bool")
            idxs = T.alloc_fragment([BS], indices_dtype)
            idxs_cmp = T.alloc_fragment([BS], indices_dtype)

            P_shared_cast = T.alloc_shared([block_H, BS], dtype)
            dP_shared_cast = T.alloc_shared([block_H, BS], dtype)
            dQ_shared = T.alloc_shared([block_H, D], dtype)

            acc_p = T.alloc_fragment([block_H, BS], accum_dtype)
            tmp_p = T.alloc_fragment([block_H, BS], accum_dtype)
            acc_dp = T.alloc_fragment([block_H, BS], accum_dtype)
            acc_dq = T.alloc_fragment([block_H, D], accum_dtype)
            acc_dkv = T.alloc_fragment([BS, D], accum_dtype)
            acc_dkv_shared = T.alloc_fragment([BS, D], accum_dtype)

            Delta_shared = T.alloc_shared([block_H], accum_dtype)
            AttnSink_shared = T.alloc_shared([block_H], accum_dtype)
            tmp_AttnSink = T.alloc_shared([block_H], accum_dtype)
            Lse_shared = T.alloc_shared([block_H], accum_dtype)

            T.copy(Q[by, s_i, bz * block_H : (bz + 1) * block_H, :D], Q_shared)
            T.copy(dO[by, s_i, bz * block_H : (bz + 1) * block_H, :D], dO_shared)
            T.copy(
                Delta[by, s_i, bz * block_H : (bz + 1) * block_H],
                Delta_shared[0:block_H],
            )
            T.copy(
                Lse[by, s_i, bz * block_H : (bz + 1) * block_H], Lse_shared[0:block_H]
            )

            T.clear(acc_dq)

            acc_p_inf = -T.infinity(accum_dtype)
            value_negone = -1

            T.vbrc(value_negone, idxs_cmp)

            for i_i in T.Pipelined(NS, num_stages=num_stages):
                T.copy(Indices[by, s_i, i_i * BS], idxs, size=[BS])
                T.vcmp(idxs, idxs_cmp, mask, "ne")
                T.clear(KV_shared)
                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_p[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, acc_p_inf)

                KV_gather = T.alloc_shared([BS, D], dtype)
                for i in T.serial(BS):
                    cur_idx = idxs[i]
                    if cur_idx != -1:
                        T.copy(KV[by, cur_idx, 0:D], KV_gather[i, :D])
                T.copy(
                    KV_gather, SparseKVBuffer[by, s_i, i_i * BS : (i_i + 1) * BS, 0:D]
                )
                T.copy(
                    SparseKVBuffer[by, s_i, i_i * BS : (i_i + 1) * BS, 0:D], KV_shared
                )

                T.gemm(Q_shared, KV_shared, acc_p, b_transpose=True)

                # P = exp2(scores * sm_scale_log2e - LSE)
                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_p[h_i, bi_i] = (
                        acc_p[h_i, bi_i] * sm_scale_mul_reciprocal_log2
                        - Lse_shared[h_i]
                    )

                T.vexp2(acc_p, acc_p, tmp_p)
                T.copy(acc_p, P_shared_cast)

                # dP = P * (dO @ KV^T - Delta)
                T.gemm(dO_shared, KV_shared, acc_dp, b_transpose=True, initC=True)

                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_dp[h_i, bi_i] = (
                        acc_p[h_i, bi_i]
                        * (acc_dp[h_i, bi_i] - Delta_shared[h_i])
                        * sm_scale
                    )

                T.copy(acc_dp, dP_shared_cast)

                # dQ += dP @ KV
                T.gemm(dP_shared_cast, KV_shared, acc_dq)

                # dKV += dP^T @ Q + P^T @ dO
                T.gemm(dP_shared_cast, Q_shared, acc_dkv, a_transpose=True, initC=True)
                T.gemm(
                    P_shared_cast,
                    dO_shared,
                    acc_dkv_shared,
                    a_transpose=True,
                    initC=True,
                )
                T.vadd(acc_dkv, acc_dkv_shared, acc_dkv)

                # Atomic store dKV with split to reduce register pressure
                for s in T.serial(split_store):
                    for bi_i in T.serial(BS // split_store):
                        row = bi_i + s * (BS // split_store)
                        cur_idx = idxs[row]
                        if cur_idx != -1:
                            T.atomic_add(dKV[by, cur_idx, :], acc_dkv[row, :])

            # Store dQ
            T.copy(acc_dq, dQ_shared)
            T.copy(dQ_shared, dQ[by, s_i, bz * block_H : (bz + 1) * block_H, :D])
            # dAttnSink[h] = -sum_{b,s}( Delta[b,s,h] * p_sink[b,s,h] )
            # where p_sink = exp(attn_sink[h]) / Z = exp2(attn_sink[h]*log2e - LSE)
            # attn_sink is a pre-scaled logit, so only convert to log2 base (no sm_scale)
            T.copy(AttnSink[bz * block_H : (bz + 1) * block_H], AttnSink_shared)
            T.vmul(AttnSink_shared, 1.44269504, AttnSink_shared)
            T.vsub(AttnSink_shared, Lse_shared, AttnSink_shared)
            T.vexp2(AttnSink_shared, AttnSink_shared, tmp_AttnSink)
            T.vmul(AttnSink_shared, Delta_shared, AttnSink_shared)
            T.vmul(AttnSink_shared, -1, AttnSink_shared)
            T.atomic_add(dAttnSink[bz * block_H : (bz + 1) * block_H], AttnSink_shared)

    return sparse_mqa_bwd_kernel


def sparse_mqa_bwd_interface(q, kv, attn_sink, o, do, topk_idxs, lse, sm_scale):
    assert q.is_contiguous() and kv.is_contiguous()
    assert topk_idxs.is_contiguous() and lse.is_contiguous()
    B, S, H, D = q.shape
    _, S_kv, _ = kv.shape
    topk = topk_idxs.shape[-1]

    block_size = 16
    padded_topk = (topk + block_size - 1) // block_size * block_size
    if padded_topk != topk:
        pad = torch.full(
            (B, S, padded_topk - topk),
            -1,
            device=topk_idxs.device,
            dtype=topk_idxs.dtype,
        )
        topk_idxs = torch.cat([topk_idxs, pad], dim=-1).contiguous()
        topk = padded_topk

    if B > 1:
        delta_flatten = preprocess(1, B * S, H, D)(
            o.view(1, B * S, H, D).contiguous(), do.view(1, B * S, H, D).contiguous()
        )
        delta = delta_flatten.view(B, S, H).contiguous()
    else:
        delta = preprocess(B, S, H, D)(o, do)
    bwd_kernel = bwd(B, S, S_kv, H, D, topk, sm_scale, block_size)
    postprocess_kernel = postprocess(B, S_kv, D, block_N=16)

    dkv = torch.zeros_like(kv, dtype=torch.float32)
    d_attn_sink = torch.zeros_like(attn_sink)
    dq = torch.zeros_like(q, dtype=torch.bfloat16)
    SparseKVBuffer = torch.empty((B, S, topk, D), device=q.device, dtype=torch.bfloat16)
    bwd_kernel(
        q,
        kv,
        do,
        attn_sink,
        topk_idxs,
        lse,
        delta,
        dq,
        dkv,
        d_attn_sink,
        SparseKVBuffer,
    )

    dkv = postprocess_kernel(dkv)

    return dq, dkv, d_attn_sink


LOG2E = 1.4426950408889634


def gather_sparse_kv(kv: torch.Tensor, topk_idxs: torch.Tensor):
    B, S, topk = topk_idxs.shape

    valid_mask = topk_idxs != -1
    safe_idxs = topk_idxs.masked_fill(~valid_mask, 0).long()

    batch_idx = torch.arange(B, device=kv.device).view(B, 1, 1).expand(B, S, topk)
    kv_sparse = kv[batch_idx, safe_idxs, :]

    kv_sparse = kv_sparse * valid_mask.unsqueeze(-1).to(kv_sparse.dtype)
    return kv_sparse, valid_mask


def ref_sparse_attn_with_lse(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    sm_scale: Optional[float] = None,
):
    q = q.float()
    kv = kv.float()
    attn_sink = attn_sink.float()

    B, S, H, D = q.shape
    topk = topk_idxs.shape[-1]

    if sm_scale is None:
        sm_scale = D**-0.5

    kv_sparse, valid_mask = gather_sparse_kv(kv, topk_idxs)

    scores = torch.einsum("bshd,bskd->bshk", q, kv_sparse) * sm_scale
    scores = scores.masked_fill(~valid_mask.unsqueeze(2), float("-inf"))

    scores_max = scores.max(dim=-1, keepdim=True).values
    scores_max = scores_max.clamp(min=-1e30)

    exp_scores = torch.exp(scores - scores_max)
    numerator = torch.einsum("bshk,bskd->bshd", exp_scores, kv_sparse)
    sum_exp = exp_scores.sum(dim=-1)

    sink_term = torch.exp(attn_sink.view(1, 1, H) - scores_max.squeeze(-1))
    denominator = sum_exp + sink_term
    o = numerator / denominator.unsqueeze(-1)

    lse_log2 = torch.log2(denominator) + scores_max.squeeze(-1) * LOG2E

    return o, lse_log2


def ref_sparse_attn_with_grad(
    q_base: torch.Tensor,
    kv_base: torch.Tensor,
    attn_sink_base: torch.Tensor,
    topk_idxs: torch.Tensor,
    do: torch.Tensor,
    sm_scale: Optional[float] = None,
):
    q_ref = q_base.detach().clone().float().requires_grad_(True)
    kv_ref = kv_base.detach().clone().float().requires_grad_(True)
    sink_ref = attn_sink_base.detach().clone().float().requires_grad_(True)

    o_ref, lse_ref = ref_sparse_attn_with_lse(
        q_ref,
        kv_ref,
        sink_ref,
        topk_idxs,
        sm_scale,
    )

    loss = (o_ref * do.float()).sum()
    loss.backward()

    return {
        "o": o_ref.detach(),
        "lse": lse_ref.detach(),
        "dq": q_ref.grad.detach(),
        "dkv": kv_ref.grad.detach(),
        "d_attn_sink": sink_ref.grad.detach(),
    }


def make_inputs(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    dim: int,
    seq_len_kv: int,
    topk: int,
    seed: int = 88888888,
    device: str = "cpu",
    use_invalid_indices: bool = False,
    topk_mode: str = "no_conflict",
):
    torch.manual_seed(seed)

    q = torch.randn(
        batch_size,
        seq_len,
        num_heads,
        dim,
        device=device,
        dtype=torch.bfloat16,
    )

    kv = torch.randn(
        batch_size,
        seq_len_kv,
        dim,
        device=device,
        dtype=torch.bfloat16,
    )

    attn_sink = torch.randn(
        num_heads,
        device=device,
        dtype=torch.float32,
    )

    if topk_mode == "no_conflict":
        assert seq_len * topk <= seq_len_kv
        topk_idxs = torch.empty(
            batch_size, seq_len, topk, device=device, dtype=torch.int32
        )
        for b in range(batch_size):
            for s in range(seq_len):
                start = s * topk
                topk_idxs[b, s, :] = torch.arange(
                    start, start + topk, device=device, dtype=torch.int32
                )
    elif topk_mode == "conflict":
        topk_idxs = torch.arange(topk, device=device, dtype=torch.int32).view(
            1, 1, topk
        )
        topk_idxs = topk_idxs.expand(batch_size, seq_len, topk).contiguous()
    elif topk_mode == "random":
        topk_idxs = torch.randint(
            low=0,
            high=seq_len_kv,
            size=(batch_size, seq_len, topk),
            device=device,
            dtype=torch.int32,
        )
    elif topk_mode == "duplicate":
        topk_idxs = torch.zeros(
            batch_size,
            seq_len_kv,
            topk,
            device=device,
            dtype=torch.int32,
        )

    sm_scale = dim**-0.5

    return (
        q.contiguous(),
        kv.contiguous(),
        attn_sink.contiguous(),
        topk_idxs.contiguous(),
        sm_scale,
    )


def run_one_bwd_case(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    dim: int,
    seq_len_kv: int,
    topk: int,
    topk_mode: str = "no_conflict",
):
    print(
        f"\n[BWD] "
        f"B={batch_size}, S={seq_len}, H={num_heads}, D={dim}, "
        f"S_kv={seq_len_kv}, topk={topk}, topk_mode={topk_mode}"
    )

    q_cpu, kv_cpu, attn_sink_cpu, topk_idxs_cpu, sm_scale = make_inputs(
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=num_heads,
        dim=dim,
        seq_len_kv=seq_len_kv,
        topk=topk,
        topk_mode=topk_mode,
        device="cpu",
    )

    torch.manual_seed(1234)
    do_cpu = torch.randn_like(q_cpu, dtype=torch.bfloat16).contiguous()

    ref_cpu = ref_sparse_attn_with_grad(
        q_base=q_cpu,
        kv_base=kv_cpu,
        attn_sink_base=attn_sink_cpu,
        topk_idxs=topk_idxs_cpu,
        do=do_cpu,
        sm_scale=sm_scale,
    )

    q_npu = q_cpu.to("npu")
    kv_npu = kv_cpu.to("npu")
    attn_sink_npu = attn_sink_cpu.to("npu")
    topk_idxs_npu = topk_idxs_cpu.to("npu")
    do_npu = do_cpu.to("npu")

    o_npu = ref_cpu["o"].to(torch.bfloat16).to("npu").contiguous()
    lse_npu = ref_cpu["lse"].float().to("npu").contiguous()

    tl_dq_npu, tl_dkv_npu, tl_d_attn_sink_npu = sparse_mqa_bwd_interface(
        q=q_npu,
        kv=kv_npu,
        attn_sink=attn_sink_npu,
        o=o_npu,
        do=do_npu,
        topk_idxs=topk_idxs_npu,
        lse=lse_npu,
        sm_scale=sm_scale,
    )

    tl_dq_cpu = tl_dq_npu.cpu()
    tl_dkv_cpu = tl_dkv_npu.cpu()
    tl_d_attn_sink_cpu = tl_d_attn_sink_npu.cpu()

    # Compare
    torch.testing.assert_close(
        tl_dq_cpu.float(), ref_cpu["dq"].float(), rtol=5e-2, atol=5e-2
    )
    print("\033[92mdq check passed.\033[0m")

    torch.testing.assert_close(
        tl_d_attn_sink_cpu.float(), ref_cpu["d_attn_sink"].float(), rtol=5e-2, atol=5e-2
    )
    print("\033[92mattn_sink check passed.\033[0m")

    torch.testing.assert_close(
        tl_dkv_cpu.float(),
        ref_cpu["dkv"].to(torch.bfloat16).float(),
        rtol=5e-2,
        atol=5e-2,
    )
    print("\033[92mdkv check passed.\033[0m")


def run_all_tests():
    cases = [
        # (batch, seqlen, heads, dim, seqlen_kv, topk, topk_mode)
        (1, 128, 8, 512, 160, 64, "random"),
        (1, 256, 16, 512, 320, 128, "random"),
        (1, 256, 64, 512, 320, 128, "random"),
        (2, 128, 8, 512, 160, 64, "random"),
        (1, 512, 8, 512, 640, 256, "random"),
        (1, 512, 8, 512, 640, 256, "random"),
    ]
    for case in cases:
        run_one_bwd_case(*case)


if __name__ == "__main__":
    tilelang.cache.clear_cache()
    os.environ["TILELANG_ASCEND_MODE"] = "Developer"
    run_all_tests()
