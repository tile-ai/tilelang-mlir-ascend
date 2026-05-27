# Copyright (c) Huawei Technologies Co., Ltd. 2026.
#
# Port of upstream tile-ai/tilelang examples/deepseek_v32/sparse_mla_fwd.py to
# Ascend NPU via mlir-ascend. Used by miles (radixark/miles) at
# miles_plugins/models/glm5/ops/tilelang_sparse_mla_fwd.py.
#
# This is the static-shape baseline. Dynamic shapes (T.dynamic) follow in a
# sibling file once this proves out end-to-end.
#
# Notable adaptations vs upstream:
#   * 3-axis grid (seq_len * REPLICATE_H, batch, kv_group) collapsed into 1 axis
#     -- is_npu=True requires exactly one block dimension. We decode inside.
#   * b_transpose= / NPU vector intrinsics (vbrc/vmul/vexp/...) instead of
#     T.Parallel elementwise loops where possible.
#   * Single-pass online softmax matches the flash-attention template that
#     already passes on this backend.
#   * REPLICATE_H == 1; H_per_block == padded_head_kv. Multi-replication will
#     be added once correctness is established here.
#
# Bisect lessons baked in (T33.P1.3):
#   * vbrc literal `0` errors at the rank check; assign `value_zero = 0` first
#     so TVM wraps it in a tir.Var (subclass of PrimExpr) and the check is
#     bypassed correctly.
#   * Scalar factory-closure values (e.g. sm_scale) need a local copy inside
#     the prim_func body to be captured cleanly by the parser.
import os
import torch
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-2, -1], target="npuir")
def sparse_mla_fwd(
    batch,
    seq_len,
    seq_len_kv,
    heads,
    dim,
    tail_dim,
    topk,
    block_M=None,
    block_N=64,
    num_stages=2,
):
    """Sparse MLA forward kernel.

    Args correspond to upstream sparse_mla_fwd; kv_group is fixed at 1 here.
    block_M defaults to heads (one Q-block per head group, matching upstream
    H_per_block when REPLICATE_H==1).
    """
    if block_M is None:
        block_M = heads
    D = dim
    DT = tail_dim
    dtype = "float16"
    accum_dtype = "float32"
    idx_dtype = "int32"
    sm_scale = (1.0 / (D + DT)) ** 0.5

    q_shape = [batch, seq_len, heads, D + DT]
    kv_shape = [batch, seq_len_kv, 1, D + DT]
    o_shape = [batch, seq_len, heads, D]
    idx_shape = [batch, seq_len, 1, topk]
    lse_shape = [
        batch,
        seq_len,
        heads,
        1,
    ]  # trailing 1 keeps rank parity with [BM,1] fragment

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        KV: T.Tensor(kv_shape, dtype),
        Indices: T.Tensor(idx_shape, idx_dtype),
        Output: T.Tensor(o_shape, dtype),
        Lse: T.Tensor(lse_shape, accum_dtype),
    ):
        with T.Kernel(batch * seq_len, is_npu=True) as (cid, _):
            b_i = cid // seq_len
            s_i = cid % seq_len

            Q_shared = T.alloc_shared([block_M, D], dtype)
            Q_tail_shared = T.alloc_shared([block_M, DT], dtype)
            KV_shared = T.alloc_shared([block_N, D], dtype)
            K_tail_shared = T.alloc_shared([block_N, DT], dtype)

            scores = T.alloc_fragment([block_M, block_N], accum_dtype)
            scores_cast = T.alloc_fragment([block_M, block_N], dtype)
            correction = T.alloc_fragment([block_M, 1], accum_dtype)
            local_max = T.alloc_fragment([block_M, 1], accum_dtype)
            local_sum = T.alloc_fragment([block_M, 1], accum_dtype)
            acc_m = T.alloc_fragment([block_M, 1], accum_dtype)
            acc_l = T.alloc_fragment([block_M, 1], accum_dtype)
            acc_o = T.alloc_fragment([block_M, D], accum_dtype)
            tmp = T.alloc_fragment([block_M, block_N], accum_dtype)
            tmp1 = T.alloc_fragment([block_M, 1], accum_dtype)
            new_max = T.alloc_fragment([block_M, 1], accum_dtype)
            scales = T.alloc_fragment([block_M, block_N], accum_dtype)
            idx_buf = T.alloc_fragment([block_N], idx_dtype)

            local_sm_scale = sm_scale
            value_zero = 0
            value_min = -T.infinity(accum_dtype)
            T.vbrc(value_zero, acc_o)
            T.vbrc(value_zero, acc_l)
            T.vbrc(value_min, acc_m)
            T.vbrc(local_sm_scale, scales)

            T.copy(Q[b_i, s_i, 0:block_M, 0:D], Q_shared)
            T.copy(Q[b_i, s_i, 0:block_M, D : D + DT], Q_tail_shared)

            for k in T.Pipelined(T.ceildiv(topk, block_N), num_stages=num_stages):
                T.copy(Indices[b_i, s_i, 0, k * block_N], idx_buf)
                for bi_i in T.serial(block_N):
                    cur_idx = idx_buf[bi_i]
                    T.copy(KV[b_i, cur_idx, 0, 0:D], KV_shared[bi_i, 0:D])
                    T.copy(KV[b_i, cur_idx, 0, D : D + DT], K_tail_shared[bi_i, 0:DT])

                T.gemm(Q_shared, KV_shared, scores, initC=True, b_transpose=True)
                T.gemm(
                    Q_tail_shared, K_tail_shared, scores, initC=False, b_transpose=True
                )

                T.vmul(scores, scales, scores)
                T.reduce_max(scores, local_max, dim=1)
                T.vmax(acc_m, local_max, new_max)
                T.vsub(acc_m, new_max, tmp1)
                T.vexp(tmp1, correction)
                T.vsub(scores, new_max, tmp)
                T.vexp(tmp, scores)
                T.reduce_sum(scores, local_sum, dim=1)
                T.vmul(acc_l, correction, acc_l)
                T.vadd(acc_l, local_sum, acc_l)
                T.vmul(acc_o, correction, acc_o)
                T.vcast(scores, scores_cast, round_mode="rint")
                T.vbrc(value_zero, tmp1)
                T.vadd(tmp1, new_max, acc_m)
                T.gemm(scores_cast, KV_shared, acc_o, initC=False)

            T.vdiv(acc_o, acc_l, acc_o)
            O_cast = T.alloc_shared([block_M, D], dtype)
            T.vcast(acc_o, O_cast, round_mode="rint")
            T.copy(O_cast, Output[b_i, s_i, 0:block_M, 0:D])

            # Lse for bwd: log(acc_l) + acc_m. Kept as [BM,1] to match fragments;
            # caller squeezes the trailing 1 to recover [B,S,H].
            Lse_shared = T.alloc_shared([block_M, 1], accum_dtype)
            tmp_lse = T.alloc_fragment([block_M, 1], accum_dtype)
            T.vln(acc_l, tmp_lse)
            T.vadd(tmp_lse, acc_m, tmp_lse)
            T.copy(tmp_lse, Lse_shared)
            T.copy(Lse_shared, Lse[b_i, s_i, 0:block_M, 0:1])

    return main


def _ref_torch(q, kv, indices, sm_scale=None):
    """Reference computed in fp32 on the same device."""
    qf = q.float()
    kvf = kv.float()
    B, S, H, DQK = q.shape
    _, SKV, G, _ = kv.shape  # G == 1 here
    assert G == 1
    _, _, _, topk = indices.shape
    # We follow upstream convention: q has dim_qk = D + DT, output uses first D.
    if sm_scale is None:
        sm_scale = (1.0 / DQK) ** 0.5
    k_full = kvf  # (B, SKV, 1, DQK)
    out = torch.zeros(B, S, H, q.shape[-1], dtype=torch.float32, device=q.device)
    for b in range(B):
        for s in range(S):
            idxs = indices[b, s, 0].long()
            kg = k_full[b, idxs, 0, :]  # (topk, DQK)
            qi = qf[b, s]  # (H, DQK)
            scores = (qi @ kg.transpose(0, 1)) * sm_scale  # (H, topk)
            mask = torch.softmax(scores, dim=-1)
            out[b, s] = mask @ kg  # (H, DQK) — caller slices the first D
    return out


def test_sparse_mla_fwd_small():
    torch.npu.set_device(0)
    B, S, SKV, H = 1, 8, 16, 16
    D, DT = 64, 16
    topk = 8
    BM, BN = H, topk

    torch.manual_seed(0)
    q = torch.randn(B, S, H, D + DT, dtype=torch.float16, device="npu") * 0.5
    kv = torch.randn(B, SKV, 1, D + DT, dtype=torch.float16, device="npu") * 0.5
    indices = torch.zeros(B, S, 1, topk, dtype=torch.int32, device="npu")
    for s in range(S):
        # only attend to past kv positions (causal-style); fall back to 0 when none.
        avail = max(1, s + 1)
        perm = torch.randperm(min(SKV, avail))[:topk]
        if len(perm) < topk:
            perm = torch.cat([perm, torch.zeros(topk - len(perm), dtype=torch.long)])
        indices[0, s, 0, :] = perm.to(torch.int32)

    print(
        f"compile sparse_mla_fwd(B={B},S={S},SKV={SKV},H={H},D={D},DT={DT},topk={topk}) ..."
    )
    kernel = sparse_mla_fwd(B, S, SKV, H, D, DT, topk, block_M=BM, block_N=BN)
    print("compile OK; running on NPU ...")
    out, lse = kernel(q, kv, indices)
    print("run OK; out shape:", tuple(out.shape), "lse shape:", tuple(lse.shape))
    print("out[0,0,0,:4] =", out[0, 0, 0, :4].cpu().tolist())
    print("lse[0,0,:4]   =", lse[0, 0, :4].cpu().tolist())

    # crude correctness via fp32 ref on cpu (skip strict assert for first pass)
    q_cpu = q.cpu()
    kv_cpu = kv.cpu()
    indices_cpu = indices.cpu()
    ref_out_cpu = _ref_torch(q_cpu, kv_cpu, indices_cpu)
    print("ref_out[0,0,0,:4] =", ref_out_cpu[0, 0, 0, :4].tolist())
    abs_err = (out.cpu().float() - ref_out_cpu[..., :D]).abs().max().item()
    print(f"max abs err vs cpu ref: {abs_err:.4f}")
    # tolerance from manual NPU runs: max abs err ~5e-4 vs fp32 cpu ref
    assert abs_err < 5e-3, f"sparse_mla_fwd accuracy regressed: {abs_err}"
    print("sparse_mla_fwd PASS")


if __name__ == "__main__":
    os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
    test_sparse_mla_fwd_small()
