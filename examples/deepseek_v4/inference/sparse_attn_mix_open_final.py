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
    # 编译期 tile 参数：
    # - block_top_k：沿 sparse KV token 维度(top_k)的分块大小，例如每次处理 32 个候选 KV。
    # - block_heads：沿 Q 的 num_heads 维度的分块大小，例如每次处理 16 个 Q heads。
    # - num_heads：Q 的注意力头数量；本 kernel 的 KV 没有 head 维度，更接近 MQA/shared-KV。
    # - dim：每个 head 的向量维度，也是 Q/K/V 点积的归约维。
    # - batch_size/seq_len/seq_len_kv/top_k：按 FA_mix 的写法作为编译期静态 shape，
    #   避免在后续 NPUIR stride-align 前丢失 reinterpret_cast/subview 的静态 stride 信息。
    # - multibuffer：mix pass 扩展 workspace 后的 ping-pong buffer 数量，同时也是 pipeline stage 数。
    # - scale：attention score 的缩放因子，默认 1/sqrt(dim)。
    if scale is None:
        scale = (1.0 / dim) ** 0.5

    assert block_heads % 2 == 0, "mix kernel maps one cube block to two vector sub-blocks"
    assert dim % 2 == 0, "mix V1 gather splits KV dim across two vector sub-blocks"

    shape_q = [batch_size, seq_len, num_heads, dim]
    shape_kv = [batch_size, seq_len_kv, dim]
    shape_o = [batch_size, seq_len, num_heads, dim]
    shape_sink = [num_heads]
    shape_topk = [batch_size, seq_len, top_k]

    # mix 模式下一个 Cube tile 负责 block_heads 个 head，两个 Vector 逻辑核
    # 通过 vid 分别处理前/后 block_heads_half 个 head，形成 CV 1:2 的划分。
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
        # 张量语义：
        # - Q[b, s, h, d]：query token s 在 head h 上的向量。
        # - KV[b, skv, d]：被所有 Q heads 共享的 key/value 向量；这里没有 kv_head 维度。
        # - TopKIndices[b, s, t]：query token s 只关注的第 t 个 KV token 下标，-1 表示无效。
        # - AttnSink[h]：每个 Q head 的 sink 项，只加入 softmax 分母，不参与 P*V。
        # - Output[b, s, h, d]：每个 query token、每个 Q head 的输出向量。
        # cid 对应一个 query 位置 (batch, seq)，同一个 cid 下用 vid=0/1
        # 切分 Vector 侧 head 子块；Cube 侧按完整 block_heads 做矩阵乘。
        with T.Kernel(batch_size * seq_len, is_npu=True) as (cid, vid):
            by = cid // seq_len
            bx = cid % seq_len
            value_zero = 0
            value_min = -T.infinity(accum_dtype)

            # Cube 侧本地 tile：q_shared/kv_shared 进入 QK 和 PV 两次 GEMM。
            # kv_shared 在本轮 top-k 块内由第一段 Cube 从 workspace 回灌，
            # 第二段 Cube 直接复用这份 L1 数据，避免重复搬运同一块 KV。
            # EnableMultiBuffer 会把 C1(s0,s1) 和 C2(s0,s1) 拆成两个 stage loop，
            # 因此 kv_shared 必须按 stage 保留，否则 C2(s0) 会读到 C1(s1) 搬入的 KV。
            q_shared = T.alloc_shared((block_heads, dim), dtype)
            kv_shared = T.alloc_shared((block_top_k, dim), dtype, multi_buffer=multibuffer)
            prob_shared = T.alloc_shared((block_heads, block_top_k), dtype)
            scores = T.alloc_fragment((block_heads, block_top_k), accum_dtype)
            scores_cast = T.alloc_shared((block_heads_half, block_top_k), dtype)
            pv_acc = T.alloc_fragment((block_heads, dim), accum_dtype)

            # Vector 侧本地 tile：负责 gather 稀疏 KV、mask、online softmax 和输出累加。
            kv_ub = T.alloc_shared((block_top_k, dim_half), dtype)
            idxs = T.alloc_fragment((block_top_k,), indices_dtype)
            # mask_ub 由 V1 生成并在 V2 使用，是跨 Vector scope 的 per-stage 临时值；
            # 它不是 scores_max/sum_exp/acc_o 这种跨 k 递推状态，因此需要 local multi-buffer。
            mask_ub = T.alloc_shared((1, block_top_k), accum_dtype, multi_buffer=multibuffer)
            scores_ub = T.alloc_shared((block_heads_half, block_top_k), accum_dtype)
            scores_max = T.alloc_shared((block_heads_half, 1), accum_dtype)
            scores_max_prev = T.alloc_shared((block_heads_half, 1), accum_dtype)
            # scores_scale 由 V2 生成并在 V3 用于修正历史 acc_o，语义等价于 FA_mix 的 correction。
            scores_scale = T.alloc_shared(
                (block_heads_half, 1), accum_dtype, multi_buffer=multibuffer
            )
            scores_sum = T.alloc_shared((block_heads_half, 1), accum_dtype)
            sum_exp = T.alloc_shared((block_heads_half, 1), accum_dtype)
            acc_o = T.alloc_shared((block_heads_half, dim), accum_dtype)
            acc_o_new = T.alloc_shared((block_heads_half, dim), accum_dtype)
            o_cast = T.alloc_shared((block_heads_half, dim), dtype)

            # workspace 是 Cube/Vector 边界的 GM 中转区，也是 mix pass 做
            # multi-buffer 扩展和自动 set/wait 同步的锚点。
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

            # 外层按 head block 遍历。这里必须是普通 serial loop，当前 mix pass
            # 会把 T.Pipelined 当成一个 C/V 流水区域，不能嵌套多个 pipeline。
            for n in T.serial(T.ceildiv(num_heads, block_heads)):
                # n 选择当前 Q head 范围：
                # [n * block_heads, min((n + 1) * block_heads, num_heads))。
                # head 维不是 attention 的归约维，不同 head 输出互相独立。
                T.vbrc(value_zero, acc_o)
                T.vbrc(value_zero, sum_exp)
                T.vbrc(value_min, scores_max)
                # 固定 cid 后，Q 的 seq 维已经是当前 bx；这里拷贝的是
                # 当前 query token 在一个 head block 上的 [block_heads, dim]。
                T.copy(Q[by, bx, n * block_heads, 0], q_shared, size=[block_heads, dim])

                # 内层按 top_k 分块做 sparse attention。该 loop 是唯一的 mix
                # pipeline 区域，后续 pass 会围绕它做 multi-buffer 和 C/V 同步。
                for k in T.Pipelined(T.ceildiv(top_k, block_top_k), num_stages=multibuffer):
                    # k 选择当前 sparse KV token 范围：
                    # [k * block_top_k, min((k + 1) * block_top_k, top_k))。
                    # top_k 是每个 query 预选出来的稀疏 KV 序列长度，不是完整 seq_len_kv。
                    real_block_top_k = T.min(top_k - k * block_top_k, block_top_k)

                    # Vector 阶段 1：读取本 query 的 top-k 索引，按索引从 KV
                    # gather 出稀疏 K/V tile；-1 表示无效位置，对应 mask=0。
                    # KV 没有 head 维，因此 V1 使用 vid 沿 dim 维二分：
                    # 两个 AIV 分别搬同一批 top-k KV 的前/后半列，写入
                    # workspace_kv 的不同 dim 区间，避免重复搬完整 KV。
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

                    # Cube 阶段 1：从 workspace 回灌 KV 到 L1，计算
                    # scores = Q * K^T，得到 block_heads x block_top_k 的分数。
                    # 形状上等价于：
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

                    # Vector 阶段 2：每个 vid 只处理一半 heads。这里做在线
                    # softmax：维护历史 max/sum，并把本块概率写回 workspace。
                    T.copy(
                        workspace_score[0,vid * block_heads_half, 0],
                        scores_ub,
                        size=[block_heads_half, block_top_k],
                    )

                    # online softmax 状态：
                    # scores_max/sum_exp/acc_o 保存已经处理过的 top_k 分块结果；
                    # scores_scale 用新旧 max 的差值修正历史分母和历史输出累加。
                    T.copy(scores_max, scores_max_prev)
                    T.vmul(scores_ub, scale, scores_ub)
                    T.reduce_max(scores_ub, scores_max, dim=1)
                    for i, j in T.Parallel(block_heads_half, block_top_k):
                        scores_ub[i, j] = T.exp(scores_ub[i, j] - scores_max[i, 0])
                    for i, j in T.Parallel(block_heads_half, block_top_k):
                        scores_ub[i, j] *= mask_ub[0, j]

                    # P 是跨到 Cube 阶段 2 的数据，优先 cast 并通过 MTE3 写出。
                    # 后续 reduce_sum/sum_exp 更新只在 Vector 内部使用，可与 P 的搬出重叠。
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

                    # Cube 阶段 2：读取两个 Vector 子块写好的概率，计算
                    # P * V，输出完整 block_heads 的本块贡献到 workspace_out。
                    # 形状上等价于：
                    # [block_heads, block_top_k] @ [block_top_k, dim] -> [block_heads, dim]。
                    T.copy(workspace_prob[0,0, 0], prob_shared, size=[block_heads, block_top_k])
                    # kv_shared 已在 QK 阶段从 workspace_kv 回灌，本阶段直接复用。
                    T.gemm(
                        prob_shared,
                        kv_shared,
                        pv_acc,
                        initC=True,
                        size=[block_heads, block_top_k, dim],
                    )
                    T.copy(pv_acc, workspace_out[0,0, 0], size=[block_heads, dim])

                    # Vector 阶段 3：取自己负责的 half-head 输出块，结合
                    # online softmax 的 scale 修正历史累加结果。
                    T.copy(
                        workspace_out[0,vid * block_heads_half, 0],
                        acc_o_new,
                        size=[block_heads_half, dim],
                    )
                    T.vmul(acc_o, scores_scale, acc_o)
                    T.vadd(acc_o, acc_o_new, acc_o)

                # top-k 所有分块结束后，把 attention sink 加进 softmax 分母；
                # sink 只改变归一化分母，不产生 value 项。
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
                # 最终每个 vid 写回自己负责的 half-head 输出。
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
    # Python wrapper 入参：
    # - q: [batch_size, seq_len, num_heads, dim]，Q 的完整多头输入。
    # - kv: [batch_size, seq_len_kv, dim]，共享 KV；没有 kv_head 维度，因此是 MQA/shared-KV 形态。
    # - attn_sink: [num_heads]，每个 Q head 一个 sink 标量。
    # - topk_idxs: [batch_size, seq_len, top_k]，每个 query token 对应的稀疏 KV 下标。
    # - softmax_scale: score 缩放因子，通常是 1/sqrt(dim)。
    block = 64
    # block_heads = 64
    multibuffer = 2
    # block 别名对应 kernel 里的 block_top_k；block_heads 对应 Q num_heads 维分块。
    # 当前配置下，Cube 每次处理 16 个 Q heads x 32 个 sparse KV tokens，
    # Vector 侧两个 vid 分别处理 8 个 Q heads。
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
        # Expert/API lowering 才会把 T.alloc_workspace 降成 memref_ext.alloc_workspace，
        # 后续 mix pass 依赖它识别 Cube/Vector 边界和 multi-buffer workspace。
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
