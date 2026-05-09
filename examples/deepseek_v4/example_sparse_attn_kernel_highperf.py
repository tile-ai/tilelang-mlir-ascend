# Copyright (c) Huawei Technologies Co., Ltd. 2025.
# Import necessary libraries
import torch

# Import tilelang modules for NPU kernel development
import tilelang
import tilelang.language as T

from typing import Optional


@tilelang.jit(
    target="npuir",
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION: False,
        tilelang.PassConfigKey.NPUIR_ENABLE_AUTO_MULTI_BUFFER: False,
    },
)
def sparse_attn_kernel(
    batch,
    heads,
    dim,
    top_k,
    num_kernels,
    sm_scale=None,
    block=64,
    multi_ws_kv=1,
    multi_ws_s=1,
    multi_ws_p=1,
    multi_ws_o=1,
):
    # Validate input parameters
    assert block % 2 == 0, "Block size must be even"
    assert dim == tilelang.math.next_power_of_2(dim), (
        f"haven't check padding correctness yet, dim={dim}"
    )

    seq_len = T.symbolic("seqLen")
    seq_len_kv = T.symbolic("seqLenKV")

    # Set softmax scale if not provided
    if sm_scale is None:
        sm_scale = (1.0 / dim) ** 0.5

    multi_l1_kv = 2
    multi_ub_inner_cross = 2

    # ws_kv: V0S -> C1L -> ...
    flag_base_V0S_C1L_kv = 0

    # ws_s: C1S -> V1L -> ...
    flag_base_C1S_V1L_s = 0

    # ws_p: V1S -> C2L -> ...
    flag_base_V1S_C2L_p = flag_base_V0S_C1L_kv + multi_ws_kv

    # ws_o: C2S -> V2L -> ...
    flag_base_C2S_V2L_o = flag_base_C1S_V1L_s + multi_ws_s

    num_physical_kernels = 24
    num_logic_kernels = batch * seq_len
    num_top_k_blocks = (top_k - 1) // block + 1

    # Define data types
    indices_dtype = "int32"
    dtype = "bfloat16"
    accum_dtype = "float"

    # Set block size for head dimension
    heads_half = heads // 2

    # Calculate half block sizes for vector operations
    block_half = block // 2

    block_dim_share = max(block, dim)
    block_heads_shared_half = max(block_half, heads_half)

    @T.macro
    def C1L(Q, workspace_kv, l1_q, l1_kv_sparse, kernel_id, task_id):
        local_id = task_id // num_top_k_blocks
        block_i_id = task_id % num_top_k_blocks

        # Calculate batch and sequence indices
        logic_kernel_id = local_id * num_kernels + kernel_id
        batch_id = logic_kernel_id // seq_len
        seq_id = logic_kernel_id % seq_len

        with T.rs("PIPE_MTE2"):
            if block_i_id == 0:
                # Load query data to L1
                T.copy(Q[batch_id, seq_id, :, :], l1_q)

            flag_V0S_C1L_kv = flag_base_V0S_C1L_kv + task_id % multi_ws_kv
            T.sync_block_wait(flag_V0S_C1L_kv)
            # Load sparse KV data to L1
            T.copy(
                workspace_kv[kernel_id, task_id % multi_ws_kv, :, :],
                l1_kv_sparse[task_id % multi_l1_kv, :, :],
            )

    @T.macro
    def C1P(
        l1_q,
        l1_kv_sparse,
        l0_c,
        kernel_id,
        task_id,
    ):
        with T.rs("PIPE_C"):
            # Perform matrix multiplication (Q @ K^T)
            T.gemm(
                l1_q[:, :],
                l1_kv_sparse[task_id % multi_l1_kv, :, :],
                l0_c[:, :block],
                initC=True,
                b_transpose=True,
            )

    @T.macro
    def C1S(
        workspace_s,
        l0_c,
        kernel_id,
        task_id,
    ):
        # Store intermediate results and synchronize
        with T.rs("PIPE_FIX"):
            T.copy(l0_c[:, :block], workspace_s[kernel_id, task_id % multi_ws_s, :, :])
            flag_C1S_V1L_s = flag_base_C1S_V1L_s + task_id % multi_ws_s
            T.sync_block_set(flag_C1S_V1L_s)

    @T.macro
    def C2L(
        workspace_p,
        l1_p,
        kernel_id,
        task_id,
    ):
        # Load intermediate results for softmax
        with T.rs("PIPE_MTE2"):
            flag_V1S_C2L_p = flag_base_V1S_C2L_p + task_id % multi_ws_p
            T.sync_block_wait(flag_V1S_C2L_p)
            T.copy(workspace_p[kernel_id, task_id % multi_ws_p, :, :], l1_p)

    @T.macro
    def C2P(
        l1_p,
        l1_kv_sparse,
        l0_c,
        kernel_id,
        task_id,
    ):
        with T.rs("PIPE_C"):
            # Perform matrix multiplication (P @ V)
            T.gemm(
                l1_p[:, :],
                l1_kv_sparse[task_id % multi_l1_kv, :, :],
                l0_c[:, :dim],
                initC=True,
            )

    @T.macro
    def C2S(
        workspace_o,
        l0_c,
        kernel_id,
        task_id,
    ):
        # Synchronize after output computation
        with T.rs("PIPE_FIX"):
            T.copy(l0_c[:, :dim], workspace_o[kernel_id, task_id % multi_ws_o, :, :])
            flag_C2S_V2L_o = flag_base_C2S_V2L_o + task_id % multi_ws_o
            T.sync_block_set(flag_C2S_V2L_o)

    @T.macro
    def V0L(
        KV,
        Indices,
        ub_indices,
        ub_kv_sparse,
        kernel_id,
        vid,
        task_id,
    ):
        value_zero = 0

        local_id = task_id // num_top_k_blocks
        block_i_id = task_id % num_top_k_blocks

        logic_kernel_id = local_id * num_kernels + kernel_id
        batch_id = logic_kernel_id // seq_len
        seq_id = logic_kernel_id % seq_len

        tail_size_i = T.min(top_k - block_i_id * block, block)
        tail_size_i_half = (tail_size_i + 1) // 2

        block_i_offset = block_i_id * block
        block_i_sub_offset = vid * tail_size_i_half
        tail_size_i_half = tail_size_i_half - (tail_size_i % 2) * vid

        with T.rs("PIPE_V"):
            if tail_size_i_half > 0:
                T.vbrc(value_zero, ub_kv_sparse[:block_half, :dim])

        with T.rs("PIPE_MTE2"):
            # Load indices and gather sparse KV data (only for first head block)
            T.copy(
                Indices[batch_id, seq_id, block_i_offset : block_i_offset + block],
                ub_indices[task_id % multi_ub_inner_cross, 0, :],
            )
            for idx_id in T.serial(T.max(tail_size_i_half, 0)):
                current_index = ub_indices[
                    task_id % multi_ub_inner_cross, 0, block_i_sub_offset + idx_id
                ]
                if current_index >= 0 and current_index < seq_len_kv:
                    T.copy(KV[batch_id, current_index, :], ub_kv_sparse[idx_id, :dim])

    @T.macro
    def V0S(
        workspace_kv,
        ub_kv_sparse,
        kernel_id,
        vid,
        task_id,
    ):
        block_i_id = task_id % num_top_k_blocks

        tail_size_i = T.min(top_k - block_i_id * block, block)
        tail_size_i_half = (tail_size_i + 1) // 2
        block_i_sub_offset = vid * tail_size_i_half
        tail_size_i_half = tail_size_i_half - (tail_size_i % 2) * vid

        # Synchronize after KV gathering
        with T.rs("PIPE_MTE3"):
            if tail_size_i_half > 0:
                T.copy(
                    ub_kv_sparse[:tail_size_i_half, :dim],
                    workspace_kv[
                        kernel_id,
                        task_id % multi_ws_kv,
                        block_i_sub_offset : block_i_sub_offset + tail_size_i_half,
                        :,
                    ],
                )
            flag_V0S_C1L_kv = flag_base_V0S_C1L_kv + task_id % multi_ws_kv
            T.sync_block_set(flag_V0S_C1L_kv)

    @T.macro
    def V1L(
        workspace_s,
        ub_cross_kernel_32,
        kernel_id,
        vid,
        task_id,
    ):
        sub_offset = vid * heads_half

        # Load attention scores from workspace
        with T.rs("PIPE_MTE2"):
            flag_C1S_V1L_s = flag_base_C1S_V1L_s + task_id % multi_ws_s
            T.sync_block_wait(flag_C1S_V1L_s)
            T.copy(
                workspace_s[
                    kernel_id,
                    task_id % multi_ws_s,
                    sub_offset : sub_offset + heads_half,
                    :block,
                ],
                ub_cross_kernel_32[:heads_half, :block],
            )

    @T.macro
    def V1P(
        # Global UB
        ub_attn_sink,
        ub_var_scores_max,
        ub_var_scores_scale,
        ub_indices,
        ub_arrange_mask,
        ub_attn_sink_tmp,
        # Local UB
        ub_var_scores_max_prev,
        ub_var_scores_sum,
        ub_cross_kernel_16,
        ub_cross_kernel_32,
        ub_var_valid_indices,
        ub_var_valid_mask,
        ub_var_valid_indices_32,
        # Offset Info
        kernel_id,
        vid,
        task_id,
    ):
        acc_s_scale = sm_scale
        value_zero = 0
        value_ignore_index = -1

        block_i_id = task_id % num_top_k_blocks
        tail_size_i = T.min(top_k - block_i_id * block, block)

        with T.rs("PIPE_V"):
            # Save previous max scores for numerical stability
            T.copy(ub_var_scores_max, ub_var_scores_max_prev)

            # Apply softmax scaling and compute max
            T.vmul(
                ub_cross_kernel_32[:, :block],
                acc_s_scale,
                ub_cross_kernel_32[:, :block],
            )
            T.reduce(
                ub_cross_kernel_32[:, :block],
                ub_var_scores_max,
                dims=[1],
                reduce_mode="max",
            )

            # Update max scores and compute scaling factors
            if block_i_id != 0:
                T.vmax(ub_var_scores_max_prev, ub_var_scores_max, ub_var_scores_max)
                T.vsub(
                    ub_var_scores_max_prev,
                    ub_var_scores_max,
                    ub_var_scores_scale[task_id % multi_ub_inner_cross, :, :],
                )
                T.vexp(
                    ub_var_scores_scale[task_id % multi_ub_inner_cross, :, :],
                    ub_var_scores_scale[task_id % multi_ub_inner_cross, :, :],
                )
            else:
                T.vbrc(
                    value_zero,
                    ub_var_scores_scale[task_id % multi_ub_inner_cross, :, :],
                )

            # Apply softmax stabilization and compute exponentials
            T.vsub(
                ub_cross_kernel_32[:, :block],
                ub_var_scores_max,
                ub_cross_kernel_32[:, :block],
            )
            T.vexp(ub_cross_kernel_32[:, :block], ub_cross_kernel_32[:, :block])

            # Create valid mask for incomplete blocks
            T.vcmp(
                ub_indices[task_id % multi_ub_inner_cross, :, :],
                value_ignore_index,
                ub_var_valid_indices,
                "ne",
            )
            if tail_size_i < block:
                tail_size_i = tail_size_i
                T.vcmp(ub_arrange_mask, tail_size_i, ub_var_valid_mask, "lt")
                T.vand(ub_var_valid_indices, ub_var_valid_mask, ub_var_valid_indices)

            # Apply valid mask
            T.vcast(ub_var_valid_indices, ub_var_valid_indices_32)
            T.vmul(
                ub_cross_kernel_32[:heads_half, :block],
                ub_var_valid_indices_32,
                ub_cross_kernel_32[:heads_half, :block],
            )
            T.vcast(
                ub_cross_kernel_32[:heads_half, :block],
                ub_cross_kernel_16[:heads_half, :block],
                round_mode="rint",
            )

    @T.macro
    def V1S(
        workspace_p,
        ub_cross_kernel_16,
        kernel_id,
        vid,
        task_id,
    ):
        sub_offset = vid * heads_half

        # Store softmax results and synchronize
        with T.rs("PIPE_MTE3"):
            T.copy(
                ub_cross_kernel_16[:heads_half, :block],
                workspace_p[
                    kernel_id,
                    task_id % multi_ws_p,
                    sub_offset : sub_offset + heads_half,
                    :,
                ],
            )
            flag_V1S_C2L_p = flag_base_V1S_C2L_p + task_id % multi_ws_p
            T.sync_block_set(flag_V1S_C2L_p)

    @T.macro
    def V1P_post(
        ub_attn_sink,
        ub_var_scores_max,
        ub_attn_sink_tmp,
        ub_var_scores_sum,
        ub_cross_kernel_32,
        kernel_id,
        vid,
        task_id,
    ):
        block_i_id = task_id % num_top_k_blocks

        # Compute sum of exponential for softmax denominator
        offset_tmp = (task_id % multi_ub_inner_cross) * heads_half
        T.reduce(
            ub_cross_kernel_32[:heads_half, :block],
            ub_var_scores_sum[offset_tmp : offset_tmp + heads_half, :],
            dims=[1],
            reduce_mode="sum",
        )

        if block_i_id == num_top_k_blocks - 1:
            T.vsub(ub_attn_sink, ub_var_scores_max, ub_attn_sink_tmp)
            T.vexp(ub_attn_sink_tmp, ub_attn_sink_tmp)

    @T.macro
    def V2L(
        workspace_o,
        ub_acc_o_new,
        kernel_id,
        vid,
        task_id,
    ):
        # Load and accumulate output values
        with T.rs("PIPE_MTE2"):
            flag_C2S_V2L_o = flag_base_C2S_V2L_o + task_id % multi_ws_o
            T.sync_block_wait(flag_C2S_V2L_o)
            T.copy(
                workspace_o[kernel_id, task_id % multi_ws_o, vid * heads_half, 0],
                ub_acc_o_new,
                size=[heads_half, dim],
            )

    @T.macro
    def V2P(
        ub_acc_o,
        ub_acc_o_new,
        ub_var_scores_scale,
        ub_cross_kernel_16,
        ub_var_logsum,
        ub_var_scores_sum,
        ub_attn_sink_tmp,
        kernel_id,
        vid,
        task_id,
    ):
        value_zero = 0

        block_i_id = task_id % num_top_k_blocks

        with T.rs("PIPE_V"):
            if block_i_id == 0:
                # Initialize softmax variables
                T.vbrc(value_zero, ub_var_logsum)
                T.vbrc(value_zero, ub_acc_o)

            # Update logsum and accumulate output
            T.vmul(
                ub_var_logsum,
                ub_var_scores_scale[task_id % multi_ub_inner_cross, :, :],
                ub_var_logsum,
            )
            offset_tmp = (task_id % multi_ub_inner_cross) * heads_half
            T.vadd(
                ub_var_logsum,
                ub_var_scores_sum[offset_tmp : offset_tmp + heads_half, :],
                ub_var_logsum,
            )

            T.vmul(
                ub_acc_o,
                ub_var_scores_scale[task_id % multi_ub_inner_cross, :, :],
                ub_acc_o,
            )
            T.vadd(ub_acc_o, ub_acc_o_new, ub_acc_o)

            if block_i_id == num_top_k_blocks - 1:
                # Normalize output by softmax denominator
                T.vadd(ub_var_logsum, ub_attn_sink_tmp, ub_var_logsum)
                T.vdiv(ub_acc_o, ub_var_logsum, ub_acc_o)
                T.vcast(
                    ub_acc_o, ub_cross_kernel_16[:heads_half, :dim], round_mode="rint"
                )

    @T.macro
    def V2S(
        Output,
        ub_cross_kernel_16,
        kernel_id,
        vid,
        task_id,
    ):
        local_id = task_id // num_top_k_blocks
        block_i_id = task_id % num_top_k_blocks

        logic_kernel_id = local_id * num_kernels + kernel_id
        batch_id = logic_kernel_id // seq_len
        seq_id = logic_kernel_id % seq_len

        sub_offset = vid * heads_half

        with T.rs("PIPE_MTE3"):
            if block_i_id == num_top_k_blocks - 1:
                T.copy(
                    ub_cross_kernel_16[:heads_half, :dim],
                    Output[batch_id, seq_id, sub_offset : sub_offset + heads_half, :],
                )

    # Define the main sparse attention kernel using TileLang
    @T.prim_func
    def SparseAttnExp(
        Q: T.Tensor([batch, seq_len, heads, dim], dtype),  # type: ignore
        KV: T.Tensor([batch, seq_len_kv, dim], dtype),  # type: ignore
        Output: T.Tensor([batch, seq_len, heads, dim], dtype),  # type: ignore
        attn_sink: T.Tensor([heads], accum_dtype),
        Indices: T.Tensor([batch, seq_len, top_k], indices_dtype),  # type: ignore
        workspace_kv: T.Tensor([num_kernels, multi_ws_kv, block, dim], dtype),
        workspace_s: T.Tensor([num_kernels, multi_ws_s, heads, block], accum_dtype),
        workspace_p: T.Tensor([num_kernels, multi_ws_p, heads, block], dtype),
        workspace_o: T.Tensor([num_kernels, multi_ws_o, heads, dim], accum_dtype),
    ):
        # Launch NPU kernel with specified number of parallel kernels
        with T.Kernel(num_physical_kernels, is_npu=True) as (kernel_id, vid):
            # Cube computation section (matrix operations)
            with T.Scope("Cube"):
                # Allocate L1 buffers for cube operations
                l1_q = T.alloc_L1([heads, dim], dtype)
                l1_p = T.alloc_L1([heads, block], dtype)
                l1_kv_sparse = T.alloc_L1([multi_l1_kv, block, dim], dtype)

                # Allocate L0 buffer for accumulation
                l0_c = T.alloc_L0C([heads, dim], accum_dtype)

                num_local_logic_kernels = T.ceildiv(
                    num_logic_kernels - kernel_id, num_kernels
                )
                num_tasks = num_local_logic_kernels * num_top_k_blocks

                for stream_id in T.serial(num_tasks + 1):
                    if stream_id < num_tasks:
                        task_id = stream_id
                        C1L(Q, workspace_kv, l1_q, l1_kv_sparse, kernel_id, task_id)
                        C1P(l1_q, l1_kv_sparse, l0_c, kernel_id, task_id)
                        C1S(workspace_s, l0_c, kernel_id, task_id)

                    if stream_id > 0:
                        task_id = stream_id - 1
                        C2L(workspace_p, l1_p, kernel_id, task_id)
                        C2P(l1_p, l1_kv_sparse, l0_c, kernel_id, task_id)
                        C2S(workspace_o, l0_c, kernel_id, task_id)

            # Vector computation section (softmax and normalization)
            with T.Scope("Vector"):
                # Allocate unified buffers for vector operations
                ub_acc_o = T.alloc_ub([heads_half, dim], accum_dtype)
                ub_attn_sink = T.alloc_ub([heads_half, 1], accum_dtype)

                ub_cross_kernel_16 = T.alloc_ub(
                    [block_heads_shared_half, block_dim_share], dtype
                )
                ub_cross_kernel_32 = T.alloc_ub(
                    [heads_half, block_dim_share], accum_dtype
                )

                # ub only used in V1P
                ub_var_scores_max_prev = T.alloc_ub([heads_half, 1], accum_dtype)
                ub_var_scores_max = T.alloc_ub([heads_half, 1], accum_dtype)
                ub_var_valid_indices = T.alloc_ub([1, block], "bool")
                ub_var_valid_mask = T.alloc_ub([1, block], "bool")
                ub_var_valid_indices_32 = T.alloc_ub([1, block], accum_dtype)

                # ub only used in V2P
                ub_var_logsum = T.alloc_ub([heads_half, 1], accum_dtype)

                # inner cross
                ub_indices = T.alloc_ub([multi_ub_inner_cross, 1, block], indices_dtype)
                ub_var_scores_sum = T.alloc_ub(
                    [multi_ub_inner_cross * heads_half, 1], accum_dtype
                )
                ub_var_scores_scale = T.alloc_ub(
                    [multi_ub_inner_cross, heads_half, 1], accum_dtype
                )

                # outer loop inner cross
                ub_attn_sink_tmp = T.alloc_ub([heads_half, 1], accum_dtype)

                ub_arrange_mask = T.alloc_ub([1, block], "int16")
                offset_heads_sub = vid * heads_half
                with T.rs("PIPE_MTE2"):
                    T.copy(
                        attn_sink[offset_heads_sub : offset_heads_sub + heads_half],
                        ub_attn_sink[:, 0],
                    )
                with T.rs("PIPE_V"):
                    T.arange(ub_arrange_mask, [0, 1], 0)

                num_local_logic_kernels = T.ceildiv(
                    num_logic_kernels - kernel_id, num_kernels
                )
                num_tasks = num_local_logic_kernels * num_top_k_blocks

                for stream_id in T.serial(num_tasks + 2):
                    if stream_id < num_tasks:
                        task_id = stream_id
                        V0L(
                            KV,
                            Indices,
                            ub_indices,
                            ub_cross_kernel_16,
                            kernel_id,
                            vid,
                            task_id,
                        )
                        V0S(workspace_kv, ub_cross_kernel_16, kernel_id, vid, task_id)

                    if stream_id > 0 and stream_id - 1 < num_tasks:
                        task_id = stream_id - 1
                        V1L(workspace_s, ub_cross_kernel_32, kernel_id, vid, task_id)
                        V1P(
                            # Global UB
                            ub_attn_sink,
                            ub_var_scores_max,
                            ub_var_scores_scale,
                            ub_indices,
                            ub_arrange_mask,
                            ub_attn_sink_tmp,
                            # Local UB
                            ub_var_scores_max_prev,
                            ub_var_scores_sum,
                            ub_cross_kernel_16,
                            ub_cross_kernel_32,
                            ub_var_valid_indices,
                            ub_var_valid_mask,
                            ub_var_valid_indices_32,
                            # Offset Info
                            kernel_id,
                            vid,
                            task_id,
                        )
                        V1S(workspace_p, ub_cross_kernel_16, kernel_id, vid, task_id)
                        V1P_post(
                            ub_attn_sink,
                            ub_var_scores_max,
                            ub_attn_sink_tmp,
                            ub_var_scores_sum,
                            ub_cross_kernel_32,
                            kernel_id,
                            vid,
                            task_id,
                        )

                    if stream_id > 1:
                        task_id = stream_id - 2
                        V2L(workspace_o, ub_cross_kernel_32, kernel_id, vid, task_id)
                        V2P(
                            ub_acc_o,
                            ub_cross_kernel_32,
                            ub_var_scores_scale,
                            ub_cross_kernel_16,
                            ub_var_logsum,
                            ub_var_scores_sum,
                            ub_attn_sink_tmp,
                            kernel_id,
                            vid,
                            task_id,
                        )
                        V2S(Output, ub_cross_kernel_16, kernel_id, vid, task_id)

    return SparseAttnExp


def sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: Optional[float] = None,
):
    num_kernels = 24
    block = 64
    multi_ws_kv = 2
    multi_ws_s = 2
    multi_ws_p = 2
    multi_ws_o = 2
    batch, seq_len, heads, dim = q.size()
    _, _, top_k = topk_idxs.size()

    sparse_attn.kernel = sparse_attn_kernel(
        batch,
        heads,
        dim,
        top_k,
        num_kernels,
        softmax_scale,
        block,
        multi_ws_kv,
        multi_ws_s,
        multi_ws_p,
        multi_ws_o,
    )
    output = torch.empty((batch, seq_len, heads, dim), dtype=q.dtype, device=q.device)
    w_kv = torch.empty(
        (num_kernels, multi_ws_kv, block, dim), dtype=q.dtype, device=q.device
    )
    w_s = torch.empty(
        (num_kernels, multi_ws_s, heads, block), dtype=attn_sink.dtype, device=q.device
    )
    w_p = torch.empty(
        (num_kernels, multi_ws_p, heads, block), dtype=q.dtype, device=q.device
    )
    w_o = torch.empty(
        (num_kernels, multi_ws_o, heads, dim), dtype=attn_sink.dtype, device=q.device
    )
    sparse_attn.kernel(q, kv, output, attn_sink, topk_idxs, w_kv, w_s, w_p, w_o)
    torch.npu.synchronize()
    return output


def gather_from_kv(KV, indices):
    """Gather key-value pairs using indices for reference implementation"""
    b, s1, k = indices.shape
    batch_idx = torch.arange(b, device=KV.device).view(b, 1, 1).expand(-1, s1, k)
    indices_flat = indices.long()
    out = KV[batch_idx, indices_flat, :].squeeze(dim=2)
    return out


def softmax_with_sink(x: torch.Tensor, attn_sink: torch.Tensor, head_dim, dim=-1):
    max_vals = torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x - max_vals)
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)

    sink_view_shape = [1] * x.dim()
    sink_view_shape[head_dim if head_dim > 0 else head_dim % x.dim()] = x.shape[
        head_dim
    ]

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
    """Reference Sparse Attention kernel implemented in PyTorch"""
    kv_sparse = gather_from_kv(kv, topk_idxs)
    mask_acc_s = torch.where((topk_idxs == -1).unsqueeze(-2), -torch.inf, 0.0)
    mask_acc_s = mask_acc_s.to(device=q.device, dtype=torch.float32)
    ref_output = (
        softmax_with_sink(
            ((q @ kv_sparse.transpose(-2, -1)).to(attn_sink.dtype) + mask_acc_s)
            * softmax_scale,
            attn_sink,
            head_dim=-2,
            dim=-1,
        ).to(q.dtype)
        @ kv_sparse
    )

    return ref_output


def rand_sparse_attn_input(
    batch_size, num_heads, seq_len, seq_len_kv, top_k, dim, seed=88888888, causal=True
):
    """Generate legalized random inputs for Sparse Attention"""
    torch.manual_seed(seed)
    base_dtype = torch.bfloat16

    # Generate inputs
    q = torch.randn((batch_size, seq_len, num_heads, dim), dtype=base_dtype).npu()
    kv = torch.randn((batch_size, seq_len_kv, dim), dtype=base_dtype).npu()
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


TEST_CASES = [
    {
        "case_id": 0,
        "params": {
            "batch_size": 1,
            "num_heads": 64,
            "seq_len": 4096,
            "seq_len_kv": 4096,
            "top_k": 128,
            "dim": 512,
        },
    },
    {
        "case_id": 1,
        "params": {
            "batch_size": 1,
            "num_heads": 64,
            "seq_len": 2999,
            "seq_len_kv": 4096,
            "top_k": 128,
            "dim": 512,
        },
    },
]


def run_test(verify_acc=True):
    """
    Run tests for all cases defined in TEST_CASES.
    If verify_acc is True, check numerical accuracy (rtol=1e-2, atol=1e-2).
    If False, only ensure the function runs without error (output is not None).
    """
    errors = []

    for case in TEST_CASES:
        case_id = case.get("case_id", "unknown")
        kwargs = case["params"]

        try:
            # 1. Generate random inputs in memory
            inputs = rand_sparse_attn_input(**kwargs)

            # 2. Compute reference output (only when accuracy check is needed)
            if verify_acc:
                expected = sparse_attn_torch(**inputs)

            # 3. Compute output of the function under test
            output = sparse_attn(**inputs)

            # 4. Verification
            if verify_acc:
                torch.testing.assert_close(expected, output, rtol=1e-2, atol=1e-2)
                print(f"case_{case_id}: \033[92mPassed.\033[0m")
            else:
                assert output is not None, "Output is None"
                print(f"case_{case_id}: \033[92mFinished.\033[0m")

        except Exception as e:
            errors.append(f"case_{case_id}: {str(e)}")
            print(f"case_{case_id}: \033[91mFailed: {e}\033[0m")

    # 5. Report errors if any
    if errors:
        error_msg = "\n".join(errors)
        raise AssertionError(f"Some cases failed:\n{error_msg}")
    else:
        print("\033[92mAll checks passed.\033[0m")


if __name__ == "__main__":
    # Specifies which NPU device to use
    torch.npu.set_device(0)
    tilelang.cache.clear_cache()

    # Run tests
    run_test(verify_acc=True)
