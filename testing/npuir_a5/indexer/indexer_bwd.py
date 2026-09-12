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
def _indexer_bwd_kernel(
    num_heads,
    head_dim,
    top_k,
    block_top_k=32,
):
    dtype = FP16
    accum_dtype = FP32
    index_dtype = INT32

    query_len = T.symbolic("query_len")
    kv_len = T.symbolic("kv_len")

    @T.prim_func
    def indexer_bwd_kernel(
        IndexQ: T.Tensor([query_len, num_heads, head_dim], dtype),
        IndexK: T.Tensor([kv_len, head_dim], dtype),
        Indices: T.Tensor([query_len, top_k], index_dtype),
        Weights: T.Tensor([query_len, num_heads], accum_dtype),
        GradOutput: T.Tensor([query_len, top_k], accum_dtype),
        dIndexQ: T.Tensor([query_len, num_heads, head_dim], dtype),
        dWeights: T.Tensor([query_len, num_heads], accum_dtype),
        dIndexK: T.Tensor([kv_len, head_dim], accum_dtype),
        K_gather: T.Tensor([query_len, block_top_k, head_dim], dtype),
    ):
        with T.Kernel(query_len, is_npu=True) as (batch_idx, _):
            # 1. Memory allocation for Shared and Local memory
            q_shared = T.alloc_shared([num_heads, head_dim], dtype)
            k_shared = T.alloc_shared([block_top_k, head_dim], dtype)
            w_shared = T.alloc_shared([num_heads, 1], accum_dtype)

            grad_out_shared = T.alloc_shared([1, block_top_k], accum_dtype)
            idx_shared = T.alloc_shared([block_top_k], index_dtype)

            dw_acc_ub = T.alloc_shared([num_heads, 1], accum_dtype)
            T.clear(dw_acc_ub)

            logits_ub = T.alloc_shared([num_heads, block_top_k], accum_dtype)
            m_grad_ub = T.alloc_shared([num_heads, block_top_k], accum_dtype)
            m_grad_f16_ub = T.alloc_shared([num_heads, block_top_k], dtype)
            mask_ub = T.alloc_shared([num_heads, block_top_k], accum_dtype)

            dk_temp_ub = T.alloc_shared([block_top_k, head_dim], accum_dtype)

            logits_frag = T.alloc_fragment([num_heads, block_top_k], accum_dtype)
            dq_frag = T.alloc_fragment([num_heads, head_dim], accum_dtype)
            dk_frag = T.alloc_fragment([block_top_k, head_dim], accum_dtype)
            T.clear(dq_frag)

            T.copy(IndexQ[batch_idx, 0, 0], q_shared, size=[num_heads, head_dim])
            T.copy(Weights[batch_idx, 0], w_shared, size=[num_heads])

            # 2. Main loop: process TopK indices in blocks
            num_iters = top_k // block_top_k
            for i_iter in T.serial(num_iters):
                offset = i_iter * block_top_k

                # Load indices and gather corresponding Key vectors from global memory
                T.copy(Indices[batch_idx, offset], idx_shared, size=[block_top_k])
                for j in T.serial(block_top_k):
                    curr_k_idx = idx_shared[j]
                    T.copy(IndexK[curr_k_idx, 0], k_shared[j, 0], size=[head_dim])

                # Workaround for CV kernel bug, this addition is mandatory
                T.vadd(k_shared, T.cast(0.0, dtype), k_shared)
                T.copy(
                    k_shared[0:block_top_k, 0:head_dim],
                    K_gather[batch_idx, 0:block_top_k, 0:head_dim],
                )
                T.copy(
                    K_gather[batch_idx, 0:block_top_k, 0:head_dim],
                    k_shared[0:block_top_k, 0:head_dim],
                )

                T.copy(
                    GradOutput[batch_idx, offset],
                    grad_out_shared,
                    size=[1, block_top_k],
                )

                # Compute Logits (Q * K^T)
                T.gemm(q_shared, k_shared, logits_frag, b_transpose=True, initC=True)
                T.copy(logits_frag, logits_ub)

                # Compute weight gradient dW: grad_out * relu(logits)
                T.vrelu(logits_ub, logits_ub)
                T.vmul(logits_ub, grad_out_shared, m_grad_ub)
                T.reduce_sum(m_grad_ub, dw_acc_ub, dim=1, clear=False)

                # Compute backpropagation intermediate gradients and apply ReLU mask
                T.vbrc(grad_out_shared, m_grad_ub)
                T.vmul(m_grad_ub, w_shared, m_grad_ub)
                T.vmul(logits_ub, 100000000.0, mask_ub)
                T.vmin(mask_ub, 1.0, mask_ub)
                T.vmul(m_grad_ub, mask_ub, m_grad_ub)
                T.vcast(m_grad_ub, m_grad_f16_ub, round_mode="round")

                # Compute matrix multiplication for dQ and dK
                T.gemm(m_grad_f16_ub, k_shared, dq_frag, initC=False)
                T.gemm(
                    m_grad_f16_ub,
                    q_shared,
                    dk_frag,
                    a_transpose=True,
                    b_transpose=False,
                    initC=True,
                )

                # Atomic accumulate and write back to dK
                T.copy(dk_frag, dk_temp_ub)
                for j in T.serial(block_top_k):
                    target_k_idx = idx_shared[j]
                    T.atomic_add(dIndexK[target_k_idx, 0], dk_temp_ub[j, 0], [head_dim])

            # 3. Final write back of dWeights and dQ results
            T.copy(dw_acc_ub, dWeights[batch_idx, 0], size=[num_heads])
            T.copy(dq_frag, dIndexQ[batch_idx, 0, 0], size=[num_heads, head_dim])

    return indexer_bwd_kernel


def indexer_bwd(q, k, indices, weights, grad_output, block_top_k=16):
    query_len, num_heads, head_dim = q.shape
    kv_len = k.shape[0]
    top_k = indices.shape[1]
    device = q.device

    assert top_k % block_top_k == 0, (
        f"top_k ({top_k}) must be a multiple of block_top_k ({block_top_k})"
    )

    dq = torch.zeros(
        (query_len, num_heads, head_dim), device=device, dtype=torch.float16
    )
    dw = torch.zeros((query_len, num_heads), device=device, dtype=torch.float32)
    dk = torch.zeros((kv_len, head_dim), device=device, dtype=torch.float32)

    k_gather = torch.zeros(
        (query_len, block_top_k, head_dim), device=device, dtype=torch.float16
    )
    kernel = _indexer_bwd_kernel(num_heads, head_dim, top_k, block_top_k)
    kernel(
        q.contiguous(),
        k.contiguous(),
        indices.contiguous(),
        weights.contiguous(),
        grad_output.contiguous(),
        dq,
        dw,
        dk,
        k_gather,
    )

    return dq, dw, dk


TEST_CASES = [
    (8, 512, 512, 128, 32),
]


def indexer_torch_ref(q, k, indices, weights, grad_output):
    """
    PyTorch reference backward implementation.
    """

    q_ref = q.detach().clone().float().cpu().requires_grad_(True)
    k_ref = k.detach().clone().float().cpu().requires_grad_(True)
    w_ref = weights.detach().clone().float().cpu().requires_grad_(True)

    indices_cpu = indices.cpu()
    grad_output_cpu = grad_output.cpu()

    query_len, num_heads, head_dim = q_ref.shape

    outputs = []

    for b in range(query_len):
        # [top_k, head_dim]
        selected_k = k_ref[indices_cpu[b]]

        # [num_heads, head_dim] @ [head_dim, top_k]
        # -> [num_heads, top_k]
        logits = torch.matmul(q_ref[b], selected_k.t())

        relu_logits = torch.relu(logits)

        # [num_heads, 1]
        w = w_ref[b].view(num_heads, 1)

        # sum over heads
        # -> [top_k]
        out = (relu_logits * w).sum(dim=0)

        outputs.append(out)

    outputs = torch.stack(outputs, dim=0)

    outputs.backward(grad_output_cpu)

    return (
        q_ref.grad,
        w_ref.grad,
        k_ref.grad,
    )


class TestIndexBwd:
    def setup_method(self):
        torch.manual_seed(42)

    @pytest.mark.parametrize(
        ("heads", "query_len", "kv_len", "dim", "top_k"),
        [
            pytest.param(
                *test, id=f"H{test[0]}-Q{test[1]}-KV{test[2]}-D{test[3]}-TOPK{test[4]}"
            )
            for test in TEST_CASES
        ],
    )
    def test_indexer_bwd_npu(
        self,
        heads,
        query_len,
        kv_len,
        dim,
        top_k,
    ):
        device = "npu"

        q = torch.randn(
            query_len,
            heads,
            dim,
            device=device,
            dtype=torch.float16,
        )

        k = torch.randn(
            kv_len,
            dim,
            device=device,
            dtype=torch.float16,
        )

        weights = torch.randn(
            query_len,
            heads,
            device=device,
            dtype=torch.float32,
        )

        # IMPORTANT:
        # allow duplicated indices to stress atomic_add
        indices = torch.randint(
            0,
            kv_len,
            (query_len, top_k),
            device=device,
            dtype=torch.int32,
        )

        grad_output = torch.randn(
            query_len,
            top_k,
            device=device,
            dtype=torch.float32,
        )

        # TileLang kernel
        dq_npu, dw_npu, dk_npu = indexer_bwd(
            q,
            k,
            indices,
            weights,
            grad_output,
            block_top_k=16,
        )

        # Torch reference
        dq_ref, dw_ref, dk_ref = indexer_torch_ref(
            q,
            k,
            indices,
            weights,
            grad_output,
        )

        dq_cpu = dq_npu.float().cpu()
        dw_cpu = dw_npu.float().cpu()
        dk_cpu = dk_npu.float().cpu()

        # Error stats
        dq_err = (dq_cpu - dq_ref).abs().max().item()
        dw_err = (dw_cpu - dw_ref).abs().max().item()
        dk_err = (dk_cpu - dk_ref).abs().max().item()

        print(f"dQ Max Error: {dq_err:.6f}")
        print(f"dW Max Error: {dw_err:.6f}")
        print(f"dK Max Error: {dk_err:.6f}")

        torch.testing.assert_close(
            dq_cpu,
            dq_ref,
            rtol=1e-2,
            atol=1e-2,
        )

        torch.testing.assert_close(
            dw_cpu,
            dw_ref,
            rtol=1e-2,
            atol=1e-2,
        )

        torch.testing.assert_close(
            dk_cpu,
            dk_ref,
            rtol=1e-2,
            atol=1e-2,
        )
