"""
Tilelang implementation of Mamba3 forward kernel,
with MIMO support.

Copyright (c) 2026, Dao AI Lab, Goombalab

"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
import tilelang
import tilelang.language as T
import os


FIXED_B = 4
FIXED_S = 2048
FIXED_H = 16
FIXED_G = 1
FIXED_ROTARY_DIM_DIVISOR = 4
FIXED_DTYPE = torch.bfloat16
REL_TOL = 0.10


case_grid = [16, 64, 4, 8, 128]


@tilelang.jit(target="npuir")
def mamba_mimo_fwd(
    B,
    S,
    H,
    G,
    N,
    P,
    R,
    hasZ,
    hasD,
    reduceO,
    return_final_state=False,
    chunk_size: int = 16,
    rotary_dim_divisor=4,
    dtype: str = "bfloat16",
    threads: int = 128,
    num_stages: int = 0,
):

    accum_dtype = "float32"

    nchunks = tilelang.cdiv(S, chunk_size)
    tail_len = S % chunk_size
    fused_chunk_size = chunk_size * R
    rotary_dim = N // rotary_dim_divisor

    O_shape = (B, S, H, P) if reduceO else (B, S, R, H, P)

    @T.prim_func
    def mamba_mimo_fwd_kernel(
        Q: T.Tensor([B, S, R, G, N], dtype),  # type: ignore
        K: T.Tensor([B, S, R, G, N], dtype),  # type: ignore
        V: T.Tensor([B, S, H, P], dtype),  # type: ignore
        O: T.Tensor(O_shape, accum_dtype),  # type: ignore
        Q_BIAS: T.Tensor([H, R, N], "float32"),  # type: ignore
        K_BIAS: T.Tensor([H, R, N], "float32"),  # type: ignore
        MIMO_V: T.Tensor([H, R, P], "float32"),  # type: ignore
        MIMO_O: T.Tensor([H, R, P], "float32"),  # type: ignore
        Z: T.Tensor([B, S, H, P], dtype),  # type: ignore
        D: T.Tensor([H], "float32"),  # type: ignore
        MIMO_Z: T.Tensor([H, R, P], "float32"),  # type: ignore
        ANGLES: T.Tensor([B, S, H, N // rotary_dim_divisor], "float32"),  # type: ignore
        DA_CS: T.Tensor([B, H, S], "float32"),  # type: ignore
        DA_CS_REV: T.Tensor([B, H, S], "float32"),  # type: ignore
        DT: T.Tensor([B, H, S + 1], "float32"),  # type: ignore
        TRAP: T.Tensor([B, H, S + 1], dtype),  # type: ignore
        SEGSUM: T.Tensor([B, H, nchunks, chunk_size, chunk_size], "float32"),  # type: ignore
        FINAL_STATE: T.Tensor([B, H, N, P], "float32"),  # type: ignore
        FINAL_K: T.Tensor([B, R, H, N], dtype),  # type: ignore
    ):
        """
        Overview:
            Fused chunked forward pass that combines MIMO projections with recurrent state updates.
            Computes interchunk and intrachunk contributions with optional D and Z paths,
            then writes output activations.

        Inputs:
            - Activations: Q, K, V.
            - Projection parameters/biases: MIMO_V (Psi), MIMO_O (Phi), optional MIMO_Z (Zeta), ANGLES,
              and Q_BIAS/K_BIAS.
            - Optional modifiers: Z, and D.
            - Discretization tensors: DA_CS, DA_CS_REV, DT, TRAP, and SEGSUM.

        Outputs:
            - O: fused forward output activations.
            - FINAL_STATE: final recurrent states (if return_final_state is True).
            - FINAL_K: final K tensor (if return_state is True, for use in decode)

        Notation:
            - Psi: MIMO X projection.
            - Phi: MIMO O projection.
            - Zeta: MIMO Z projection.
            - Trap: convex-combination modulator used in exponential-trapezoidal discretization.
        """

        with T.Kernel(H * B, is_npu=True) as (idx, _):
            # --- Kernel Setup ---
            # GQA support: map V head to Q/K head
            # (i_h, i_b, _):
            i_h = idx % H
            i_b = idx // H
            i_h_qk = i_h // (H // G)

            # --- Buffer Allocation ---
            q_shared = T.alloc_shared([fused_chunk_size, N], accum_dtype)
            k_shared = T.alloc_shared([fused_chunk_size, N], accum_dtype)
            q_bias_frag = T.alloc_fragment([1, R, N], accum_dtype)
            k_bias_frag = T.alloc_fragment([1, R, N], accum_dtype)

            PsiV_shared = T.alloc_shared([fused_chunk_size, P], accum_dtype)
            v_shared = T.alloc_shared([chunk_size, 1, P], accum_dtype)
            states_accum_cast_shared = T.alloc_shared([N, P], accum_dtype)
            qk_intrachunk_shared = T.alloc_shared(
                [fused_chunk_size, fused_chunk_size], accum_dtype
            )
            qk_dot_full_shared = T.alloc_shared(
                [fused_chunk_size, fused_chunk_size], accum_dtype
            )

            # --- Per-Head Constants / Running State ---
            states_frag = T.alloc_fragment([N, P], accum_dtype)
            T.clear(states_frag)

            phi_frag_intrachunk = T.alloc_fragment([1, R, P], accum_dtype)
            if reduceO:
                T.copy(MIMO_O[i_h, :, :], phi_frag_intrachunk)
            Psi_frag = T.alloc_fragment([1, R, P], accum_dtype)
            T.copy(MIMO_V[i_h, :, :], Psi_frag)

            T.copy(Q_BIAS[i_h, :, :], q_bias_frag)
            T.copy(K_BIAS[i_h, :, :], k_bias_frag)

            # --- Chunk Loop ---
            for i in T.Pipelined(0, nchunks, num_stages=num_stages):
                chunk_start = i * chunk_size

                segsum = T.alloc_fragment([chunk_size, chunk_size], "float32")
                T.copy(SEGSUM[i_b, i_h, i, :, :], segsum)

                # --- Discretization Factors (Shifted Gamma + Trap Scale) ---
                trap_shifted_frag = T.alloc_fragment([chunk_size], accum_dtype)
                T.copy(
                    TRAP[i_b, i_h, chunk_start + 1 : chunk_start + chunk_size + 1],
                    trap_shifted_frag,
                )
                T.vmul(trap_shifted_frag, -1, trap_shifted_frag)
                T.vsigmoid(trap_shifted_frag, trap_shifted_frag)
                dt_shifted_frag = T.alloc_fragment([chunk_size], accum_dtype)
                T.copy(
                    DT[i_b, i_h, chunk_start + 1 : chunk_start + chunk_size + 1],
                    dt_shifted_frag,
                )
                shifted_gamma_frag = T.alloc_fragment([chunk_size], accum_dtype)
                T.clear(shifted_gamma_frag)
                for cs in T.serial(chunk_size):
                    if chunk_start + cs < (S - 1):
                        shifted_gamma_frag[cs] = (
                            dt_shifted_frag[cs] * trap_shifted_frag[cs]
                        )

                shifted_gamma_shared = T.alloc_shared([chunk_size], accum_dtype)
                T.copy(shifted_gamma_frag, shifted_gamma_shared)

                trap_frag = T.alloc_fragment([chunk_size], "float32")
                T.copy(
                    TRAP[i_b, i_h, chunk_start : chunk_start + chunk_size], trap_frag
                )
                T.vsigmoid(trap_frag, trap_frag)
                dt_frag = T.alloc_fragment([chunk_size], accum_dtype)
                T.copy(DT[i_b, i_h, chunk_start : chunk_start + chunk_size], dt_frag)
                gamma_frag = T.alloc_fragment([chunk_size], accum_dtype)

                T.vmul(dt_frag, trap_frag, gamma_frag)

                trap_scale_frag = T.alloc_fragment([chunk_size], accum_dtype)

                T.vadd(gamma_frag, shifted_gamma_shared, trap_scale_frag)

                trap_scale_shared = T.alloc_shared([chunk_size], accum_dtype)
                T.copy(trap_scale_frag, trap_scale_shared)

                # --- Up-Project V and Prepare Biased Q/K ---
                PsiV_frag = T.alloc_fragment([chunk_size, R, P], accum_dtype)

                T.copy(V[i_b, chunk_start : chunk_start + chunk_size, i_h, :], v_shared)

                T.vmul(v_shared, Psi_frag, PsiV_frag)
                PsiV_reshaped_frag = T.alloc_fragment(
                    [fused_chunk_size, P], accum_dtype
                )
                T.reshape(PsiV_frag, PsiV_reshaped_frag)
                T.copy(PsiV_reshaped_frag, PsiV_shared)

                q_frag = T.alloc_fragment([chunk_size, R, N], accum_dtype)
                T.copy(
                    Q[i_b, chunk_start : chunk_start + chunk_size, :, i_h_qk, :], q_frag
                )

                T.vadd(q_frag, q_bias_frag, q_frag)
                q_frag_reshape = T.alloc_shared([fused_chunk_size, N], accum_dtype)
                T.reshape(q_frag, q_frag_reshape)
                T.copy(q_frag_reshape, q_shared)

                k_frag = T.alloc_fragment([chunk_size, R, N], accum_dtype)
                T.copy(
                    K[i_b, chunk_start : chunk_start + chunk_size, :, i_h_qk, :], k_frag
                )

                T.vadd(k_frag, k_bias_frag, k_frag)
                k_shared_reshape = T.alloc_shared([fused_chunk_size, N], accum_dtype)
                T.reshape(k_frag, k_shared_reshape)
                T.copy(k_shared_reshape, k_shared)

                # --- Cache Diagonal qk_dot Path ---
                # Keep full qk_dot in shared memory because we reuse same-step R x R blocks later.
                qk_dot_frag = T.alloc_fragment(
                    [fused_chunk_size, fused_chunk_size], dtype=accum_dtype
                )
                T.gemm(q_shared, k_shared, qk_dot_frag, b_transpose=True, initC=True)
                T.copy(qk_dot_frag, qk_dot_full_shared)

                # --- Rotary Q + Interchunk Contribution ---
                # NOTE: angles are casted to fp32 for numerical stability.
                angles_frag = T.alloc_shared([chunk_size, 1, rotary_dim], "float32")
                T.copy(
                    ANGLES[i_b, chunk_start : chunk_start + chunk_size, i_h, :],
                    angles_frag,
                )
                angles_frag_cos = T.alloc_fragment(
                    [chunk_size, 1, rotary_dim], "float32"
                )
                angles_frag_sin = T.alloc_fragment(
                    [chunk_size, 1, rotary_dim], "float32"
                )
                angles_frag_cos_r = T.alloc_fragment(
                    [chunk_size, R, rotary_dim], "float32"
                )
                angles_frag_sin_r = T.alloc_fragment(
                    [chunk_size, R, rotary_dim], "float32"
                )
                angles_frag_cos_r_reshaped = T.alloc_fragment(
                    [fused_chunk_size, rotary_dim], "float32"
                )
                angles_frag_sin_r_reshaped = T.alloc_fragment(
                    [fused_chunk_size, rotary_dim], "float32"
                )

                T.vcos(angles_frag, angles_frag_cos)
                T.vsin(angles_frag, angles_frag_sin)
                T.vbrc(angles_frag_cos, angles_frag_cos_r)
                T.vbrc(angles_frag_sin, angles_frag_sin_r)
                T.reshape(angles_frag_cos_r, angles_frag_cos_r_reshaped)
                T.reshape(angles_frag_sin_r, angles_frag_sin_r_reshaped)

                q_first_sin = T.alloc_fragment(
                    [fused_chunk_size, rotary_dim], accum_dtype
                )
                q_second_sin = T.alloc_fragment(
                    [fused_chunk_size, rotary_dim], accum_dtype
                )
                q_first_cos = T.alloc_fragment(
                    [fused_chunk_size, rotary_dim], accum_dtype
                )
                q_second_cos = T.alloc_fragment(
                    [fused_chunk_size, rotary_dim], accum_dtype
                )

                T.copy(q_shared[:, 0:rotary_dim], q_first_sin)
                T.copy(q_shared[:, N // 2 : N // 2 + rotary_dim], q_second_sin)
                T.vmul(q_first_sin, angles_frag_cos_r_reshaped, q_first_cos)
                T.vmul(q_second_sin, angles_frag_cos_r_reshaped, q_second_cos)
                T.vmul(q_first_sin, angles_frag_sin_r_reshaped, q_first_sin)
                T.vmul(q_second_sin, angles_frag_sin_r_reshaped, q_second_sin)
                T.vsub(q_first_cos, q_second_sin, q_shared[:, 0:rotary_dim])
                T.vadd(
                    q_first_sin,
                    q_second_cos,
                    q_shared[:, N // 2 : N // 2 + rotary_dim],
                )

                o_mimo_accum_frag = T.alloc_fragment([fused_chunk_size, P], accum_dtype)
                T.copy(states_frag, states_accum_cast_shared)
                T.gemm(
                    q_shared, states_accum_cast_shared, o_mimo_accum_frag, initC=True
                )

                # --- Rotary K + Trap Scaling + Intrachunk Contribution ---
                k_first_sin = T.alloc_fragment(
                    [fused_chunk_size, rotary_dim], accum_dtype
                )
                k_second_sin = T.alloc_fragment(
                    [fused_chunk_size, rotary_dim], accum_dtype
                )
                k_first_cos = T.alloc_fragment(
                    [fused_chunk_size, rotary_dim], accum_dtype
                )
                k_second_cos = T.alloc_fragment(
                    [fused_chunk_size, rotary_dim], accum_dtype
                )

                T.copy(k_shared[:, 0:rotary_dim], k_first_sin)
                T.copy(k_shared[:, N // 2 : N // 2 + rotary_dim], k_second_sin)
                T.vmul(k_first_sin, angles_frag_cos_r_reshaped, k_first_cos)
                T.vmul(k_second_sin, angles_frag_cos_r_reshaped, k_second_cos)
                T.vmul(k_first_sin, angles_frag_sin_r_reshaped, k_first_sin)
                T.vmul(k_second_sin, angles_frag_sin_r_reshaped, k_second_sin)
                T.vsub(k_first_cos, k_second_sin, k_shared[:, 0:rotary_dim])
                T.vadd(
                    k_first_sin,
                    k_second_cos,
                    k_shared[:, N // 2 : N // 2 + rotary_dim],
                )

                if i == nchunks - 1 and return_final_state:
                    seq_boundary = T.min(chunk_start + chunk_size, S) - chunk_start
                    last_step = seq_boundary - 1
                    for r, n in T.Parallel(R, N):
                        FINAL_K[i_b, r, i_h, n] = k_shared[last_step * R + r, n]

                k_trap_scaled_frag = T.alloc_fragment(
                    [fused_chunk_size, N], accum_dtype
                )
                T.copy(k_shared, k_trap_scaled_frag)

                for cs in T.serial(chunk_size):
                    T.vmul(
                        k_trap_scaled_frag[cs * R : cs * R + R, :],
                        trap_scale_shared[cs],
                        k_trap_scaled_frag[cs * R : cs * R + R, :],
                    )
                T.copy(k_trap_scaled_frag, k_shared)

                qk_intrachunk_frag = T.alloc_fragment(
                    [fused_chunk_size, fused_chunk_size], accum_dtype
                )
                T.gemm(
                    q_shared, k_shared, qk_intrachunk_frag, b_transpose=True, initC=True
                )

                # Causal mask over chunk steps (include same-step block; qkv subtract below).
                da_cs__or__exp_da_cs_shared = T.alloc_shared(
                    [chunk_size, 1, 1], "float32"
                )
                T.copy(
                    DA_CS[i_b, i_h, chunk_start : chunk_start + chunk_size],
                    da_cs__or__exp_da_cs_shared,
                )
                qk_intrachunk_masked_frag = T.alloc_fragment(
                    [fused_chunk_size, fused_chunk_size], accum_dtype
                )
                qk_intrachunk_frag_reshaped = T.alloc_fragment(
                    [chunk_size, R, chunk_size, R], accum_dtype
                )
                segsum_reshaped = T.alloc_fragment(
                    [chunk_size, 1, chunk_size, 1], "float32"
                )

                T.reshape(qk_intrachunk_frag, qk_intrachunk_frag_reshaped)
                T.reshape(segsum, segsum_reshaped)
                T.vexp(segsum_reshaped, segsum_reshaped)

                for csr_i in T.serial(chunk_size):
                    for csr_j in T.serial(chunk_size):
                        if csr_i < csr_j:
                            segsum_reshaped[csr_i, 0, csr_j, 0] = 0.0

                T.vmul(
                    qk_intrachunk_frag_reshaped,
                    segsum_reshaped,
                    qk_intrachunk_frag_reshaped,
                )
                T.reshape(qk_intrachunk_frag_reshaped, qk_intrachunk_masked_frag)

                # Exponentiate da_cs__or__exp_da_cs_shared so that later usage does not have to:

                T.vexp(da_cs__or__exp_da_cs_shared, da_cs__or__exp_da_cs_shared)

                exp_da_cs_frag = T.alloc_fragment([chunk_size, 1, 1], dtype="float32")
                o_mimo_accum_frag_reshaped = T.alloc_fragment(
                    [chunk_size, R, P], accum_dtype
                )
                T.reshape(o_mimo_accum_frag, o_mimo_accum_frag_reshaped)
                T.copy(da_cs__or__exp_da_cs_shared, exp_da_cs_frag)

                T.vmul(
                    o_mimo_accum_frag_reshaped,
                    exp_da_cs_frag,
                    o_mimo_accum_frag_reshaped,
                )

                T.reshape(o_mimo_accum_frag_reshaped, o_mimo_accum_frag)

                T.copy(qk_intrachunk_masked_frag, qk_intrachunk_shared)

                tmp = T.alloc_fragment([fused_chunk_size, P], dtype=accum_dtype)
                T.gemm(qk_intrachunk_shared, PsiV_shared, tmp, initC=True)
                T.vadd(o_mimo_accum_frag, tmp, o_mimo_accum_frag)

                # --- Subtract qkv correction (pre-rotary qk_dot * shifted_gamma) ---
                qkdot_psiv_frag = T.alloc_fragment([chunk_size, R, P], accum_dtype)
                T.clear(qkdot_psiv_frag)

                # Apply shifted gamma
                qk_dot_tmp_2d = T.alloc_shared([R, R], accum_dtype)
                qk_dot_tmp = T.alloc_shared([R, R, 1], accum_dtype)
                PsiV_tmp = T.alloc_shared([1, R, P], accum_dtype)
                PsiV_tmp_reshaped = T.alloc_fragment([R, R, P], accum_dtype)
                kv_dot_tmp = T.alloc_shared([R, 1, P], accum_dtype)
                kv_dot_tmp_reshaped = T.alloc_fragment([R, P], accum_dtype)
                for cs in T.serial(chunk_size):
                    T.copy(
                        qk_dot_full_shared[cs * R : cs * R + R, cs * R : cs * R + R],
                        qk_dot_tmp_2d,
                    )
                    T.reshape(qk_dot_tmp_2d, qk_dot_tmp)
                    T.copy(PsiV_shared[cs * R : cs * R + R, :], PsiV_tmp[0, :, :])
                    T.vbrc(PsiV_tmp, PsiV_tmp_reshaped)
                    T.vmul(qk_dot_tmp, PsiV_tmp_reshaped, PsiV_tmp_reshaped)
                    T.reduce(PsiV_tmp_reshaped, kv_dot_tmp, dims=1, reduce_mode="sum")
                    T.vmul(kv_dot_tmp, shifted_gamma_frag[cs], kv_dot_tmp)
                    T.reshape(kv_dot_tmp, kv_dot_tmp_reshaped)
                    T.copy(kv_dot_tmp_reshaped, qkdot_psiv_frag[cs, :, :])

                qkdot_psiv_reshaped_frag = T.alloc_fragment(
                    [fused_chunk_size, P], accum_dtype
                )
                T.reshape(qkdot_psiv_frag, qkdot_psiv_reshaped_frag)
                T.vsub(o_mimo_accum_frag, qkdot_psiv_reshaped_frag, o_mimo_accum_frag)

                if hasD:
                    PsiV_D_frag = T.alloc_fragment([fused_chunk_size, P], "float32")
                    T.copy(PsiV_shared, PsiV_D_frag)
                    d_i_h = D[i_h]

                    T.vmul(PsiV_D_frag, d_i_h, PsiV_D_frag)

                    T.vadd(o_mimo_accum_frag, PsiV_D_frag, o_mimo_accum_frag)
                # --- Optional Z Gating + Down-Projection ---
                if reduceO:
                    lqk_PsiV_reshaped_frag = T.alloc_fragment(
                        [chunk_size, R, P], accum_dtype
                    )
                    T.reshape(o_mimo_accum_frag, lqk_PsiV_reshaped_frag)
                    if hasZ:
                        z_frag = T.alloc_fragment([chunk_size, 1, P], accum_dtype)
                        T.copy(
                            Z[i_b, chunk_start : chunk_start + chunk_size, i_h, :],
                            z_frag,
                        )
                        z_expanded_frag = T.alloc_fragment(
                            [chunk_size, R, P], accum_dtype
                        )
                        o_gated = T.alloc_fragment([chunk_size, R, P], accum_dtype)
                        o_gated_tanh = T.alloc_fragment([chunk_size, R, P], accum_dtype)

                        mimoZ_shared = T.alloc_shared([1, R, P], accum_dtype)
                        T.copy(MIMO_Z[i_h, :, :], mimoZ_shared)
                        T.vmul(mimoZ_shared, 0.5, mimoZ_shared)
                        T.vmul(z_frag, mimoZ_shared, o_gated)

                        T.vtanh(o_gated, o_gated_tanh)
                        o_gated_mul_tmp = T.alloc_shared(
                            [chunk_size, R, P], accum_dtype
                        )
                        T.vmul(o_gated, o_gated_tanh, o_gated_mul_tmp)
                        T.vadd(o_gated, o_gated_mul_tmp, z_expanded_frag)

                        lqk_tmp = T.alloc_shared([chunk_size, R, P], accum_dtype)
                        T.vmul(phi_frag_intrachunk, z_expanded_frag, lqk_tmp)
                        T.vmul(lqk_tmp, lqk_PsiV_reshaped_frag, lqk_PsiV_reshaped_frag)
                    else:
                        for cs, r, p in T.Parallel(chunk_size, R, P):
                            lqk_PsiV_reshaped_frag[cs, r, p] *= phi_frag_intrachunk[
                                0, r, p
                            ]
                    lqk_PsiV_reshaped_shared = T.alloc_shared(
                        [chunk_size, R, P], accum_dtype
                    )
                    T.copy(lqk_PsiV_reshaped_frag, lqk_PsiV_reshaped_shared)
                    o_frag = T.alloc_fragment([chunk_size, 1, P], accum_dtype)
                    T.clear(o_frag)
                    for r in T.serial(R):
                        T.vadd(lqk_PsiV_reshaped_shared[:, r, :], o_frag, o_frag)
                    T.copy(
                        o_frag, O[i_b, chunk_start : chunk_start + chunk_size, i_h, :]
                    )
                else:
                    if hasZ:
                        z_frag = T.alloc_fragment([chunk_size, 1, P], accum_dtype)
                        T.copy(
                            Z[i_b, chunk_start : chunk_start + chunk_size, i_h, :],
                            z_frag,
                        )
                        z_expanded_frag = T.alloc_fragment(
                            [chunk_size, R, P], accum_dtype
                        )
                        o_gated = T.alloc_fragment([chunk_size, R, P], accum_dtype)
                        o_gated_tanh = T.alloc_fragment([chunk_size, R, P], accum_dtype)

                        mimoZ_shared = T.alloc_shared([1, R, P], accum_dtype)
                        T.copy(MIMO_Z[i_h, :, :], mimoZ_shared)

                        T.vmul(mimoZ_shared, 0.5, mimoZ_shared)
                        T.vmul(z_frag, mimoZ_shared, o_gated)

                        T.vtanh(o_gated, o_gated_tanh)
                        o_gated_mul_tmp = T.alloc_fragment(
                            [chunk_size, R, P], accum_dtype
                        )
                        T.vmul(o_gated, o_gated_tanh, o_gated_mul_tmp)
                        T.vadd(o_gated, o_gated_mul_tmp, z_expanded_frag)

                        o_mimo_accum_reshaped_frag = T.alloc_fragment(
                            [chunk_size, R, P], accum_dtype
                        )
                        T.reshape(o_mimo_accum_frag, o_mimo_accum_reshaped_frag)
                        T.vmul(
                            o_mimo_accum_reshaped_frag,
                            z_expanded_frag,
                            o_mimo_accum_reshaped_frag,
                        )
                        lqk_PsiV_reshaped_shared = T.alloc_shared(
                            [chunk_size, R, P], accum_dtype
                        )
                        T.copy(o_mimo_accum_reshaped_frag, lqk_PsiV_reshaped_shared)
                        T.copy(
                            lqk_PsiV_reshaped_shared,
                            O[i_b, chunk_start : chunk_start + chunk_size, :, i_h, :],
                        )
                    else:
                        lqk_PsiV_reshaped_shared = T.alloc_shared(
                            [chunk_size, R, P], accum_dtype
                        )
                        for cs, r, p in T.Parallel(chunk_size, R, P):
                            lqk_PsiV_reshaped_shared[cs, r, p] = o_mimo_accum_frag[
                                cs * R + r, p
                            ]
                        T.copy(
                            lqk_PsiV_reshaped_shared,
                            O[i_b, chunk_start : chunk_start + chunk_size, :, i_h, :],
                        )

                # --- Recurrent State Update ---
                # DA_CS_REV scales per-step K contributions for state accumulation.
                dA_cs_rev_frag = T.alloc_fragment([chunk_size], accum_dtype)
                T.copy(
                    DA_CS_REV[i_b, i_h, chunk_start : chunk_start + chunk_size],
                    dA_cs_rev_frag,
                )
                T.vexp(dA_cs_rev_frag, dA_cs_rev_frag)

                k_state_frag = T.alloc_fragment([fused_chunk_size, N], accum_dtype)
                T.copy(k_shared, k_state_frag)

                for cs in T.serial(chunk_size):
                    T.vmul(
                        k_state_frag[cs * R : cs * R + R, :],
                        dA_cs_rev_frag[cs],
                        k_state_frag[cs * R : cs * R + R, :],
                    )

                # DA_CS(last) applies the chunk-level decay to the carried state.
                da_cs_sum = T.alloc_fragment([1, 1], "float32")
                if tail_len > 0 and i == nchunks - 1:
                    T.copy(DA_CS[i_b, i_h, S - 1], da_cs_sum)
                    if return_final_state:
                        for csr, n in T.Parallel(fused_chunk_size, N):
                            if csr >= tail_len * R:
                                k_state_frag[csr, n] = 0
                else:
                    T.copy(DA_CS[i_b, i_h, chunk_start + chunk_size - 1], da_cs_sum)
                T.vexp(da_cs_sum, da_cs_sum)
                T.vmul(states_frag, da_cs_sum[0, 0], states_frag)
                T.gemm(
                    k_state_frag,
                    PsiV_shared,
                    states_frag,
                    a_transpose=True,
                    initC=False,
                )

            # --- Save Last State (if applicable) ---
            if return_final_state:
                T.copy(states_frag, FINAL_STATE[i_b, i_h, :, :])

    return mamba_mimo_fwd_kernel


def mamba_mimo_forward(
    q,
    k,
    v,
    q_bias,
    k_bias,
    mimo_v,
    mimo_o,
    z,
    D,
    mimo_z,
    angles,
    dA_cs,
    dA_cs_rev,
    dt,
    trap,
    segsum,
    chunk_size,
    rotary_dim_divisor,
    dtype,
    return_state=False,
    threads=128,
    num_stages=0,
):
    B, S, R, G, N = q.shape
    H, P = v.shape[-2], v.shape[-1]
    if isinstance(dtype, torch.dtype):
        tl_dtype = str(dtype).replace("torch.", "")
    else:
        tl_dtype = dtype
    reduceO = mimo_o is not None
    kernel = mamba_mimo_fwd(
        B,
        S,
        H,
        G,
        N,
        P,
        R,
        z is not None,
        D is not None,
        reduceO,
        return_final_state=return_state,
        chunk_size=chunk_size,
        rotary_dim_divisor=rotary_dim_divisor,
        dtype=tl_dtype,
        threads=threads,
        num_stages=num_stages,
    )
    # print(kernel.get_kernel_source()) # NOTE: prints compiled CUDA code
    if reduceO:
        o = torch.empty((B, S, H, P), device="npu", dtype=torch.float32)
    else:
        o = torch.empty((B, S, R, H, P), device="npu", dtype=torch.float32)
    # Kernel always declares all tensor parameters; pass dummies for None args
    mimo_o_arg = (
        mimo_o
        if reduceO
        else torch.empty((H, R, P), device=q.device, dtype=torch.float32)
    )
    z_arg = (
        z if z is not None else torch.empty((B, S, H, P), device=q.device, dtype=dtype)
    )
    D_arg = (
        D if D is not None else torch.empty((H,), device=q.device, dtype=torch.float32)
    )
    mimo_z_arg = (
        mimo_z
        if mimo_z is not None
        else torch.empty((H, R, P), device=q.device, dtype=torch.float32)
    )

    h = (
        torch.empty((B, H, N, P), device="npu", dtype=torch.float32)
        if return_state
        else None
    )
    k_final = (
        torch.empty((B, R, H, N), device="npu", dtype=dtype) if return_state else None
    )

    dt_arg = F.pad(dt, (0, 1), value=0.0)
    trap_arg = F.pad(trap, (0, 1), value=0.0)

    kernel(
        q,
        k,
        v,
        o,
        q_bias,
        k_bias,
        mimo_v,
        mimo_o_arg,
        z_arg,
        D_arg,
        mimo_z_arg,
        angles,
        dA_cs,
        dA_cs_rev,
        dt_arg,
        trap_arg,
        segsum,
        h,
        k_final,
    )
    return o, h, k_final


def _pad_zeros(t: Optional[Tensor], pad_len: int, dim: int) -> Optional[Tensor]:
    """Append ``pad_len`` zero-slices along ``dim``.  Returns ``t`` if pad_len==0."""
    if t is None or pad_len == 0:
        return t
    shape = list(t.shape)
    shape[dim] = pad_len
    return torch.cat([t, torch.zeros(shape, device=t.device, dtype=t.dtype)], dim=dim)


def mamba3_MIMO_chunk_ref(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_bias: Tensor,
    k_bias: Tensor,
    mimo_v: Tensor,
    mimo_o: Optional[Tensor],
    z: Optional[Tensor],
    mimo_z: Optional[Tensor],
    angles: Tensor,
    dA_cs: Tensor,
    dA_cs_rev: Tensor,
    dt: Tensor,
    trap: Tensor,
    D: Optional[Tensor],
    chunk_size: int = 64,
    rotary_dim_divisor: int = 4,
    return_final_state: bool = False,
    dtype: torch.dtype = torch.float32,
    rotate_pairwise: bool = False,
    contract_mimo_out: bool = True,
    cu_seqlens: Optional[Tensor] = None,
) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    # Local copy of the reference program so tests remain valid even if module-level
    # debug/reference helpers are removed from shipped kernels.
    from einops import rearrange, repeat

    # --- Varlen path: loop per-sequence, delegate to the single-sequence path ---
    if cu_seqlens is not None:
        NS = cu_seqlens.shape[0] - 1
        out_parts = []
        for i in range(NS):
            start = int(cu_seqlens[i].item())
            end = int(cu_seqlens[i + 1].item())
            seq_len = end - start
            out_i, _, _ = mamba3_MIMO_chunk_ref(
                q[:, start:end],
                k[:, start:end],
                v[:, start:end],
                q_bias,
                k_bias,
                mimo_v,
                mimo_o,
                z[:, start:end] if z is not None else None,
                mimo_z,
                angles[:, start:end],
                dA_cs[:, :, start:end],
                dA_cs_rev[:, :, start:end],
                dt[:, :, start:end],
                trap[:, :, start:end],
                D,
                chunk_size=chunk_size,
                rotary_dim_divisor=rotary_dim_divisor,
                return_final_state=False,
                dtype=dtype,
                rotate_pairwise=rotate_pairwise,
                contract_mimo_out=contract_mimo_out,
                cu_seqlens=None,
            )
            out_parts.append(out_i[:, :seq_len])
        return torch.cat(out_parts, dim=1), None, None

    # --- Single-sequence path ---
    # Pad to the next multiple of chunk_size so the chunked rearranges are valid
    # for sequences whose length is not a multiple of chunk_size.
    orig_seqlen = q.shape[1]
    pad_len = (chunk_size - orig_seqlen % chunk_size) % chunk_size
    if pad_len > 0:
        q = _pad_zeros(q, pad_len, dim=1)
        k = _pad_zeros(k, pad_len, dim=1)
        v = _pad_zeros(v, pad_len, dim=1)
        angles = _pad_zeros(angles, pad_len, dim=1)
        z = _pad_zeros(z, pad_len, dim=1)
        dA_cs = _pad_zeros(dA_cs, pad_len, dim=2)
        dA_cs_rev = _pad_zeros(dA_cs_rev, pad_len, dim=2)
        dt = _pad_zeros(dt, pad_len, dim=2)
        trap = _pad_zeros(trap, pad_len, dim=2)

    nchunks = q.shape[1] // chunk_size
    q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
    if z is not None:
        z = z.to(dtype)
        mimo_z = mimo_z.to(dtype)
    if D is not None:
        D = D.to(dtype)
    q_bias, k_bias = q_bias.to(dtype), k_bias.to(dtype)
    mimo_v = mimo_v.to(dtype)
    if contract_mimo_out:
        assert mimo_o is not None
        mimo_o = mimo_o.to(dtype)
    if dA_cs is not None:
        dA_cs, dA_cs_rev = dA_cs.to(dtype), dA_cs_rev.to(dtype)
        dA_cs = rearrange(dA_cs, "b h (n c) -> b h n c", c=chunk_size)
        dA_cs_rev = rearrange(dA_cs_rev, "b h (n c) -> b h n c", c=chunk_size)

    batch, seqlen, mimo_rank, nheads_qk, dstate = q.shape
    nheads = v.shape[-2]
    if nheads_qk != nheads:
        q = repeat(q, "b s r h_qk d -> b s r (h_qk g) d", g=nheads // nheads_qk)
        k = repeat(k, "b s r h_qk d -> b s r (h_qk g) d", g=nheads // nheads_qk)

    angles = angles.to(dtype) if angles is not None else None
    trap = trap.to(dtype) if trap is not None else None
    dt = dt.to(dtype) if dt is not None else None

    q_bias = rearrange(q_bias, "h r d -> r h d")
    k_bias = rearrange(k_bias, "h r d -> r h d")
    q = q + q_bias[None, None, :, :, :]
    k = k + k_bias[None, None, :, :, :]

    qk_dot = torch.einsum("bsRhd,bsrhd->bsRrh", q, k)

    if angles is not None:
        angles = angles.unsqueeze(2)
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)

        def apply_rotary_emb(tensor: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
            if rotate_pairwise:
                # Pairwise convention used by mamba3_MIMO_step_ref / debug_mimo_step.py.
                tensor_reshaped = tensor.view(*tensor.shape[:-1], -1, 2)
                tensor_0 = tensor_reshaped[..., 0]
                tensor_1 = tensor_reshaped[..., 1]
                rotated_0 = tensor_0 * cos - tensor_1 * sin
                rotated_1 = tensor_0 * sin + tensor_1 * cos
                return torch.stack([rotated_0, rotated_1], dim=-1).view_as(tensor)
            # Kernel-aligned convention (kept as default for existing tests).
            tensor_reshaped = tensor.view(*tensor.shape[:-1], 2, -1)
            tensor_0 = tensor_reshaped[..., 0, :]
            tensor_1 = tensor_reshaped[..., 1, :]
            rotated_0 = tensor_0 * cos - tensor_1 * sin
            rotated_1 = tensor_0 * sin + tensor_1 * cos
            return torch.stack([rotated_0, rotated_1], dim=-2).view_as(tensor)

        def apply_rotary_emb_rotate_half(
            tensor: Tensor, cos: Tensor, sin: Tensor
        ) -> Tensor:
            tensor_reshaped = tensor.view(*tensor.shape[:-1], 4, -1)
            tensor_0 = tensor_reshaped[..., 0, :]
            tensor_1 = tensor_reshaped[..., 2, :]
            rotated_0 = tensor_0 * cos - tensor_1 * sin
            rotated_1 = tensor_0 * sin + tensor_1 * cos
            return torch.stack(
                [
                    rotated_0,
                    tensor_reshaped[..., 1, :],
                    rotated_1,
                    tensor_reshaped[..., 3, :],
                ],
                dim=-2,
            ).view_as(tensor)

        if rotary_dim_divisor == 4:
            q = apply_rotary_emb_rotate_half(q, cos_angles, sin_angles)
            k = apply_rotary_emb_rotate_half(k, cos_angles, sin_angles)
        elif rotary_dim_divisor == 2:
            q = apply_rotary_emb(q, cos_angles, sin_angles)
            k = apply_rotary_emb(k, cos_angles, sin_angles)
        else:
            raise ValueError(f"Invalid rotary_dim_divisor: {rotary_dim_divisor}")

    final_k = k[:, -1].contiguous().clone() if return_final_state else None

    trap = torch.nn.functional.sigmoid(trap)
    gamma = dt * trap
    dt_shifted = torch.nn.functional.pad(dt[:, :, 1:], (0, 1), value=0.0)
    trap_shifted = torch.nn.functional.pad(trap[:, :, 1:], (0, 1), value=0.0)
    shifted_gamma = dt_shifted * (1 - trap_shifted)
    factor = gamma + shifted_gamma
    k = torch.einsum("bsrhn,bhs->bsrhn", k, factor)
    qk_dot = torch.einsum("bsrRh,bhs->bsrRh", qk_dot, shifted_gamma)

    v = torch.einsum("bthd,hrd->btrhd", v, mimo_v)

    def segsum_unstable(x: Tensor) -> Tensor:
        x_segsum = x[..., :, None] - x[..., None, :]
        mask = torch.tril(
            torch.ones(x.size(-1), x.size(-1), device=x.device, dtype=torch.bool),
            diagonal=0,
        )
        return x_segsum.masked_fill(~mask, -torch.inf)

    mimo_mask_outer = segsum_unstable(dA_cs)
    mimo_mask_inner = torch.ones(
        mimo_rank, mimo_rank, dtype=torch.bool, device=q.device
    )
    mimo_mask = torch.kron(mimo_mask_outer, mimo_mask_inner[None, None, None, :, :])

    q = rearrange(q, "b (n c) r h d -> b h n (c r) d", c=chunk_size)
    k_scaled = rearrange(k, "b (n c) r h d -> b h n c r d", c=chunk_size)
    k_scaled = torch.einsum("bhncrd,bhnc->bhncrd", k_scaled, torch.exp(dA_cs_rev))
    k_scaled = rearrange(k_scaled, "b h n c r d -> b h n (c r) d", c=chunk_size)
    k = rearrange(k, "b (n c) r h d -> b h n (c r) d", c=chunk_size)
    v = rearrange(v, "b (n c) r h d -> b h n (c r) d", c=chunk_size)
    kv = k_scaled.transpose(-1, -2) @ v

    curr_state = torch.zeros_like(kv[:, :, 0, :, :])
    for n in range(nchunks):
        curr_dA_sum = dA_cs[:, :, n, -1]
        next_state = (torch.exp(curr_dA_sum[:, :, None, None]) * curr_state) + kv[
            :, :, n, :, :
        ]
        kv[:, :, n, :, :] = curr_state
        curr_state = next_state

    final_state = next_state.float() if return_final_state else None

    q_inter = q * torch.exp(
        repeat(dA_cs, "b h n c -> b h n (c r)", r=mimo_rank).unsqueeze(-1)
    )
    inter = q_inter @ kv
    intra = ((q @ k.transpose(-1, -2)) * torch.exp(mimo_mask)) @ v
    o = inter + intra
    o = rearrange(o, "b h n (c r) d -> b h n c r d", r=mimo_rank)

    v = rearrange(v, "b h n (c r) d -> b h (n c) r d", r=mimo_rank)
    qk_dot = rearrange(qk_dot, "b t R r h -> b h t R r")
    qkv = torch.einsum("bhtRr,bhtrp->bhtRp", qk_dot, v)
    qkv = rearrange(qkv, "b h (n c) r d -> b h n c r d", c=chunk_size)
    o -= qkv

    if D is not None:
        vd = torch.einsum("bhtrp,h->bhtrp", v, D)
        vd = rearrange(vd, "b h (n c) r d -> b h n c r d", c=chunk_size)
        o += vd

    if z is not None:
        z = torch.einsum("bthd,hrd->btrhd", z, mimo_z)
        z = rearrange(z, "b (n c) r h d -> b h n c r d", c=chunk_size)
        o = o * torch.nn.functional.silu(z)

    if contract_mimo_out:
        assert mimo_o is not None
        o = torch.einsum("bhncrd,hrd->bhncd", o, mimo_o)
        out = rearrange(o, "b h n c d -> b (n c) h d")
        return out[:, :orig_seqlen], final_state, final_k

    out = rearrange(o, "b h n c r d -> b (n c) r h d")
    return out[:, :orig_seqlen], final_state, final_k


def compute_dacs_segsum_ref(
    da: torch.Tensor,  # [B, H, S]
    chunk_size: int,
):
    """Dense reference for compute_dacs_segsum_triton.

    Requires S to be a multiple of chunk_size.  Returns (da_cs, da_cs_rev, segsum).
    """
    from einops import repeat

    B, H, S = da.shape
    nchunks = S // chunk_size

    da_reshaped = da.view(B, H, nchunks, chunk_size)
    da_cs = torch.cumsum(da_reshaped, dim=-1)
    da_cs_sum = torch.sum(da_reshaped, dim=-1)
    da_cs_rev = da_cs_sum[..., None] - da_cs

    segsum = repeat(da_reshaped, "... d -> ... d e", e=chunk_size)
    mask = torch.tril(
        torch.ones(chunk_size, chunk_size, device=da_cs.device, dtype=bool), diagonal=-1
    )
    segsum = segsum.masked_fill(~mask, 0)
    segsum = torch.cumsum(segsum, dim=-2)

    return da_cs.view(B, H, S), da_cs_rev.view(B, H, S), segsum


def build_inputs(
    *,
    n: int,
    p: int,
    r: int,
    chunk_size: int,
    seed: int,
    b: int = FIXED_B,
    s: int = FIXED_S,
    h: int = FIXED_H,
    g: int = FIXED_G,
    dtype: torch.dtype = FIXED_DTYPE,
    has_z: bool = True,
    has_d: bool = True,
    rotary_dim_divisor: int = FIXED_ROTARY_DIM_DIVISOR,
) -> dict:
    assert s % chunk_size == 0
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)

    q = torch.randn((b, s, r, g, n), device="npu", dtype=dtype)
    k = torch.randn((b, s, r, g, n), device="npu", dtype=dtype)
    v = torch.randn((b, s, h, p), device="npu", dtype=dtype)

    q_bias = torch.randn((h, r, n), device="npu", dtype=torch.float32)
    k_bias = torch.randn((h, r, n), device="npu", dtype=torch.float32)
    mimo_v = torch.randn((h, r, p), device="npu", dtype=torch.float32) / r
    mimo_o = torch.randn((h, r, p), device="npu", dtype=torch.float32) / r

    z = torch.randn_like(v) if has_z else None
    mimo_z = torch.randn_like(mimo_v) if has_z else None
    d = torch.randn((h,), device="npu", dtype=torch.float32) if has_d else None

    angles = torch.rand(
        (b, s, h, n // rotary_dim_divisor), device="npu", dtype=torch.float32
    )
    dt = F.softplus(-3.0 + torch.randn((b, h, s), device="npu", dtype=torch.float32))
    a = torch.rand((b, h, s), device="npu", dtype=torch.float32)
    dA = (-dt * a).detach()
    dA_cs, dA_cs_rev, segsum = compute_dacs_segsum_ref(dA, chunk_size)
    trap = torch.rand((b, h, s), device="npu", dtype=dtype)
    dout = torch.randn_like(v)

    return {
        "q": q,
        "k": k,
        "v": v,
        "q_bias": q_bias,
        "k_bias": k_bias,
        "mimo_v": mimo_v,
        "mimo_o": mimo_o,
        "z": z,
        "mimo_z": mimo_z,
        "D": d,
        "angles": angles,
        "dt": dt,
        "dA": dA,
        "dA_cs": dA_cs,
        "dA_cs_rev": dA_cs_rev,
        "segsum": segsum,
        "trap": trap,
        "dout": dout,
        "chunk_size": chunk_size,
        "rotary_dim_divisor": rotary_dim_divisor,
    }


def assert_stable_rel(
    ours: Tensor,
    ref: Tensor,
    *,
    label: str,
    cfg: str,
    rel_tol: float = REL_TOL,
) -> None:

    def max_rel_err(ours: Tensor, ref: Tensor, eps: float = 1e-5) -> float:
        ours_f = ours.float()
        ref_f = ref.float()
        num = (ours_f - ref_f).abs().max()
        den = ref_f.abs().max().clamp_min(eps)
        return float((num / den).item())

    ours_f = ours.float()
    ref_f = ref.float()
    rel = max_rel_err(ours_f, ref_f)
    close_mask = torch.isclose(ours_f, ref_f, rtol=0.1, atol=0.1)
    bad_frac = float((~close_mask).float().mean().item())
    max_abs = float((ours_f - ref_f).abs().max().item())
    print(
        f"[debug] {label} ({cfg}) "
        f"stable_max_rel={rel:.6f} max_abs={max_abs:.6e} "
        f"bad_frac(rtol=0.1,atol=0.1)={bad_frac:.6f}"
    )
    if rel < rel_tol:
        return

    raise AssertionError(
        f"{label} stable_max_rel >= {rel_tol} for {cfg}: \n"
        f"stable_max_rel={rel:.6f}, max_abs= {max_abs:.6e}\n"
        f"diag_bad_frac_at_rtol0.1_atol0.1= {bad_frac:.6f}\n"
    )


def test_mimo_fwd_reduceO(n: int, p: int, r: int, chunk_size: int):

    inputs = build_inputs(
        n=n,
        p=p,
        r=r,
        chunk_size=chunk_size,
        seed=42,
    )

    out_tilelang, _, _ = mamba_mimo_forward(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["q_bias"],
        inputs["k_bias"],
        inputs["mimo_v"],
        inputs["mimo_o"],
        inputs["z"],
        inputs["D"],
        inputs["mimo_z"],
        inputs["angles"],
        inputs["dA_cs"],
        inputs["dA_cs_rev"],
        inputs["dt"],
        inputs["trap"],
        inputs["segsum"],
        chunk_size=chunk_size,
        rotary_dim_divisor=inputs["rotary_dim_divisor"],
        dtype=FIXED_DTYPE,
    )

    q = inputs["q"].cpu()
    k = inputs["k"].cpu()
    v = inputs["v"].cpu()
    q_bias = inputs["q_bias"].cpu()
    k_bias = inputs["k_bias"].cpu()
    mimo_v = inputs["mimo_v"].cpu()
    mimo_o = inputs["mimo_o"].cpu()
    z = inputs["z"].cpu()
    mimo_z = inputs["mimo_z"].cpu()
    angles = inputs["angles"].cpu()
    dA_cs = inputs["dA_cs"].cpu()
    dA_cs_rev = inputs["dA_cs_rev"].cpu()
    dt = inputs["dt"].cpu()
    trap = inputs["trap"].cpu()
    D = inputs["D"].cpu()

    out_ref_fp32, _, _ = mamba3_MIMO_chunk_ref(
        q,
        k,
        v,
        q_bias,
        k_bias,
        mimo_v,
        mimo_o,
        z,
        mimo_z,
        angles,
        dA_cs,
        dA_cs_rev,
        dt,
        trap,
        D,
        chunk_size=chunk_size,
        rotary_dim_divisor=inputs["rotary_dim_divisor"],
        dtype=torch.float32,
        return_final_state=True,
    )

    print(f"Test mamba3 mimo fwd reduceO: N={n}, P={p}, R={r}, chunk={chunk_size}")
    out_tilelang = out_tilelang.float().cpu()

    assert_stable_rel(
        out_tilelang,
        out_ref_fp32,
        label="O",
        cfg=f"N={n}, P={p}, R={r}, chunk={chunk_size}",
    )


def test_mimo_fwd(n: int, p: int, r: int, chunk_size: int):
    # not reduceO test, mimo_o is None
    inputs = build_inputs(
        n=n,
        p=p,
        r=r,
        chunk_size=chunk_size,
        seed=42,
    )

    out_tilelang, _, _ = mamba_mimo_forward(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["q_bias"],
        inputs["k_bias"],
        inputs["mimo_v"],
        None,
        inputs["z"],
        inputs["D"],
        inputs["mimo_z"],
        inputs["angles"],
        inputs["dA_cs"],
        inputs["dA_cs_rev"],
        inputs["dt"],
        inputs["trap"],
        inputs["segsum"],
        chunk_size=chunk_size,
        rotary_dim_divisor=inputs["rotary_dim_divisor"],
        dtype=FIXED_DTYPE,
    )

    q = inputs["q"].cpu()
    k = inputs["k"].cpu()
    v = inputs["v"].cpu()
    q_bias = inputs["q_bias"].cpu()
    k_bias = inputs["k_bias"].cpu()
    mimo_v = inputs["mimo_v"].cpu()
    z = inputs["z"].cpu()
    mimo_z = inputs["mimo_z"].cpu()
    angles = inputs["angles"].cpu()
    dA_cs = inputs["dA_cs"].cpu()
    dA_cs_rev = inputs["dA_cs_rev"].cpu()
    dt = inputs["dt"].cpu()
    trap = inputs["trap"].cpu()
    D = inputs["D"].cpu()

    out_ref_fp32, _, _ = mamba3_MIMO_chunk_ref(
        q,
        k,
        v,
        q_bias,
        k_bias,
        mimo_v,
        None,
        z,
        mimo_z,
        angles,
        dA_cs,
        dA_cs_rev,
        dt,
        trap,
        D,
        chunk_size=chunk_size,
        rotary_dim_divisor=inputs["rotary_dim_divisor"],
        dtype=torch.float32,
        return_final_state=False,
        contract_mimo_out=False,
    )

    print(f"Test mamba3 mimo fwd: N={n}, P={p}, R={r}, chunk={chunk_size}")
    out_tilelang = out_tilelang.float().cpu()

    assert_stable_rel(
        out_tilelang,
        out_ref_fp32,
        label="O",
        cfg=f"N={n}, P={p}, R={r}, chunk={chunk_size}",
    )


if __name__ == "__main__":
    os.environ["TILELANG_ASCEND_MODE"] = "Dev"
    torch.npu.set_device(0)
    tilelang.cache.clear_cache()
    # test reduceO
    test_mimo_fwd_reduceO(case_grid[0], case_grid[1], case_grid[2], case_grid[3])
    # test Non-reduceO
    test_mimo_fwd(case_grid[0], case_grid[1], case_grid[2], case_grid[3])
    print("\033[92mAll check passed.\033[0m")
