"""Ascend 950 Dense Attention Prefill Forward reference skeleton (TileLang).

Spec Card
---------
- **Dynamicity**: Single compiled binary serves arbitrary runtime ``seq_q`` and
  ``seq_k`` (both multiples of 128, e.g. 1K, 2K, 8K, 64K) via ``T.dynamic``.
  JIT cache key is ``(batch, heads_q, heads_kv, head_dim, dtype_name,
  softmax_scale, core_count)`` — never ``seq_q`` or ``seq_k``.
- **Supported Scope**:
  - ``dtype``: ``torch.bfloat16`` or ``torch.float16`` (Q, K, V, Output share
    dtype; LSE is always ``torch.float32``).
  - ``head_dim``: ``128`` (flagship 128x128x128 tile, 3-stage KV L1, V lead=2).
  - Layouts: native BSHD ``Q/Output: [B, Sq, Hq, 128]``,
    ``K/V: [B, Sk, Hkv, 128]``, ``LSE: [B, Hq, Sq, 1]`` (natural-log
    ``ln(sum_k exp(s_k * scale))``).
  - Sequence lengths: ``seq_q > 0``, ``seq_k > 0``, ``seq_q % 128 == 0``,
    ``seq_k % 128 == 0`` (supports ``tiles_k == 1``, ``2``, and ``>= 3``).
  - GQA / MQA / MHA: any ``heads_q % heads_kv == 0``.
  - Configurable ``core_count`` (default ``32``).
- **Target Arch & Expected Tier**: Ascend 950 (1:2 Cube/Vector ratio);
  flagship dense prefill forward architecture verified at ~95%–96%
  frequency-corrected MFU (``aic_mac ~0.99``, ``aiv_vec ~0.88``) via:
  - Transposed QK (``S^T = K @ Q^T``) + hardware N-split FIXPIPE + columnwise
    FP32 256B SimdVF online softmax (zero SimtVF, zero ``vcmax``/``vcadd``).
  - Compact-NZ probability packing via ``S.vdintlv`` + ``S.vsstb`` + hardware
    L1-to-L0A transpose PV GEMM.
  - Fused ``S.vmadd`` output accumulation, single-vector reciprocal epilogue,
    and in-VF natural-log LSE via ``S.vln``.
  - Explicit 6-stage Cube / 2-stage Vector persistent cross-Q-tile ring
    pipeline with O(1) KV-head-major Zig-Zag wave scheduling for high L2 reuse.
- **Unsupported Boundaries**:
  - ``float32`` inputs, ``head_dim != 128``, unaligned ``seq_q``/``seq_k``
    (not divisible by 128), causal/sparse masks (extension hooks marked with
    ``# HOOK:``), and backward pass are outside this dense forward skeleton.
- **Related KB Playbook & Patterns**:
  - ``families/attention/PLAYBOOK.md``
  - ``patterns/softmax-col-fp32.md``
  - ``patterns/cv-explicit-ring-pipeline.md``
  - ``patterns/simd-vf-256b-fusion.md``
  - ``patterns/cube-fixpipe-hol-reordering.md``
"""

import argparse
import hashlib
import math
import os
import re
from typing import Any

import tilelang
import tilelang.ascend.language as T
import torch
import torch.nn.functional as F
from tilelang.ascend.language import simd as S
from tilelang.layout import make_ascend_compact_nz_layout

# Flagship Ascend 950 dense BF16/FP16 D=128 geometry.
_BR = 128
_BC = 128
_KC = 128
_HEAD_DIM = 128
_KV_STAGES = 3
_V_LEAD = 2
_ROWS = _BR // 2  # 64 rows per Vector sub-core (1:2 Cube:Vector ratio)
_HALF_K = _BC // 2  # 64 K columns per compact-NZ half
_VL = 64  # FP32 256B SIMD vector length
_NEG_SENTINEL = -1.0e30

_DTYPE_TO_NAME: dict[torch.dtype, str] = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
}


def _discover_cxx_flags() -> list[str]:
    """Locate host libstdc++ headers for ``bisheng --cce-aicore-only``."""
    flags: list[str] = []
    cxx_root = "/usr/include/c++"
    if os.path.isdir(cxx_root):
        versions = sorted(name for name in os.listdir(cxx_root) if name[:1].isdigit())
        if versions:
            base = os.path.join(cxx_root, versions[-1])
            flags.append("-I" + base)
            triples = [
                name
                for name in sorted(os.listdir(base))
                if os.path.isfile(os.path.join(base, name, "bits", "c++config.h"))
            ]
            triples.sort(key=lambda name: ("linux" not in name, name))
            if triples:
                flags.append("-I" + os.path.join(base, triples[0]))
    return flags


_CXX_FLAGS: list[str] = _discover_cxx_flags()


def _tile_dtype(dtype_name: str) -> str:
    """Validate and return a flagship TileLang half-precision dtype string."""
    if dtype_name in ("bfloat16", "float16"):
        return dtype_name
    raise ValueError(
        f"Unsupported dtype_name={dtype_name!r}; flagship dense kernel supports 'bfloat16' and 'float16'."
    )


@T.macro
def _manual_sync(event, index):
    T.ascend_set_flag(event, index)
    T.ascend_wait_flag(event, index)


@T.macro
def _manual_pair_set(pipe, flag):
    # Intra-block mode 4: distinct flag bank for each of the two AIV sub-cores.
    T.ascend_cross_core_set_flag(4, pipe, flag)
    T.ascend_cross_core_set_flag(4, pipe, flag + 16)


@T.macro
def _manual_pair_wait(pipe, flag):
    T.ascend_cross_core_wait_flag(4, pipe, flag)
    T.ascend_cross_core_wait_flag(4, pipe, flag + 16)


@tilelang.jit(
    target="ascend",
    compile_flags=_CXX_FLAGS,
    pass_configs={
        "tl.enable_auto_schedule": False,
        "tl.disable_shared_memory_reuse": True,
    },
)
def dense_attention_fwd_kernel(
    batch: int,
    heads_q: int,
    heads_kv: int,
    head_dim: int,
    dtype_name: str,
    softmax_scale: float,
    core_count: int = 32,
) -> Any:
    """Build and cache the Ascend 950 dense prefill forward TileLang kernel.

    Runtime tensor arguments (in-place Output and LSE):
        ``(Q, K, V, Output, LSE)``
    where ``seq_q`` and ``seq_k`` are runtime dynamic dimensions.
    """
    dtype = _tile_dtype(dtype_name)
    if head_dim != _HEAD_DIM:
        raise ValueError(
            f"Unsupported head_dim={head_dim}; flagship dense kernel requires head_dim=={_HEAD_DIM}."
        )
    if batch <= 0 or heads_q <= 0 or heads_kv <= 0:
        raise ValueError("batch, heads_q, and heads_kv must be positive integers.")
    if heads_q % heads_kv != 0:
        raise ValueError(f"heads_q ({heads_q}) must be divisible by heads_kv ({heads_kv}).")
    if core_count <= 0:
        raise ValueError(f"core_count must be positive, got {core_count}.")
    scale = float(softmax_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"softmax_scale must be finite and positive, got {softmax_scale}.")

    BR = _BR
    BC = _BC
    KC = _KC
    KV_STAGES = _KV_STAGES
    V_LEAD = _V_LEAD
    ROWS = _ROWS
    HALF_K = _HALF_K
    VL = _VL
    D = _HEAD_DIM

    groups = heads_q // heads_kv
    total_kv_heads = batch * heads_kv
    rot_step = groups if groups < core_count else 1
    neg = T.float32(_NEG_SENTINEL)

    # Runtime dynamic sequence dimensions (single binary for all multiples of 128).
    seq_q = T.dynamic("seq_q")
    seq_k = T.dynamic("seq_k")
    tiles_q = seq_q // BR
    tiles_k = seq_k // BC

    q_shape = [batch, seq_q, heads_q, D]
    kv_shape = [batch, seq_k, heads_kv, D]
    lse_shape = [batch, heads_q, seq_q, 1]

    @T.macro
    def softmax_fast(
        score,
        prob,
        maximum,
        denominator,
        alpha,
        ts,
        step_seq,
        first,
    ):
        with T.SimdVF():
            full = S.pset(32, "PAT_ALL")
            b16 = S.pset(16, "PAT_VL128")
            ss = step_seq % 3
            pslot = step_seq % 2
            dst0 = S.make_ubuf_ptr(
                T.access_ptr(prob[pslot, 0, 0, 0], "w", HALF_K + 1, ROWS), dtype
            )
            dst1 = S.make_ubuf_ptr(
                T.access_ptr(prob[pslot, 0, 32, 0], "w", HALF_K + 1, ROWS), dtype
            )

            # Pass 1: 4-way unrolled columnwise max over K=0..BC-1 on S^T[BC, ROWS].
            # Zero vcmax, zero vsts_1st, zero vlds_brc_elem, zero mem_bar.
            # HOOK: attention bias / mask on S^T — to add an additive bias or mask
            # (e.g. causal / ALiBi / sliding window), scale & apply bias/vsel to
            # score[ss, k, 0] here, write back + S.mem_bar("VST_VLD"), and read
            # pre-scaled scores directly in Pass 2.
            m0 = S.alloc_local((1,), "float32")
            m1 = S.alloc_local((1,), "float32")
            m2 = S.alloc_local((1,), "float32")
            m3 = S.alloc_local((1,), "float32")
            m0[0] = S.vld(score[ss, 0, 0], "NORM")
            m1[0] = S.vld(score[ss, 1, 0], "NORM")
            m2[0] = S.vld(score[ss, 2, 0], "NORM")
            m3[0] = S.vld(score[ss, 3, 0], "NORM")
            for k in range(4, BC, 4):
                m0[0] = S.vmax(m0[0], S.vld(score[ss, k, 0], "NORM"), full)
                m1[0] = S.vmax(m1[0], S.vld(score[ss, k + 1, 0], "NORM"), full)
                m2[0] = S.vmax(m2[0], S.vld(score[ss, k + 2, 0], "NORM"), full)
                m3[0] = S.vmax(m3[0], S.vld(score[ss, k + 3, 0], "NORM"), full)
            m0[0] = S.vmax(S.vmax(m0[0], m2[0], full), S.vmax(m1[0], m3[0], full), full)
            m_cur = S.vmuls(m0[0], T.float32(scale), full)
            if first:
                m_new = m_cur
                S.vsts(maximum[ts, 0], m_new, full, "NORM_B32")
            else:
                m_old = S.vld(maximum[ts, 0], "NORM")
                m_new = S.vmax(m_cur, m_old, full)
                a = S.vexpdif(m_old, m_new, full)
                l_scaled = S.vmul(S.vld(denominator[ts, 0], "NORM"), a, full)
                S.vsts(alpha[ss, 0], a, full, "NORM_B32")
                S.vsts(maximum[ts, 0], m_new, full, "NORM_B32")

            # Pass 2: 4-way unrolled scale + vexpdif + columnwise sum + compact-NZ vsstb.
            s0 = S.alloc_local((1,), "float32")
            s1 = S.alloc_local((1,), "float32")
            s2 = S.alloc_local((1,), "float32")
            s3 = S.alloc_local((1,), "float32")
            z = S.vdup(T.float32(0.0), "float32", full)
            s0[0] = z
            s1[0] = z
            s2[0] = z
            s3[0] = z
            for i in range(32):
                x0 = S.vmuls(S.vld(score[ss, i, 0], "NORM"), T.float32(scale), full)
                x1 = S.vmuls(S.vld(score[ss, i + 32, 0], "NORM"), T.float32(scale), full)
                x2 = S.vmuls(S.vld(score[ss, i + 64, 0], "NORM"), T.float32(scale), full)
                x3 = S.vmuls(S.vld(score[ss, i + 96, 0], "NORM"), T.float32(scale), full)
                e0 = S.vexpdif(x0, m_new, full)
                e1 = S.vexpdif(x1, m_new, full)
                e2 = S.vexpdif(x2, m_new, full)
                e3 = S.vexpdif(x3, m_new, full)
                c0 = S.vcvt(e0, dtype, full, sat=False, part=0)
                c2 = S.vcvt(e2, dtype, full, sat=False, part=0)
                de0, _ = S.vdintlv(c0, c2)
                c1 = S.vcvt(e1, dtype, full, sat=False, part=0)
                c3 = S.vcvt(e3, dtype, full, sat=False, part=0)
                de1, _ = S.vdintlv(c1, c3)
                dst0 = S.vsstb(
                    de0, dst0, T.int32(((HALF_K + 1) << 16) | 1), b16, update=True
                )
                s0[0] = S.vadd(s0[0], e0, full)
                s2[0] = S.vadd(s2[0], e2, full)
                dst1 = S.vsstb(
                    de1, dst1, T.int32(((HALF_K + 1) << 16) | 1), b16, update=True
                )
                s1[0] = S.vadd(s1[0], e1, full)
                s3[0] = S.vadd(s3[0], e3, full)
            row_sum_v = S.vadd(S.vadd(s0[0], s2[0], full), S.vadd(s1[0], s3[0], full), full)
            if first:
                S.vsts(denominator[ts, 0], row_sum_v, full, "NORM_B32")
            else:
                S.vsts(denominator[ts, 0], S.vadd(l_scaled, row_sum_v, full), full, "NORM_B32")

    @T.macro
    def update_output(
        pv,
        accum,
        output,
        lse_ub,
        alpha,
        denominator,
        maximum,
        safe_denom,
        ts,
        step_seq,
        first,
        last,
    ):
        with T.SimdVF():
            full = S.pset(32, "PAT_ALL")
            if last:
                l_vec = S.vld(denominator[ts, 0], "NORM")
                m_vec = S.vld(maximum[ts, 0], "NORM")
                alive = S.vcmps(l_vec, T.float32(0.0), full, op="gt")
                one_vec = S.vdup(T.float32(1.0), "float32", full)
                neg_vec = S.vdup(neg, "float32", full)
                safe_l = S.vsel(l_vec, one_vec, alive)
                inv_l = S.vdiv(one_vec, safe_l, full)
                lse_val = S.vsel(S.vadd(S.vln(safe_l, full), m_vec, full), neg_vec, alive)
                S.vsts(lse_ub[0, 0], lse_val, full, "NORM_B32")
                S.vsts(safe_denom[0], inv_l, full, "NORM_B32")
                S.mem_bar("VST_VLD")

            if first and not last:
                for r in range(0, ROWS, 2):
                    c00 = S.vld(pv[step_seq % 2, r, 0], "NORM")
                    c01 = S.vld(pv[step_seq % 2, r, VL], "NORM")
                    c10 = S.vld(pv[step_seq % 2, r + 1, 0], "NORM")
                    c11 = S.vld(pv[step_seq % 2, r + 1, VL], "NORM")
                    S.vsts(accum[r, 0], c00, full, "NORM_B32")
                    S.vsts(accum[r, VL], c01, full, "NORM_B32")
                    S.vsts(accum[r + 1, 0], c10, full, "NORM_B32")
                    S.vsts(accum[r + 1, VL], c11, full, "NORM_B32")
            elif not first and not last:
                acc00 = S.alloc_local((1,), "float32")
                acc01 = S.alloc_local((1,), "float32")
                acc10 = S.alloc_local((1,), "float32")
                acc11 = S.alloc_local((1,), "float32")
                for r in range(0, ROWS, 2):
                    a0 = S.vld(alpha[step_seq % 3, r], "BRC_B32")
                    a1 = S.vld(alpha[step_seq % 3, r + 1], "BRC_B32")
                    acc00[0] = S.vld(accum[r, 0], "NORM")
                    acc01[0] = S.vld(accum[r, VL], "NORM")
                    acc10[0] = S.vld(accum[r + 1, 0], "NORM")
                    acc11[0] = S.vld(accum[r + 1, VL], "NORM")
                    cur00 = S.vld(pv[step_seq % 2, r, 0], "NORM")
                    cur01 = S.vld(pv[step_seq % 2, r, VL], "NORM")
                    cur10 = S.vld(pv[step_seq % 2, r + 1, 0], "NORM")
                    cur11 = S.vld(pv[step_seq % 2, r + 1, VL], "NORM")
                    S.vmadd(acc00[0], a0, cur00, full)
                    S.vmadd(acc01[0], a0, cur01, full)
                    S.vmadd(acc10[0], a1, cur10, full)
                    S.vmadd(acc11[0], a1, cur11, full)
                    S.vsts(accum[r, 0], acc00[0], full, "NORM_B32")
                    S.vsts(accum[r, VL], acc01[0], full, "NORM_B32")
                    S.vsts(accum[r + 1, 0], acc10[0], full, "NORM_B32")
                    S.vsts(accum[r + 1, VL], acc11[0], full, "NORM_B32")
            elif first and last:
                # HOOK: custom epilogue / quant — fuse output scaling, activation, or
                # quantization here before storing the final normalized rows.
                for r in range(0, ROWS, 2):
                    inv0 = S.vld(safe_denom[r], "BRC_B32")
                    inv1 = S.vld(safe_denom[r + 1], "BRC_B32")
                    c00 = S.vld(pv[step_seq % 2, r, 0], "NORM")
                    c01 = S.vld(pv[step_seq % 2, r, VL], "NORM")
                    c10 = S.vld(pv[step_seq % 2, r + 1, 0], "NORM")
                    c11 = S.vld(pv[step_seq % 2, r + 1, VL], "NORM")
                    S.vsts(
                        output[r, 0], S.vcvt(S.vmul(c00, inv0, full), dtype, full), full, "PK_B32"
                    )
                    S.vsts(
                        output[r, VL], S.vcvt(S.vmul(c01, inv0, full), dtype, full), full, "PK_B32"
                    )
                    S.vsts(
                        output[r + 1, 0],
                        S.vcvt(S.vmul(c10, inv1, full), dtype, full),
                        full,
                        "PK_B32",
                    )
                    S.vsts(
                        output[r + 1, VL],
                        S.vcvt(S.vmul(c11, inv1, full), dtype, full),
                        full,
                        "PK_B32",
                    )
            else:
                # HOOK: custom epilogue / quant — fuse output scaling, activation, or
                # quantization here before storing the final normalized rows.
                acc00 = S.alloc_local((1,), "float32")
                acc01 = S.alloc_local((1,), "float32")
                acc10 = S.alloc_local((1,), "float32")
                acc11 = S.alloc_local((1,), "float32")
                for r in range(0, ROWS, 2):
                    a0 = S.vld(alpha[step_seq % 3, r], "BRC_B32")
                    a1 = S.vld(alpha[step_seq % 3, r + 1], "BRC_B32")
                    inv0 = S.vld(safe_denom[r], "BRC_B32")
                    inv1 = S.vld(safe_denom[r + 1], "BRC_B32")
                    acc00[0] = S.vld(accum[r, 0], "NORM")
                    acc01[0] = S.vld(accum[r, VL], "NORM")
                    acc10[0] = S.vld(accum[r + 1, 0], "NORM")
                    acc11[0] = S.vld(accum[r + 1, VL], "NORM")
                    cur00 = S.vld(pv[step_seq % 2, r, 0], "NORM")
                    cur01 = S.vld(pv[step_seq % 2, r, VL], "NORM")
                    cur10 = S.vld(pv[step_seq % 2, r + 1, 0], "NORM")
                    cur11 = S.vld(pv[step_seq % 2, r + 1, VL], "NORM")
                    S.vmadd(acc00[0], a0, cur00, full)
                    S.vmadd(acc01[0], a0, cur01, full)
                    S.vmadd(acc10[0], a1, cur10, full)
                    S.vmadd(acc11[0], a1, cur11, full)
                    S.vsts(
                        output[r, 0],
                        S.vcvt(S.vmul(acc00[0], inv0, full), dtype, full),
                        full,
                        "PK_B32",
                    )
                    S.vsts(
                        output[r, VL],
                        S.vcvt(S.vmul(acc01[0], inv0, full), dtype, full),
                        full,
                        "PK_B32",
                    )
                    S.vsts(
                        output[r + 1, 0],
                        S.vcvt(S.vmul(acc10[0], inv1, full), dtype, full),
                        full,
                        "PK_B32",
                    )
                    S.vsts(
                        output[r + 1, VL],
                        S.vcvt(S.vmul(acc11[0], inv1, full), dtype, full),
                        full,
                        "PK_B32",
                    )

    @T.prim_func
    def dense_attention_fwd_main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        Output: T.Tensor(q_shape, dtype),
        LSE: T.Tensor(lse_shape, "float32"),
    ):
        with T.Kernel(core_count) as cid:
            T.assume(seq_q > 0)
            T.assume(seq_k > 0)
            T.assume(seq_q % BR == 0)
            T.assume(seq_k % BC == 0)

            # Cube on-chip ring buffers
            q_l1 = T.alloc_l1((2, BR, D), dtype)
            k_l1 = T.alloc_l1((KV_STAGES, BC, D), dtype)
            v_l1 = T.alloc_l1((KV_STAGES, D, BC), dtype)
            p_l1 = T.alloc_l1((2, BC, BR), dtype)
            a_l0 = T.alloc_l0a((2, BR, KC), dtype)
            b_l0 = T.alloc_l0b((2, D, KC), dtype)
            qk_l0 = T.alloc_l0c((2, BC, BR), "float32")
            pv_l0 = T.alloc_l0c((2, BR, D), "float32")

            # Vector UB ring buffers (~249 KB total, within 256 KB UB budget)
            score = T.alloc_shared((3, BC, ROWS), "float32")
            prob = T.alloc_shared((2, 2, HALF_K + 1, ROWS), dtype)
            T.annotate_layout({prob: make_ascend_compact_nz_layout(prob)})
            pv = T.alloc_shared((2, ROWS, D), "float32")
            accum = T.alloc_shared((ROWS, D), "float32")
            output = T.alloc_shared((ROWS, D), dtype)
            maximum = T.alloc_shared((3, ROWS), "float32")
            denominator = T.alloc_shared((3, ROWS), "float32")
            alpha = T.alloc_shared((3, ROWS), "float32")
            safe_denom = T.alloc_shared((ROWS,), "float32")
            lse_ub = T.alloc_shared((ROWS, 1), "float32")

            # O(1) KV-head-major Zig-Zag wave scheduling (no host metadata tensors).
            # Every wave w < num_waves - 1 assigns a valid tid < total_tasks to every
            # core; only the final wave may be partial. Hence a core's valid waves form
            # a contiguous prefix 0 .. my_tasks - 1, and total KV steps = my_tasks * tiles_k.
            total_tasks = total_kv_heads * tiles_q * groups
            num_waves = T.ceildiv(total_tasks, core_count)
            last_w = num_waves - 1
            rot_idx_last = (last_w >> 1) ^ (last_w >> 3) ^ (last_w >> 5) ^ (last_w >> 7)
            rot_last = T.floormod(cid + rot_idx_last * rot_step, core_count)
            slot_last = T.if_then_else((last_w & 1) == 0, rot_last, core_count - 1 - rot_last)
            my_tasks = T.if_then_else(
                last_w * core_count + slot_last < total_tasks,
                num_waves,
                num_waves - 1,
            )
            # HOOK: causal / sliding-window / sparse KRange bounds — for variable KV
            # spans per Q tile, sum each assigned task's [k_lo, k_hi) length into
            # `total` here and load per-task (k_lo, k_span) inside each cursor below.
            total = T.alloc_var("int32", init=0)
            total = my_tasks * tiles_k

            # --- Persistent Cross-Q-Tile Cube/Vector Ring Pipeline ---
            with T.Cube():
                if total > 0:
                    # Seed initial free-slot tokens:
                    # - MTE1_MTE2[0..2]: v_l1[0..2], [3..5]: k_l1[0..2], [6..7]: q_l1[0..1]
                    # - FIX_M[0..1]: pv_l0[0..1], [2..3]: qk_l0[0..1]
                    # - M_MTE1[0..1]: a_l0/b_l0[0..1]
                    # - Cross-core mode-4 PIPE_MTE1 flags {5,21} / {12,28}: p_l1[0..1] free
                    for e in T.serial(8):
                        T.ascend_set_flag("MTE1_MTE2", e)
                    for e in T.serial(4):
                        T.ascend_set_flag("FIX_M", e)
                    for e in T.serial(2):
                        T.ascend_set_flag("M_MTE1", e)
                        T.ascend_cross_core_set_flag(4, "PIPE_MTE1", 5 + e * 16)
                        T.ascend_cross_core_set_flag(4, "PIPE_MTE1", 12 + e * 16)

                    # Cursor K/Q (step)
                    tk = T.alloc_var("int32", init=0)
                    kk = T.alloc_var("int32", init=0)
                    k_bz = T.alloc_var("int32", init=0)
                    k_head = T.alloc_var("int32", init=0)
                    k_kvh = T.alloc_var("int32", init=0)
                    k_qt = T.alloc_var("int32", init=0)

                    # Cursor QK (step - 1)
                    tq = T.alloc_var("int32", init=0)
                    kq = T.alloc_var("int32", init=0)

                    # Cursor V (step - V_LEAD)
                    tv = T.alloc_var("int32", init=0)
                    kv = T.alloc_var("int32", init=0)
                    v_bz = T.alloc_var("int32", init=0)
                    v_kvh = T.alloc_var("int32", init=0)

                    l0_seq = T.alloc_var("int32", init=0)

                    for step in T.serial(total + 5):
                        # 1. Feed older V(step - V_LEAD) before prefetching future Q/K.
                        if step >= V_LEAD and step < total + V_LEAD:
                            g = step - V_LEAD
                            if kv == 0:
                                rot_iv = (tv >> 1) ^ (tv >> 3) ^ (tv >> 5) ^ (tv >> 7)
                                rot_v = T.floormod(cid + rot_iv * rot_step, core_count)
                                slot_v = T.if_then_else(
                                    (tv & 1) == 0, rot_v, core_count - 1 - rot_v
                                )
                                tid_v = tv * core_count + slot_v
                                qg_v = tid_v // groups
                                kv_idx_v = qg_v // tiles_q
                                v_bz = kv_idx_v // heads_kv
                                v_kvh = kv_idx_v - v_bz * heads_kv
                            v_kb = kv
                            T.assume(0 <= v_bz < batch)
                            T.assume(0 <= v_kvh < heads_kv)
                            T.assume(0 <= v_kb < tiles_k)
                            T.ascend_wait_flag("MTE1_MTE2", g % KV_STAGES)
                            T.copy(
                                V[v_bz, v_kb * BC : (v_kb + 1) * BC, v_kvh, :],
                                v_l1[g % KV_STAGES, :, :],
                                transpose=True,
                            )
                            T.ascend_set_flag("MTE2_MTE1", g % KV_STAGES)
                            kv += 1
                            if kv == tiles_k:
                                kv = 0
                                tv += 1

                        # 2. Prefetch Q/K(step).
                        if step < total:
                            if kk == 0:
                                rot_ik = (tk >> 1) ^ (tk >> 3) ^ (tk >> 5) ^ (tk >> 7)
                                rot_k = T.floormod(cid + rot_ik * rot_step, core_count)
                                slot_k = T.if_then_else(
                                    (tk & 1) == 0, rot_k, core_count - 1 - rot_k
                                )
                                tid_k = tk * core_count + slot_k
                                qg_k = tid_k // groups
                                lh_k = tid_k - qg_k * groups
                                kv_idx_k = qg_k // tiles_q
                                k_qt = qg_k - kv_idx_k * tiles_q
                                k_bz = kv_idx_k // heads_kv
                                k_kvh = kv_idx_k - k_bz * heads_kv
                                k_head = k_kvh * groups + lh_k
                                T.assume(0 <= k_bz < batch)
                                T.assume(0 <= k_head < heads_q)
                                T.assume(0 <= k_qt < tiles_q)
                                qslot_k = T.floormod(tk, 2)
                                T.assume(0 <= qslot_k < 2)
                                T.ascend_wait_flag("MTE1_MTE2", 6 + qslot_k)
                                T.copy(
                                    Q[k_bz, k_qt * BR : (k_qt + 1) * BR, k_head, :],
                                    q_l1[qslot_k, :, :],
                                    l2_cache_ctrl="NOTALLOC_KEEP",
                                )
                                T.ascend_set_flag("MTE2_MTE1", 6 + qslot_k)
                            k_kb = kk
                            T.assume(0 <= k_bz < batch)
                            T.assume(0 <= k_kvh < heads_kv)
                            T.assume(0 <= k_kb < tiles_k)
                            T.ascend_wait_flag("MTE1_MTE2", 3 + step % KV_STAGES)
                            T.copy(
                                K[k_bz, k_kb * BC : (k_kb + 1) * BC, k_kvh, :],
                                k_l1[step % KV_STAGES, :, :],
                            )
                            T.ascend_set_flag("MTE2_MTE1", 3 + step % KV_STAGES)
                            kk += 1
                            if kk == tiles_k:
                                kk = 0
                                tk += 1

                        # 3. Execute QK GEMM(step - 1): S^T = K @ Q^T -> [128_k, 128_q].
                        if step >= 1 and step < total + 1:
                            g = step - 1
                            qslot_q = T.floormod(tq, 2)
                            T.assume(0 <= qslot_q < 2)
                            T.ascend_wait_flag("FIX_M", 2 + g % 2)
                            if kq == 0:
                                T.ascend_wait_flag("MTE2_MTE1", 6 + qslot_q)
                            T.ascend_wait_flag("MTE2_MTE1", 3 + g % KV_STAGES)
                            ls = T.floormod(l0_seq, 2)
                            T.assume(0 <= ls < 2)
                            T.ascend_wait_flag("M_MTE1", ls)
                            T.copy(k_l1[g % KV_STAGES, :, :], a_l0[ls, :, :])
                            T.copy(q_l1[qslot_q, :, :], b_l0[ls, :, :])
                            _manual_sync("MTE1_M", ls)
                            T.gemm(
                                a_l0[ls, :, :],
                                b_l0[ls, :, :],
                                qk_l0[g % 2, :, :],
                                transpose_B=True,
                                clear_accum=True,
                            )
                            T.ascend_set_flag("M_MTE1", ls)
                            l0_seq += 1
                            T.ascend_set_flag("M_FIX", 2 + g % 2)
                            T.ascend_set_flag("MTE1_MTE2", 3 + g % KV_STAGES)
                            kq += 1
                            if kq == tiles_k:
                                T.ascend_set_flag("MTE1_MTE2", 6 + qslot_q)
                                kq = 0
                                tq += 1

                        # 4. Publish ready QK score(step - 2) BEFORE waiting on older PV
                        #    (eliminates FIXPIPE head-of-line blocking).
                        if step >= 2 and step < total + 2:
                            g = step - 2
                            _manual_pair_wait("PIPE_FIX", g % 3)
                            T.ascend_wait_flag("M_FIX", 2 + g % 2)
                            T.dual_copy(qk_l0[g % 2, :, :], score[g % 3, :, :])
                            _manual_pair_set("PIPE_FIX", 6 + g % 3)
                            T.ascend_set_flag("FIX_M", 2 + g % 2)

                        # 5. Execute PV GEMM(step - 4): (P^T)^T @ V -> [128_q, 128_d].
                        if step >= 4 and step < total + 4:
                            g = step - 4
                            ready = T.if_then_else(g % 2 == 0, 11, 13)
                            free = T.if_then_else(g % 2 == 0, 5, 12)
                            T.ascend_wait_flag("FIX_M", g % 2)
                            T.ascend_wait_flag("MTE2_MTE1", g % KV_STAGES)
                            _manual_pair_wait("PIPE_MTE1", ready)
                            ls = T.floormod(l0_seq, 2)
                            T.assume(0 <= ls < 2)
                            T.ascend_wait_flag("M_MTE1", ls)
                            T.copy(p_l1[g % 2, :, :], a_l0[ls, :, :], transpose=True)
                            T.copy(v_l1[g % KV_STAGES, :, :], b_l0[ls, :, :])
                            _manual_sync("MTE1_M", ls)
                            T.gemm(
                                a_l0[ls, :, :],
                                b_l0[ls, :, :],
                                pv_l0[g % 2, :, :],
                                transpose_B=True,
                                clear_accum=True,
                            )
                            T.ascend_set_flag("M_MTE1", ls)
                            l0_seq += 1
                            T.ascend_set_flag("M_FIX", g % 2)
                            _manual_pair_set("PIPE_MTE1", free)
                            T.ascend_set_flag("MTE1_MTE2", g % KV_STAGES)

                        # 6. Publish PV output(step - 5) to Vector UB.
                        if step >= 5:
                            g = step - 5
                            _manual_pair_wait("PIPE_FIX", 3 + g % 2)
                            T.ascend_wait_flag("M_FIX", g % 2)
                            T.dual_copy(pv_l0[g % 2, :, :], pv[g % 2, :, :])
                            _manual_pair_set("PIPE_FIX", 9 + g % 2)
                            T.ascend_set_flag("FIX_M", g % 2)

                    # Drain all Cube intra-core and cross-core tokens once.
                    for e in T.serial(8):
                        T.ascend_wait_flag("MTE1_MTE2", e)
                    for e in T.serial(4):
                        T.ascend_wait_flag("FIX_M", e)
                    for sub in T.serial(2):
                        T.ascend_wait_flag("M_MTE1", sub)
                        for e in T.serial(5):
                            T.ascend_cross_core_wait_flag(4, "PIPE_FIX", e + sub * 16)

            with T.Vector() as sid:
                if total > 0:
                    # Seed initial Vector tokens:
                    # - Cross-core mode-4 PIPE_V[0..2]: score[0..2] free, [3..4]: pv[0..1] free
                    # - MTE3_V[0..1]: prob[0..1] free, [2]: output/lse_ub free
                    for e in T.serial(5):
                        T.ascend_cross_core_set_flag(4, "PIPE_V", e)
                    for e in T.serial(3):
                        T.ascend_set_flag("MTE3_V", e)

                    # Softmax cursor (step)
                    st = T.alloc_var("int32", init=0)
                    skv = T.alloc_var("int32", init=0)

                    # Output cursor (step - 3)
                    ot = T.alloc_var("int32", init=0)
                    okv = T.alloc_var("int32", init=0)
                    o_bz = T.alloc_var("int32", init=0)
                    o_head = T.alloc_var("int32", init=0)
                    o_qt = T.alloc_var("int32", init=0)

                    for step in T.serial(total + 3):
                        # 1. Consume PV(step - 3) BEFORE softmax(step) overwrites alpha[step % 3]
                        #    and maximum/denominator[st % 3].
                        if step >= 3:
                            g = step - 3
                            if okv == 0:
                                rot_io = (ot >> 1) ^ (ot >> 3) ^ (ot >> 5) ^ (ot >> 7)
                                rot_o = T.floormod(cid + rot_io * rot_step, core_count)
                                slot_o = T.if_then_else(
                                    (ot & 1) == 0, rot_o, core_count - 1 - rot_o
                                )
                                tid_o = ot * core_count + slot_o
                                qg_o = tid_o // groups
                                lh_o = tid_o - qg_o * groups
                                kv_idx_o = qg_o // tiles_q
                                o_qt = qg_o - kv_idx_o * tiles_q
                                o_bz = kv_idx_o // heads_kv
                                o_kvh_l = kv_idx_o - o_bz * heads_kv
                                o_head = o_kvh_l * groups + lh_o
                            o_last = okv == tiles_k - 1
                            ots = T.floormod(ot, 3)
                            T.assume(0 <= ots < 3)
                            T.ascend_cross_core_wait_flag(4, "PIPE_V", 9 + g % 2)
                            if o_last:
                                T.ascend_wait_flag("MTE3_V", 2)
                                if okv == 0:
                                    update_output(
                                        pv,
                                        accum,
                                        output,
                                        lse_ub,
                                        alpha,
                                        denominator,
                                        maximum,
                                        safe_denom,
                                        ots,
                                        g,
                                        True,
                                        True,
                                    )
                                else:
                                    update_output(
                                        pv,
                                        accum,
                                        output,
                                        lse_ub,
                                        alpha,
                                        denominator,
                                        maximum,
                                        safe_denom,
                                        ots,
                                        g,
                                        False,
                                        True,
                                    )
                            elif okv == 0:
                                update_output(
                                    pv,
                                    accum,
                                    output,
                                    lse_ub,
                                    alpha,
                                    denominator,
                                    maximum,
                                    safe_denom,
                                    ots,
                                    g,
                                    True,
                                    False,
                                )
                            else:
                                update_output(
                                    pv,
                                    accum,
                                    output,
                                    lse_ub,
                                    alpha,
                                    denominator,
                                    maximum,
                                    safe_denom,
                                    ots,
                                    g,
                                    False,
                                    False,
                                )
                            T.ascend_cross_core_set_flag(4, "PIPE_V", 3 + g % 2)
                            if o_last:
                                T.assume(0 <= o_bz < batch)
                                T.assume(0 <= o_head < heads_q)
                                T.assume(0 <= o_qt < tiles_q)
                                row_s = o_qt * BR + sid * ROWS
                                _manual_sync("V_MTE3", 2)
                                T.copy(
                                    output,
                                    Output[o_bz, row_s : row_s + ROWS, o_head, :],
                                    l2_cache_ctrl="NOTALLOC_KEEP",
                                )
                                T.copy(
                                    lse_ub,
                                    LSE[o_bz, o_head, row_s : row_s + ROWS, :],
                                )
                                T.ascend_set_flag("MTE3_V", 2)
                                okv = 0
                                ot += 1
                            else:
                                okv += 1

                        # 2. Run online softmax(step) -> direct compact-NZ P -> UB-to-L1 copy.
                        if step < total:
                            sts = T.floormod(st, 3)
                            T.assume(0 <= sts < 3)
                            T.ascend_wait_flag("MTE3_V", step % 2)
                            T.ascend_cross_core_wait_flag(4, "PIPE_V", 6 + step % 3)
                            if skv == 0:
                                softmax_fast(
                                    score,
                                    prob,
                                    maximum,
                                    denominator,
                                    alpha,
                                    sts,
                                    step,
                                    True,
                                )
                            else:
                                softmax_fast(
                                    score,
                                    prob,
                                    maximum,
                                    denominator,
                                    alpha,
                                    sts,
                                    step,
                                    False,
                                )
                            T.ascend_set_flag("V_MTE3", step % 2)
                            T.ascend_cross_core_set_flag(4, "PIPE_V", step % 3)
                            T.ascend_cross_core_wait_flag(
                                4, "PIPE_MTE3", T.if_then_else(step % 2 == 0, 5, 12)
                            )
                            T.ascend_wait_flag("V_MTE3", step % 2)
                            T.dual_copy(
                                prob[step % 2, 0, :HALF_K, :],
                                p_l1[step % 2, 0:HALF_K, :],
                            )
                            T.dual_copy(
                                prob[step % 2, 1, :HALF_K, :],
                                p_l1[step % 2, HALF_K:BC, :],
                            )
                            T.ascend_cross_core_set_flag(
                                4, "PIPE_MTE3", T.if_then_else(step % 2 == 0, 11, 13)
                            )
                            T.ascend_set_flag("MTE3_V", step % 2)
                            skv += 1
                            if skv == tiles_k:
                                skv = 0
                                st += 1

                    # Drain Vector intra-core and cross-core tokens once.
                    for e in T.serial(3):
                        T.ascend_wait_flag("MTE3_V", e)
                    T.ascend_cross_core_wait_flag(4, "PIPE_MTE3", 5)
                    T.ascend_cross_core_wait_flag(4, "PIPE_MTE3", 12)

    return dense_attention_fwd_main


def dense_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float | None = None,
    core_count: int = 32,
    *,
    out: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run dense prefill forward attention on Ascend 950 (native BSHD layout).

    The compiled kernel is cached by
    ``(batch, heads_q, heads_kv, head_dim, dtype_name, softmax_scale, core_count)``
    and dynamically reused across arbitrary runtime ``seq_q`` and ``seq_k``
    multiples of 128 without recompilation.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Q, K, V must be 4D BSHD tensors: [B, S, H, D].")
    if q.device != k.device or q.device != v.device:
        raise ValueError("Q, K, V must reside on the same device.")
    if q.dtype not in _DTYPE_TO_NAME or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("Q, K, V must share dtype torch.bfloat16 or torch.float16.")

    batch, seq_q, heads_q, head_dim = q.shape
    batch_k, seq_k, heads_kv, head_dim_k = k.shape
    if batch_k != batch or v.shape != k.shape or head_dim_k != head_dim:
        raise ValueError("Q, K, V must match on batch and head_dim, and K/V shapes must match.")
    if head_dim != _HEAD_DIM:
        raise ValueError(
            f"Unsupported head_dim={head_dim}; flagship dense kernel requires head_dim=={_HEAD_DIM}."
        )
    if min(batch, seq_q, seq_k, heads_q, heads_kv) <= 0:
        raise ValueError("All tensor dimensions must be positive.")
    if heads_q % heads_kv != 0:
        raise ValueError(f"heads_q ({heads_q}) must be divisible by heads_kv ({heads_kv}).")
    if seq_q % _BR != 0 or seq_k % _BC != 0:
        raise ValueError(
            f"seq_q ({seq_q}) and seq_k ({seq_k}) must be divisible by {_BR}/{_BC}."
        )
    if batch * heads_q * (seq_q // _BR) > 2**31 - 1:
        raise ValueError("Total Q tile grid exceeds signed 32-bit index range.")
    if core_count <= 0:
        raise ValueError(f"core_count must be positive, got {core_count}.")

    scale = head_dim**-0.5 if softmax_scale is None else float(softmax_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"softmax_scale must be finite and positive, got {softmax_scale}.")

    q_c = q if q.is_contiguous() else q.contiguous()
    k_c = k if k.is_contiguous() else k.contiguous()
    v_c = v if v.is_contiguous() else v.contiguous()

    expected_lse_shape = (batch, heads_q, seq_q, 1)
    if out is None:
        out_t = torch.empty_like(q_c)
    else:
        if out.shape != q_c.shape or out.dtype != q_c.dtype or out.device != q_c.device:
            raise ValueError("Pre-allocated `out` must match Q shape, dtype, and device.")
        if not out.is_contiguous():
            raise ValueError("Pre-allocated `out` must be contiguous.")
        out_t = out

    if lse is None:
        lse_t = torch.empty(expected_lse_shape, dtype=torch.float32, device=q_c.device)
    else:
        if (
            tuple(lse.shape) != expected_lse_shape
            or lse.dtype != torch.float32
            or lse.device != q_c.device
        ):
            raise ValueError(
                f"Pre-allocated `lse` must have shape {expected_lse_shape}, dtype float32, and match Q device."
            )
        if not lse.is_contiguous():
            raise ValueError("Pre-allocated `lse` must be contiguous.")
        lse_t = lse

    dtype_name = _DTYPE_TO_NAME[q_c.dtype]
    kernel = dense_attention_fwd_kernel(
        batch,
        heads_q,
        heads_kv,
        head_dim,
        dtype_name,
        scale,
        core_count,
    )
    kernel(q_c, k_c, v_c, out_t, lse_t)
    return out_t, lse_t


def reference_dense_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure FP32 PyTorch reference for dense BSHD attention forward.

    Returns ``(out, lse)`` in ``torch.float32`` with shapes ``[B, Sq, Hq, D]``
    and ``[B, Hq, Sq, 1]``, matching the exact natural-log LSE semantics
    ``ln(sum_k exp(s_k * scale))`` of the Ascend kernel.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Q, K, V must be 4D BSHD tensors: [B, S, H, D].")
    batch, _, heads_q, head_dim = q.shape
    batch_k, _, heads_kv, head_dim_k = k.shape
    if batch_k != batch or v.shape != k.shape or head_dim_k != head_dim:
        raise ValueError("Q, K, V shape mismatch.")
    if heads_q % heads_kv != 0:
        raise ValueError(f"heads_q ({heads_q}) must be divisible by heads_kv ({heads_kv}).")

    scale = head_dim**-0.5 if softmax_scale is None else float(softmax_scale)
    groups = heads_q // heads_kv

    q_f = q.to(torch.float32)
    k_f = k.to(torch.float32)
    v_f = v.to(torch.float32)
    if groups > 1:
        k_f = k_f.repeat_interleave(groups, dim=2)
        v_f = v_f.repeat_interleave(groups, dim=2)

    q_bh = q_f.permute(0, 2, 1, 3)  # [B, Hq, Sq, D]
    k_bh = k_f.permute(0, 2, 1, 3)  # [B, Hq, Sk, D]
    v_bh = v_f.permute(0, 2, 1, 3)  # [B, Hq, Sk, D]

    scores = torch.matmul(q_bh, k_bh.transpose(-1, -2)) * scale
    m = scores.amax(dim=-1, keepdim=True)
    p = torch.exp(scores - m)
    denom = p.sum(dim=-1, keepdim=True)
    lse = torch.log(denom) + m  # [B, Hq, Sq, 1] FP32
    out_bh = torch.matmul(p / denom, v_bh)
    out = out_bh.permute(0, 2, 1, 3).contiguous()  # [B, Sq, Hq, D] FP32
    return out, lse


# ---------------------------------------------------------------------------
# Self-check CLI (--lower-only, --cpu-ref-test, --npu-test)
# ---------------------------------------------------------------------------


def _schedule_core_tasks_cpu(
    batch: int,
    heads_q: int,
    heads_kv: int,
    tiles_q: int,
    core_count: int,
    cid: int,
) -> list[tuple[int, int, int, int]]:
    """Mirror the kernel's O(1) KV-head-major Zig-Zag wave scheduler on CPU."""
    groups = heads_q // heads_kv
    rot_step = groups if groups < core_count else 1
    total_tasks = batch * heads_kv * tiles_q * groups
    num_waves = (total_tasks + core_count - 1) // core_count
    last_w = num_waves - 1
    rot_idx_last = (last_w >> 1) ^ (last_w >> 3) ^ (last_w >> 5) ^ (last_w >> 7)
    rot_last = (cid + rot_idx_last * rot_step) % core_count
    slot_last = rot_last if (last_w & 1) == 0 else (core_count - 1 - rot_last)
    my_tasks = num_waves if (last_w * core_count + slot_last) < total_tasks else (num_waves - 1)

    assigned: list[tuple[int, int, int, int]] = []
    for t in range(my_tasks):
        rot_idx = (t >> 1) ^ (t >> 3) ^ (t >> 5) ^ (t >> 7)
        rot = (cid + rot_idx * rot_step) % core_count
        slot = rot if (t & 1) == 0 else (core_count - 1 - rot)
        tid = t * core_count + slot
        qg = tid // groups
        lh = tid - qg * groups
        kv_idx = qg // tiles_q
        qt = qg - kv_idx * tiles_q
        bz = kv_idx // heads_kv
        kvh = kv_idx - bz * heads_kv
        head = kvh * groups + lh
        assigned.append((bz, kvh, head, qt))
    return assigned


def capture_lowered_asc(
    batch: int = 1,
    heads_q: int = 32,
    heads_kv: int = 8,
    dtype_name: str = "bfloat16",
    softmax_scale: float | None = None,
    core_count: int = 32,
) -> str:
    """Lower ``dense_attention_fwd_kernel`` to Ascend C (.asc) without invoking bisheng."""
    import tilelang.contrib.bisheng as bisheng

    scale = _HEAD_DIM**-0.5 if softmax_scale is None else float(softmax_scale)
    orig_compile = bisheng.compile_ascend
    captured: list[str] = []

    def _intercept(code: str, *args: Any, **kwargs: Any) -> bytearray:
        captured.append(code)
        raise RuntimeError("__CAPTURE_LOWERED_ASC_DONE__")

    bisheng.compile_ascend = _intercept
    try:
        dense_attention_fwd_kernel(
            batch,
            heads_q,
            heads_kv,
            _HEAD_DIM,
            dtype_name,
            scale,
            core_count,
        )
    except RuntimeError as exc:
        if captured and "__CAPTURE_LOWERED_ASC_DONE__" in str(exc):
            return captured[0]
        raise
    finally:
        bisheng.compile_ascend = orig_compile

    raise RuntimeError(
        "Kernel was already cached before capture; call capture_lowered_asc with an uncached key."
    )


def _audit_asc_code(code: str, label: str) -> dict[str, Any]:
    """Audit lowered .asc code for flagship SIMD VF and dynamic-shape invariants."""
    simt_vf = len(re.findall(r"\b__simt_vf__\b|\bsimt_vf\b", code))
    simd_vf = len(re.findall(r"\b__simd_vf__\b", code))
    vexpdif = len(re.findall(r"\bvexpdif\b", code))
    vln = len(re.findall(r"\bvln\b", code))
    vsstb = len(re.findall(r"\bvsstb\b", code))
    sig_match = re.search(
        r'extern\s+"C"\s+__global__[^{]*\bvoid\s+\w+\([^)]*\)', code, re.S
    )
    signature = " ".join(sig_match.group(0).split()) if sig_match else "<not found>"
    has_seq_q = "seq_q" in signature
    has_seq_k = "seq_k" in signature
    sha256 = hashlib.sha256(code.encode("utf-8")).hexdigest()
    lines = code.count("\n") + 1

    print(f"[lower-only] {label}")
    print(f"  lines={lines}  sha256={sha256}")
    print(
        f"  simt_vf={simt_vf}  simd_vf={simd_vf}  vexpdif={vexpdif}  vln={vln}  vsstb={vsstb}"
    )
    print(f"  dynamic_seq_q={has_seq_q}  dynamic_seq_k={has_seq_k}")
    print(f"  kernel_sig: {signature}")

    assert simt_vf == 0, f"{label}: expected simt_vf == 0, got {simt_vf}"
    assert simd_vf > 0, f"{label}: expected simd_vf > 0, got {simd_vf}"
    assert vexpdif > 0, f"{label}: expected vexpdif > 0, got {vexpdif}"
    assert vln > 0, f"{label}: expected vln > 0, got {vln}"
    assert vsstb > 0, f"{label}: expected vsstb > 0, got {vsstb}"
    assert has_seq_q and has_seq_k, (
        f"{label}: missing dynamic seq_q/seq_k in kernel signature: {signature}"
    )

    return {
        "lines": lines,
        "sha256": sha256,
        "simt_vf": simt_vf,
        "simd_vf": simd_vf,
        "vexpdif": vexpdif,
        "vln": vln,
        "vsstb": vsstb,
        "signature": signature,
    }


def _run_lower_only(dump_dir: str | None = None) -> None:
    configs = [
        ("bfloat16", 32),
        ("bfloat16", 1),
        ("float16", 32),
        ("float16", 1),
    ]
    if dump_dir is not None:
        os.makedirs(dump_dir, exist_ok=True)
    for dtype_name, cores in configs:
        code = capture_lowered_asc(
            batch=1,
            heads_q=32,
            heads_kv=8,
            dtype_name=dtype_name,
            core_count=cores,
        )
        label = f"dtype={dtype_name} cores={cores} (B=1 Hq=32 Hkv=8 D=128 dynamic Sq/Sk)"
        _audit_asc_code(code, label)
        if dump_dir is not None:
            out_path = os.path.join(dump_dir, f"prefill_fwd_dense_{dtype_name}_c{cores}.asc")
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write(code)
            print(f"  dumped -> {out_path}")
    print("[lower-only] ALL CHECKS PASSED.")


def _run_cpu_ref_test() -> None:
    torch.manual_seed(0)

    # 1. Verify reference_dense_attention_fwd against PyTorch SDPA (MHA + GQA + BF16/FP16).
    ref_cases = [
        (2, 256, 256, 8, 8, torch.bfloat16),
        (2, 256, 512, 16, 4, torch.bfloat16),
        (1, 128, 128, 4, 1, torch.float16),
        (1, 384, 256, 8, 2, torch.float16),
    ]
    for b, sq, sk, hq, hkv, dt in ref_cases:
        q = torch.randn((b, sq, hq, _HEAD_DIM), dtype=dt)
        k = torch.randn((b, sk, hkv, _HEAD_DIM), dtype=dt)
        v = torch.randn((b, sk, hkv, _HEAD_DIM), dtype=dt)
        scale = _HEAD_DIM**-0.5

        out_ref, lse_ref = reference_dense_attention_fwd(q, k, v, softmax_scale=scale)
        assert out_ref.dtype == torch.float32 and lse_ref.dtype == torch.float32
        assert tuple(out_ref.shape) == (b, sq, hq, _HEAD_DIM)
        assert tuple(lse_ref.shape) == (b, hq, sq, 1)

        groups = hq // hkv
        q_bh = q.float().permute(0, 2, 1, 3)
        k_bh = k.float().repeat_interleave(groups, dim=2).permute(0, 2, 1, 3)
        v_bh = v.float().repeat_interleave(groups, dim=2).permute(0, 2, 1, 3)
        sdpa = F.scaled_dot_product_attention(q_bh, k_bh, v_bh, scale=scale)
        sdpa_bshd = sdpa.permute(0, 2, 1, 3).contiguous()
        max_err = (out_ref - sdpa_bshd).abs().max().item()

        scores = torch.matmul(q_bh, k_bh.transpose(-1, -2)) * scale
        lse_check = torch.logsumexp(scores, dim=-1, keepdim=True)
        lse_err = (lse_ref - lse_check).abs().max().item()

        print(
            f"[cpu-ref-test] SDPA B={b} Sq={sq} Sk={sk} Hq={hq} Hkv={hkv} "
            f"dtype={dt}: out_max_err={max_err:.3e} lse_max_err={lse_err:.3e}"
        )
        assert max_err < 1e-5, f"SDPA mismatch: {max_err}"
        assert lse_err < 1e-5, f"LSE mismatch: {lse_err}"

    # 2. Verify O(1) persistent scheduler covers every (b, hq, qt) tile exactly once.
    sched_cases = [
        # (B, Hq, Hkv, Sq, Sk, core_count)
        (1, 1, 1, 128, 128, 32),  # fewer tasks than cores
        (1, 4, 2, 256, 128, 32),  # partial first wave
        (1, 32, 8, 128, 256, 32),  # exact 1 full wave
        (1, 32, 8, 256, 384, 32),  # exact 2 full waves (Zig-Zag)
        (2, 32, 8, 512, 512, 32),  # multi-batch multi-wave GQA
        (3, 24, 6, 384, 256, 28),  # non-power-of-two core_count + partial last wave
        (2, 16, 1, 640, 256, 36),  # MQA + 36 cores
        (1, 8, 8, 1024, 128, 1),  # single core
        (2, 64, 8, 384, 256, 32),  # groups < core_count with many waves
        (1, 64, 1, 256, 256, 32),  # groups >= core_count -> rot_step=1
    ]
    for b, hq, hkv, sq, sk, cores in sched_cases:
        tiles_q = sq // _BR
        groups = hq // hkv
        expected = {
            (bz, head // groups, head, qt)
            for bz in range(b)
            for head in range(hq)
            for qt in range(tiles_q)
        }
        seen: list[tuple[int, int, int, int]] = []
        counts: list[int] = []
        for cid in range(cores):
            ct = _schedule_core_tasks_cpu(b, hq, hkv, tiles_q, cores, cid)
            seen.extend(ct)
            counts.append(len(ct))
        assert len(seen) == len(expected), (
            f"Count mismatch for {(b, hq, hkv, sq, sk, cores)}: got {len(seen)}, expected {len(expected)}"
        )
        assert set(seen) == expected, (
            f"Tile coverage mismatch (duplicate/missing) for {(b, hq, hkv, sq, sk, cores)}"
        )
        assert max(counts) - min(counts) <= 1, (
            f"Unbalanced core task counts for {(b, hq, hkv, sq, sk, cores)}: {counts}"
        )
        print(
            f"[cpu-ref-test] sched B={b} Hq={hq} Hkv={hkv} Sq={sq} Sk={sk} "
            f"cores={cores}: tasks={len(seen)} per_core=[{min(counts)},{max(counts)}] OK"
        )

    print("[cpu-ref-test] ALL CHECKS PASSED.")


def _run_npu_test(core_count: int = 32) -> None:
    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("--npu-test requires torch_npu on an Ascend NPU host.") from exc

    if not torch.npu.is_available():
        raise RuntimeError("No Ascend NPU device available.")

    device = torch.device("npu:0")
    torch.manual_seed(42)

    batch, heads_q, heads_kv = 1, 16, 4
    scale = _HEAD_DIM**-0.5
    # Multiple dynamic (Sq, Sk) shapes including tiles_k == 1, 2, 3, and larger —
    # all served by a SINGLE compiled kernel binary!
    dynamic_shapes = [
        (128, 128),  # tiles_k == 1
        (128, 256),  # tiles_k == 2
        (256, 384),  # tiles_k == 3
        (512, 1024),  # tiles_k == 8
    ]

    for dtype in (torch.bfloat16, torch.float16):
        dtype_name = _DTYPE_TO_NAME[dtype]
        atol = 2e-2 if dtype == torch.bfloat16 else 5e-3
        lse_atol = 1e-2 if dtype == torch.bfloat16 else 2e-3

        compiled_first = dense_attention_fwd_kernel(
            batch, heads_q, heads_kv, _HEAD_DIM, dtype_name, scale, core_count
        )
        for sq, sk in dynamic_shapes:
            compiled_now = dense_attention_fwd_kernel(
                batch, heads_q, heads_kv, _HEAD_DIM, dtype_name, scale, core_count
            )
            assert compiled_now is compiled_first, "Expected dynamic reuse of single compiled kernel!"

            q = torch.randn((batch, sq, heads_q, _HEAD_DIM), dtype=dtype, device=device)
            k = torch.randn((batch, sk, heads_kv, _HEAD_DIM), dtype=dtype, device=device)
            v = torch.randn((batch, sk, heads_kv, _HEAD_DIM), dtype=dtype, device=device)

            # NaN-prefill output buffers to catch any unwritten tile or row.
            out = torch.full_like(q, float("nan"))
            lse = torch.full(
                (batch, heads_q, sq, 1), float("nan"), dtype=torch.float32, device=device
            )
            dense_attention_fwd(
                q, k, v, softmax_scale=scale, core_count=core_count, out=out, lse=lse
            )
            torch.npu.synchronize()

            assert not torch.isnan(out).any(), f"NaN detected in Output for Sq={sq}, Sk={sk}"
            assert not torch.isnan(lse).any(), f"NaN detected in LSE for Sq={sq}, Sk={sk}"

            out_ref, lse_ref = reference_dense_attention_fwd(
                q.cpu(), k.cpu(), v.cpu(), softmax_scale=scale
            )
            out_err = (out.float().cpu() - out_ref).abs().max().item()
            lse_err = (lse.cpu() - lse_ref).abs().max().item()
            print(
                f"[npu-test] dtype={dtype_name} Sq={sq} Sk={sk}: "
                f"out_max_err={out_err:.4e} (tol={atol}) lse_max_err={lse_err:.4e} (tol={lse_atol})"
            )
            assert out_err <= atol, f"Output error {out_err} exceeds {atol}"
            assert lse_err <= lse_atol, f"LSE error {lse_err} exceeds {lse_atol}"

    print("[npu-test] ALL CHECKS PASSED.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ascend 950 Dense Attention Prefill Forward reference skeleton self-checks."
    )
    parser.add_argument(
        "--lower-only",
        action="store_true",
        help="Lower to Ascend C (.asc) without invoking bisheng and audit VF/dynamic invariants.",
    )
    parser.add_argument(
        "--dump-dir",
        type=str,
        default=None,
        help="Optional directory (e.g. /tmp/dense_asc) to save captured .asc files with --lower-only.",
    )
    parser.add_argument(
        "--cpu-ref-test",
        action="store_true",
        help="Verify FP32 reference vs PyTorch SDPA and audit O(1) persistent task coverage.",
    )
    parser.add_argument(
        "--npu-test",
        action="store_true",
        help="Run NaN-prefilled accuracy checks on an Ascend NPU across dynamic Sq/Sk.",
    )
    parser.add_argument(
        "--core-count",
        type=int,
        default=32,
        help="Number of Ascend Cube cores for --npu-test (default: 32).",
    )
    args = parser.parse_args()

    if not (args.lower_only or args.cpu_ref_test or args.npu_test):
        parser.print_help()
        return

    if args.cpu_ref_test:
        _run_cpu_ref_test()
    if args.lower_only:
        _run_lower_only(dump_dir=args.dump_dir)
    if args.npu_test:
        _run_npu_test(core_count=args.core_count)


if __name__ == "__main__":
    main()
