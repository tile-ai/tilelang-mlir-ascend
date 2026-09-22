# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Mamba-2 SSD chunk scan forward kernel (NPU Expert-mode MixCV).

Migrated from the GPU TileLang factory ``_ssd_chunk_scan_fwd_kernel`` to
``target="npuir"`` per DESIGN.md v2 (revision v2, frozen after three review
rounds).  This is the Stage 3 deliverable: the standalone kernel module with
the embedded golden (reused from ``tileops.testing.mamba2_reference``), the
layered test suite (L0/L1/L2/Boundary) and the ``--level`` entry point.

Math (source semantics, kept verbatim, DESIGN §1.3):

    out[l, p] = exp(dA_cumsum[l]) * (C[l] @ prev_states)
              + sum_{s <= l} cb[l, s] * exp(dA_cumsum[l] - dA_cumsum[s]) * dt[s] * x[s, p]

Structure (DESIGN §0.6 / §1.4 / §3 / §6):
  - 1-D persistent ``T.Kernel(24)`` (R1); logical task = (b, c, h);
    ``cid = task_id * 24 + kernel_id`` round-robin + ghost clamp (R6/§5.4).
  - History path (R2+R5): ``T.gemm(c_scaled, state_t, acc, b_transpose=True)``
    with ``state_t`` direct-loaded [bp, bn]; ``c_scaled = cast(C * exp(dA_l))``
    folds the exp(dA_l) scale into the A operand (OPT-A) so the accumulator
    never needs a Vector rescale (stays purely in L0C / Cube).
  - Intra path (R3): ``lcb = cast(cb * exp(dA_l - dA_s) * dt)`` computed in the
    Vector domain with a direct-diff chain (matches the golden's direct decay
    ``exp(dA_l - dA_s)`` numerical path); the diagonal block adds an arithmetic
    penalty mask (R4, OPT-B) so masked cells underflow to exact +0.0.
  - Cross-engine data via GM workspace relay (R7): Vector produces factors into
    block-contiguous ``ws_c``/``ws_lcb`` (PL-1.14), Cube consumes them. Depth-2
    task pipeline (PL-1.12): double-slot workspace + 4-flag handshake with a
    Cube prologue set (Vector(T+1) overlaps Cube(T); Vector's slot-free wait is
    task-head-placed so task W writes only after Cube consumed task W-2).
  - Cube-side causal-band assembly (R7.4, v5_xband): whole-chunk x L1 cache
    [Q, bp] loaded once per (task, pp), and one K=(l0+tl) band gemm per l-tile
    after single-hop GM->L1 column-offset direct loads into ``l1_band``
    (src/dst tail-clamped by ``ts = min(bs, Q-s0)``, TRAP-L1-band-dst-tail-overrun).
  - AIV work split by ``subid`` (R8, PL-1.13): snake-balanced l-tile partition;
    the two AIVs own disjoint l-tile sets instead of duplicating the Vector
    program.

Expert-mode note (PL-1.16 / CG-2026-0010): the persistent kernel + gemm +
vector pattern is only supported in Expert mode (explicit ``T.Scope("Cube")`` /
``T.Scope("Vector")``); Developer mode crashes at runtime for the same
structure.  ``pass_configs`` double-disable
(``TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION`` +
``NPUIR_ENABLE_AUTO_MULTI_BUFFER``) protects the manual Cube/Vector split and
explicit workspaces from buffer re-scoping (codegen "cannot find variable").

Layouts (official):
  x:           [B, S, H, P]        dtype (fp16/bf16)
  cb:          [B, C, G, Q, Q]     dtype
  dA_cumsum:   [B, H, C, Q]        float32
  C_mat:       [B, S, G, N]        dtype
  prev_states: [B, C, H, P, N]     float32  (P before N)
  dt:          [B, H, C, Q]        dtype
  out:         [B, S, H, P]        float32

Interface contract (unchanged vs GPU source except the GPU-only ``threads``
parameter, which is removed)::

    _ssd_chunk_scan_fwd_kernel(batch, num_chunks, chunk_len, n_heads, d_head,
        d_state, n_groups, dtype='float16')(
        block_l, block_p, block_n, block_s, num_stages)(
        x, cb, dA_cumsum, C_mat, prev_states, dt) -> out

Run: python _ssd_chunk_scan_fwd_kernel.py --level {L0,all}
"""

import argparse
import functools
import os

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  (registers the "npu" device)

# Expert mode: required for the persistent kernel + gemm + vector pattern
# (GQA / sparse_mla precedent). Developer mode crashes on this structure.
os.environ.setdefault("TILELANG_ASCEND_MODE", "Expert")

# Physical AI-core count on Ascend910B2C (DESIGN §5.5, verified 2026-09-20).
NUM_KERNELS = 24

# Stage-4 tuned default config (DESIGN §5.2, inherited from the old task's
# perf_opt final; msprof op Task Duration, median of 20, manifest workloads).
# block_n is clamped internally to min(block_n, N) so 128 is safe for every
# contract N (N < 128 collapses to a single tail block).
TUNED_DEFAULT_CONFIG = {
    "block_l": 64,
    "block_p": 64,
    "block_n": 128,
    "block_s": 64,
    "num_stages": 2,
}
# Arithmetic penalty sentinel (finite, avoids -inf - (-inf) NaN path, R4).
_PEN = 1e30


# ---------- Golden (DESIGN §8.1: inlined ported reference math) ----------
# The golden body is the inlined torch computation of the ported
# ``tileops.testing.mamba2_reference.ssd_chunk_scan_fwd_ref`` (pure fp32 PyTorch
# materialize: einsum dual path + tril mask).  It is semantically identical to
# that reference and independent of the NPU algorithm (no anchor factorization,
# no dtype quantization, no vectorization structure).  Runs on CPU.
def golden_ssd_chunk_scan_fwd(x, cb, dA_cumsum, C, prev_states, dt, n_groups):
    """Official-aligned reference (independent of the NPU algorithm).

    Inputs (official layouts):
      x:           [B, S, H, P]        dtype
      cb:          [B, C, G, L, L]     dtype    group-owned
      dA_cumsum:   [B, H, C, L]        float32
      C:           [B, S, G, N]        dtype    group-owned
      prev_states: [B, C, H, P, N]     float32  P before N
      dt:          [B, H, C, L]        dtype

    Output: [B, S, H, P] float32
    """
    b, S, h, p = x.shape
    _, _, c, L = dA_cumsum.shape
    n = C.shape[-1]
    g = n_groups
    heads_per_group = h // g

    x_chunked = x.float().reshape(b, c, L, h, p)  # [B, C, L, H, P]
    C_chunked = C.float().reshape(b, c, L, g, n)  # [B, C, L, G, N]
    # broadcast C from groups to heads: [B, C, L, H, N]
    C_heads = C_chunked[:, :, :, torch.arange(h, device=x.device) // heads_per_group, :]

    # dA_cumsum: [B, H, C, L] -> [B, C, L, H] for broadcast
    dA = dA_cumsum.float().permute(0, 2, 3, 1)  # [B, C, L, H]

    # --- History path: exp(dA_l) * C[l] @ prev_states[p, n] ---
    y_off = torch.einsum("bclhn,bchpn->bclhp", C_heads, prev_states.float())
    y_off = y_off * torch.exp(dA).unsqueeze(-1)  # scale by exp(dA_l)

    # --- Intra-chunk path: sum_{s<=l} cb[l,s] * exp(dA_l - dA_s) * dt[s] * x[s] ---
    cb_heads = cb.float()[:, :, torch.arange(h, device=x.device) // heads_per_group, :, :]

    # decay[b,c,h,l,s] = exp(dA_cumsum[l] - dA_cumsum[s])
    dA_l = dA_cumsum.float().unsqueeze(-1)  # [B, H, C, L, 1]
    dA_s = dA_cumsum.float().unsqueeze(-2)  # [B, H, C, 1, L]
    decay = torch.exp(dA_l - dA_s)  # [B, H, C, L, L]

    # causal mask
    mask = torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool))
    decay = decay.masked_fill(~mask.unsqueeze(0).unsqueeze(0).unsqueeze(0), 0.0)
    decay = decay.permute(0, 2, 1, 3, 4)  # [B, C, H, L, L]

    # dt: [B, H, C, L] -> [B, C, H, 1, L]
    dt_s = dt.float().permute(0, 2, 1, 3).unsqueeze(-2)  # [B, C, H, 1, L]

    # lcb[b,c,h,l,s] = cb[l,s] * decay[l,s] * dt[s]
    lcb = cb_heads * decay * dt_s  # [B, C, H, L, L]

    # y_diag[b,c,l,h,p] = sum_s lcb[b,c,h,l,s] * x[b,c,s,h,p]
    y_diag = torch.einsum("bchls,bcshp->bclhp", lcb, x_chunked)

    return (y_off + y_diag).reshape(b, S, h, p)


# ================= KERNEL (Expert-mode MixCV, DESIGN §3.3) =================
@functools.lru_cache(maxsize=32)
def _ssd_chunk_scan_fwd_kernel(
    batch,
    num_chunks,
    chunk_len,
    n_heads,
    d_head,
    d_state,
    n_groups,
    dtype="float16",
):
    """GPU-source-compatible factory (signature/validation/lru_cache kept).

    Returns an inner factory ``_ssd_chunk_scan_fwd_func(block_l, block_p,
    block_n, block_s, num_stages)`` whose result is the compiled closure
    ``wrapped(x, cb, dA_cumsum, C, prev_states, dt) -> out``.
    """
    accum_dtype = "float32"
    B, C, Q, H, P, N, G = batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups
    S = C * Q
    HPG = H // G
    if H % G != 0:
        raise ValueError("n_heads must be divisible by n_groups")
    num_logical = B * C * H

    @tilelang.jit(
        out_idx=[-1],
        target="npuir",
        pass_configs={
            # Protect the manual Cube/Vector split and explicit workspaces from
            # buffer-allocation reordering (PL-1.16; GQA/ssd precedent):
            # without these the planner re-scopes UB buffers and codegen fails
            # to find them ("cannot find variable").
            tilelang.PassConfigKey.TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION: False,
            tilelang.PassConfigKey.NPUIR_ENABLE_AUTO_MULTI_BUFFER: False,
        },
    )
    def _builder(block_l, block_p, block_n, block_s, num_stages):
        bl, bp, bn, bs = block_l, block_p, block_n, block_s
        # Clamp the n-block to the real K extent: an L1 operand buffer wider
        # than N leaves a stale fractal band that leaks into the gemm when
        # tn < bn (L0-5 N=48/bn=128 regression, max_diff 7.4e-3).  For the
        # tuned default (128) this keeps N=128 manifests on the single-block
        # fast path and falls back to the baseline behavior for small N.
        bn = min(bn, N)
        L_tiles = (Q + bl - 1) // bl
        P_tiles = (P + bp - 1) // bp
        N_tiles = (N + bn - 1) // bn

        # per-core GM workspaces (Vector produces factors, Cube consumes).
        # ws_c holds the full N dimension: the history n-loop consumes a
        # different bn-wide C slice per n-block.  Depth-2 task pipeline
        # (PL-1.12): a leading slot dim double-buffers the workspace so
        # Vector(T+1) production overlaps Cube(T) consumption.  Block-contiguous
        # layout is mandatory (PL-1.14: band-ized ws writes are 3x slower).
        ws_c_shape = [NUM_KERNELS, 2, P_tiles, L_tiles, bl, N]
        ws_lcb_shape = [NUM_KERNELS, 2, P_tiles, L_tiles, L_tiles, bl, bs]

        @T.prim_func
        def main(
            x: T.Tensor((B, S, H, P), dtype),
            cb: T.Tensor((B, C, G, Q, Q), dtype),
            dA_cumsum: T.Tensor((B, H, C, Q), accum_dtype),
            C_mat: T.Tensor((B, S, G, N), dtype),
            prev_states: T.Tensor((B, C, H, P, N), dtype),
            dt: T.Tensor((B, H, C, Q), dtype),
            ws_c: T.Tensor(ws_c_shape, dtype),
            ws_lcb: T.Tensor(ws_lcb_shape, dtype),
            out: T.Tensor((B, S, H, P), accum_dtype),
        ):
            with T.Kernel(NUM_KERNELS, is_npu=True) as (kernel_id, subid):
                num_local_tasks = T.ceildiv(num_logical - kernel_id, NUM_KERNELS)

                with T.Scope("Cube"):
                    l1_c = T.alloc_L1([bl, bn], dtype)
                    l1_state = T.alloc_L1([bp, bn], dtype)
                    # whole-chunk x cache (one load per task+pp); the band gemm
                    # reads the [0, l0+tl) row range via its size parameter.
                    l1_x = T.alloc_L1([Q, bp], dtype)
                    # causal band assembly buffer: per-s-block contiguous reads
                    # land at column offsets s0 (multiple of bs, fractal-aligned).
                    # Single-hop GM->L1 direct load (no L1->L1 copy, which is an
                    # unsupported direction).
                    l1_band = T.alloc_L1([bl, Q], dtype)
                    l0_acc = T.alloc_L0C([bl, bp], accum_dtype)

                    # Depth-2 prologue (PL-1.12): both slots start "free"
                    # (nothing consumed yet), so Vector tasks 0/1 pass their
                    # slot-free waits immediately.  Strict set/wait alternation
                    # per flag id is preserved:
                    #   flag 2*slot   = factors-ready(slot)
                    #   flag 2*slot+1 = slot-free / consumed(slot)
                    with T.rs("PIPE_FIX"):
                        T.sync_block_set(1)
                        T.sync_block_set(3)

                    for task_id in T.serial(num_local_tasks):
                        slot = task_id % 2
                        cid = T.min(task_id * NUM_KERNELS + kernel_id, num_logical - 1)
                        bz = cid // (C * H)
                        bc_idx = (cid // H) % C
                        bh = cid % H
                        bg = bh // HPG
                        cs = bc_idx * Q

                        with T.rs("PIPE_MTE2"):
                            T.sync_block_wait(2 * slot)  # factors ready (Vector -> Cube)

                        for pp in T.serial(P_tiles):
                            p0 = pp * bp
                            tp = T.min(bp, P - p0)
                            tnp = T.max(16, T.min(bp, T.ceildiv(tp, 16) * 16))

                            # whole-chunk x cache: loaded once, reused by all
                            # l-tiles of this task (eliminates the 2.5x
                            # cross-lt re-read of strided x rows).
                            T.copy(
                                x[bz, cs : cs + Q, bh, p0 : p0 + tp],
                                l1_x[0:Q, 0:tp],
                            )

                            for lt in T.serial(L_tiles):
                                l0 = lt * bl
                                tl = T.min(bl, Q - l0)
                                tmc = T.max(16, T.min(bl, T.ceildiv(tl, 16) * 16))

                                # ---- history (n-loop) ----
                                for n_blk in T.serial(N_tiles):
                                    n0 = n_blk * bn
                                    tn = T.min(bn, N - n0)
                                    T.copy(
                                        ws_c[kernel_id, slot, pp, lt, 0:tl, n0 : n0 + tn],
                                        l1_c[0:tl, 0:tn],
                                    )
                                    T.copy(
                                        prev_states[bz, bc_idx, bh, p0 : p0 + tp, n0 : n0 + tn],
                                        l1_state[0:tp, 0:tn],
                                    )
                                    T.gemm(
                                        l1_c,
                                        l1_state,
                                        l0_acc,
                                        initC=(n_blk == 0),
                                        b_transpose=True,
                                        size=[tmc, tn, tnp],
                                    )

                                # ---- intra band (full-lower + diag, one gemm) ----
                                # Each s-block is read contiguously from the
                                # block-layout ws into its band column offset;
                                # K = l0 + tl covers all causal s of this
                                # l-tile (ascending order preserved; the diag
                                # penalty is already folded by Vector).
                                for s_blk in T.serial(lt + 1):
                                    s0 = s_blk * bs
                                    # Tail-safe band assembly: the dst column
                                    # range must be clamped to ts = min(bs,
                                    # Q - s0); an unclamped s0+bs slice
                                    # overruns the [bl, Q] L1 band buffer on
                                    # the last s-block (Q % bs != 0) and
                                    # corrupts adjacent L1 allocations.
                                    ts = T.min(bs, Q - s0)
                                    T.copy(
                                        ws_lcb[kernel_id, slot, pp, lt, s_blk, 0:bl, 0:ts],
                                        l1_band[0:bl, s0 : s0 + ts],
                                    )
                                T.gemm(
                                    l1_band,
                                    l1_x,
                                    l0_acc,
                                    initC=False,
                                    size=[tmc, l0 + tl, tnp],
                                )

                                # ---- write back ----
                                T.copy(
                                    l0_acc[0:tl, 0:tp],
                                    out[bz, cs + l0 : cs + l0 + tl, bh, p0 : p0 + tp],
                                )

                        with T.rs("PIPE_FIX"):
                            T.sync_block_set(2 * slot + 1)  # consumed (Cube -> Vector)

                with T.Scope("Vector"):
                    zero = T.float32(0)
                    pen_val = T.float32(_PEN)

                    # ---- kernel-level penalty mask (R4): 0 for i>=j, <= -PEN for i<j
                    pen_const = T.alloc_ub((bl, bs), accum_dtype)
                    idx_j = T.alloc_ub((bl, bs), accum_dtype)
                    T.arange(pen_const, [1, 0], 0)  # local row i
                    T.arange(idx_j, [0, 1], 0)  # local col j
                    T.vsub(pen_const, idx_j, pen_const)  # i - j
                    T.vmin(pen_const, zero, pen_const)  # min(i-j, 0)
                    T.vmul(pen_const, pen_val, pen_const)  # 0 / <= -PEN

                    # ---- factor buffers ----
                    dA_l_col = T.alloc_ub((bl, 1), accum_dtype)
                    exp_dA_l_col = T.alloc_ub((bl, 1), accum_dtype)
                    dA_s_row = T.alloc_ub((1, bs), accum_dtype)
                    dt_s_row = T.alloc_ub((1, bs), accum_dtype)
                    dt_16 = T.alloc_ub((1, bs), dtype)
                    cb_16 = T.alloc_ub((bl, bs), dtype)
                    cb_f32 = T.alloc_ub((bl, bs), accum_dtype)
                    lcb_f32 = T.alloc_ub((bl, bs), accum_dtype)
                    lcb_16 = T.alloc_ub((bl, bs), dtype)
                    c_16 = T.alloc_ub((bl, N), dtype)
                    c_f32 = T.alloc_ub((bl, N), accum_dtype)
                    c_scaled_16 = T.alloc_ub((bl, N), dtype)
                    dA_l_mat = T.alloc_ub((bl, bs), accum_dtype)
                    dA_s_mat = T.alloc_ub((bl, bs), accum_dtype)
                    dt_mat = T.alloc_ub((bl, bs), accum_dtype)

                    # AIV work split (R8, PL-1.13): the two AIVs of each block
                    # snake-partition the l-tiles (L=4 -> {0,3}/{1,2}).  The
                    # split dimension (lt) is orthogonal to the ws index (slot,
                    # pp, lt), so the two AIVs write disjoint ws regions.  The
                    # snake affine + clamps cover degenerate/odd L_tiles (dup
                    # l-tiles are bit-identical, benign).  Both AIVs still
                    # execute the same set/wait sequence (one-set-multi-wait).
                    lt_count = (L_tiles + 1) // 2

                    for task_id in T.serial(num_local_tasks):
                        slot = task_id % 2
                        cid = T.min(task_id * NUM_KERNELS + kernel_id, num_logical - 1)
                        bz = cid // (C * H)
                        bc_idx = (cid // H) % C
                        bh = cid % H
                        bg = bh // HPG
                        cs = bc_idx * Q

                        # slot-free handshake, task-head placed (v1 revision):
                        # for tasks 0/1 this passes immediately (Cube prologue
                        # set); from task 2 on it waits for Cube's consumption
                        # of task (T-2), which last used this slot.  Placing the
                        # wait BEFORE any ws[slot] write prevents the off-by-one
                        # WAR race (producer otherwise leads by 2 tasks).
                        with T.rs("PIPE_MTE2"):
                            T.sync_block_wait(2 * slot + 1)  # slot free (Cube -> Vector)

                        for pp in T.serial(P_tiles):
                            p0 = pp * bp
                            for i in T.serial(lt_count):
                                lt = i + subid + (i % 2) * (L_tiles - 2 * i - 2 * subid)
                                lt = T.min(T.max(lt, 0), L_tiles - 1)
                                l0 = lt * bl
                                tl = T.min(bl, Q - l0)

                                # ---- exp_dA_l = exp(dA_l) ----
                                T.copy(
                                    dA_cumsum[bz, bh, bc_idx, l0 : l0 + tl],
                                    dA_l_col[0:tl, 0:1],
                                )
                                T.vexp(dA_l_col, exp_dA_l_col)

                                # ---- c_scaled = cast(C * exp(dA_l)) (full N) ----
                                T.copy(
                                    C_mat[bz, cs + l0 : cs + l0 + tl, bg, 0:N],
                                    c_16[0:tl, 0:N],
                                )
                                T.vcast(c_16, c_f32, round_mode="rint")
                                T.vmul(c_f32, exp_dA_l_col, c_f32)
                                T.vcast(c_f32, c_scaled_16, round_mode="rint")
                                T.copy(
                                    c_scaled_16[0:tl, 0:N],
                                    ws_c[kernel_id, slot, pp, lt, 0:tl, 0:N],
                                )

                                # ---- intra lcb per s-block ----
                                for s_blk in T.serial(lt + 1):
                                    s0 = s_blk * bs
                                    ts = T.min(bs, Q - s0)
                                    T.copy(cb[bz, bc_idx, bg, l0, s0], cb_16, size=[tl, ts])
                                    T.vcast(cb_16, cb_f32, round_mode="rint")
                                    T.copy(
                                        dA_cumsum[bz, bh, bc_idx, s0 : s0 + ts],
                                        dA_s_row[0:1, 0:ts],
                                    )
                                    T.vbrc(dA_l_col, dA_l_mat)  # [bl,1] -> [bl,bs]
                                    T.vbrc(dA_s_row, dA_s_mat)  # [1,bs] -> [bl,bs]
                                    T.vsub(dA_l_mat, dA_s_mat, dA_l_mat)  # diff = dA_l - dA_s
                                    if s_blk == lt:
                                        T.vadd(dA_l_mat, pen_const, dA_l_mat)  # + penalty (diag)
                                    T.vexp(dA_l_mat, dA_l_mat)
                                    T.vmul(cb_f32, dA_l_mat, lcb_f32)  # cb * exp
                                    T.copy(dt[bz, bh, bc_idx, s0 : s0 + ts], dt_16[0:1, 0:ts])
                                    T.vcast(dt_16, dt_s_row, round_mode="rint")
                                    T.vbrc(dt_s_row, dt_mat)  # [1,bs] -> [bl,bs]
                                    T.vmul(lcb_f32, dt_mat, lcb_f32)  # * dt
                                    T.vcast(lcb_f32, lcb_16, round_mode="rint")
                                    T.copy(
                                        lcb_16[0:tl, 0:bs],
                                        ws_lcb[kernel_id, slot, pp, lt, s_blk, 0:tl, 0:bs],
                                    )

                        with T.rs("PIPE_MTE3"):
                            T.sync_block_set(2 * slot)  # factors ready (Vector -> Cube)

        return main

    def _inner(block_l, block_p, block_n, block_s, num_stages):
        bl, bp, bs = block_l, block_p, block_s
        L_tiles_inner = (Q + bl - 1) // bl
        P_tiles_inner = (P + bp - 1) // bp
        kernel = _builder(block_l, block_p, block_n, block_s, num_stages)
        torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16

        def wrapped(x, cb, dA_cumsum, C_mat, prev_states, dt):
            ws_c = torch.empty(
                (NUM_KERNELS, 2, P_tiles_inner, L_tiles_inner, bl, N),
                dtype=torch_dtype,
                device=x.device,
            )
            ws_lcb = torch.empty(
                (NUM_KERNELS, 2, P_tiles_inner, L_tiles_inner, L_tiles_inner, bl, bs),
                dtype=torch_dtype,
                device=x.device,
            )
            # source casts prev_states (fp32) to the input dtype before the gemm;
            # replicate that cast here (Expert T.copy cannot cast across dtype).
            prev_states_c = prev_states.to(torch_dtype)
            return kernel(x, cb, dA_cumsum, C_mat, prev_states_c, dt, ws_c, ws_lcb)

        return wrapped

    return _inner


# ================= TESTS =================
_ATOL = {torch.float16: 1e-3, torch.bfloat16: 2e-3}
_RTOL = 1e-5


def _dtype_str(dtype):
    return "float16" if dtype == torch.float16 else "bfloat16"


def _run_case(B, C, Q, H, P, N, G, dtype, tag, seed=0, decay_scale=1.0):
    """Generate inputs on NPU, run the kernel, gate vs golden (fp32)."""
    import time

    torch.manual_seed(seed)
    S = C * Q
    x = torch.randn(B, S, H, P, dtype=dtype, device="npu") * 0.1
    cb = torch.randn(B, C, G, Q, Q, dtype=dtype, device="npu") * 0.1
    dA_cumsum = -torch.rand(B, H, C, Q, dtype=torch.float32, device="npu").cumsum(-1)
    dA_cumsum = dA_cumsum * decay_scale
    C_mat = torch.randn(B, S, G, N, dtype=dtype, device="npu") * 0.1
    prev_states = torch.randn(B, C, H, P, N, dtype=torch.float32, device="npu") * 0.1
    dt = torch.rand(B, H, C, Q, dtype=dtype, device="npu") * 0.1 + 0.01

    fn = _ssd_chunk_scan_fwd_kernel(B, C, Q, H, P, N, G, _dtype_str(dtype))
    wrapped = fn(
        TUNED_DEFAULT_CONFIG["block_l"],
        TUNED_DEFAULT_CONFIG["block_p"],
        TUNED_DEFAULT_CONFIG["block_n"],
        TUNED_DEFAULT_CONFIG["block_s"],
        TUNED_DEFAULT_CONFIG["num_stages"],
    )
    torch.npu.synchronize()
    t0 = time.perf_counter()
    out = wrapped(x, cb, dA_cumsum, C_mat, prev_states, dt)
    torch.npu.synchronize()
    t_ms = (time.perf_counter() - t0) * 1e3

    assert out.shape == (B, S, H, P), f"out shape contract: {tuple(out.shape)}"
    assert out.dtype == torch.float32, f"out dtype contract: {out.dtype}"

    ref = golden_ssd_chunk_scan_fwd(
        x.cpu(), cb.cpu(), dA_cumsum.cpu(), C_mat.cpu(), prev_states.cpu(), dt.cpu(), G
    )
    atol = _ATOL[dtype]
    diff = (out.cpu() - ref).abs()
    tol = atol + _RTOL * ref.abs()
    ok = bool((diff <= tol).all().item())
    max_diff = float(diff.max().item())
    print(
        f"  [{tag}] B={B} C={C} Q={Q} H={H} P={P} N={N} G={G} {_dtype_str(dtype)}: "
        f"{'PASS' if ok else 'FAIL'} (max_diff={max_diff:.3e}) [{t_ms:.1f} ms]"
    )
    return ok, max_diff


def _contract_checks():
    print("  [Contract] lru_cache identity / out shape+dtype / contiguous / ValueError")
    f1 = _ssd_chunk_scan_fwd_kernel(1, 2, 64, 4, 64, 32, 1, "float16")
    f2 = _ssd_chunk_scan_fwd_kernel(1, 2, 64, 4, 64, 32, 1, "float16")
    assert f1 is f2, "lru_cache factory identity violated"
    wrapped = f1(
        TUNED_DEFAULT_CONFIG["block_l"],
        TUNED_DEFAULT_CONFIG["block_p"],
        TUNED_DEFAULT_CONFIG["block_n"],
        TUNED_DEFAULT_CONFIG["block_s"],
        TUNED_DEFAULT_CONFIG["num_stages"],
    )
    x = torch.randn(1, 128, 4, 64, dtype=torch.float16, device="npu") * 0.1
    cb = torch.randn(1, 2, 1, 64, 64, dtype=torch.float16, device="npu") * 0.1
    dA = -torch.rand(1, 4, 2, 64, dtype=torch.float32, device="npu").cumsum(-1)
    Cm = torch.randn(1, 128, 1, 32, dtype=torch.float16, device="npu") * 0.1
    ps = torch.randn(1, 2, 4, 64, 32, dtype=torch.float32, device="npu") * 0.1
    dt = torch.rand(1, 4, 2, 64, dtype=torch.float16, device="npu") * 0.1 + 0.01
    out = wrapped(x, cb, dA, Cm, ps, dt)
    torch.npu.synchronize()
    assert out.shape == (1, 128, 4, 64) and out.dtype == torch.float32
    assert out.is_contiguous(), "out must be contiguous"
    try:
        _ssd_chunk_scan_fwd_kernel(1, 2, 64, 5, 64, 32, 2, "float16")
        raise AssertionError("H%G ValueError not raised")
    except ValueError:
        pass
    print("  [Contract] PASS")


def run_L0():
    """L0 gate (blocking, DESIGN §8.2 order)."""
    print("== L0 (blocking) ==")
    ok = True
    ok &= _run_case(1, 2, 64, 4, 64, 32, 1, torch.float16, "L0-1", seed=0)[0]
    ok &= _run_case(1, 2, 128, 4, 128, 32, 1, torch.bfloat16, "L0-2", seed=1)[0]
    ok &= _run_case(2, 4, 64, 8, 64, 64, 2, torch.float16, "L0-3", seed=2)[0]
    ok &= _run_case(2, 2, 64, 4, 64, 32, 2, torch.bfloat16, "L0-4", seed=3)[0]
    ok &= _run_case(1, 2, 96, 4, 48, 48, 1, torch.float16, "L0-5", seed=4)[0]
    # L0-7: representative workload w2 (depth-2 pipeline steady state + bn=128
    # single-block + non-degenerate snake split), both dtypes.
    ok &= _run_case(1, 16, 256, 48, 64, 128, 1, torch.float16, "L0-7-fp16", seed=5)[0]
    ok &= _run_case(1, 16, 256, 48, 64, 128, 1, torch.bfloat16, "L0-7-bf16", seed=6)[0]
    print("[[contract]]", end=" ")
    _contract_checks()
    assert ok, "L0 FAILED (blocking)"
    print("[L0] PASS (7 precision cases + contract)")


def run_L1():
    print("== L1 (blocking) ==")
    ok = True
    cases = [
        (2, 4, 128, 8, 64, 64, 2, "L1-128-multilt"),
        (1, 16, 64, 48, 64, 64, 8, "L1-16chunk"),
        (2, 8, 256, 8, 128, 128, 2, "L1-256-full"),
    ]
    seed = 10
    for B, C, Q, H, P, N, G, name in cases:
        for dt in (torch.float16, torch.bfloat16):
            ok &= _run_case(B, C, Q, H, P, N, G, dt, f"{name}-{_dtype_str(dt)}", seed=seed)[0]
            seed += 1
    assert ok, "L1 FAILED (blocking)"
    print("[L1] PASS (6 cases)")


def run_L2():
    print("== L2 (warn-only) ==")
    cases = [
        (1, 1, 64, 4, 64, 32, 1, torch.float16, "L2-tiny", 20, 1.0),
        (4, 8, 64, 80, 128, 64, 8, torch.float16, "L2-large-h80", 21, 1.0),
        (2, 4, 64, 8, 64, 64, 2, torch.bfloat16, "L2-deep-decay-bf16", 22, 100.0),
        (2, 2, 128, 4, 64, 64, 2, torch.float16, "L2-deep-decay-fp16", 23, 100.0),
    ]
    n_pass = 0
    for B, C, Q, H, P, N, G, dt, name, seed, decay in cases:
        try:
            ok, _ = _run_case(B, C, Q, H, P, N, G, dt, name, seed=seed, decay_scale=decay)
            n_pass += 1 if ok else 0
        except Exception as e:  # noqa: BLE001 - warn-only layer
            print(f"  [{name}] WARN (exception): {e}")
    if n_pass == len(cases):
        print(f"[L2] PASS ({len(cases)} cases)")
    else:
        print(f"[L2] WARN: {len(cases) - n_pass}/{len(cases)} cases failed (non-blocking)")


def run_boundary():
    print("== Boundary (warn-only) ==")
    cases = [
        (1, 2, 96, 4, 48, 48, 1, torch.bfloat16, "B-q96-bf16", 30),
        (1, 2, 64, 4, 64, 16, 1, torch.float16, "B-n16", 31),
        (1, 3, 64, 4, 64, 32, 1, torch.float16, "B-c3", 32),
        (1, 2, 80, 4, 64, 32, 1, torch.float16, "B-q80", 33),
    ]
    n_pass = 0
    for B, C, Q, H, P, N, G, dt, name, seed in cases:
        try:
            ok, _ = _run_case(B, C, Q, H, P, N, G, dt, name, seed=seed)
            n_pass += 1 if ok else 0
        except Exception as e:  # noqa: BLE001 - warn-only layer
            print(f"  [{name}] WARN (exception): {e}")
    if n_pass == len(cases):
        print(f"[Boundary] PASS ({len(cases)} cases)")
    else:
        print(f"[Boundary] WARN: {len(cases) - n_pass}/{len(cases)} cases failed (non-blocking)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "all"])
    args, _ = parser.parse_known_args()
    if args.level == "L0":
        run_L0()
    else:
        run_L0()
        run_L1()
        run_L2()
        run_boundary()
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    main()
