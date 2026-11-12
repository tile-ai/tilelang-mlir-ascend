"""Grouped NCL convolution: exact input packing, Cube FP32 accumulation, one cast."""

import functools

import tilelang
import tilelang.ascend.language as T
import torch  # noqa: F401

from tileops._kernels.convolution.call_spec import Conv1dFwdInterface
from tileops._kernels.kernel_base import Kernel


@functools.lru_cache(maxsize=128)
def build(c):
    batch, groups, ci, co = c.n, c.groups, c.c_in_g, c.c_out_g
    length, out, kw = c.l_in, c.out_l, c.kernel_l
    stride, pad, dilation = c.stride_l, c.pad_left, c.dilation_l
    dtype = str(c.dtype).split(".")[-1]
    k = ci * kw
    kb = (
        256
        if c.has_bias and k >= 1024 and k % 256 == 0
        else (16 if k <= 16 else (k if 128 < k <= 256 and k % 16 == 0 else 128))
    )
    kp, mp, np = (k + kb - 1) // kb * kb, (co + 63) // 64 * 64, (out + 127) // 128 * 128
    xtotal = batch * groups * kp * np
    wtotal = groups * mp * kp
    xb, wb = min(64, (xtotal + 255) // 256), min(64, (wtotal + 255) // 256)

    @T.prim_func
    def pack_x(
        X: T.Tensor((batch, groups * ci, length), dtype),
        P: T.Tensor((batch * groups, kp, np), dtype),
    ):
        with T.Kernel(xb) as bx:  # noqa: SIM117
            with T.SimtVF(threads=256):
                for step in T.serial(T.ceildiv(xtotal, xb * 256)):
                    for lane in T.Parallel(256):
                        index = (step * xb + bx) * 256 + lane
                        if index < xtotal:
                            j = index % np
                            r = index // np % kp
                            bg = index // (np * kp)
                            pos = j * stride - pad + (r % kw) * dilation
                            P[bg, r, j] = T.cast(0, dtype)
                            if r < k and j < out and pos >= 0 and pos < length:
                                P[bg, r, j] = X[bg // groups, bg % groups * ci + r // kw, pos]

    @T.prim_func
    def pack_w(W: T.Tensor((groups * co, ci, kw), dtype), Q: T.Tensor((groups, mp, kp), dtype)):
        with T.Kernel(wb) as bx:  # noqa: SIM117
            with T.SimtVF(threads=256):
                for step in T.serial(T.ceildiv(wtotal, wb * 256)):
                    for lane in T.Parallel(256):
                        index = (step * wb + bx) * 256 + lane
                        if index < wtotal:
                            r = index % kp
                            m = index // kp % mp
                            g = index // (kp * mp)
                            Q[g, m, r] = T.cast(0, dtype)
                            if r < k and m < co:
                                Q[g, m, r] = W[g * co + m, r // kw, r % kw]

    wide = (
        co % 128 == 0
        and 128 < k <= 256
        and np % 256 == 0
        and batch * groups * (co // 128) * (np // 256) >= 32
    )
    bm, bn, bk = (128, 256, kb) if wide else (64, 128, kb)
    mt, nt = mp // bm, np // bn
    tiles = batch * groups * mt * nt
    blocks = min(32, tiles)
    direct = use_direct_output(c)
    cm, cn = (co, out) if direct else (mp, np)
    cdtype = dtype if direct else "float32"

    @T.prim_func
    def contract(
        Q: T.Tensor((groups, mp, kp), dtype),
        P: T.Tensor((batch * groups, kp, np), dtype),
        C: T.Tensor((batch * groups, cm, cn), cdtype),
    ):
        with T.Kernel(blocks) as bx:
            a = T.alloc_l1((bm, bk), dtype)
            b = T.alloc_l1((bn, bk), dtype)
            acc = T.alloc_l0c((bm, bn), "float32")
            for step in T.serial(T.ceildiv(tiles, blocks)):
                tile = step * blocks + bx
                if tile < tiles:
                    bg = tile // (mt * nt)
                    mi = tile // nt % mt * bm
                    ni = tile % nt * bn
                    am = bm
                    an = bn
                    for ki in T.serial(T.ceildiv(kp, bk)):
                        ak = bk
                        T.copy(Q[bg % groups, mi : mi + am, ki * bk : ki * bk + ak], a[0:am, 0:ak])
                        T.copy(
                            P[bg, ki * bk : ki * bk + ak, ni : ni + an],
                            b[0:an, 0:ak],
                            transpose=True,
                        )
                        T.gemm(
                            a[0:am, 0:ak],
                            b[0:an, 0:ak],
                            acc[0:am, 0:an],
                            transpose_B=True,
                            clear_accum=ki == 0,
                        )
                    if direct:
                        live_n = T.min(bn, out - ni)
                        T.copy(acc[0:am, 0:live_n], C[bg, mi : mi + am, ni : ni + live_n])
                    else:
                        T.copy(acc[0:am, 0:an], C[bg, mi : mi + am, ni : ni + an])

    @T.prim_func
    def contract_pipeline(
        Q: T.Tensor((groups, mp, kp), dtype),
        P: T.Tensor((batch * groups, kp, np), dtype),
        C: T.Tensor((batch * groups, cm, cn), cdtype),
    ):
        with T.Kernel(blocks) as bx:
            a = T.alloc_l1((64, kb), dtype)
            b = T.alloc_l1((128, kb), dtype)
            acc = T.alloc_l0c((64, 128), "float32")
            T.annotate_buffer_versions({a: 2, b: 2})
            for step in T.serial(T.ceildiv(tiles, blocks)):
                tile = step * blocks + bx
                if tile < tiles:
                    bg = tile // (mt * nt)
                    mi = tile // nt % mt * 64
                    ni = tile % nt * 128
                    am = 64
                    an = 128
                    for ki in T.Pipelined(T.ceildiv(kp, kb), num_stages=2):
                        ak = kb
                        T.copy(Q[bg % groups, mi : mi + am, ki * kb : ki * kb + ak], a[0:am, 0:ak])
                        T.copy(
                            P[bg, ki * kb : ki * kb + ak, ni : ni + an],
                            b[0:an, 0:ak],
                            transpose=True,
                        )
                        T.gemm(
                            a[0:am, 0:ak],
                            b[0:an, 0:ak],
                            acc[0:am, 0:an],
                            transpose_B=True,
                            clear_accum=ki == 0,
                        )
                    if direct:
                        live_n = T.min(128, out - ni)
                        T.copy(acc[0:am, 0:live_n], C[bg, mi : mi + am, ni : ni + live_n])
                    else:
                        T.copy(acc[0:am, 0:an], C[bg, mi : mi + am, ni : ni + an])

    @T.prim_func
    def contract_outer_pipeline(
        Q: T.Tensor((groups, mp, kp), dtype),
        P: T.Tensor((batch * groups, kp, np), dtype),
        C: T.Tensor((batch * groups, cm, cn), cdtype),
    ):
        with T.Kernel(blocks) as bx:
            a = T.alloc_l1((bm, bk), dtype)
            b = T.alloc_l1((bn, bk), dtype)
            acc = T.alloc_l0c((bm, bn), "float32")
            T.annotate_buffer_versions({a: 2, b: 2, acc: 1})
            for step in T.Pipelined(T.ceildiv(tiles, blocks), num_stages=2):
                tile = step * blocks + bx
                if tile < tiles:
                    bg = tile // (mt * nt)
                    mi = tile // nt % mt * bm
                    ni = tile % nt * bn
                    am = bm
                    an = bn
                    for ki in T.serial(T.ceildiv(kp, bk)):
                        ak = bk
                        T.copy(Q[bg % groups, mi : mi + am, ki * bk : ki * bk + ak], a[0:am, 0:ak])
                        T.copy(
                            P[bg, ki * bk : ki * bk + ak, ni : ni + an],
                            b[0:an, 0:ak],
                            transpose=True,
                        )
                        T.gemm(
                            a[0:am, 0:ak],
                            b[0:an, 0:ak],
                            acc[0:am, 0:an],
                            transpose_B=True,
                            clear_accum=ki == 0,
                        )
                    if direct:
                        live_n = T.min(bn, out - ni)
                        T.copy(acc[0:am, 0:live_n], C[bg, mi : mi + am, ni : ni + live_n])
                    else:
                        T.copy(acc[0:am, 0:an], C[bg, mi : mi + am, ni : ni + an])

    total = batch * groups * co * out
    fb = min(64, (total + 255) // 256)
    has_bias = c.has_bias

    @T.prim_func
    def finish(
        C: T.Tensor((batch * groups, mp, np), "float32"),
        Bias: T.Tensor((groups * co,), dtype),
        Y: T.Tensor((batch, groups * co, out), dtype),
    ):
        with T.Kernel(fb) as bx:  # noqa: SIM117
            with T.SimtVF(threads=256):
                for step in T.serial(T.ceildiv(total, fb * 256)):
                    for lane in T.Parallel(256):
                        index = (step * fb + bx) * 256 + lane
                        if index < total:
                            j = index % out
                            channel = index // out % (groups * co)
                            n = index // (out * groups * co)
                            value = T.alloc_var("float32")
                            value = C[n * groups + channel // co, channel % co, j]
                            if has_bias:
                                value = value + T.cast(Bias[channel], "float32")
                            Y[n, channel, j] = T.cast(value, dtype)

    channel_dma = (
        stride == 1
        and pad == 0
        and dilation == 1
        and 128 < k <= 256
        and kp == k
        and 2 <= kw <= 16
        and length <= 4096
        and kw * np <= 16384
        and kw * np % 256 == 0
        and batch * groups * ci >= 64
    )
    channel_dma = channel_dma or (
        stride == 1
        and kp == k
        and 2 <= kw <= 16
        and length <= 2048
        and kw * np <= 16384
        and kw * np % 256 == 0
        and batch * groups * ci >= 64
    )
    input_kernel = (
        build_input_channel_dma(c)
        if channel_dma
        else (
            build_input_dma(c)
            if c.stride_l == 1 and c.pad_left == 0
            else tilelang.compile(pack_x, target="ascend", out_idx=[1])
        )
    )
    compiled = (
        input_kernel,
        tilelang.compile(pack_w, target="ascend", out_idx=[1]),
        (build_deep_mixed_contract(c) if use_deep_mixed_finish(c) else build_mixed_contract(c))
        if use_mixed_finish(c)
        else tilelang.compile(
            contract_outer_pipeline
            if wide and direct
            else (contract_pipeline if kp // kb >= 4 else contract),
            target="ascend",
            out_idx=[2],
        ),
    )
    final = (
        build_finish_dma(c)
        if use_dma_finish(c)
        else tilelang.compile(finish, target="ascend", out_idx=[2])
    )
    return (*compiled, final)


def use_direct_output(c):
    k = c.c_in_g * c.kernel_l
    np = (c.out_l + 127) // 128 * 128
    return not c.has_bias and (
        (c.c_out_g % 64 == 0 and (k + 127) // 128 >= 4)
        or (
            c.c_out_g % 128 == 0
            and 128 < k <= 256
            and np % 256 == 0
            and c.n * c.groups * (c.c_out_g // 128) * (np // 256) >= 32
        )
    )


def use_mixed_finish(c):
    k = c.c_in_g * c.kernel_l
    np = (c.out_l + 127) // 128 * 128
    return use_deep_mixed_finish(c) or (
        c.has_bias
        and c.c_out_g % 128 == 0
        and 128 < k <= 256
        and np % 256 == 0
        and c.n * c.groups * (c.c_out_g // 128) * (np // 256) >= 32
    )


@functools.lru_cache(maxsize=128)
def build_mixed_contract(c):
    batch, groups, ci, co = c.n, c.groups, c.c_in_g, c.c_out_g
    out, kw = c.out_l, c.kernel_l
    dtype = str(c.dtype).split(".")[-1]
    k = ci * kw
    kb = k if 128 < k <= 256 and k % 16 == 0 else 128
    kp, mp, np = (k + kb - 1) // kb * kb, (co + 63) // 64 * 64, (out + 127) // 128 * 128
    bm, bn = 128, 256
    mt, nt = mp // bm, np // bn
    tiles = batch * groups * mt * nt
    blocks = min(32, tiles)

    @T.prim_func
    def contract_mixed(
        Q: T.Tensor((groups, mp, kp), dtype),
        P: T.Tensor((batch * groups, kp, np), dtype),
        Bias: T.Tensor((groups * co,), dtype),
        Y: T.Tensor((batch * groups, co, out), dtype),
    ):
        with T.MixedKernel(blocks) as (bx, sid):
            a = T.alloc_l1((bm, kb), dtype)
            b = T.alloc_l1((bn, kb), dtype)
            acc = T.alloc_l0c((bm, bn), "float32")
            values = T.alloc_shared((bm // 2, bn), "float32")
            result = T.alloc_shared((bm // 2, bn), dtype)
            bias_raw = T.alloc_shared((bm,), dtype)
            T.annotate_buffer_versions({a: 2, b: 2, acc: 1, values: 1, result: 1, bias_raw: 1})
            for step in T.Pipelined(T.ceildiv(tiles, blocks), num_stages=2):
                tile = step * blocks + bx
                if tile < tiles:
                    bg = tile // (mt * nt)
                    mi = tile // nt % mt * bm
                    ni = tile % nt * bn
                    valid_n = T.min(bn, out - ni)
                    T.copy(Bias[bg % groups * co + mi : bg % groups * co + mi + bm], bias_raw)
                    for ki in T.serial(T.ceildiv(kp, kb)):
                        T.copy(Q[bg % groups, mi : mi + bm, ki * kb : ki * kb + kb], a)
                        T.copy(P[bg, ki * kb : ki * kb + kb, ni : ni + bn], b, transpose=True)
                        T.gemm(a, b, acc, transpose_B=True, clear_accum=ki == 0)
                    T.dual_copy(acc, values)
                    with T.SimdVF():
                        full = T.simd.pset(32)
                        full16 = T.simd.pset(16)
                        for row in T.serial(bm // 2):
                            packed_bias = T.simd.vld(
                                bias_raw[row + sid * (bm // 2)], dist="BRC_B16"
                            )
                            bias_vector = T.simd.vcvt(packed_bias, "float32", full16, part=0)
                            for part in T.serial(bn // 64):
                                value = T.simd.vld(values[row, part * 64])
                                value = T.simd.vadd(value, bias_vector, full)
                                narrow = T.simd.vcvt(value, dtype, full, sat=False)
                                T.simd.vsts(result[row, part * 64], narrow, full, dist="PK_B32")
                    T.dual_copy(
                        result[0 : bm // 2, 0:valid_n], Y[bg, mi : mi + bm, ni : ni + valid_n]
                    )

    return tilelang.compile(contract_mixed, target="ascend", out_idx=[3])


def use_deep_mixed_finish(c):
    k = c.c_in_g * c.kernel_l
    kb = 256 if k >= 1024 and k % 256 == 0 else (16 if k <= 16 else 128)
    return c.has_bias and c.c_out_g % 64 == 0 and (k + kb - 1) // kb >= 4


@functools.lru_cache(maxsize=128)
def build_deep_mixed_contract(c):
    batch, groups, ci, co = c.n, c.groups, c.c_in_g, c.c_out_g
    out, kw = c.out_l, c.kernel_l
    dtype = str(c.dtype).split(".")[-1]
    k = ci * kw
    kb = 256 if k >= 1024 and k % 256 == 0 else (16 if k <= 16 else 128)
    kp, mp, np = (k + kb - 1) // kb * kb, (co + 63) // 64 * 64, (out + 127) // 128 * 128
    bm, bn = 64, 128
    mt, nt = mp // bm, np // bn
    tiles = batch * groups * mt * nt
    blocks = min(32, tiles)

    @T.prim_func
    def contract_mixed(
        Q: T.Tensor((groups, mp, kp), dtype),
        P: T.Tensor((batch * groups, kp, np), dtype),
        Bias: T.Tensor((groups * co,), dtype),
        Y: T.Tensor((batch * groups, co, out), dtype),
    ):
        with T.MixedKernel(blocks) as (bx, sid):
            a = T.alloc_l1((bm, kb), dtype)
            b = T.alloc_l1((bn, kb), dtype)
            acc = T.alloc_l0c((bm, bn), "float32")
            values = T.alloc_shared((bm // 2, bn), "float32")
            result = T.alloc_shared((bm // 2, bn), dtype)
            bias_raw = T.alloc_shared((bm,), dtype)
            T.annotate_buffer_versions({a: 2, b: 2})
            for step in T.serial(T.ceildiv(tiles, blocks)):
                tile = step * blocks + bx
                if tile < tiles:
                    bg = tile // (mt * nt)
                    mi = tile // nt % mt * bm
                    ni = tile % nt * bn
                    valid_n = T.min(bn, out - ni)
                    T.copy(Bias[bg % groups * co + mi : bg % groups * co + mi + bm], bias_raw)
                    for ki in T.Pipelined(T.ceildiv(kp, kb), num_stages=2):
                        T.copy(Q[bg % groups, mi : mi + bm, ki * kb : ki * kb + kb], a)
                        T.copy(P[bg, ki * kb : ki * kb + kb, ni : ni + bn], b, transpose=True)
                        T.gemm(a, b, acc, transpose_B=True, clear_accum=ki == 0)
                    T.dual_copy(acc, values)
                    with T.SimdVF():
                        full = T.simd.pset(32)
                        full16 = T.simd.pset(16)
                        for row in T.serial(bm // 2):
                            packed_bias = T.simd.vld(
                                bias_raw[row + sid * (bm // 2)], dist="BRC_B16"
                            )
                            bias_vector = T.simd.vcvt(packed_bias, "float32", full16, part=0)
                            for part in T.serial(bn // 64):
                                value = T.simd.vld(values[row, part * 64])
                                value = T.simd.vadd(value, bias_vector, full)
                                narrow = T.simd.vcvt(value, dtype, full, sat=False)
                                T.simd.vsts(result[row, part * 64], narrow, full, dist="PK_B32")
                    T.dual_copy(
                        result[0 : bm // 2, 0:valid_n], Y[bg, mi : mi + bm, ni : ni + valid_n]
                    )

    return tilelang.compile(contract_mixed, target="ascend", out_idx=[3])


def use_dma_finish(c):
    total = c.n * c.c_out * c.out_l
    return total % 16 == 0 and c.out_l >= 128 and (c.c_out_g % 64 == 0 or c.n * c.groups == 1)


@functools.lru_cache(maxsize=128)
def build_finish_dma(c):
    dtype = str(c.dtype).split(".")[-1]
    co, out = c.c_out_g, c.out_l
    mp, np = (co + 63) // 64 * 64, (out + 127) // 128 * 128
    total = c.n * c.c_out * out
    ct = c.n * c.groups * mp * np
    tile = 4096
    nt = (out + tile - 1) // tile
    jobs = c.n * c.c_out * nt
    blocks = min(64, jobs)
    has_bias = c.has_bias

    @T.prim_func
    def finish_dma(
        C: T.Tensor((ct,), "float32"),
        Bias: T.Tensor((c.c_out,), dtype),
        Y: T.Tensor((total,), dtype),
    ):
        with T.Kernel(blocks) as bx:
            values = T.alloc_shared((tile,), "float32")
            output = T.alloc_shared((tile,), dtype)
            bias_raw = T.alloc_shared((128,), dtype)
            # Valid prefixes overwrite these values; padded SIMD lanes stay in UB.
            with T.SimdVF():
                full = T.simd.pset(32)
                zero = T.simd.vdup(T.float32(0), "float32", full)
                for part in T.serial(tile // 64):
                    T.simd.vsts(values[part * 64], zero, full)
            for step in T.serial(T.ceildiv(jobs, blocks)):
                job = step * blocks + bx
                if job < jobs:
                    row = job // nt
                    col = job % nt * tile
                    valid = T.min(tile, out - col)
                    source = (row // co * mp + row % co) * np + col
                    T.copy(C[source : source + valid], values[0:valid])
                    channel = row % c.c_out
                    bias_base = channel // 128 * 128
                    if has_bias:
                        bias_count = T.min(128, c.c_out - bias_base)
                        T.copy(Bias[bias_base : bias_base + bias_count], bias_raw[0:bias_count])
                    with T.SimdVF():
                        full = T.simd.pset(32)
                        if has_bias:
                            full16 = T.simd.pset(16)
                            packed_bias = T.simd.vld(bias_raw[channel - bias_base], dist="BRC_B16")
                            bias_vector = T.simd.vcvt(packed_bias, "float32", full16, part=0)
                        for part in T.serial(tile // 64):
                            value = T.simd.vld(values[part * 64])
                            if has_bias:
                                value = T.simd.vadd(value, bias_vector, full)
                            narrow = T.simd.vcvt(value, dtype, full, sat=False)
                            T.simd.vsts(output[part * 64], narrow, full, "PK_B32")
                    offset = row * out + col
                    T.copy(output[0:valid], Y[offset : offset + valid])

    return tilelang.compile(finish_dma, target="ascend", out_idx=[2])


@functools.lru_cache(maxsize=128)
def build_input_dma(c):
    dtype = str(c.dtype).split(".")[-1]
    ci, kw, length, out = c.c_in_g, c.kernel_l, c.l_in, c.out_l
    k = ci * kw
    kb = 16 if k <= 16 else (k if 128 < k <= 256 and k % 16 == 0 else 128)
    kp, np = (k + kb - 1) // kb * kb, (out + 127) // 128 * 128
    width = 2048
    nt = (np + width - 1) // width
    tiles = c.n * c.groups * kp * nt
    blocks = min(64, tiles)

    @T.prim_func
    def pack_x_dma(
        X: T.Tensor((c.n, c.c_in, length), dtype), P: T.Tensor((c.n * c.groups, kp, np), dtype)
    ):
        with T.Kernel(blocks) as bx:
            raw = T.reshape(X, (c.n * c.c_in * length,))
            shared = T.alloc_shared((width,), dtype)
            for step in T.serial(T.ceildiv(tiles, blocks)):
                tile = step * blocks + bx
                if tile < tiles:
                    col = tile % nt * width
                    row = tile // nt % kp
                    bg = tile // (nt * kp)
                    valid = T.min(width, np - col)
                    with T.SimdVF():
                        full = T.simd.pset(16)
                        zero = T.simd.vdup(T.cast(0, dtype), dtype, full)
                        for part in T.serial(width // 128):
                            T.simd.vsts(shared[part * 128], zero, full, "NORM_B16")
                    if row < k:
                        pos = col - c.pad_left + (row % kw) * c.dilation_l
                        lo = T.max(0, -pos)
                        hi = T.min(T.min(valid, out - col), length - pos)
                        if lo < hi:
                            start = (bg * ci + row // kw) * length + pos + lo
                            T.copy(raw[start : start + hi - lo], shared[lo:hi])
                    T.copy(shared[0:valid], P[bg, row, col : col + valid])

    return tilelang.compile(pack_x_dma, target="ascend", out_idx=[1])


@functools.lru_cache(maxsize=128)
def build_input_channel_dma(c):
    dtype = str(c.dtype).split(".")[-1]
    ci, kw, length, out = c.c_in_g, c.kernel_l, c.l_in, c.out_l
    k = ci * kw
    kb = 16 if k <= 16 else (k if 128 < k <= 256 and k % 16 == 0 else 128)
    kp, np = (k + kb - 1) // kb * kb, (out + 127) // 128 * 128
    jobs = c.n * c.groups * ci
    blocks = min(64, jobs)
    count = kw * np

    @T.prim_func
    def pack_x_channel(
        X: T.Tensor((c.n, c.c_in, length), dtype), P: T.Tensor((c.n * c.groups, kp, np), dtype)
    ):
        with T.Kernel(blocks) as bx:
            raw = T.alloc_shared((length,), dtype)
            packed = T.alloc_shared((count,), dtype)
            dest = T.reshape(P, (c.n * c.groups * kp * np,))
            for step in T.serial(T.ceildiv(jobs, blocks)):
                job = step * blocks + bx
                if job < jobs:
                    bg = job // ci
                    channel = job % ci
                    T.copy(X[bg // c.groups, bg % c.groups * ci + channel, 0:length], raw)
                    if c.pad_left == 0 and c.dilation_l == 1:
                        with T.SimdVF():
                            full = T.simd.pset(16)
                            zero = T.simd.vdup(T.cast(0, dtype), dtype, full)
                            for r in T.serial(kw):
                                for part in T.serial(np // 128):
                                    col = part * 128
                                    indices = T.simd.vci(T.cast(col + r, "int16"), "int16")
                                    safe_indices = T.simd.vmins(
                                        indices, T.cast(length - 1, "int16"), full
                                    )
                                    value = T.simd.vgather2(
                                        T.access_ptr(raw[0], "r", extent=length),
                                        T.reinterpret("uint16x128", safe_indices),
                                        full,
                                    )
                                    valid = T.simd.update_mask(T.min(128, T.max(0, out - col)), 16)
                                    value = T.simd.vsel(value, zero, valid)
                                    T.simd.vsts(packed[r * np + col], value, full, "NORM_B16")
                    else:
                        with T.SimtVF(threads=256):
                            for part in T.serial(count // 256):
                                for lane in T.Parallel(256):
                                    index = part * 256 + lane
                                    r = index // np
                                    col = index % np
                                    pos = col - c.pad_left + r * c.dilation_l
                                    value = T.alloc_var(dtype)
                                    value = T.cast(0, dtype)
                                    if col < out and pos >= 0 and pos < length:
                                        value = raw[pos]
                                    packed[index] = value
                    offset = (bg * kp + channel * kw) * np
                    T.copy(packed, dest[offset : offset + count])

    return tilelang.compile(pack_x_channel, target="ascend", out_idx=[1])


class AscendConv1dKernel(Kernel, Conv1dFwdInterface):
    devices = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call)

    def __init__(self, call):
        super().__init__()
        self.call = call
        self.pack_x, self.pack_w, self.contract, self.finish = build(call)

    def forward(self, x, weight, bias=None):
        packed = self.pack_x(x)
        k = self.call.c_in_g * self.call.kernel_l
        if self.call.c_out_g % 64 == 0 and (k % 128 == 0 or (128 < k <= 256 and k % 16 == 0)):
            # No padding is needed: the original contiguous grouped OIK layout
            # is exactly (groups, padded_output_channels, padded_reduction).
            weights = weight.view(self.call.groups, self.call.c_out_g, k)
        else:
            weights = self.pack_w(weight)
        if use_mixed_finish(self.call):
            return self.contract(weights, packed, bias).view(
                self.call.n, self.call.c_out, self.call.out_l
            )
        acc = self.contract(weights, packed)
        if use_direct_output(self.call):
            return acc.view(self.call.n, self.call.c_out, self.call.out_l)
        # No bias reads are generated in the no-bias specialization.
        if use_dma_finish(self.call):
            return self.finish(
                acc.reshape(-1), bias if bias is not None else weight.reshape(-1)[: self.call.c_out]
            ).view(self.call.n, self.call.c_out, self.call.out_l)
        return self.finish(acc, bias if bias is not None else weight.reshape(-1)[: self.call.c_out])
