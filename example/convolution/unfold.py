"""Tiled DMA unfolding with vector decimation for strided convolutions."""

import functools

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S


@functools.lru_cache(maxsize=128)
def _unfold_kernel(n, ci, groups, size, out, window, stride, pad, dilation, dtype):
    depth, height, width = size
    od, oh, ow = out
    kd, kh, kw = window
    sd, sh, sw = stride
    pd, ph, pw = pad
    dd, dh, dw = dilation
    reduction = ci // groups * kd * kh * kw
    pitch = (reduction + 15) // 16 * 16
    cols = min(ow, max(1, 8192 // sw))
    raw_width = (cols - 1) * sw + 1
    raw_pitch = (raw_width + 15) // 16 * 16
    rows = min(oh, max(1, 16384 // (raw_pitch * sh)))
    raw_height = (rows - 1) * sh + 1
    output_pitch = (cols + 127) // 128 * 128
    row_tiles = (oh + rows - 1) // rows
    col_tiles = (ow + cols - 1) // cols
    # A zero leading pad may still have a trailing pad (even-kernel "same").
    direct = (
        sh == 1
        and sw == 1
        and ph == 0
        and pw == 0
        and oh <= height - (kh - 1) * dh
        and ow <= width - (kw - 1) * dw
    )
    jobs = n * groups * pitch * od * row_tiles * col_tiles
    cores = min(64, jobs)

    @T.prim_func
    def main(
        X: T.Tensor((n, ci, depth, height, width), dtype),
        P: T.Tensor((n, groups, pitch, od, oh, ow), dtype),
    ):
        with T.Kernel(cores) as core:
            raw = T.alloc_shared((raw_height, raw_pitch), dtype)
            decimated = T.alloc_shared((rows, output_pitch), dtype)
            T.annotate_buffer_versions({raw: 2, decimated: 2})
            for step in T.Pipelined(T.ceildiv(jobs, cores), num_stages=2):
                job = step * cores + core
                if job < jobs:
                    ox = job % col_tiles * cols
                    oy = job // col_tiles % row_tiles * rows
                    oz = job // (col_tiles * row_tiles) % od
                    r = job // (col_tiles * row_tiles * od) % pitch
                    group = job // (col_tiles * row_tiles * od * pitch) % groups
                    batch = job // (col_tiles * row_tiles * od * pitch * groups)
                    z = oz * sd - pd + r // (kh * kw) % kd * dd
                    y = oy * sh - ph + r // kw % kh * dh
                    x = ox * sw - pw + r % kw * dw
                    count_y = T.min(rows, oh - oy)
                    count_x = T.min(cols, ow - ox)
                    begin_y, begin_x = T.max(0, y), T.max(0, x)
                    valid_y = T.max(0, T.min(height, y + raw_height) - begin_y)
                    valid_x = T.max(0, T.min(width, x + raw_width) - begin_x)
                    if r < reduction and z >= 0 and z < depth and valid_y > 0 and valid_x > 0:
                        T.copy(
                            X[
                                batch,
                                group * (ci // groups) + r // (kd * kh * kw),
                                z,
                                begin_y : begin_y + valid_y,
                                begin_x : begin_x + valid_x,
                            ],
                            raw[:valid_y, :valid_x],
                        )
                    else:
                        with T.SimtVF(threads=256):
                            for i, j in T.Parallel(raw_height, raw_pitch):
                                raw[i, j] = T.cast(0, dtype)
                    if direct:
                        T.copy(
                            raw[:count_y, :count_x],
                            P[batch, group, r, oz, oy : oy + count_y, ox : ox + count_x],
                        )
                    else:
                        with T.SimdVF():
                            full = S.pset(16)
                            lane = S.vci(0, "int16")
                            zero = S.vdup(T.cast(0, dtype), dtype, full)
                            for i in T.serial(rows):
                                iy = i * sh + y - begin_y
                                for part in T.serial(T.ceildiv(cols, 128)):
                                    ix = S.vadds(
                                        S.vmuls(lane, T.int16(sw), full),
                                        T.cast(part * 128 * sw + x - begin_x, "int16"),
                                        full,
                                    )
                                    mask = S.pand(
                                        S.vcmps(ix, T.int16(0), full, "ge"),
                                        S.vcmps(ix, T.cast(valid_x, "int16"), full, "lt"),
                                        full,
                                    )
                                    if (
                                        r < reduction
                                        and z >= 0
                                        and z < depth
                                        and iy >= 0
                                        and iy < valid_y
                                        and valid_x > 0
                                    ):
                                        offsets = S.vmaxs(
                                            S.vmins(ix, T.cast(valid_x - 1, "int16"), full),
                                            T.int16(0),
                                            full,
                                        )
                                        values = S.vsel(
                                            S.vgather2(
                                                raw[iy, 0],
                                                T.reinterpret(offsets, "uint16x128"),
                                                full,
                                            ),
                                            zero,
                                            mask,
                                        )
                                        S.vsts(decimated[i, part * 128], values, full)
                                    else:
                                        S.vsts(decimated[i, part * 128], zero, full)
                        T.copy(
                            decimated[:count_y, :count_x],
                            P[batch, group, r, oz, oy : oy + count_y, ox : ox + count_x],
                        )

    return tilelang.compile(main, target="ascend", out_idx=-1)
