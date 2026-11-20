"""Stage a token's frequency row in UB and reuse it across a block of heads.

Adapted from examples/posembedding/rope_llama31/kernels/table_low_precision.py.
NPUIR vector operations replace the Ascend 950 SIMT/SIMD intrinsics from the
source PR. Both pairing conventions accumulate in float32 and cast once.
"""

import functools
import os

import tilelang
import tilelang.language as T


@functools.lru_cache(maxsize=64)
def build_staged_rotation(
    tokens, heads, dim, rotary, table_rows, dtype, interleaved=False, indexed=False
):
    if os.environ.get("TILELANG_ASCEND_MODE", "").lower() != "developer":
        raise RuntimeError("set TILELANG_ASCEND_MODE=Developer before compiling RoPE")
    half = rotary // 2
    padded = (half + 15) // 16 * 16
    # Keep each tile bounded even for wide heads; no divisibility requirement.
    block_heads = min(heads, max(1, min(8, 1024 // padded)))
    head_blocks = (heads + block_heads - 1) // block_heads
    tasks = tokens * head_blocks
    blocks = min(48, tasks)
    positions_size = tokens if indexed else 1
    tail = max(1, dim - rotary)

    @T.prim_func
    def main(
        X: T.Tensor((tokens, heads, dim), dtype),
        Cos: T.Tensor((table_rows, half), dtype),
        Sin: T.Tensor((table_rows, half), dtype),
        Positions: T.Tensor((positions_size,), "int32"),
        Y: T.Tensor((tokens, heads, dim), dtype),
    ):
        with T.Kernel(blocks, is_npu=True) as (pid, _):
            raw = T.alloc_ub((block_heads, 2 * padded), dtype)
            lo_io = T.alloc_ub((block_heads, padded), dtype)
            hi_io = T.alloc_ub((block_heads, padded), dtype)
            table_io = T.alloc_ub((1, padded), dtype)
            lo = T.alloc_ub((block_heads, padded), "float32")
            hi = T.alloc_ub((block_heads, padded), "float32")
            cosine = T.alloc_ub((1, padded), "float32")
            sine = T.alloc_ub((1, padded), "float32")
            out_lo = T.alloc_ub((block_heads, padded), "float32")
            out_hi = T.alloc_ub((block_heads, padded), "float32")
            tmp = T.alloc_ub((block_heads, padded), "float32")
            suffix = T.alloc_ub((block_heads, tail), dtype)

            for step in T.serial(T.ceildiv(tasks, blocks)):
                task = pid + step * blocks
                with T.If(task < tasks), T.Then():
                    token = task // head_blocks
                    head_start = task % head_blocks * block_heads
                    rows = T.min(block_heads, heads - head_start)
                    position = token % table_rows
                    if indexed:
                        # The wrapper checks in int64 before narrowing; clamp also
                        # protects direct kernel callers from out-of-range reads.
                        position = T.min(T.max(Positions[token], 0), table_rows - 1)
                    T.vbrc(0, table_io)
                    T.copy(Cos[position, :half], table_io[0, :half])
                    if dtype == "float32":
                        T.copy(table_io, cosine)
                    else:
                        T.vcast(table_io, cosine, round_mode="rint")
                    T.copy(Sin[position, :half], table_io[0, :half])
                    if dtype == "float32":
                        T.copy(table_io, sine)
                    else:
                        T.vcast(table_io, sine, round_mode="rint")

                    if interleaved:
                        T.vbrc(0, raw)
                        T.copy(
                            X[token, head_start : head_start + rows, :rotary],
                            raw[:rows, :rotary],
                        )
                        T.deinterleave(raw, lo_io, hi_io, channel_nums=2)
                    else:
                        T.vbrc(0, lo_io)
                        T.vbrc(0, hi_io)
                        T.copy(
                            X[token, head_start : head_start + rows, :half],
                            lo_io[:rows, :half],
                        )
                        T.copy(
                            X[token, head_start : head_start + rows, half:rotary],
                            hi_io[:rows, :half],
                        )
                    if dtype == "float32":
                        T.copy(lo_io, lo)
                        T.copy(hi_io, hi)
                    else:
                        T.vcast(lo_io, lo, round_mode="rint")
                        T.vcast(hi_io, hi, round_mode="rint")
                    T.vmul(lo, cosine, out_lo)
                    T.vmul(hi, sine, tmp)
                    T.vsub(out_lo, tmp, out_lo)
                    T.vmul(hi, cosine, out_hi)
                    T.vmul(lo, sine, tmp)
                    T.vadd(out_hi, tmp, out_hi)
                    if dtype == "float32":
                        T.copy(out_lo, lo_io)
                        T.copy(out_hi, hi_io)
                    else:
                        T.vcast(out_lo, lo_io, round_mode="rint")
                        T.vcast(out_hi, hi_io, round_mode="rint")
                    if interleaved:
                        T.interleave(lo_io, hi_io, raw, channel_nums=2)
                        T.copy(
                            raw[:rows, :rotary],
                            Y[token, head_start : head_start + rows, :rotary],
                        )
                    else:
                        T.copy(
                            lo_io[:rows, :half],
                            Y[token, head_start : head_start + rows, :half],
                        )
                        T.copy(
                            hi_io[:rows, :half],
                            Y[token, head_start : head_start + rows, half:rotary],
                        )
                    if rotary < dim:
                        T.copy(
                            X[token, head_start : head_start + rows, rotary:dim],
                            suffix[:rows, :tail],
                        )
                        T.copy(
                            suffix[:rows, :tail],
                            Y[token, head_start : head_start + rows, rotary:dim],
                        )

    return tilelang.compile(
        main,
        target="npuir",
        out_idx=[4],
        pass_configs={
            tilelang.PassConfigKey.NPUIR_ENABLE_AUTO_MULTI_BUFFER: True,
            tilelang.PassConfigKey.NPUIR_DISABLE_HIVM_AUTO_INJECT_SYNC: False,
        },
    )
