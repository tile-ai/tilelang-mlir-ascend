# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The language interface for tl programs."""

from typing import Optional

# from .parser import *
# now is fully compatible with the upstream
# tir script
# TODO(lei): remove this import once the
# upstream tir script is fully compatible
from tvm.script.parser.tir import *
from .tir import (
    prim_func,  # noqa: F401
)
from .tir.ir import *  # noqa: F401
from tilelang.layout import Layout, Fragment  # noqa: F401
from .proxy import (
    ptr,  # noqa: F401
    make_tensor,  # noqa: F401
    Buffer,  # noqa: F401
    Tensor,  # noqa: F401
    FragmentBuffer,  # noqa: F401
    SharedBuffer,  # noqa: F401
    LocalBuffer,  # noqa: F401
)
from .parallel import Parallel  # noqa: F401
from .pipeline import Pipelined  # noqa: F401
from .frame import has_let_value, get_let_value  # noqa: F401
from .kernel import (
    Kernel,  # noqa: F401
    KernelLaunchFrame,  # noqa: F401
    get_thread_binding,  # noqa: F401
    get_thread_bindings,  # noqa: F401
    get_block_binding,  # noqa: F401
    get_block_bindings,  # noqa: F401
)
from .warpgroup import ws  # noqa: F401
from .allocate import (
    alloc_local,  # noqa: F401
    alloc_shared,  # noqa: F401
    alloc_fragment,  # noqa: F401
    alloc_var,  # noqa: F401
    alloc_L0A,  # noqa: F401
    alloc_L0B,  # noqa: F401
    alloc_L0C,  # noqa: F401
    alloc_L1,  # noqa: F401
    alloc_ub,  # noqa: F401
)
from .copy import copy, c2d_im2col  # noqa: F401, F811
from .reduce import (
    reduce,  # noqa: F401
    reduce_max,  # noqa: F401
    reduce_min,  # noqa: F401
    reduce_sum,  # noqa: F401
    reduce_abssum,  # noqa: F401
    reduce_absmax,  # noqa: F401
    cumsum,  # noqa: F401
)

# from .print import print  # noqa: F401
from .customize import (
    # atomic_add,  # noqa: F401
    # atomic_addx2,  # noqa: F401
    # atomic_addx4,  # noqa: F401
    dp4a,  # noqa: F401
    clamp,  # noqa: F401
    reshape,  # noqa: F401
    view,  # noqa: F401
)
from .customize_npuir import (
    npuir_add,  # noqa: F401
    npuir_add as vadd,  # noqa: F401
    npuir_sub,  # noqa: F401
    npuir_sub as vsub,  # noqa: F401
    npuir_max,  # noqa: F401
    npuir_max as vmax,  # noqa: F401
    npuir_min,  # noqa: F401
    npuir_min as vmin,  # noqa: F401
    npuir_mul,  # noqa: F401
    npuir_mul as vmul,  # noqa: F401
    npuir_div,  # noqa: F401
    npuir_div as vdiv,  # noqa: F401
    npuir_or,  # noqa: F401
    npuir_or as vor,  # noqa: F401
    npuir_and,  # noqa: F401
    npuir_and as vand,  # noqa: F401
    npuir_xor,  # noqa: F401
    npuir_xor as vxor,  # noqa: F401
    npuir_pow,  # noqa: F401
    npuir_pow as vpow,  # noqa: F401
    npuir_shl,  # noqa: F401
    npuir_shl as vshl,  # noqa: F401
    npuir_shr,  # noqa: F401
    npuir_shr as vshr,  # noqa: F401
    npuir_exp,  # noqa: F401
    npuir_exp as vexp,  # noqa: F401
    npuir_dot,  # noqa: F401
    npuir_dot as gemm,  # noqa: F401
    npuir_ln,  # noqa: F401
    npuir_ln as vln,  # noqa: F401
    npuir_exp2,  # noqa: F401
    npuir_exp2 as vexp2,  # noqa: F401
    npuir_log2,  # noqa: F401
    npuir_log2 as vlog2,  # noqa: F401
    npuir_load_nd2nz,  # noqa: F401
    npuir_load_nd2nz as load_nd2nz,  # noqa: F401
    npuir_store_nz2nd,  # noqa: F401
    npuir_store_nz2nd as store_nz2nd,  # noqa: F401
    npuir_store_fixpipe,  # noqa: F401
    npuir_store_fixpipe as store_fixpipe,  # noqa: F401
    npuir_brc,  # noqa: F401
    npuir_brc as vbrc,  # noqa: F401
    npuir_fill,  # noqa: F401
    npuir_fill as fill,  # noqa: F401
    npuir_clear,  # noqa: F401
    npuir_clear as clear,  # noqa: F401
    npuir_cast,  # noqa: F401
    npuir_cast as vcast,  # noqa: F401
    npuir_reduce,  # noqa: F401
    npuir_reduce as reduce,  # noqa: F401, F811
    reduce_max,  # noqa: F401, F811
    reduce_min,  # noqa: F401, F811
    reduce_sum,  # noqa: F401, F811
    reduce_abssum,  # noqa: F401, F811
    reduce_absmax,  # noqa: F401, F811
    npuir_cumsum,  # noqa: F401
    npuir_cumsum as cumsum,  # noqa: F401, F811
    npuir_clamp,  # noqa: F401
    npuir_clamp as vclamp,  # noqa: F401
    npuir_atomic_add,  # noqa: F401
    npuir_atomic_add as atomic_add,  # noqa: F401
    npuir_atomic_addx4,  # noqa: F401
    npuir_atomic_addx4 as atomic_addx4,  # noqa: F401
    npuir_relu,  # noqa: F401
    npuir_relu as vrelu,  # noqa: F401
    npuir_sigmoid,  # noqa: F401
    npuir_sigmoid as vsigmoid,  # noqa: F401
    npuir_select,  # noqa: F401
    npuir_select as vselect,  # noqa: F401
    npuir_cmp,  # noqa: F401
    npuir_cmp as vcmp,  # noqa: F401
    npuir_sqrt,  # noqa: F401
    npuir_sqrt as vsqrt,  # noqa: F401
    npuir_rsqrt,  # noqa: F401
    npuir_rsqrt as vrsqrt,  # noqa: F401
    npuir_abs,  # noqa: F401
    npuir_abs as vabs,  # noqa: F401
    npuir_rec,  # noqa: F401
    npuir_rec as vrec,  # noqa: F401
    npuir_not,  # noqa: F401
    npuir_not as vnot,  # noqa: F401
    npuir_gather,  # noqa: F401
    npuir_gather as gather,  # noqa: F401
    npuir_interleave,  # noqa: F401
    npuir_interleave as interleave,  # noqa: F401
    npuir_deinterleave,  # noqa: F401
    npuir_deinterleave as deinterleave,  # noqa: F401
    npuir_transpose,  # noqa: F401
    npuir_transpose as transpose,  # noqa: F401
    npuir_arange,  # noqa: F401
    npuir_arange as arange,  # noqa: F401
    npuir_concat,  # noqa: F401
    npuir_concat as concat,  # noqa: F401
    npuir_pad,  # noqa: F401
    npuir_pad as pad,  # noqa: F401
    npuir_flip,  # noqa: F401
    npuir_flip as flip,  # noqa: F401
    npuir_bitcast,  # noqa: F401
    npuir_bitcast as vbitcast,  # noqa: F401
    npuir_vcos,  # noqa: F401
    npuir_vcos as vcos,  # noqa: F401
    npuir_vsin,  # noqa: F401
    npuir_vsin as vsin,  # noqa: F401
    npuir_verf,  # noqa: F401
    npuir_verf as verf,  # noqa: F401
    npuir_vtanh,  # noqa: F401
    npuir_vtanh as vtanh,  # noqa: F401
    rs,  # noqa: F401
    set_flag,  # noqa: F401
    wait_flag,  # noqa: F401
    pipe_barrier,  # noqa: F401
    block_barrier,  # noqa: F401
    subblock_barrier,  # noqa: F401
    sync_block_set,  # noqa: F401
    sync_block_wait,  # noqa: F401
    Scope,  # noqa: F401
    npuir_print as print,  # noqa: F401
    npuir_reshape,  # noqa: F401
    npuir_reshape as reshape,  # noqa: F401, F811
)
from .logical import any_of, all_of  # noqa: F401
from .builtin import *  # noqa: F401

from .memscope import *  # noqa: F401


def symbolic(name: str, dtype: str = "int32"):
    return tir.Var(name, dtype)


def dynamic(name: str, dtype: str = "int32"):
    """Alias for `symbolic` matching upstream tile-ai/tilelang's API name.

    Accepts comma- or whitespace-separated names to declare several vars
    at once: `B, M, N = T.dynamic("B, M, N")`. Forwards each name to
    `symbolic()` which returns a `tir.Var`.
    """
    import re

    if "," in name:
        names = re.split(r"\s*,\s*", name)
        return tuple(symbolic(n, dtype) for n in names)
    if " " in name:
        names = re.split(r"\s+", name)
        return tuple(symbolic(n, dtype) for n in names)
    return symbolic(name, dtype)


def use_swizzle(panel_size: int, order: str = "row", enable: bool = True):
    # If order is row, use rasterization2DRow, otherwise use rasterization2DColumn
    # The panel size is the number of threads in a warp
    # Use to improve the L2 Cache Locality
    device_func = "rasterization2DRow" if order == "row" else "rasterization2DColumn"
    return (
        attr(None, "threadblock_swizzle_pattern", f"tl::{device_func}<{panel_size}>")
        if enable
        else None
    )


def annotate_layout(layout_map: Dict):
    """Annotate the layout of the buffer

    Args:
        layout_map (Dict): a dictionary of buffer to layout

    Returns:
        block_attr: a block attribute

    Example:
        @T.prim_func
        def main(
                A: T.Tensor((M, N), dtype),
                B: T.Tensor((M, N), dtype),
        ):
            # Initialize Kernel Context
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_N), dtype)

                T.annotate_layout({A_shared: layout})
                for i, j in T.Parallel(block_M, block_N):
                    A_shared[i, j] = A[by * block_M + i, bx * block_N + j]

                for i, j in T.Parallel(block_M, block_N):
                    B[by * block_M + i, bx * block_N + j] = A_shared[i, j]

        return main
    """
    # layout_map is a dictionary of buffer to layout
    layout_map = {buffer.data: layout for buffer, layout in layout_map.items()}
    return block_attr({"layout_map": layout_map})


def annotate_padding(padding_map: Dict):
    """Annotate the padding of the buffer

    Args:
        padding_map (dict): a dictionary of buffer to padding value

    Returns:
        block_attr: a block attribute

    Example:
        @T.prim_func
        def main(
                A: T.Tensor((M, N), dtype),
                B: T.Tensor((M, N), dtype),
        ):
            # Initialize Kernel Context
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_N), dtype)

                T.annotate_padding({A_shared: pad_value})
                for i, j in T.Parallel(block_M, block_N):
                    A_shared[i, j] = A[by * block_M + i - 10, bx * block_N + j]

                for i, j in T.Parallel(block_M, block_N):
                    B[by * block_M + i, bx * block_N + j] = A_shared[i, j]

        return main
    """
    # padding_map is a dictionary of buffer to padding value
    _padding_map = {}
    for buffer, padding_value in padding_map.items():
        # assert not global
        assert buffer.scope() != "global", (
            "padding can only be applied to global buffers"
        )
        _padding_map[buffer.data] = padding_value
    return block_attr({"padding_map": _padding_map})


def import_source(source: Optional[str] = None):
    # source is the source code to be imported
    return block_attr({"pragma_import_c": source}) if source is not None else None
