"""Ascend 910 (A2) language facade.

The shared TileLang surface plus A2-only allocation and npuir operations.
"""

from __future__ import annotations

from tilelang.language import *  # noqa: F401,F403
from tilelang.language import alloc_shared  # noqa: F401
from tilelang.language.kernel import launch_kernel  # noqa: F401

from . import npuir
from .copy import copy  # noqa: F401
from .kernel import Kernel  # noqa: F401
from .npuir import *  # noqa: F401,F403
from .npuir import (  # noqa: F401
    npuir_abs,
    npuir_add,
    npuir_and,
    npuir_arange,
    npuir_atomic_add,
    npuir_bitcast,
    npuir_brc,
    npuir_cast,
    npuir_cmp,
    npuir_concat,
    npuir_cumsum,
    npuir_deinterleave,
    npuir_div,
    npuir_dot,
    npuir_exp,
    npuir_exp2,
    npuir_flip,
    npuir_gather,
    npuir_interleave,
    npuir_ln,
    npuir_load_nd2nz,
    npuir_max,
    npuir_min,
    npuir_mul,
    npuir_not,
    npuir_or,
    npuir_pad,
    npuir_pow,
    npuir_rec,
    npuir_reduce,
    npuir_relu,
    npuir_reshape,
    npuir_rsqrt,
    npuir_select,
    npuir_shl,
    npuir_shr,
    npuir_sigmoid,
    npuir_sort,
    npuir_sqrt,
    npuir_store_fixpipe,
    npuir_store_nz2nd,
    npuir_sub,
    npuir_transpose,
    npuir_vcos,
    npuir_verf,
    npuir_vsin,
    npuir_vtanh,
    npuir_xor,
)


def alloc_ub(shape, dtype, scope="shared"):
    """A2's unified buffer is the tilelang shared buffer."""
    return alloc_shared(shape, dtype, scope=scope)


def alloc_l1(shape, dtype):
    from tilelang.language import alloc_l1 as _alloc_l1

    return _alloc_l1(shape, dtype)


vadd = npuir_add
vsub = npuir_sub
vmul = npuir_mul
vdiv = npuir_div
vmax = npuir_max
vmin = npuir_min
vor = npuir_or
vand = npuir_and
vxor = npuir_xor
vpow = npuir_pow
vshl = npuir_shl
vshr = npuir_shr
vexp = npuir_exp
vexp2 = npuir_exp2
vrelu = npuir_relu
vsigmoid = npuir_sigmoid
vln = npuir_ln
vlog2 = npuir_log2
vsqrt = npuir_sqrt
vrsqrt = npuir_rsqrt
vabs = npuir_abs
vrec = npuir_rec
vnot = npuir_not
vselect = npuir_select
vcmp = npuir_cmp
vdot = npuir_dot
vbrc = npuir_brc
vcast = npuir_cast
vreduce = npuir_reduce
vcumsum = npuir_cumsum
vsort = npuir_sort
vtranspose = npuir_transpose
vinterleave = npuir_interleave
vdeinterleave = npuir_deinterleave
varange = npuir_arange
vconcat = npuir_concat
vpad = npuir_pad
vflip = npuir_flip
vbitcast = npuir_bitcast
vcos = npuir_vcos
vsin = npuir_vsin
verf = npuir_verf
vtanh = npuir_vtanh


from tvm.tirx.script.builder.ir import alloc_buffer as _a2_alloc_buffer


def alloc_L1(shape, dtype):
    return _a2_alloc_buffer(shape, dtype, scope="shared.dyn")


def alloc_L0A(shape, dtype):
    return _a2_alloc_buffer(shape, dtype, scope="wmma.matrix_a")


def alloc_L0B(shape, dtype):
    return _a2_alloc_buffer(shape, dtype, scope="wmma.matrix_b")


def alloc_L0C(shape, dtype):
    return _a2_alloc_buffer(shape, dtype, scope="wmma.accumulator")
