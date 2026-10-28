// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file src/ascend910/op/ops.cc
 * \brief Register the A2 npuir operators.
 */

#include <tvm/ir/op.h>
#include <tvm/tirx/op_attr_types.h>

namespace tvm {
namespace tl {

using namespace tirx;

#define TILELANG_A2_OP(Name)                                                   \
  TVM_REGISTER_OP("tl." #Name)                                                 \
      .set_attr<TCallEffectKind>("TCallEffectKind",                            \
                                 Integer(CallEffectKind::kOpaque))

TILELANG_A2_OP(copy);
TVM_REGISTER_OP("tir.infinity")
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure));
TVM_REGISTER_OP("tirx.infinity")
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure));
TVM_REGISTER_OP("tirx.nan")
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure));
TVM_REGISTER_OP("tir.exp").set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));
TVM_REGISTER_OP("tir.exp2").set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));
TVM_REGISTER_OP("tir.log").set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));
TVM_REGISTER_OP("tir.log2").set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));
TILELANG_A2_OP(npuir_abs);
TILELANG_A2_OP(npuir_add);
TILELANG_A2_OP(npuir_and);
TILELANG_A2_OP(npuir_arange);
TILELANG_A2_OP(npuir_atomic_add);
TILELANG_A2_OP(npuir_bitcast);
TILELANG_A2_OP(npuir_brc);
TILELANG_A2_OP(npuir_cast);
TILELANG_A2_OP(npuir_cmp);
TILELANG_A2_OP(npuir_concat);
TILELANG_A2_OP(npuir_cumsum);
TILELANG_A2_OP(npuir_debug_print_buffer_value);
TILELANG_A2_OP(npuir_debug_print_var);
TILELANG_A2_OP(npuir_deinterleave);
TILELANG_A2_OP(npuir_div);
TILELANG_A2_OP(npuir_dot);
TILELANG_A2_OP(npuir_exp);
TILELANG_A2_OP(npuir_flip);
TILELANG_A2_OP(npuir_gather);
TILELANG_A2_OP(npuir_indirect_load);
TILELANG_A2_OP(npuir_interleave);
TILELANG_A2_OP(npuir_ln);
TILELANG_A2_OP(npuir_load_nd2nz);
TILELANG_A2_OP(npuir_max);
TILELANG_A2_OP(npuir_min);
TILELANG_A2_OP(npuir_mul);
TILELANG_A2_OP(npuir_not);
TILELANG_A2_OP(npuir_or);
TILELANG_A2_OP(npuir_pad);
TILELANG_A2_OP(npuir_pipe_barrier);
TILELANG_A2_OP(npuir_pow);
TILELANG_A2_OP(npuir_rec);
TILELANG_A2_OP(npuir_reduce);
TILELANG_A2_OP(npuir_relu);
TILELANG_A2_OP(npuir_reshape);
TILELANG_A2_OP(npuir_rsqrt);
TILELANG_A2_OP(npuir_select);
TILELANG_A2_OP(npuir_set_flag);
TILELANG_A2_OP(npuir_shl);
TILELANG_A2_OP(npuir_shr);
TILELANG_A2_OP(npuir_sigmoid);
TILELANG_A2_OP(npuir_sort);
TILELANG_A2_OP(npuir_sqrt);
TILELANG_A2_OP(npuir_store_fixpipe);
TILELANG_A2_OP(npuir_store_nz2nd);
TILELANG_A2_OP(npuir_sub);
TILELANG_A2_OP(npuir_sync_block);
TILELANG_A2_OP(npuir_sync_block_set);
TILELANG_A2_OP(npuir_sync_block_wait);
TILELANG_A2_OP(npuir_transpose);
TILELANG_A2_OP(npuir_vcos);
TILELANG_A2_OP(npuir_verf);
TILELANG_A2_OP(npuir_vsin);
TILELANG_A2_OP(npuir_vtanh);
TILELANG_A2_OP(npuir_wait_flag);
TILELANG_A2_OP(npuir_xor);
TILELANG_A2_OP(ascend_copy);
#undef TILELANG_A2_OP

} // namespace tl
} // namespace tvm
