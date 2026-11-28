// RUN: tilelangir-opt %s --tilelangir-convert-tl-to-hivm | FileCheck %s

// Expert-mode add subset with explicit cross-pipe sync (set + wait):
// TLIR -> HIVM (no HACC / FFTS / workspace).

// CHECK: module attributes {{{.*}}hivm.module_core_type = #hivm.module_core_type<AIV>
// CHECK-LABEL: func.func @elemwise_add_set_wait_flag
// CHECK-SAME: memref<64xf32, #hivm.address_space<gm>>
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-NOT: tl.
// CHECK: %[[UB_A:.*]] = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
// CHECK: %[[UB_B:.*]] = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
// CHECK: %[[UB_C:.*]] = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
// CHECK: memref.copy %{{.*}}, %[[UB_A]]
// CHECK: memref.copy %{{.*}}, %[[UB_B]]
// CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, %{{.*}}]
// CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, %{{.*}}]
// CHECK: hivm.hir.vadd ins(%[[UB_A]], %[[UB_B]] :{{.*}}) outs(%[[UB_C]] :
// CHECK: hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, %{{.*}}]
// CHECK: hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, %{{.*}}]
// CHECK: memref.copy %[[UB_C]], %{{.*}}

func.func @elemwise_add_set_wait_flag(%gm_a: memref<64xf32, #tl.address_space<gm>>,
                        %gm_b: memref<64xf32, #tl.address_space<gm>>,
                        %gm_c: memref<64xf32, #tl.address_space<gm>>)
    attributes {tl.core_type = #tl.core_type<aiv>} {
  %ub_a = tl.alloc : memref<64xf32, #tl.address_space<ub>>
  %ub_b = tl.alloc : memref<64xf32, #tl.address_space<ub>>
  %ub_c = tl.alloc : memref<64xf32, #tl.address_space<ub>>
  %c0_i64 = arith.constant 0 : i64
  tl.copy %gm_a -> %ub_a : memref<64xf32, #tl.address_space<gm>>, memref<64xf32, #tl.address_space<ub>>
  tl.copy %gm_b -> %ub_b : memref<64xf32, #tl.address_space<gm>>, memref<64xf32, #tl.address_space<ub>>
  tl.set_flag[#tl.pipe<pipe_mte2>, #tl.pipe<pipe_v>, %c0_i64] : i64
  tl.wait_flag[#tl.pipe<pipe_mte2>, #tl.pipe<pipe_v>, %c0_i64] : i64
  tl.add ins(%ub_a, %ub_b) outs(%ub_c) : memref<64xf32, #tl.address_space<ub>>
  tl.set_flag[#tl.pipe<pipe_v>, #tl.pipe<pipe_mte3>, %c0_i64] : i64
  tl.wait_flag[#tl.pipe<pipe_v>, #tl.pipe<pipe_mte3>, %c0_i64] : i64
  tl.copy %ub_c -> %gm_c : memref<64xf32, #tl.address_space<ub>>, memref<64xf32, #tl.address_space<gm>>
  return
}
