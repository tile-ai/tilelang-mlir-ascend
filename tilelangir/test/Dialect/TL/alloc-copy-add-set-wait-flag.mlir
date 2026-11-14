// RUN: tilelangir-opt %s | FileCheck %s

// CHECK-LABEL: func.func @elemwise_add_set_wait_flag
// CHECK-SAME: #tl.core_type<aiv>
// CHECK: %[[UB_A:.*]] = tl.alloc : memref<64xf32, #tl.address_space<ub>>
// CHECK: %[[UB_B:.*]] = tl.alloc : memref<64xf32, #tl.address_space<ub>>
// CHECK: %[[UB_C:.*]] = tl.alloc : memref<64xf32, #tl.address_space<ub>>
// CHECK: tl.copy %{{.*}} -> %[[UB_A]] : memref<64xf32, #tl.address_space<gm>>, memref<64xf32, #tl.address_space<ub>>
// CHECK: tl.copy %{{.*}} -> %[[UB_B]] : memref<64xf32, #tl.address_space<gm>>, memref<64xf32, #tl.address_space<ub>>
// CHECK: tl.set_flag[<pipe_mte2>, <pipe_v>, %{{.*}}] : i64
// CHECK: tl.wait_flag[<pipe_mte2>, <pipe_v>, %{{.*}}] : i64
// CHECK: tl.add ins(%[[UB_A]], %[[UB_B]]) outs(%[[UB_C]]) : memref<64xf32, #tl.address_space<ub>>
// CHECK: tl.set_flag[<pipe_v>, <pipe_mte3>, %{{.*}}] : i64
// CHECK: tl.wait_flag[<pipe_v>, <pipe_mte3>, %{{.*}}] : i64
// CHECK: tl.copy %[[UB_C]] -> %{{.*}} : memref<64xf32, #tl.address_space<ub>>, memref<64xf32, #tl.address_space<gm>>

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
