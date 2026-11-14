// RUN: tilelangir-opt %s | FileCheck %s

// CHECK-LABEL: func.func @elemwise_vbitcast
// CHECK-SAME: #tl.core_type<aiv>
// CHECK: %[[UB_A:.*]] = tl.alloc : memref<64xf32, #tl.address_space<ub>>
// CHECK: %[[UB_C:.*]] = tl.alloc : memref<64xi32, #tl.address_space<ub>>
// CHECK: tl.copy %{{.*}} -> %[[UB_A]] : memref<64xf32, #tl.address_space<gm>>, memref<64xf32, #tl.address_space<ub>>
// CHECK: %[[RES:.*]] = tl.vbitcast %[[UB_A]] : memref<64xf32, #tl.address_space<ub>> to memref<64xi32, #tl.address_space<ub>>

func.func @elemwise_vbitcast(%gm_a: memref<64xf32, #tl.address_space<gm>>,
                             %gm_c: memref<64xi32, #tl.address_space<gm>>)
    attributes {tl.core_type = #tl.core_type<aiv>} {
  %ub_a = tl.alloc : memref<64xf32, #tl.address_space<ub>>
  %ub_c = tl.alloc : memref<64xi32, #tl.address_space<ub>>
  tl.copy %gm_a -> %ub_a : memref<64xf32, #tl.address_space<gm>>, memref<64xf32, #tl.address_space<ub>>
  %res = tl.vbitcast %ub_a : memref<64xf32, #tl.address_space<ub>> to memref<64xi32, #tl.address_space<ub>>
  tl.copy %res -> %gm_c : memref<64xi32, #tl.address_space<ub>>, memref<64xi32, #tl.address_space<gm>>
  return
}
