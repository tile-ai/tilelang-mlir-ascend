// RUN: tilelangir-opt %s | FileCheck %s

// CHECK-LABEL: func.func @vrelu_basic
func.func @vrelu_basic(%gm_a: memref<64xf32, #tl.address_space<gm>>,
                       %gm_b: memref<64xf32, #tl.address_space<gm>>)
    attributes {tl.core_type = #tl.core_type<aiv>} {
  // CHECK: %[[UB_A:.*]] = tl.alloc : memref<64xf32, #tl.address_space<ub>>
  // CHECK: %[[UB_B:.*]] = tl.alloc : memref<64xf32, #tl.address_space<ub>>
  %ub_a = tl.alloc : memref<64xf32, #tl.address_space<ub>>
  %ub_b = tl.alloc : memref<64xf32, #tl.address_space<ub>>
  // CHECK: tl.copy %{{.*}} -> %[[UB_A]]
  tl.copy %gm_a -> %ub_a : memref<64xf32, #tl.address_space<gm>>, memref<64xf32, #tl.address_space<ub>>
  // CHECK: tl.vrelu %[[UB_A]], %[[UB_B]] : memref<64xf32, #tl.address_space<ub>>
  tl.vrelu %ub_a, %ub_b : memref<64xf32, #tl.address_space<ub>>
  // CHECK: tl.copy %[[UB_B]] -> %{{.*}}
  tl.copy %ub_b -> %gm_b : memref<64xf32, #tl.address_space<ub>>, memref<64xf32, #tl.address_space<gm>>
  return
}
