// RUN: tilelangir-opt %s | FileCheck %s

// Power variant of the Appendix A add example from
// docs/fyp-tilelang-mlir-dialect-proposal.md

// CHECK-LABEL: func.func @elemwise_pow
// CHECK-SAME: #tl.core_type<aiv>
// CHECK: %[[UB_A:.*]] = tl.alloc : memref<64xi32, #tl.address_space<ub>>
// CHECK: %[[UB_B:.*]] = tl.alloc : memref<64xi32, #tl.address_space<ub>>
// CHECK: %[[UB_C:.*]] = tl.alloc : memref<64xi32, #tl.address_space<ub>>
// CHECK: tl.copy %{{.*}} -> %[[UB_A]] : memref<64xi32, #tl.address_space<gm>>, memref<64xi32, #tl.address_space<ub>>
// CHECK: tl.copy %{{.*}} -> %[[UB_B]] : memref<64xi32, #tl.address_space<gm>>, memref<64xi32, #tl.address_space<ub>>
// CHECK: tl.pow ins(%[[UB_A]], %[[UB_B]]) outs(%[[UB_C]]) : memref<64xi32, #tl.address_space<ub>>
// CHECK: tl.copy %[[UB_C]] -> %{{.*}} : memref<64xi32, #tl.address_space<ub>>, memref<64xi32, #tl.address_space<gm>>

func.func @elemwise_pow(%gm_a: memref<64xi32, #tl.address_space<gm>>,
                        %gm_b: memref<64xi32, #tl.address_space<gm>>,
                        %gm_c: memref<64xi32, #tl.address_space<gm>>)
    attributes {tl.core_type = #tl.core_type<aiv>} {
  %ub_a = tl.alloc : memref<64xi32, #tl.address_space<ub>>
  %ub_b = tl.alloc : memref<64xi32, #tl.address_space<ub>>
  %ub_c = tl.alloc : memref<64xi32, #tl.address_space<ub>>
  tl.copy %gm_a -> %ub_a : memref<64xi32, #tl.address_space<gm>>, memref<64xi32, #tl.address_space<ub>>
  tl.copy %gm_b -> %ub_b : memref<64xi32, #tl.address_space<gm>>, memref<64xi32, #tl.address_space<ub>>
  tl.pow ins(%ub_a, %ub_b) outs(%ub_c) : memref<64xi32, #tl.address_space<ub>>
  tl.copy %ub_c -> %gm_c : memref<64xi32, #tl.address_space<ub>>, memref<64xi32, #tl.address_space<gm>>
  return
}
