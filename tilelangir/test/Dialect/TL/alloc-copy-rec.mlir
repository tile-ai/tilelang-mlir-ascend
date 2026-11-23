// RUN: tilelangir-opt %s | FileCheck %s

// Appendix A from docs/fyp-tilelang-mlir-dialect-proposal.md

// CHECK-LABEL: func.func @elemwise_rec
// CHECK-SAME: #tl.core_type<aiv>
// CHECK: %[[UB_A:.*]] = tl.alloc : memref<64xf32, #tl.address_space<ub>>
// CHECK: %[[UB_B:.*]] = tl.alloc : memref<64xf32, #tl.address_space<ub>>
// CHECK: tl.copy %{{.*}} -> %[[UB_A]] : memref<64xf32, #tl.address_space<gm>>, memref<64xf32, #tl.address_space<ub>>
// CHECK: tl.rec ins(%[[UB_A]]) outs(%[[UB_B]]) : memref<64xf32, #tl.address_space<ub>>
// CHECK: tl.copy %[[UB_B]] -> %{{.*}} : memref<64xf32, #tl.address_space<ub>>, memref<64xf32, #tl.address_space<gm>>

func.func @elemwise_rec(%gm_a: memref<64xf32, #tl.address_space<gm>>,
                        %gm_b: memref<64xf32, #tl.address_space<gm>>)
    attributes {tl.core_type = #tl.core_type<aiv>} {
  %ub_a = tl.alloc : memref<64xf32, #tl.address_space<ub>>
  %ub_b = tl.alloc : memref<64xf32, #tl.address_space<ub>>
  tl.copy %gm_a -> %ub_a : memref<64xf32, #tl.address_space<gm>>, memref<64xf32, #tl.address_space<ub>>
  tl.rec ins(%ub_a) outs(%ub_b) : memref<64xf32, #tl.address_space<ub>>
  tl.copy %ub_b -> %gm_b : memref<64xf32, #tl.address_space<ub>>, memref<64xf32, #tl.address_space<gm>>
  return
}
