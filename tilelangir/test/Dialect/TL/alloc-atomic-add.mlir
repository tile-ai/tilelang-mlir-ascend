// RUN: tilelangir-opt %s | FileCheck %s

// Expert-mode atomic_add subset: dst (GM) += src (UB), no copy-back.

// CHECK-LABEL: func.func @elemwise_atomic_add
// CHECK-SAME: #tl.core_type<aiv>
// CHECK: %[[UB_A:.*]] = tl.alloc : memref<64xf32, #tl.address_space<ub>>
// CHECK: tl.copy %{{.*}} -> %[[UB_A]] : memref<64xf32, #tl.address_space<gm>>, memref<64xf32, #tl.address_space<ub>>
// CHECK: tl.atomic_add %[[UB_A]] -> %{{.*}} : memref<64xf32, #tl.address_space<ub>>, memref<64xf32, #tl.address_space<gm>>

func.func @elemwise_atomic_add(%gm_a: memref<64xf32, #tl.address_space<gm>>,
                               %gm_b: memref<64xf32, #tl.address_space<gm>>)
    attributes {tl.core_type = #tl.core_type<aiv>} {
  %ub_a = tl.alloc : memref<64xf32, #tl.address_space<ub>>
  tl.copy %gm_a -> %ub_a : memref<64xf32, #tl.address_space<gm>>, memref<64xf32, #tl.address_space<ub>>
  tl.atomic_add %ub_a -> %gm_b : memref<64xf32, #tl.address_space<ub>>, memref<64xf32, #tl.address_space<gm>>
  return
}
