// RUN: tilelangir-opt %s | FileCheck %s

// Elementwise cast: src and dst may differ in element type.

// CHECK-LABEL: func.func @elemwise_cast
// CHECK-SAME: #tl.core_type<aiv>
// CHECK: %[[UB_A:.*]] = tl.alloc : memref<64xf32, #tl.address_space<ub>>
// CHECK: %[[UB_C:.*]] = tl.alloc : memref<64xf16, #tl.address_space<ub>>
// CHECK: tl.copy %{{.*}} -> %[[UB_A]] : memref<64xf32, #tl.address_space<gm>>, memref<64xf32, #tl.address_space<ub>>
// CHECK: tl.cast ins(%[[UB_A]]) outs(%[[UB_C]]) round_mode("rint") : memref<64xf32, #tl.address_space<ub>>, memref<64xf16, #tl.address_space<ub>>
// CHECK: tl.copy %[[UB_C]] -> %{{.*}} : memref<64xf16, #tl.address_space<ub>>, memref<64xf16, #tl.address_space<gm>>

func.func @elemwise_cast(%gm_a: memref<64xf32, #tl.address_space<gm>>,
                         %gm_c: memref<64xf16, #tl.address_space<gm>>)
    attributes {tl.core_type = #tl.core_type<aiv>} {
  %ub_a = tl.alloc : memref<64xf32, #tl.address_space<ub>>
  %ub_c = tl.alloc : memref<64xf16, #tl.address_space<ub>>
  tl.copy %gm_a -> %ub_a : memref<64xf32, #tl.address_space<gm>>, memref<64xf32, #tl.address_space<ub>>
  tl.cast ins(%ub_a) outs(%ub_c) round_mode("rint") : memref<64xf32, #tl.address_space<ub>>, memref<64xf16, #tl.address_space<ub>>
  tl.copy %ub_c -> %gm_c : memref<64xf16, #tl.address_space<ub>>, memref<64xf16, #tl.address_space<gm>>
  return
}
