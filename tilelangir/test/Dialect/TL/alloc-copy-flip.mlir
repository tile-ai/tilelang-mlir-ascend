// RUN: tilelangir-opt %s | FileCheck %s

// CHECK-LABEL: func.func @elemwise_flip
// CHECK-SAME: #tl.core_type<aiv>
// CHECK: %[[UB_A:.*]] = tl.alloc : memref<128x256xf16, #tl.address_space<ub>>
// CHECK: %[[UB_B:.*]] = tl.alloc : memref<128x256xf16, #tl.address_space<ub>>
// CHECK: tl.copy %{{.*}} -> %[[UB_A]] : memref<128x256xf16, #tl.address_space<gm>>, memref<128x256xf16, #tl.address_space<ub>>
// CHECK: tl.flip ins(%[[UB_A]]) outs(%[[UB_B]]) {axis = 1 : ui64} : memref<128x256xf16, #tl.address_space<ub>>
// CHECK: tl.copy %[[UB_B]] -> %{{.*}} : memref<128x256xf16, #tl.address_space<ub>>, memref<128x256xf16, #tl.address_space<gm>>

func.func @elemwise_flip(%gm_a: memref<128x256xf16, #tl.address_space<gm>>,
                         %gm_b: memref<128x256xf16, #tl.address_space<gm>>)
    attributes {tl.core_type = #tl.core_type<aiv>} {
  %ub_a = tl.alloc : memref<128x256xf16, #tl.address_space<ub>>
  %ub_b = tl.alloc : memref<128x256xf16, #tl.address_space<ub>>
  tl.copy %gm_a -> %ub_a : memref<128x256xf16, #tl.address_space<gm>>, memref<128x256xf16, #tl.address_space<ub>>
  tl.flip ins(%ub_a) outs(%ub_b) {axis = 1 : ui64} : memref<128x256xf16, #tl.address_space<ub>>
  tl.copy %ub_b -> %gm_b : memref<128x256xf16, #tl.address_space<ub>>, memref<128x256xf16, #tl.address_space<gm>>
  return
}
