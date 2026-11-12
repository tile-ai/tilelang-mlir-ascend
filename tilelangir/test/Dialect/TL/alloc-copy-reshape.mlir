// RUN: tilelangir-opt %s | FileCheck %s

// Reshape: src and result share element type and total size, differ in shape.

// CHECK-LABEL: func.func @reshape_view
// CHECK-SAME: #tl.core_type<aiv>
// CHECK: %[[UB_A:.*]] = tl.alloc : memref<8x16xf16, #tl.address_space<ub>>
// CHECK: tl.copy %{{.*}} -> %[[UB_A]] : memref<8x16xf16, #tl.address_space<gm>>, memref<8x16xf16, #tl.address_space<ub>>
// CHECK: %[[UB_B:.*]] = tl.reshape %[[UB_A]] : memref<8x16xf16, #tl.address_space<ub>> to memref<16x8xf16, #tl.address_space<ub>>
// CHECK: tl.copy %[[UB_B]] -> %{{.*}} : memref<16x8xf16, #tl.address_space<ub>>, memref<16x8xf16, #tl.address_space<gm>>

func.func @reshape_view(%gm_a: memref<8x16xf16, #tl.address_space<gm>>,
                       %gm_b: memref<16x8xf16, #tl.address_space<gm>>)
    attributes {tl.core_type = #tl.core_type<aiv>} {
  %ub_a = tl.alloc : memref<8x16xf16, #tl.address_space<ub>>
  tl.copy %gm_a -> %ub_a : memref<8x16xf16, #tl.address_space<gm>>, memref<8x16xf16, #tl.address_space<ub>>
  %ub_b = tl.reshape %ub_a : memref<8x16xf16, #tl.address_space<ub>> to memref<16x8xf16, #tl.address_space<ub>>
  tl.copy %ub_b -> %gm_b : memref<16x8xf16, #tl.address_space<ub>>, memref<16x8xf16, #tl.address_space<gm>>
  return
}
