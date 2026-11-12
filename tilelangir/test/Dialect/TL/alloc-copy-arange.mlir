// RUN: tilelangir-opt %s | FileCheck %s

// CHECK-LABEL: func.func @elemwise_arange
// CHECK-SAME: #tl.core_type<aiv>
// CHECK: %[[UB_A:.*]] = tl.alloc : memref<8x8xf32, #tl.address_space<ub>>
// CHECK: tl.arange outs(%[[UB_A]]) offset(%{{.*}}) strides(%{{.*}}, %{{.*}}) : memref<8x8xf32, #tl.address_space<ub>>
// CHECK: tl.copy %[[UB_A]] -> %{{.*}} : memref<8x8xf32, #tl.address_space<ub>>, memref<8x8xf32, #tl.address_space<gm>>

func.func @elemwise_arange(%gm_a: memref<8x8xf32, #tl.address_space<gm>>,
                           %off: index, %s0: index, %s1: index)
    attributes {tl.core_type = #tl.core_type<aiv>} {
  %ub_a = tl.alloc : memref<8x8xf32, #tl.address_space<ub>>
  tl.arange outs(%ub_a) offset(%off) strides(%s0, %s1) : memref<8x8xf32, #tl.address_space<ub>>
  tl.copy %ub_a -> %gm_a : memref<8x8xf32, #tl.address_space<ub>>, memref<8x8xf32, #tl.address_space<gm>>
  return
}
