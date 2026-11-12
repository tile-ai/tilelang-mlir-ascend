// RUN: tilelangir-opt %s --tilelangir-convert-tl-to-hivm | FileCheck %s

// NOTE: hivm.hir.varange's exact printed syntax lives upstream in bishengir;
// tighten these CHECK lines after your first real run below.

// CHECK: module attributes {{{.*}}hivm.module_core_type = #hivm.module_core_type<AIV>
// CHECK-LABEL: func.func @elemwise_arange
// CHECK-SAME: memref<8x8xf32, #hivm.address_space<gm>>
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-NOT: tl.
// CHECK: %[[UB_A:.*]] = memref.alloc() : memref<8x8xf32, #hivm.address_space<ub>>
// CHECK: hivm.hir.varange
// CHECK-SAME: %[[UB_A]]
// CHECK: memref.copy %[[UB_A]], %{{.*}}

func.func @elemwise_arange(%gm_a: memref<8x8xf32, #tl.address_space<gm>>,
                           %off: index, %s0: index, %s1: index)
    attributes {tl.core_type = #tl.core_type<aiv>} {
  %ub_a = tl.alloc : memref<8x8xf32, #tl.address_space<ub>>
  tl.arange outs(%ub_a) offset(%off) strides(%s0, %s1) : memref<8x8xf32, #tl.address_space<ub>>
  tl.copy %ub_a -> %gm_a : memref<8x8xf32, #tl.address_space<ub>>, memref<8x8xf32, #tl.address_space<gm>>
  return
}
