// RUN: tilelangir-opt %s --tilelangir-convert-tl-to-hivm | FileCheck %s

// Expert-mode atomic_add subset: TLIR -> HIVM (no HACC / FFTS / workspace).

// CHECK: module attributes {{{.*}}hivm.module_core_type = #hivm.module_core_type<AIV>
// CHECK-LABEL: func.func @elemwise_atomic_add
// CHECK-SAME: memref<64xf32, #hivm.address_space<gm>>
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-NOT: tl.
// CHECK: %[[UB_A:.*]] = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
// CHECK: memref.copy %{{.*}}, %[[UB_A]]
// CHECK: hivm.hir.store ins(%[[UB_A]]{{.*}}) outs(%{{.*}}) atomic = <add>

func.func @elemwise_atomic_add(%gm_a: memref<64xf32, #tl.address_space<gm>>,
                               %gm_b: memref<64xf32, #tl.address_space<gm>>)
    attributes {tl.core_type = #tl.core_type<aiv>} {
  %ub_a = tl.alloc : memref<64xf32, #tl.address_space<ub>>
  tl.copy %gm_a -> %ub_a : memref<64xf32, #tl.address_space<gm>>, memref<64xf32, #tl.address_space<ub>>
  tl.atomic_add %ub_a -> %gm_b : memref<64xf32, #tl.address_space<ub>>, memref<64xf32, #tl.address_space<gm>>
  return
}
