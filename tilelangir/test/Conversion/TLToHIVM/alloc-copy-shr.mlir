// RUN: tilelangir-opt %s --tilelangir-convert-tl-to-hivm | FileCheck %s

// Expert-mode shr subset: TLIR -> HIVM (no HACC / FFTS / workspace).

// CHECK: module attributes {{{.*}}hivm.module_core_type = #hivm.module_core_type<AIV>
// CHECK-LABEL: func.func @elemwise_shr
// CHECK-SAME: memref<64xi32, #hivm.address_space<gm>>
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-NOT: tl.
// CHECK: %[[UB_A:.*]] = memref.alloc() : memref<64xi32, #hivm.address_space<ub>>
// CHECK: %[[UB_B:.*]] = memref.alloc() : memref<64xi32, #hivm.address_space<ub>>
// CHECK: %[[UB_C:.*]] = memref.alloc() : memref<64xi32, #hivm.address_space<ub>>
// CHECK: memref.copy %{{.*}}, %[[UB_A]]
// CHECK: memref.copy %{{.*}}, %[[UB_B]]
// CHECK: hivm.hir.vshr ins(%[[UB_A]], %[[UB_B]] :{{.*}}) outs(%[[UB_C]] :{{.*}}) round : true
// CHECK: memref.copy %[[UB_C]], %{{.*}}

func.func @elemwise_shr(%gm_a: memref<64xi32, #tl.address_space<gm>>,
                        %gm_b: memref<64xi32, #tl.address_space<gm>>,
                        %gm_c: memref<64xi32, #tl.address_space<gm>>)
    attributes {tl.core_type = #tl.core_type<aiv>} {
  %ub_a = tl.alloc : memref<64xi32, #tl.address_space<ub>>
  %ub_b = tl.alloc : memref<64xi32, #tl.address_space<ub>>
  %ub_c = tl.alloc : memref<64xi32, #tl.address_space<ub>>
  tl.copy %gm_a -> %ub_a : memref<64xi32, #tl.address_space<gm>>, memref<64xi32, #tl.address_space<ub>>
  tl.copy %gm_b -> %ub_b : memref<64xi32, #tl.address_space<gm>>, memref<64xi32, #tl.address_space<ub>>
  tl.shr ins(%ub_a, %ub_b) outs(%ub_c) round(true) : memref<64xi32, #tl.address_space<ub>>
  tl.copy %ub_c -> %gm_c : memref<64xi32, #tl.address_space<ub>>, memref<64xi32, #tl.address_space<gm>>
  return
}
