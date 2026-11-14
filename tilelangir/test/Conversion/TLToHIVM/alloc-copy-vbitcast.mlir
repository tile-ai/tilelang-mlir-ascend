// RUN: tilelangir-opt %s --tilelangir-convert-tl-to-hivm | FileCheck %s

// Expert-mode vbitcast subset: TLIR -> HIVM (no HACC / FFTS / workspace).

// CHECK: module attributes {{{.*}}hivm.module_core_type = #hivm.module_core_type<AIV>
// CHECK-LABEL: func.func @elemwise_vbitcast
// CHECK-SAME: memref<64xf32, #hivm.address_space<gm>>
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-NOT: tl.
// CHECK: %[[UB_A:.*]] = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
// CHECK: %[[UB_C:.*]] = memref.alloc() : memref<64xi32, #hivm.address_space<ub>>
// CHECK: memref.copy %{{.*}}, %[[UB_A]]
// CHECK: %[[RES:.*]] = hivm.hir.bitcast %[[UB_A]] : memref<64xf32, #hivm.address_space<ub>> -> memref<64xi32, #hivm.address_space<ub>>
// CHECK: memref.copy %[[RES]], %{{.*}}

func.func @elemwise_vbitcast(%gm_a: memref<64xf32, #tl.address_space<gm>>,
                             %gm_c: memref<64xi32, #tl.address_space<gm>>)
    attributes {tl.core_type = #tl.core_type<aiv>} {
  %ub_a = tl.alloc : memref<64xf32, #tl.address_space<ub>>
  %ub_c = tl.alloc : memref<64xi32, #tl.address_space<ub>>
  tl.copy %gm_a -> %ub_a : memref<64xf32, #tl.address_space<gm>>, memref<64xf32, #tl.address_space<ub>>
  %res = tl.vbitcast %ub_a : memref<64xf32, #tl.address_space<ub>> to memref<64xi32, #tl.address_space<ub>>
  tl.copy %res -> %gm_c : memref<64xi32, #tl.address_space<ub>>, memref<64xi32, #tl.address_space<gm>>
  return
}
