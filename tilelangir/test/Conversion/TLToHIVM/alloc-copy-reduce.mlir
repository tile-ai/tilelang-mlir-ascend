// RUN: tilelangir-opt %s --tilelangir-convert-tl-to-hivm | FileCheck %s

// Expert-mode reduce subset: TLIR -> HIVM (no HACC / FFTS / workspace).
// Keepdims semantics: dst has the same rank as src, with reduced dims of size 1.

// CHECK: module attributes {{{.*}}hivm.module_core_type = #hivm.module_core_type<AIV>
// CHECK-LABEL: func.func @reduce_sum
// CHECK-SAME: memref<4x8xf32, #hivm.address_space<gm>>
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-NOT: tl.
// CHECK: %[[UB_SRC:.*]] = memref.alloc() : memref<4x8xf32, #hivm.address_space<ub>>
// CHECK: %[[UB_DST:.*]] = memref.alloc() : memref<4x1xf32, #hivm.address_space<ub>>
// CHECK: memref.copy %{{.*}}, %[[UB_SRC]]
// CHECK: hivm.hir.vreduce <sum> ins(%[[UB_SRC]] :{{.*}}) outs(%[[UB_DST]] :{{.*}}) reduce_dims = [1]
// CHECK: memref.copy %[[UB_DST]], %{{.*}}

func.func @reduce_sum(%gm_src: memref<4x8xf32, #tl.address_space<gm>>,
                      %gm_dst: memref<4x1xf32, #tl.address_space<gm>>)
    attributes {tl.core_type = #tl.core_type<aiv>} {
  %ub_src = tl.alloc : memref<4x8xf32, #tl.address_space<ub>>
  %ub_dst = tl.alloc : memref<4x1xf32, #tl.address_space<ub>>
  tl.copy %gm_src -> %ub_src : memref<4x8xf32, #tl.address_space<gm>>, memref<4x8xf32, #tl.address_space<ub>>
  tl.reduce ins(%ub_src) outs(%ub_dst) <sum> dims = [1] : memref<4x8xf32, #tl.address_space<ub>>, memref<4x1xf32, #tl.address_space<ub>>
  tl.copy %ub_dst -> %gm_dst : memref<4x1xf32, #tl.address_space<ub>>, memref<4x1xf32, #tl.address_space<gm>>
  return
}
