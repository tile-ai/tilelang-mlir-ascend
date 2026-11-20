// RUN: tilelangir-opt %s --tilelangir-convert-tl-to-hivm | FileCheck %s

// Expert-mode gather subset: TLIR -> HIVM (no HACC / FFTS / workspace).

// CHECK: module attributes {{{.*}}hivm.module_core_type = #hivm.module_core_type<AIV>
// CHECK-LABEL: func.func @gather_example
// CHECK-SAME: memref<64xf32, #hivm.address_space<gm>>
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-NOT: tl.
// CHECK: %[[UB_SRC:.*]] = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
// CHECK: %[[UB_IDX:.*]] = memref.alloc() : memref<64xi32, #hivm.address_space<ub>>
// CHECK: %[[UB_DST:.*]] = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
// CHECK: memref.copy %{{.*}}, %[[UB_SRC]]
// CHECK: memref.copy %{{.*}}, %[[UB_IDX]]
// CHECK: hivm.hir.vgather ins(%[[UB_SRC]] :{{.*}}) indices(%[[UB_IDX]] :{{.*}}) outs(%[[UB_DST]] :
// CHECK: memref.copy %[[UB_DST]], %{{.*}}

func.func @gather_example(%gm_src: memref<64xf32, #tl.address_space<gm>>,
                           %gm_idx: memref<64xi32, #tl.address_space<gm>>,
                           %gm_dst: memref<64xf32, #tl.address_space<gm>>)
    attributes {tl.core_type = #tl.core_type<aiv>} {
  %ub_src = tl.alloc : memref<64xf32, #tl.address_space<ub>>
  %ub_idx = tl.alloc : memref<64xi32, #tl.address_space<ub>>
  %ub_dst = tl.alloc : memref<64xf32, #tl.address_space<ub>>
  tl.copy %gm_src -> %ub_src : memref<64xf32, #tl.address_space<gm>>, memref<64xf32, #tl.address_space<ub>>
  tl.copy %gm_idx -> %ub_idx : memref<64xi32, #tl.address_space<gm>>, memref<64xi32, #tl.address_space<ub>>
  tl.gather ins(%ub_src, %ub_idx) outs(%ub_dst) : memref<64xf32, #tl.address_space<ub>>, memref<64xi32, #tl.address_space<ub>>, memref<64xf32, #tl.address_space<ub>>
  tl.copy %ub_dst -> %gm_dst : memref<64xf32, #tl.address_space<ub>>, memref<64xf32, #tl.address_space<gm>>
  return
}
