// RUN: tilelangir-opt %s --tilelangir-convert-tl-to-hivm | FileCheck %s

// Expert-mode pad subset: TLIR -> HIVM (no HACC / FFTS / workspace).

// CHECK: module attributes {{{.*}}hivm.module_core_type = #hivm.module_core_type<AIV>
// CHECK-LABEL: func.func @elemwise_pad
// CHECK-SAME: memref<60xf32, #hivm.address_space<gm>>
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-NOT: tl.
// CHECK: %[[UB_SRC:.*]] = memref.alloc() : memref<60xf32, #hivm.address_space<ub>>
// CHECK: %[[UB_DST:.*]] = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
// CHECK: memref.copy %{{.*}}, %[[UB_SRC]]
// CHECK: hivm.hir.vpad ins(%[[UB_SRC]]{{.*}}) outs(%[[UB_DST]]{{.*}})
// CHECK: memref.copy %[[UB_DST]], %{{.*}}

func.func @elemwise_pad(%gm_src: memref<60xf32, #tl.address_space<gm>>,
                        %gm_dst: memref<64xf32, #tl.address_space<gm>>)
    attributes {tl.core_type = #tl.core_type<aiv>} {
  %ub_src = tl.alloc : memref<60xf32, #tl.address_space<ub>>
  %ub_dst = tl.alloc : memref<64xf32, #tl.address_space<ub>>
  %pad_val = arith.constant 0.0 : f32
  tl.copy %gm_src -> %ub_src : memref<60xf32, #tl.address_space<gm>>, memref<60xf32, #tl.address_space<ub>>
  tl.pad ins(%ub_src) outs(%ub_dst) pad_value(%pad_val : f32) low() high() {static_low = array<i64: 0>, static_high = array<i64: 4>} : memref<60xf32, #tl.address_space<ub>>, memref<64xf32, #tl.address_space<ub>>
  tl.copy %ub_dst -> %gm_dst : memref<64xf32, #tl.address_space<ub>>, memref<64xf32, #tl.address_space<gm>>
  return
}
