// RUN: tilelangir-opt %s | FileCheck %s

// Pad case study: pads a 1-D buffer from 60 elements to 64 elements along
// its only dimension (low=0, high=4), mirroring the add/sub tests'
// alloc -> copy -> op -> copy structure.

// CHECK-LABEL: func.func @elemwise_pad
// CHECK-SAME: #tl.core_type<aiv>
// CHECK: %[[UB_SRC:.*]] = tl.alloc : memref<60xf32, #tl.address_space<ub>>
// CHECK: %[[UB_DST:.*]] = tl.alloc : memref<64xf32, #tl.address_space<ub>>
// CHECK: %[[PADVAL:.*]] = arith.constant
// CHECK: tl.copy %{{.*}} -> %[[UB_SRC]] : memref<60xf32, #tl.address_space<gm>>, memref<60xf32, #tl.address_space<ub>>
// CHECK: tl.pad ins(%[[UB_SRC]]) outs(%[[UB_DST]]) pad_value(%[[PADVAL]] : f32) low() high() {{.*}} : memref<60xf32, #tl.address_space<ub>>, memref<64xf32, #tl.address_space<ub>>
// CHECK: tl.copy %[[UB_DST]] -> %{{.*}} : memref<64xf32, #tl.address_space<ub>>, memref<64xf32, #tl.address_space<gm>>

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
