// RUN: tilelangir-opt %s | FileCheck %s

// Round-trip test for tl.gather: retrieves elements from src using
// indices, stores them in dst. indices has a different element type
// (i32) from src/dst (f32), so tl.gather has no AllTypesMatch constraint
// (unlike tl.add).

// CHECK-LABEL: func.func @gather_example
// CHECK-SAME: #tl.core_type<aiv>
// CHECK: %[[UB_SRC:.*]] = tl.alloc : memref<64xf32, #tl.address_space<ub>>
// CHECK: %[[UB_IDX:.*]] = tl.alloc : memref<64xi32, #tl.address_space<ub>>
// CHECK: %[[UB_DST:.*]] = tl.alloc : memref<64xf32, #tl.address_space<ub>>
// CHECK: tl.copy %{{.*}} -> %[[UB_SRC]] : memref<64xf32, #tl.address_space<gm>>, memref<64xf32, #tl.address_space<ub>>
// CHECK: tl.copy %{{.*}} -> %[[UB_IDX]] : memref<64xi32, #tl.address_space<gm>>, memref<64xi32, #tl.address_space<ub>>
// CHECK: tl.gather ins(%[[UB_SRC]], %[[UB_IDX]]) outs(%[[UB_DST]]) : memref<64xf32, #tl.address_space<ub>>, memref<64xi32, #tl.address_space<ub>>, memref<64xf32, #tl.address_space<ub>>
// CHECK: tl.copy %[[UB_DST]] -> %{{.*}} : memref<64xf32, #tl.address_space<ub>>, memref<64xf32, #tl.address_space<gm>>

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
