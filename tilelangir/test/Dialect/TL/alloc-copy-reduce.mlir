// RUN: tilelangir-opt %s | FileCheck %s

// CHECK-LABEL: func.func @reduce_sum
// CHECK-SAME: #tl.core_type<aiv>
// CHECK: %[[UB_SRC:.*]] = tl.alloc : memref<4x8xf32, #tl.address_space<ub>>
// CHECK: %[[UB_DST:.*]] = tl.alloc : memref<4x1xf32, #tl.address_space<ub>>
// CHECK: tl.copy %{{.*}} -> %[[UB_SRC]] : memref<4x8xf32, #tl.address_space<gm>>, memref<4x8xf32, #tl.address_space<ub>>
// CHECK: tl.reduce ins(%[[UB_SRC]]) outs(%[[UB_DST]]) <sum> dims = [1] : memref<4x8xf32, #tl.address_space<ub>>, memref<4x1xf32, #tl.address_space<ub>>
// CHECK: tl.copy %[[UB_DST]] -> %{{.*}} : memref<4x1xf32, #tl.address_space<ub>>, memref<4x1xf32, #tl.address_space<gm>>

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
