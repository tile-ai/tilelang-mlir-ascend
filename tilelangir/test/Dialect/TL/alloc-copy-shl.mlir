// RUN: tilelangir-opt %s | FileCheck %s

// Integer elementwise shift-left: scalar rhs, matches HIVM vshl constraints.

// CHECK-LABEL: func.func @elemwise_shl
// CHECK-SAME: #tl.core_type<aiv>
// CHECK: %[[UB_A:.*]] = tl.alloc : memref<64xi32, #tl.address_space<ub>>
// CHECK: %[[UB_C:.*]] = tl.alloc : memref<64xi32, #tl.address_space<ub>>
// CHECK: tl.copy %{{.*}} -> %[[UB_A]] : memref<64xi32, #tl.address_space<gm>>, memref<64xi32, #tl.address_space<ub>>
// CHECK: tl.shl ins(%[[UB_A]], %{{.*}}) outs(%[[UB_C]]) : memref<64xi32, #tl.address_space<ub>>, i32
// CHECK: tl.copy %[[UB_C]] -> %{{.*}} : memref<64xi32, #tl.address_space<ub>>, memref<64xi32, #tl.address_space<gm>>

func.func @elemwise_shl(%gm_a: memref<64xi32, #tl.address_space<gm>>,
                        %shift: i32,
                        %gm_c: memref<64xi32, #tl.address_space<gm>>)
    attributes {tl.core_type = #tl.core_type<aiv>} {
  %ub_a = tl.alloc : memref<64xi32, #tl.address_space<ub>>
  %ub_c = tl.alloc : memref<64xi32, #tl.address_space<ub>>
  tl.copy %gm_a -> %ub_a : memref<64xi32, #tl.address_space<gm>>, memref<64xi32, #tl.address_space<ub>>
  tl.shl ins(%ub_a, %shift) outs(%ub_c) : memref<64xi32, #tl.address_space<ub>>, i32
  tl.copy %ub_c -> %gm_c : memref<64xi32, #tl.address_space<ub>>, memref<64xi32, #tl.address_space<gm>>
  return
}
