// RUN: tilelangir-opt %s | FileCheck %s

// Verifies the newly added L0A/L0B/L0C/L1 address spaces for tl.alloc and
// tl.copy. tl.add is NOT exercised across these spaces here because
// TL_AddOp requires lhs/rhs/out to share one identical type
// (AllTypesMatch<["lhs","rhs","out"]>), which includes address space --
// L0A, L0B and L0C are distinct types and cannot be mixed in one tl.add.
// Real GEMM kernels move L1 -> L0A/L0B/L0C via T.gemm (a higher-level op
// outside this minimal TLIR subset), not via tl.add.

// CHECK-LABEL: func.func @alloc_copy_l0abc
// CHECK-SAME: #tl.core_type<aic>
// CHECK: %[[L0A:.*]] = tl.alloc : memref<64x64xf32, #tl.address_space<l0a>>
// CHECK: %[[L0B:.*]] = tl.alloc : memref<64x64xf32, #tl.address_space<l0b>>
// CHECK: %[[L0C:.*]] = tl.alloc : memref<64x64xf32, #tl.address_space<l0c>>
// CHECK: tl.copy %{{.*}} -> %[[L0A]] : memref<64x64xf32, #tl.address_space<gm>>, memref<64x64xf32, #tl.address_space<l0a>>
// CHECK: tl.copy %{{.*}} -> %[[L0B]] : memref<64x64xf32, #tl.address_space<gm>>, memref<64x64xf32, #tl.address_space<l0b>>
// CHECK: tl.copy %{{.*}} -> %[[L0C]] : memref<64x64xf32, #tl.address_space<gm>>, memref<64x64xf32, #tl.address_space<l0c>>

func.func @alloc_copy_l0abc(%gm_a: memref<64x64xf32, #tl.address_space<gm>>,
                            %gm_b: memref<64x64xf32, #tl.address_space<gm>>,
                            %gm_c: memref<64x64xf32, #tl.address_space<gm>>)
    attributes {tl.core_type = #tl.core_type<aic>} {
  %l0a = tl.alloc : memref<64x64xf32, #tl.address_space<l0a>>
  %l0b = tl.alloc : memref<64x64xf32, #tl.address_space<l0b>>
  %l0c = tl.alloc : memref<64x64xf32, #tl.address_space<l0c>>
  tl.copy %gm_a -> %l0a : memref<64x64xf32, #tl.address_space<gm>>, memref<64x64xf32, #tl.address_space<l0a>>
  tl.copy %gm_b -> %l0b : memref<64x64xf32, #tl.address_space<gm>>, memref<64x64xf32, #tl.address_space<l0b>>
  tl.copy %gm_c -> %l0c : memref<64x64xf32, #tl.address_space<gm>>, memref<64x64xf32, #tl.address_space<l0c>>
  return
}

// CHECK-LABEL: func.func @alloc_copy_add_l1
// CHECK: %[[L1_A:.*]] = tl.alloc : memref<64xf32, #tl.address_space<l1>>
// CHECK: %[[L1_B:.*]] = tl.alloc : memref<64xf32, #tl.address_space<l1>>
// CHECK: %[[L1_C:.*]] = tl.alloc : memref<64xf32, #tl.address_space<l1>>
// CHECK: tl.copy %{{.*}} -> %[[L1_A]] : memref<64xf32, #tl.address_space<gm>>, memref<64xf32, #tl.address_space<l1>>
// CHECK: tl.copy %{{.*}} -> %[[L1_B]] : memref<64xf32, #tl.address_space<gm>>, memref<64xf32, #tl.address_space<l1>>
// CHECK: tl.add ins(%[[L1_A]], %[[L1_B]]) outs(%[[L1_C]]) : memref<64xf32, #tl.address_space<l1>>
// CHECK: tl.copy %[[L1_C]] -> %{{.*}} : memref<64xf32, #tl.address_space<l1>>, memref<64xf32, #tl.address_space<gm>>

func.func @alloc_copy_add_l1(%gm_a: memref<64xf32, #tl.address_space<gm>>,
                             %gm_b: memref<64xf32, #tl.address_space<gm>>,
                             %gm_c: memref<64xf32, #tl.address_space<gm>>)
    attributes {tl.core_type = #tl.core_type<aiv>} {
  %l1_a = tl.alloc : memref<64xf32, #tl.address_space<l1>>
  %l1_b = tl.alloc : memref<64xf32, #tl.address_space<l1>>
  %l1_c = tl.alloc : memref<64xf32, #tl.address_space<l1>>
  tl.copy %gm_a -> %l1_a : memref<64xf32, #tl.address_space<gm>>, memref<64xf32, #tl.address_space<l1>>
  tl.copy %gm_b -> %l1_b : memref<64xf32, #tl.address_space<gm>>, memref<64xf32, #tl.address_space<l1>>
  tl.add ins(%l1_a, %l1_b) outs(%l1_c) : memref<64xf32, #tl.address_space<l1>>
  tl.copy %l1_c -> %gm_c : memref<64xf32, #tl.address_space<l1>>, memref<64xf32, #tl.address_space<gm>>
  return
}
