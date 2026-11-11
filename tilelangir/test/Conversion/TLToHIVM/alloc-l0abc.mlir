// RUN: tilelangir-opt %s --tilelangir-convert-tl-to-hivm | FileCheck %s

// Lowered L0A/L0B/L0C/L1 address spaces: TLIR -> HIVM. Mirrors
// alloc-l0abc.mlir in Dialect/TL; see that file for why tl.add is not
// exercised across L0A/L0B/L0C.

// CHECK: module attributes {{{.*}}hivm.module_core_type = #hivm.module_core_type<AIC>
// CHECK-LABEL: func.func @alloc_copy_l0abc
// CHECK-SAME: memref<64x64xf32, #hivm.address_space<gm>>
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIC>
// CHECK-NOT: tl.
// CHECK: %[[L0A:.*]] = memref.alloc() : memref<64x64xf32, #hivm.address_space<ca>>
// CHECK: %[[L0B:.*]] = memref.alloc() : memref<64x64xf32, #hivm.address_space<cb>>
// CHECK: %[[L0C:.*]] = memref.alloc() : memref<64x64xf32, #hivm.address_space<cc>>
// CHECK: memref.copy %{{.*}}, %[[L0A]]
// CHECK: memref.copy %{{.*}}, %[[L0B]]
// CHECK: memref.copy %{{.*}}, %[[L0C]]

// CHECK-LABEL: func.func @alloc_copy_add_l1
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-NOT: tl.
// CHECK: %[[L1_A:.*]] = memref.alloc() : memref<64xf32, #hivm.address_space<cbuf>>
// CHECK: %[[L1_B:.*]] = memref.alloc() : memref<64xf32, #hivm.address_space<cbuf>>
// CHECK: %[[L1_C:.*]] = memref.alloc() : memref<64xf32, #hivm.address_space<cbuf>>
// CHECK: memref.copy %{{.*}}, %[[L1_A]]
// CHECK: memref.copy %{{.*}}, %[[L1_B]]
// CHECK: hivm.hir.vadd ins(%[[L1_A]], %[[L1_B]] :{{.*}}) outs(%[[L1_C]] :
// CHECK: memref.copy %[[L1_C]], %{{.*}}

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
