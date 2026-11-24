// RUN: tilelangir-opt %s --tilelangir-convert-tl-to-hivm | FileCheck %s

// TLIR -> HIVM lowering for tl.sort -> hivm.hir.vsort.

// Ascending sort checks.
// CHECK: module attributes {{{.*}}hivm.module_core_type = #hivm.module_core_type<AIV>
// CHECK-LABEL: func.func @sort_ub
// CHECK-SAME: memref<64xf32, #hivm.address_space<ub>>
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK: hivm.hir.vsort ins(%{{.*}} : memref<64xf32, #hivm.address_space<ub>>) outs(%{{.*}}, %{{.*}} : memref<64xf32, #hivm.address_space<ub>>, memref<64xi32, #hivm.address_space<ub>>) descending = false sort_axis = -1

// Descending sort checks.
// CHECK-LABEL: func.func @sort_descending
// CHECK-SAME: memref<64xf32, #hivm.address_space<ub>>
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK: hivm.hir.vsort ins(%{{.*}} : memref<64xf32, #hivm.address_space<ub>>) outs(%{{.*}}, %{{.*}} : memref<64xf32, #hivm.address_space<ub>>, memref<64xi32, #hivm.address_space<ub>>) descending = true sort_axis = -1

func.func @sort_ub(%src: memref<64xf32, #tl.address_space<ub>>,
                    %val: memref<64xf32, #tl.address_space<ub>>,
                    %idx: memref<64xi32, #tl.address_space<ub>>)
    attributes {tl.core_type = #tl.core_type<aiv>} {
  tl.sort ins(%src) outs(%val, %idx) descending = false sort_axis = -1
      : memref<64xf32, #tl.address_space<ub>>, memref<64xf32, #tl.address_space<ub>>, memref<64xi32, #tl.address_space<ub>>
  return
}

func.func @sort_descending(%src: memref<64xf32, #tl.address_space<ub>>,
                           %val: memref<64xf32, #tl.address_space<ub>>,
                           %idx: memref<64xi32, #tl.address_space<ub>>)
    attributes {tl.core_type = #tl.core_type<aiv>} {
  tl.sort ins(%src) outs(%val, %idx) descending = true sort_axis = -1
      : memref<64xf32, #tl.address_space<ub>>, memref<64xf32, #tl.address_space<ub>>, memref<64xi32, #tl.address_space<ub>>
  return
}
