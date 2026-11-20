// RUN: tilelangir-opt %s | FileCheck %s

// CHECK-LABEL: func.func @elemwise_concat
// CHECK-SAME: #tl.core_type<aiv>
// CHECK: tl.concat ins(%{{.*}}, %{{.*}}) outs(%{{.*}}) axis(1)

func.func @elemwise_concat(%a: memref<8x4xf32, #tl.address_space<ub>>,
                           %b: memref<8x4xf32, #tl.address_space<ub>>,
                           %dst: memref<8x8xf32, #tl.address_space<ub>>,
                           %gm: memref<8x8xf32, #tl.address_space<gm>>)
    attributes {tl.core_type = #tl.core_type<aiv>} {
  tl.concat ins(%a, %b) outs(%dst) axis(1) : memref<8x4xf32, #tl.address_space<ub>>, memref<8x4xf32, #tl.address_space<ub>> -> memref<8x8xf32, #tl.address_space<ub>>
  tl.copy %dst -> %gm : memref<8x8xf32, #tl.address_space<ub>>, memref<8x8xf32, #tl.address_space<gm>>
  return
}
