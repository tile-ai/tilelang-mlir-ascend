// RUN: tilelangir-opt %s --tilelangir-convert-tl-to-hivm | FileCheck %s

// NOTE: hivm concat op's real name/syntax must be confirmed upstream;
// tighten these CHECK lines after your first real run.

// CHECK-LABEL: func.func @elemwise_concat
// CHECK-NOT: tl.
// CHECK: hivm.hir.vconcat
// CHECK: memref.copy

func.func @elemwise_concat(%a: memref<8x4xf32, #tl.address_space<ub>>,
                           %b: memref<8x4xf32, #tl.address_space<ub>>,
                           %dst: memref<8x8xf32, #tl.address_space<ub>>,
                           %gm: memref<8x8xf32, #tl.address_space<gm>>)
    attributes {tl.core_type = #tl.core_type<aiv>} {
  tl.concat ins(%a, %b) outs(%dst) axis(1) : memref<8x4xf32, #tl.address_space<ub>>, memref<8x4xf32, #tl.address_space<ub>> -> memref<8x8xf32, #tl.address_space<ub>>
  tl.copy %dst -> %gm : memref<8x8xf32, #tl.address_space<ub>>, memref<8x8xf32, #tl.address_space<gm>>
  return
}
