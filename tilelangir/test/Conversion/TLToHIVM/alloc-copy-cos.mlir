// RUN: tilelangir-opt --tilelangir-convert-tl-to-hivm %s | FileCheck %s
// CHECK-LABEL: func.func @elemwise_cos
// CHECK: hivm.hir.vcos ins(%{{.*}}) outs(%{{.*}})
func.func @elemwise_cos(%gm_a: memref<64xf32, #tl.address_space<gm>>, %gm_b: memref<64xf32, #tl.address_space<gm>>)
    attributes {tl.core_type = #tl.core_type<aiv>} {
  %ub_a = tl.alloc : memref<64xf32, #tl.address_space<ub>>
  %ub_b = tl.alloc : memref<64xf32, #tl.address_space<ub>>
  tl.copy %gm_a -> %ub_a : memref<64xf32, #tl.address_space<gm>>, memref<64xf32, #tl.address_space<ub>>
  tl.cos ins(%ub_a) outs(%ub_b) : memref<64xf32, #tl.address_space<ub>>
  tl.copy %ub_b -> %gm_b : memref<64xf32, #tl.address_space<ub>>, memref<64xf32, #tl.address_space<gm>>
  return
}
