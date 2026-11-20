// RUN: tilelangir-opt %s --tilelangir-convert-tl-to-hivm | FileCheck %s

// CHECK: hivm.module_core_type = #hivm.module_core_type<AIV>
// CHECK-LABEL: func.func @test_transpose
// CHECK-NOT: tl.transpose
// CHECK: hivm.hir.vtranspose ins(%{{.*}} : memref<32x8xf32, #hivm.address_space<ub>>) outs(%{{.*}} : memref<8x32xf32, #hivm.address_space<ub>>) permutation = [1, 0]

func.func @test_transpose() {
  %src = tl.alloc : memref<32x8xf32, #tl.address_space<ub>>
  %dst = tl.alloc : memref<8x32xf32, #tl.address_space<ub>>
  tl.transpose ins(%src) outs(%dst) permutation = [1, 0] : memref<32x8xf32, #tl.address_space<ub>>, memref<8x32xf32, #tl.address_space<ub>>
  return
}
