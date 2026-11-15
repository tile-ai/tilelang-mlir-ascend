// RUN: tilelangir-opt %s | FileCheck %s

// CHECK-LABEL: func.func @test_transpose
// CHECK: %[[SRC:.*]] = tl.alloc : memref<32x8xf32, #tl.address_space<ub>>
// CHECK: %[[DST:.*]] = tl.alloc : memref<8x32xf32, #tl.address_space<ub>>
// CHECK: tl.transpose ins(%[[SRC]]) outs(%[[DST]]) permutation = [1, 0] : memref<32x8xf32, #tl.address_space<ub>>, memref<8x32xf32, #tl.address_space<ub>>

func.func @test_transpose() {
  %src = tl.alloc : memref<32x8xf32, #tl.address_space<ub>>
  %dst = tl.alloc : memref<8x32xf32, #tl.address_space<ub>>
  tl.transpose ins(%src) outs(%dst) permutation = [1, 0] : memref<32x8xf32, #tl.address_space<ub>>, memref<8x32xf32, #tl.address_space<ub>>
  return
}
