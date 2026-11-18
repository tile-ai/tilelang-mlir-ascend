// RUN: tilelangir-opt %s | FileCheck %s

func.func @print_buf(%buf: memref<64xf32, #tl.address_space<ub>>) {
  // CHECK: tl.print %{{.*}} prefix "C_VEC" hex false : memref<64xf32, #tl.address_space<ub>>
  tl.print %buf prefix "C_VEC" hex false : memref<64xf32, #tl.address_space<ub>>
  return
}

func.func @print_var(%v: i32) {
  // CHECK: tl.print %{{.*}} prefix "idx" hex false : i32
  tl.print %v prefix "idx" hex false : i32
  return
}
