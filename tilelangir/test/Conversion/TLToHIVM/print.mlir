// RUN: tilelangir-opt %s --tilelangir-convert-tl-to-hivm | FileCheck %s
// CHECK: hivm.hir.debug {{.*}}debugtype = "print"{{.*}}
func.func @print_buf_lowers(%buf: memref<64xf32, #tl.address_space<ub>>)
    attributes {tl.core_type = #tl.core_type<aiv>} {
  tl.print %buf prefix "C_VEC" hex false : memref<64xf32, #tl.address_space<ub>>
  return
}

// CHECK: hivm.hir.debug {{.*}}debugtype = "print"{{.*}}
func.func @print_var_lowers(%v: i32)
    attributes {tl.core_type = #tl.core_type<aiv>} {
  tl.print %v prefix "idx" hex false : i32
  return
}
