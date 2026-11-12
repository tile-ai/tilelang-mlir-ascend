// RUN: tilelangir-opt --split-input-file --verify-diagnostics %s

func.func @print_invalid_elem_type(%buf: memref<64xcomplex<f32>, #tl.address_space<ub>>) {
  // expected-error@+1 {{buffer operand element type must be integer or float}}
  tl.print %buf prefix "bad" hex false : memref<64xcomplex<f32>, #tl.address_space<ub>>
  return
}
