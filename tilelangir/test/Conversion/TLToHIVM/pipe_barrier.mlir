// RUN: tilelangir-opt %s --tilelangir-convert-tl-to-hivm | FileCheck %s
// CHECK: hivm.hir.pipe_barrier[<PIPE_V>]
func.func @pipe_barrier_lowers() attributes {tl.core_type = #tl.core_type<aiv>} {
  tl.pipe_barrier "V"
  return
}
