// RUN: tilelangir-opt %s | FileCheck %s
// CHECK-LABEL: func.func @pipe_barrier_roundtrip
func.func @pipe_barrier_roundtrip() attributes {tl.core_type = #tl.core_type<aiv>} {
  // CHECK: tl.pipe_barrier "V"
  tl.pipe_barrier "V"
  return
}
