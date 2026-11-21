// RUN: tilelangir-opt %s | FileCheck %s
// CHECK-LABEL: func.func @sync_block_roundtrip
func.func @sync_block_roundtrip() attributes {tl.core_type = #tl.core_type<aiv>} {
  // CHECK: tl.sync_block_set <inter_block> "Vector" 0
  tl.sync_block_set <inter_block> "Vector" 0
  // CHECK: tl.sync_block_wait "Vector" 0
  tl.sync_block_wait "Vector" 0
  return
}
