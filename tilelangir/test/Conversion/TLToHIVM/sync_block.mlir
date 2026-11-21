// RUN: tilelangir-opt %s --tilelangir-convert-tl-to-hivm | FileCheck %s
// CHECK: hivm.hir.sync_block_set
// CHECK-SAME: VECTOR
// CHECK-SAME: flag = 0
// CHECK: hivm.hir.sync_block_wait
// CHECK-SAME: VECTOR
// CHECK-SAME: flag = 0
func.func @sync_block_lowers() attributes {tl.core_type = #tl.core_type<aiv>} {
  tl.sync_block_set <inter_block> "Vector" 0
  tl.sync_block_wait "Vector" 0
  return
}
