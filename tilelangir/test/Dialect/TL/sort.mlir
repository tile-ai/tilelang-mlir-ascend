// RUN: tilelangir-opt %s | FileCheck %s

// TL-dialect round-trip test for tl.sort.

// CHECK-LABEL: func.func @sort_ub
// CHECK-SAME: #tl.core_type<aiv>
// CHECK: tl.sort ins(%{{.*}}) outs(%{{.*}}, %{{.*}}) descending = false sort_axis = -1 : memref<64xf32, #tl.address_space<ub>>, memref<64xf32, #tl.address_space<ub>>, memref<64xi32, #tl.address_space<ub>>

func.func @sort_ub(%src: memref<64xf32, #tl.address_space<ub>>,
                    %val: memref<64xf32, #tl.address_space<ub>>,
                    %idx: memref<64xi32, #tl.address_space<ub>>)
    attributes {tl.core_type = #tl.core_type<aiv>} {
  tl.sort ins(%src) outs(%val, %idx) descending = false sort_axis = -1
      : memref<64xf32, #tl.address_space<ub>>, memref<64xf32, #tl.address_space<ub>>, memref<64xi32, #tl.address_space<ub>>
  return
}

// -----

// CHECK-LABEL: func.func @sort_tail_rank2
// CHECK: tl.sort ins(%{{.*}}) outs(%{{.*}}, %{{.*}}) descending = true sort_axis = 1 : memref<8x8xf32, #tl.address_space<ub>>, memref<8x8xf32, #tl.address_space<ub>>, memref<8x8xi32, #tl.address_space<ub>>

func.func @sort_tail_rank2(%src: memref<8x8xf32, #tl.address_space<ub>>,
                            %val: memref<8x8xf32, #tl.address_space<ub>>,
                            %idx: memref<8x8xi32, #tl.address_space<ub>>)
    attributes {tl.core_type = #tl.core_type<aiv>} {
  tl.sort ins(%src) outs(%val, %idx) descending = true sort_axis = 1
      : memref<8x8xf32, #tl.address_space<ub>>, memref<8x8xf32, #tl.address_space<ub>>, memref<8x8xi32, #tl.address_space<ub>>
  return
}
