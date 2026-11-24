// RUN: tilelangir-opt %s --tilelangir-convert-tl-to-hivm | FileCheck %s

// Expert-mode reshape subset: TLIR -> HIVM (no HACC / FFTS / workspace).
// tl.reshape lowers directly to memref.reinterpret_cast; no HIVM op needed.

// CHECK: module attributes {{{.*}}hivm.module_core_type = #hivm.module_core_type<AIV>
// CHECK-LABEL: func.func @reshape_view
// CHECK-SAME: memref<8x16xf16, #hivm.address_space<gm>>
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-NOT: tl.
// CHECK: %[[UB_A:.*]] = memref.alloc() : memref<8x16xf16, #hivm.address_space<ub>>
// CHECK: memref.copy %{{.*}}, %[[UB_A]]
// CHECK: %[[UB_B:.*]] = memref.reinterpret_cast %[[UB_A]]
// CHECK: memref.copy %[[UB_B]], %{{.*}}

func.func @reshape_view(%gm_a: memref<8x16xf16, #tl.address_space<gm>>,
                        %gm_b: memref<16x8xf16, #tl.address_space<gm>>)
    attributes {tl.core_type = #tl.core_type<aiv>} {
  %ub_a = tl.alloc : memref<8x16xf16, #tl.address_space<ub>>
  tl.copy %gm_a -> %ub_a : memref<8x16xf16, #tl.address_space<gm>>, memref<8x16xf16, #tl.address_space<ub>>
  %ub_b = tl.reshape %ub_a : memref<8x16xf16, #tl.address_space<ub>> to memref<16x8xf16, #tl.address_space<ub>>
  tl.copy %ub_b -> %gm_b : memref<16x8xf16, #tl.address_space<ub>>, memref<16x8xf16, #tl.address_space<gm>>
  return
}
