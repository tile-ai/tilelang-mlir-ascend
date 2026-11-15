// RUN: tilelangir-opt %s --tilelangir-convert-tl-to-hivm | FileCheck %s

// Lowering check: local/local.var/shared.dyn/shared TL address spaces
// lower to the correct HIVM address spaces (ub/ub/l1/ub).

// CHECK-LABEL: func.func @alloc_scopes
// CHECK-NOT: tl.
// CHECK: %[[LOCAL:.*]] = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
// CHECK: %[[LOCAL_VAR:.*]] = memref.alloc() : memref<1xf32, #hivm.address_space<ub>>
// CHECK: %[[SHARED_DYN:.*]] = memref.alloc() : memref<64xf32, #hivm.address_space<cbuf>>
// CHECK: %[[SHARED:.*]] = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>

func.func @alloc_scopes()
    attributes {tl.core_type = #tl.core_type<aiv>} {
  %local = tl.alloc : memref<64xf32, #tl.address_space<ub>>
  %local_var = tl.alloc : memref<1xf32, #tl.address_space<ub>>
  %shared_dyn = tl.alloc : memref<64xf32, #tl.address_space<l1>>
  %shared = tl.alloc : memref<64xf32, #tl.address_space<ub>>
  return
}
