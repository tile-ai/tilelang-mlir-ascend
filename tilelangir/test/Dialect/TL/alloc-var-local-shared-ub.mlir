// RUN: tilelangir-opt %s | FileCheck %s

// Scope-to-address-space mapping test for alloc_local, alloc_var,
// alloc_shared and alloc_ub (local/local.var/shared.dyn/shared TIR scopes).

// CHECK-LABEL: func.func @alloc_scopes
// CHECK: %[[LOCAL:.*]] = tl.alloc : memref<64xf32, #tl.address_space<ub>>
// CHECK: %[[LOCAL_VAR:.*]] = tl.alloc : memref<1xf32, #tl.address_space<ub>>
// CHECK: %[[SHARED_DYN:.*]] = tl.alloc : memref<64xf32, #tl.address_space<l1>>
// CHECK: %[[SHARED:.*]] = tl.alloc : memref<64xf32, #tl.address_space<ub>>

func.func @alloc_scopes()
    attributes {tl.core_type = #tl.core_type<aiv>} {
  %local = tl.alloc : memref<64xf32, #tl.address_space<ub>>
  %local_var = tl.alloc : memref<1xf32, #tl.address_space<ub>>
  %shared_dyn = tl.alloc : memref<64xf32, #tl.address_space<l1>>
  %shared = tl.alloc : memref<64xf32, #tl.address_space<ub>>
  return
}
