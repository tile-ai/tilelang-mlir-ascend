//===- TLIROps.td - TLIR op definitions (alloc/copy/add slice) ----------===//
//
// Minimal op set needed to take the TileLang `vec_add` benchmark kernel
// (examples/elementwise/vec_add_1d.py) from TIR through TLIR down to the
// A5 target dialects.
//
// v1 simplification vs. the real codegen (CreateHIVMBinaryVectorOp in
// codegen_npuir_api_a5.cc): no scalar-operand support, no broadcasting.
// tl.add's verifier enforces identical lhs/rhs/out shapes so a kernel
// needing broadcast fails loudly at TLIR-construction time instead of
// silently mis-lowering.
//
//===----------------------------------------------------------------------===//

#ifndef TILELANGIR_DIALECT_TL_TLIROPS_TD
#define TILELANGIR_DIALECT_TL_TLIROPS_TD

include "TLIRDialect.td"
include "TLIRAttrs.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinTypes.td"

def TLIR_AllocOp : TLIR_Op<"alloc", []> {
  let summary = "Allocate a tile buffer in a given address space";
  let description = [{
    Corresponds to `T.alloc_ub` / `T.alloc_shared` in TileLang. Lowers to
    `memref.alloc` in tlir-to-npuir.

    Example:
      %ub = tl.alloc : memref<4096xf32, #tl.address_space<ub>>
  }];
  let results = (outs AnyMemRef:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def TLIR_CopyOp : TLIR_Op<"copy", []> {
  let summary = "Copy between two tile buffers";
  let description = [{
    Corresponds to `T.copy(src, dst)`. The real codegen's
    AscendCopyCodegen builds this from a `T.region(...)` call and does
    considerably more (subview construction, ND2NZ layout handling for
    kernel #5 in the benchmark set); v1 covers the structurally
    equivalent same-shape case only.

    Example:
      tl.copy %gm_a -> %ub_a : memref<4096xf32>
  }];
  let arguments = (ins AnyMemRef:$source, AnyMemRef:$target);
  let assemblyFormat = "$source `->` $target attr-dict `:` type($source) `,` type($target)";
  let hasVerifier = 1;
}

def TLIR_AddOp : TLIR_Op<"add", []> {
  let summary = "Elementwise tile addition (vector core)";
  let description = [{
    Corresponds to the `tl.npuir_add` TVM Op / TileLang's
    `T.npuir_add(A_VEC, B_VEC, C_VEC)`. On A5, lowers to
    `linalg.elemwise_binary { fun = #linalg.binary_fn<add> }`
    (see TLIRToNPUIR.cpp for why this differs from the A2/A3
    `hivm.VAddOp` path).

    Example:
      tl.add ins(%ub_a, %ub_b) outs(%ub_c) : memref<4096xf32>
  }];
  let arguments = (ins AnyMemRef:$lhs, AnyMemRef:$rhs, AnyMemRef:$out);
  let assemblyFormat =
      "`ins` `(` $lhs `,` $rhs `)` `outs` `(` $out `)` attr-dict `:` "
      "type($lhs) `,` type($rhs) `,` type($out)";
  let hasVerifier = 1;
}

#endif // TILELANGIR_DIALECT_TL_TLIROPS_TD
