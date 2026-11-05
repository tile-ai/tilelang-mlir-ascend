#ifndef TILELANGIR_DIALECT_TL_TLOPS_TD
#define TILELANGIR_DIALECT_TL_TLOPS_TD

include "tilelangir/Dialect/TL/TLAttrs.td"

def TL_AllocOp : TL_Op<"alloc", [MemoryEffects<[MemAlloc]>]> {
  let summary = "Allocate a tile buffer in an address space";
  let description = [{
    Allocates a memref whose memory space is typically `#tl.address_space<...>`.
    Example:
      %ub = tl.alloc : memref<64xf32, #tl.address_space<ub>>
  }];
  let results = (outs AnyMemRef:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def TL_CopyOp : TL_Op<"copy", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "DMA-style copy between tile buffers";
  let description = [{
    Copies between memrefs of the same shape and element type.
    Address spaces may differ (e.g. GM -> UB).
    Example:
      tl.copy %gm -> %ub : memref<64xf32>
  }];
  let arguments = (ins AnyMemRef:$source, AnyMemRef:$destination);
  let assemblyFormat =
      "$source `->` $destination attr-dict `:` type($source) `,` type($destination)";
  let hasVerifier = 1;
}

def TL_AddOp : TL_Op<"add", [
    MemoryEffects<[MemRead, MemWrite]>,
    AllTypesMatch<["lhs", "rhs", "out"]>
]> {
  let summary = "Elementwise add of two tile buffers";
  let description = [{
    Destination-style vector add. All operands share shape and element type.
    Example:
      tl.add ins(%a, %b) outs(%c) : memref<64xf32>
  }];
  let arguments = (ins AnyMemRef:$lhs, AnyMemRef:$rhs, AnyMemRef:$out);
  let assemblyFormat =
      "`ins` `(` $lhs `,` $rhs `)` `outs` `(` $out `)` attr-dict `:` type($lhs)";
  let hasVerifier = 1;
}

def TL_ShlOp : TL_Op<"shl", [
    MemoryEffects<[MemRead, MemWrite]>,
    AllTypesMatch<["lhs", "out"]>
]> {
  let summary = "Elementwise left-shift of a tile buffer by a scalar";
  let description = [{
    Destination-style vector shift-left: out = lhs << rhs. lhs and out share
    shape and element type (integer only: i16/i32/i64, per HIVM vshl
    constraints). rhs is a scalar shift amount, not a tile buffer, matching
    the HIVM ScalarOnlyHWTrait constraint on vshl's second operand.
    Example:
      tl.shl ins(%a, %shift) outs(%c) : memref<64xi32>, i32
  }];
  let arguments = (ins AnyMemRef:$lhs, AnyType:$rhs, AnyMemRef:$out);
  let assemblyFormat =
      "`ins` `(` $lhs `,` $rhs `)` `outs` `(` $out `)` attr-dict `:` type($lhs) `,` type($rhs)";
  let hasVerifier = 1;
}

#endif // TILELANGIR_DIALECT_TL_TLOPS_TD
