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

def TL_ShrOp : TL_Op<"shr", [
    MemoryEffects<[MemRead, MemWrite]>,
    AllTypesMatch<["lhs", "rhs", "out"]>
]> {
  let summary = "Elementwise right-shift of a tile buffer, with optional rounding";
  let description = [{
    Destination-style vector shift-right: out = lhs >> rhs (arithmetic,
    with optional rounding). lhs, rhs and out share shape and element type
    (integer only: i16/i32/i64, per HIVM vshr constraints).

    KNOWN LIMITATION: rhs is modeled here as a tensor operand (not scalar),
    matching what the existing NpuirShr/npuir_shr TIR call layer actually
    supports today. This does NOT match HIVM vshr's true ScalarOnlyHWTrait
    on its second operand; conversion to HIVM is only correct when rhs is,
    in practice, a single-element buffer. Full scalar-rhs support requires
    fixing NpuirShr's C++ constructor and the npuir_shr Python wrapper,
    which is out of scope here.
    Example:
      tl.shr ins(%a, %shift) outs(%c) round(true) : memref<64xi32>
  }];
  let arguments = (ins AnyMemRef:$lhs, AnyMemRef:$rhs, AnyMemRef:$out,
      DefaultValuedAttr<BoolAttr, "true">:$round);
  let assemblyFormat =
      "`ins` `(` $lhs `,` $rhs `)` `outs` `(` $out `)` `round` `(` $round `)` attr-dict `:` type($lhs)";
  let hasVerifier = 1;
}

#endif // TILELANGIR_DIALECT_TL_TLOPS_TD
