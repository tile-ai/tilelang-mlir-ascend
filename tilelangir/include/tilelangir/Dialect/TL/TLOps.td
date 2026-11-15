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

def TL_TransposeOp : TL_Op<"transpose", [
    MemoryEffects<[MemRead, MemWrite]>
]> {
  let summary = "Permute the dimensions of a tile buffer";
  let description = [{
    Destination-style transpose. src and dst must have the same rank and
    element type; shapes need not match since transpose permutes dimensions.
    Example:
      tl.transpose ins(%src) outs(%dst) permutation = [1, 0] : memref<32x8xf32>, memref<8x32xf32>
  }];
  let arguments = (ins
    AnyMemRef:$src,
    AnyMemRef:$dst,
    DenseI64ArrayAttr:$permutation
  );
  let assemblyFormat =
      "`ins` `(` $src `)` `outs` `(` $dst `)` `permutation` `=` $permutation attr-dict `:` type($src) `,` type($dst)";
  let hasVerifier = 1;
}

#endif // TILELANGIR_DIALECT_TL_TLOPS_TD
