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

def TL_SubOp : TL_Op<"sub", [
  MemoryEffects<[MemRead, MemWrite]>,
  AllTypesMatch<["lhs", "rhs", "out"]>
]> {
  let summary = "Elementwise subtract of two tile buffers";
  let description = [{
    Destination-style vector sub: out = lhs - rhs. All operands share shape
    and element type.

    Example:
      tl.sub ins(%a, %b) outs(%c) : memref<64xf32>
  }];
  let arguments = (ins AnyMemRef:$lhs, AnyMemRef:$rhs, AnyMemRef:$out);
  let assemblyFormat =
    "`ins` `(` $lhs `,` $rhs `)` `outs` `(` $out `)` attr-dict `:` type($lhs)";
  let hasVerifier = 1;
}

class TL_BinaryOp<string mnemonic> : TL_Op<mnemonic, [
  MemoryEffects<[MemRead, MemWrite]>,
  AllTypesMatch<["lhs", "rhs", "out"]>
]> {
  let arguments = (ins AnyMemRef:$lhs, AnyMemRef:$rhs, AnyMemRef:$out);
  let assemblyFormat =
    "`ins` `(` $lhs `,` $rhs `)` `outs` `(` $out `)` attr-dict `:` type($lhs)";
  let hasVerifier = 1;
}

def TL_MulOp : TL_BinaryOp<"mul"> {
  let summary = "Elementwise multiply of two tile buffers";
  let description = [{ Destination-style vector mul: out = lhs * rhs. }];
}

def TL_DivOp : TL_BinaryOp<"div"> {
  let summary = "Elementwise divide of two tile buffers";
  let description = [{ Destination-style vector div: out = lhs / rhs. }];
}

def TL_MaxOp : TL_BinaryOp<"max"> {
  let summary = "Elementwise maximum of two tile buffers";
  let description = [{ Destination-style vector max: out = max(lhs, rhs). }];
}

def TL_MinOp : TL_BinaryOp<"min"> {
  let summary = "Elementwise minimum of two tile buffers";
  let description = [{ Destination-style vector min: out = min(lhs, rhs). }];
}

def TL_AtomicAddOp : TL_Op<"atomic_add", [
    MemoryEffects<[MemRead, MemWrite]>
]> {
  let summary = "Atomic elementwise add into a tile buffer (dst += src)";
  let description = [{
    Atomic destination-style vector add: dst += src, lowered to
    `hivm.hir.store ... atomic = <add>`. Operands must share shape and
    element type, but (unlike `tl.add`) their address spaces may differ:
    `src` is typically UB and `dst` GM, so no round-trip copy back out of
    UB is needed.

    Example:
      tl.atomic_add %src -> %dst : memref<64xf32, #tl.address_space<ub>>, memref<64xf32, #tl.address_space<gm>>
  }];
  let arguments = (ins AnyMemRef:$src, AnyMemRef:$dst);
  let assemblyFormat =
      "$src `->` $dst attr-dict `:` type($src) `,` type($dst)";
  let hasVerifier = 1;
}

#endif // TILELANGIR_DIALECT_TL_TLOPS_TD
