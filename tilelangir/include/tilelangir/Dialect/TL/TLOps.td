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

def TL_CastOp : TL_Op<"cast", [
    MemoryEffects<[MemRead, MemWrite]>
]> {
  let summary = "Elementwise type conversion of a tile buffer";
  let description = [{
    Destination-style vector cast: dst = cast(src, round_mode). src and dst
    share shape but may differ in element type (that is the point of a
    cast); legal src/dst dtype pairs are constrained by the HIVM vcast
    hardware op, not by this verifier. round_mode is one of:
    round/rint/floor/ceil/trunc/odd, matching the set accepted by
    tilelang.language.customize_npuir.npuir_cast on the Python side (a
    strict subset of HIVM's RoundMode enum, which also has
    truncwithoverflow).
    Example:
      tl.cast ins(%a) outs(%c) round_mode("rint") : memref<64xf32>, memref<64xf16>
  }];
  let arguments = (ins AnyMemRef:$src, AnyMemRef:$dst, StrAttr:$round_mode);
  let assemblyFormat =
      "`ins` `(` $src `)` `outs` `(` $dst `)` `round_mode` `(` $round_mode `)` attr-dict `:` type($src) `,` type($dst)";
  let hasVerifier = 1;
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

def TL_PrintOp : TL_Op<"print", [MemoryEffects<[MemRead]>]> {
  let summary = "Debug-print a scalar or tile buffer (device-side)";
  let description = [{
    ...
  }];
  let arguments = (ins
    AnyType:$src,
    StrAttr:$prefix,
    BoolAttr:$hex
  );
  let results = (outs);
  let assemblyFormat =
      "$src `prefix` $prefix `hex` $hex attr-dict `:` type($src)";
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

class TL_UnaryOp<string mnemonic> : TL_Op<mnemonic, [
  MemoryEffects<[MemRead, MemWrite]>,
  AllTypesMatch<["src", "dst"]>
]> {
  let arguments = (ins AnyMemRef:$src, AnyMemRef:$dst);
  let assemblyFormat =
    "$src `,` $dst attr-dict `:` type($src)";
  let hasVerifier = 1;
}

def TL_VReluOp : TL_UnaryOp<"vrelu"> {
  let summary = "Elementwise ReLU activation";
  let description = [{
    Destination-style elementwise ReLU: dst = max(src, 0).

    Example:
      tl.vrelu %src, %dst : memref<64xf32>
  }];
 }
 
def TL_AndOp : TL_BinaryOp<"and"> {
  let summary = "Elementwise bitwise AND of two tile buffers";
  let description = [{ Destination-style vector and: out = lhs & rhs. }];
}

#endif // TILELANGIR_DIALECT_TL_TLOPS_TD
