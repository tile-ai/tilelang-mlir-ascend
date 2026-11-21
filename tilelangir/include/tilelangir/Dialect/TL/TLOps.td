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


def TL_SortOp : TL_Op<"sort", [
    MemoryEffects<[MemRead, MemWrite]>
]> {
  let summary = "Sort a tile buffer along an axis, producing sorted values and indices";
  let description = [{
    Destination-style vector sort. Sorts the sort_axis of src in ascending
    or descending order, writing the sorted values into dst_value and the
    original indices of each sorted element into dst_index. Mirrors
    HIVM's hivm.hir.vsort op (and tl.npuir_sort at the TIR level).

    Constraints:
      - src and dst_value must have the same shape and element type.
      - src and dst_index must have the same shape.
      - dst_index must have an integer element type.

    Example:
      tl.sort ins(%src) outs(%dst_value, %dst_index)
          descending = false sort_axis = -1
          : memref<64xf32>, memref<64xf32>, memref<64xi32>
  }];
  let arguments = (ins AnyMemRef:$src,
                       AnyMemRef:$dst_value,
                       AnyMemRef:$dst_index,
                       DefaultValuedAttr<BoolAttr, "false">:$descending,
                       DefaultValuedAttr<I64Attr, "-1">:$sort_axis);
  let assemblyFormat = [{
    `ins` `(` $src `)` `outs` `(` $dst_value `,` $dst_index `)`
    `descending` `=` $descending `sort_axis` `=` $sort_axis
    attr-dict `:` type($src) `,` type($dst_value) `,` type($dst_index)
  }];
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

class TL_UnaryOp<string mnemonic> : TL_Op<mnemonic, [
  MemoryEffects<[MemRead, MemWrite]>,
  AllTypesMatch<["src", "out"]>
]> {
  let arguments = (ins AnyMemRef:$src, AnyMemRef:$out);
  let assemblyFormat =
    "`ins` `(` $src `)` `outs` `(` $out `)` attr-dict `:` type($src)";
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

def TL_RsqrtOp : TL_UnaryOp<"rsqrt"> {
  let summary = "Elementwise reciprocal square root of a tile buffer";
  let description = [{ Destination-style vector rsqrt: out = 1 / sqrt(src). Lowers to hivm.hir.vrsqrt, which requires f16/f32 operands. }];
}

def TL_SqrtOp : TL_UnaryOp<"sqrt"> {
  let summary = "Elementwise square root of a tile buffer";
  let description = [{ Destination-style vector sqrt: out = sqrt(src). Lowers to hivm.hir.vsqrt, which requires f16/f32 operands. }];
}

def TL_AbsOp : TL_UnaryOp<"abs"> {
  let summary = "Elementwise absolute value of a tile buffer";
  let description = [{ Destination-style vector abs: out = abs(src). Lowers to hivm.hir.vabs. }];
}

def TL_NotOp : TL_UnaryOp<"not"> {
  let summary = "Elementwise logical/bitwise not of a tile buffer";
  let description = [{ Destination-style vector not: out = not(src). Lowers to hivm.hir.vnot. }];
}

def TL_FlipOp : TL_Op<"flip", [
    MemoryEffects<[MemRead, MemWrite]>,
    AllTypesMatch<["src", "dst"]>
]> {
  let summary = "Flip a tile buffer along an axis";
  let description = [{
    Destination-style vector flip: reverses `src` along `axis` and writes the
    result to `dst`. `src` and `dst` share shape and element type.

    Example:
      tl.flip ins(%a) outs(%b) {axis = 1 : ui64} : memref<128x256xf16>
  }];
  let arguments = (ins AnyMemRef:$src, AnyMemRef:$dst, UI64Attr:$axis);
  let assemblyFormat =
      "`ins` `(` $src `)` `outs` `(` $dst `)` attr-dict `:` type($src)";
  let hasVerifier = 1;
}
def TL_ConcatOp : TL_Op<"concat", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Concatenate tile buffers along a given axis";
  let description = [{
    Destination-style concat: writes the variadic `srcs` into `dst`,
    laid out contiguously along dimension `axis`. All `srcs` and `dst`
    must share rank and element type; every non-axis dimension must
    match across all operands, and the axis-dimension sizes of `srcs`
    must sum to `dst`'s axis-dimension size.
    Example:
      tl.concat ins(%a, %b) outs(%dst) axis(1) : memref<8x4xf32>, memref<8x4xf32> -> memref<8x8xf32>
  }];
  let arguments = (ins Variadic<AnyMemRef>:$srcs, AnyMemRef:$dst, I64Attr:$axis);
  let assemblyFormat =
      "`ins` `(` $srcs `)` `outs` `(` $dst `)` `axis` `(` $axis `)` attr-dict `:` type($srcs) `->` type($dst)";
  let hasVerifier = 1;
}
def TL_ArangeOp : TL_Op<"arange", [MemoryEffects<[MemWrite]>]> {
  let summary = "Fill a tile buffer with a strided index sequence";
  let description = [{
    Destination-style vector arange: fills `dst` with
    offset, offset + stride0, offset + stride0 + stride1, ... with one
    stride per dimension of `dst`. Mirrors `hivm.hir.varange`.
    Example:
      tl.arange outs(%dst) offset(%off) strides(%s0, %s1) : memref<8x8xf32>
  }];
  let arguments = (ins AnyMemRef:$dst, Index:$offset, Variadic<Index>:$strides);
  let assemblyFormat =
      "`outs` `(` $dst `)` `offset` `(` $offset `)` `strides` `(` $strides `)` attr-dict `:` type($dst)";
  let hasVerifier = 1;
}
def TL_PowOp : TL_BinaryOp<"pow"> {
  let summary = "Elementwise power of two tile buffers";
  let description = [{ Destination-style vector pow: out = pow(lhs, rhs). Lowers to hivm.hir.vpow, which requires i32 operands. }];
}
  
def TL_XorOp : TL_BinaryOp<"xor"> {
  let summary = "Elementwise bitwise XOR of two tile buffers";
  let description = [{ Destination-style vector xor: out = lhs ^ rhs. }];
}

def TL_OrOp : TL_BinaryOp<"or"> {
  let summary = "Elementwise bitwise OR of two tile buffers";
  let description = [{ Destination-style vector or: out = lhs | rhs. }];
 }
  
def TL_AndOp : TL_BinaryOp<"and"> {
  let summary = "Elementwise bitwise AND of two tile buffers";
  let description = [{ Destination-style vector and: out = lhs & rhs. }];
}

def TL_ReshapeOp : TL_Op<"reshape", [Pure]> {
  let summary = "Reinterpret a tile buffer under a new static shape";
  let description = [{
    Pure view op: result = reshape(src). src and result must share the
    same element type and the same total element count, but may differ
    in shape and rank. Unlike tl.add/tl.cast this is not destination-
    passing: there is no pre-allocated dst operand, because reshape never
    moves data -- it is lowered to a contiguous memref.reinterpret_cast
    of src, matching what ReshapeCodegen already does on the NPUIR path.
    Example:
      %view = tl.reshape %a : memref<8x16xf16> to memref<16x8xf16>
  }];
  let arguments = (ins AnyMemRef:$src);
  let results = (outs AnyMemRef:$result);
  let assemblyFormat =
      "$src attr-dict `:` type($src) `to` type($result)";
  let hasVerifier = 1;
}


#endif // TILELANGIR_DIALECT_TL_TLOPS_TD

