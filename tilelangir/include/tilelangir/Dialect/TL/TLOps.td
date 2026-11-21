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

def TL_PipeBarrierOp : TL_Op<"pipe_barrier"> {
  let summary = "Barrier across a given pipe";
  let description = [{
    Blocks all following instructions on `pipe` until all previous
    instructions on that pipe finish. `pipe` is carried as a string,
    mirroring NpuirPipeBarrier::pipe_type on the TVM side.
    Example:
      tl.pipe_barrier "v"
  }];
  let arguments = (ins StrAttr:$pipe);
  let assemblyFormat = "$pipe attr-dict";
  let hasVerifier = 0;
}

def TL_SyncBlockSetOp : TL_Op<"sync_block_set"> {
  let summary = "Set an inter/intra-block synchronization flag";
  let description = [{
    Lowers to hivm.hir.sync_block_set. The target core type is derived from
    the enclosing function's `tl.core_type` (not stored on this op).
    `flag_id` is restricted to a static (compile-time) value in this subset.
    Example:
      tl.sync_block_set <inter_block> "Vector" 0
  }];
  let arguments = (ins TL_SyncInstrModeAttr:$sync_instr_mode, StrAttr:$pipe,
                       I64Attr:$flag_id);
  let assemblyFormat = "$sync_instr_mode $pipe $flag_id attr-dict";
}

def TL_SyncBlockWaitOp : TL_Op<"sync_block_wait"> {
  let summary = "Wait for an inter/intra-block synchronization flag";
  let description = [{
    Lowers to hivm.hir.sync_block_wait. Target core type is derived the same
    way as tl.sync_block_set.
    Example:
      tl.sync_block_wait "Vector" 0
  }];
  let arguments = (ins StrAttr:$pipe, I64Attr:$flag_id);
  let assemblyFormat = "$pipe $flag_id attr-dict";
}

#endif // TILELANGIR_DIALECT_TL_TLOPS_TD
