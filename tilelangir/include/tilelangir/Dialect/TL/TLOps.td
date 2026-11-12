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

#endif // TILELANGIR_DIALECT_TL_TLOPS_TD
