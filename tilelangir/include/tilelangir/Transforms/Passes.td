// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tilelangir/include/tilelangir/Transforms/Passes.td
 * \brief TileLangIR pass TableGen definitions (CV split, vectorize).
 *
 */

#ifndef TILELANGIR_TRANSFORMS_PASSES_TD
#define TILELANGIR_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def TileLangIRCVSplit : Pass<"tilelangir-cv-split", "::mlir::ModuleOp"> {
  let summary = "TileLangIR CV split pass (custom)";
  let description = [{
    Splits or transforms operations for CV (e.g. coefficient/variable separation).
    Implementation to be added.
  }];
}

def TileLangIRVectorize : Pass<"tilelangir-vectorize", "::mlir::ModuleOp"> {
  let summary = "TileLangIR vectorize pass (custom)";
  let description = [{
    Applies vectorization transformations. Implementation to be added.
  }];
}

def ConvertTLToHIVM : Pass<"tilelangir-convert-tl-to-hivm", "::mlir::ModuleOp"> {
  let summary = "Convert TLIR (tl) to HIVM expert-mode subset";
  let description = [{
    Expert-mode add case study:
      tl.alloc -> memref.alloc
      tl.copy  -> memref.copy
      tl.add   -> hivm.hir.vadd
    Also rewrites #tl.address_space to #hivm.address_space and
    tl.core_type to hivm.func_core_type / hivm.module_core_type.
    Does not inject HACC / FFTS / workspace ABI.
  }];
  let dependentDialects = [
    "mlir::tl::TLDialect",
    "mlir::hivm::HIVMDialect",
    "mlir::memref::MemRefDialect",
    "mlir::func::FuncDialect"
  ];
}

#endif // TILELANGIR_TRANSFORMS_PASSES_TD
