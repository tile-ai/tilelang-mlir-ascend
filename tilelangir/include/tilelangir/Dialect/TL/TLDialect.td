#ifndef TILELANGIR_DIALECT_TL_TLDIALECT_TD
#define TILELANGIR_DIALECT_TL_TLDIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Restricted TileLang dialect (TLIR) for Ascend A5 lowering.
// See docs/fyp-tilelang-mlir-dialect-proposal.md and tilelangir/docs/tl-dialect.md.

def TL_Dialect : Dialect {
  let name = "tl";
  let summary = "Restricted TileLang tile-level dialect (TLIR)";
  let description = [{
    Captures tile-level compute, memory movement, and launch metadata
    between TileLang TIR and HIVM HIR. v1 is a subset: alloc, copy, add.
  }];
  let cppNamespace = "::mlir::tl";
  let useDefaultAttributePrinterParser = 1;
  let dependentDialects = ["memref::MemRefDialect"];
}

class TL_Op<string mnemonic, list<Trait> traits = []>
    : Op<TL_Dialect, mnemonic, traits>;

#endif // TILELANGIR_DIALECT_TL_TLDIALECT_TD
