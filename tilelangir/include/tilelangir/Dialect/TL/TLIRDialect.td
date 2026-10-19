//===- TLIRDialect.td - TLIR dialect definition -----------*- tablegen -*-===//
//
// Restricted TileLang MLIR dialect ("tl"), scoped for the `add` op slice
// of the FYP: dialect design -> import from TIR -> lowering to A5 NPUIR
// target dialects.
//
//===----------------------------------------------------------------------===//

#ifndef TILELANGIR_DIALECT_TL_TLIRDIALECT_TD
#define TILELANGIR_DIALECT_TL_TLIRDIALECT_TD

include "mlir/IR/OpBase.td"

def TLIR_Dialect : Dialect {
  let name = "tl";
  let summary = "Restricted TileLang IR dialect (TLIR)";
  let description = [{
    TLIR is an intermediate representation layer between TVM TIR (as
    emitted by TileLang for Ascend) and the A5 NPUIR target dialects
    (linalg for elementwise vector ops, memref for buffers/DMA). It makes
    tile-level compute and memory-movement intent explicit before
    hardware-specific lowering.

    This restricted slice covers only the ops needed to represent and
    lower the elementwise `add` benchmark kernel:
      tl.alloc, tl.copy, tl.add

    Note: unlike the A2/A3 codegen path (codegen_npuir_api.cc,
    codegen_npuir_dev.cc), which emits `mlir::hivm::VAddOp` directly for
    `tl.npuir_add`, the A5 path (codegen_npuir_api_a5.cc,
    codegen_npuir_dev_a5.cc) lowers the same TileLang op via
    `ElemwiseOp<linalg::BinaryFn::add>`, i.e. upstream MLIR's `linalg`
    dialect. `tl.add`'s lowering pattern (see TLIRToNPUIR.cpp) targets
    the A5 behaviour.
  }];
  let cppNamespace = "::mlir::tlir";
  let useDefaultAttributePrinterParser = 1;
}

class TLIR_Op<string mnemonic, list<Trait> traits = []>
    : Op<TLIR_Dialect, mnemonic, traits>;

#endif // TILELANGIR_DIALECT_TL_TLIRDIALECT_TD
