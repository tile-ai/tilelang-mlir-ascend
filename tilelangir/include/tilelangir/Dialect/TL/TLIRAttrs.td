//===- TLIRAttrs.td - TLIR attribute definitions ----------*- tablegen -*-===//

#ifndef TILELANGIR_DIALECT_TL_TLIRATTRS_TD
#define TILELANGIR_DIALECT_TL_TLIRATTRS_TD

include "TLIRDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

class TLIR_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<TLIR_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

// TODO(fyp): address-space plumbing is target-specific and differs
// between the AscendNPU-IR and AscendNPU-IR-Dev submodules; this is a
// placeholder pending a read of how GenSubviewFromRegion / buffer scope
// ("local.UB" etc.) actually maps to memref memory-space attrs on A5.
def TLIR_GM : I32EnumAttrCase<"gm", 0>;
def TLIR_UB : I32EnumAttrCase<"ub", 1>;
def TLIR_L1 : I32EnumAttrCase<"l1", 2>;
def TLIR_L0 : I32EnumAttrCase<"l0", 3>;

def AddressSpaceKind : I32EnumAttr<"AddressSpaceKind", "TLIR address space",
    [TLIR_GM, TLIR_UB, TLIR_L1, TLIR_L0]> {
  let cppNamespace = "::mlir::tlir";
  let genSpecializedAttr = 0;
}

def TLIR_AddressSpaceAttr : TLIR_Attr<"AddressSpace", "address_space"> {
  let summary = "Ascend on-chip / off-chip memory space";
  let parameters = (ins "AddressSpaceKind":$value);
  let assemblyFormat = "`<` $value `>`";
}

#endif // TILELANGIR_DIALECT_TL_TLIRATTRS_TD
