#ifndef TILELANGIR_DIALECT_TL_TLATTRS_TD
#define TILELANGIR_DIALECT_TL_TLATTRS_TD

include "tilelangir/Dialect/TL/TLDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

class TL_I32Enum<string name, string description, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, description, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tl";
}

class TL_I32EnumAttr<string mnemonic, TL_I32Enum enumInfo>
    : EnumAttr<TL_Dialect, enumInfo, mnemonic> {
  let assemblyFormat = "`<` $value `>`";
}

def TL_AddressSpaceGM : I32EnumAttrCase<"GM", 0, "gm">;
def TL_AddressSpaceUB : I32EnumAttrCase<"UB", 1, "ub">;
def TL_AddressSpaceL1 : I32EnumAttrCase<"L1", 2, "l1">;
def TL_AddressSpaceL0 : I32EnumAttrCase<"L0", 3, "l0">;

def TL_AddressSpace : TL_I32Enum<"AddressSpace", "TileLang buffer address space", [
  TL_AddressSpaceGM,
  TL_AddressSpaceUB,
  TL_AddressSpaceL1,
  TL_AddressSpaceL0
]>;

def TL_AddressSpaceAttr : TL_I32EnumAttr<"address_space", TL_AddressSpace>;

def TL_LayoutND : I32EnumAttrCase<"ND", 0, "nd">;
def TL_LayoutNZ : I32EnumAttrCase<"NZ", 1, "nz">;

def TL_Layout : TL_I32Enum<"Layout", "TileLang buffer layout", [
  TL_LayoutND,
  TL_LayoutNZ
]>;

def TL_LayoutAttr : TL_I32EnumAttr<"layout", TL_Layout>;

def TL_CoreTypeAIC : I32EnumAttrCase<"AIC", 0, "aic">;
def TL_CoreTypeAIV : I32EnumAttrCase<"AIV", 1, "aiv">;
def TL_CoreTypeMIX : I32EnumAttrCase<"MIX", 2, "mix">;

def TL_CoreType : TL_I32Enum<"CoreType", "TileLang launch core type", [
  TL_CoreTypeAIC,
  TL_CoreTypeAIV,
  TL_CoreTypeMIX
]>;

def TL_CoreTypeAttr : TL_I32EnumAttr<"core_type", TL_CoreType>;

#endif // TILELANGIR_DIALECT_TL_TLATTRS_TD
