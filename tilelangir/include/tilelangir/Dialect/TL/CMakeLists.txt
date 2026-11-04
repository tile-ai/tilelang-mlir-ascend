tilelangir_mlir_tablegen(TLDialect.h.inc
  TD_FILE TLOps.td
  ARGS -gen-dialect-decls -dialect=tl
  DEPENDS TLDialect.td TLAttrs.td)
tilelangir_mlir_tablegen(TLDialect.cpp.inc
  TD_FILE TLOps.td
  ARGS -gen-dialect-defs -dialect=tl
  DEPENDS TLDialect.td TLAttrs.td)
tilelangir_mlir_tablegen(TLEnums.h.inc
  TD_FILE TLAttrs.td
  ARGS -gen-enum-decls
  DEPENDS TLDialect.td)
tilelangir_mlir_tablegen(TLEnums.cpp.inc
  TD_FILE TLAttrs.td
  ARGS -gen-enum-defs
  DEPENDS TLDialect.td)
tilelangir_mlir_tablegen(TLAttrs.h.inc
  TD_FILE TLAttrs.td
  ARGS -gen-attrdef-decls -attrdefs-dialect=tl
  DEPENDS TLDialect.td)
tilelangir_mlir_tablegen(TLAttrs.cpp.inc
  TD_FILE TLAttrs.td
  ARGS -gen-attrdef-defs -attrdefs-dialect=tl
  DEPENDS TLDialect.td)
tilelangir_mlir_tablegen(TLOps.h.inc
  TD_FILE TLOps.td
  ARGS -gen-op-decls
  DEPENDS TLDialect.td TLAttrs.td)
tilelangir_mlir_tablegen(TLOps.cpp.inc
  TD_FILE TLOps.td
  ARGS -gen-op-defs
  DEPENDS TLDialect.td TLAttrs.td)
tilelangir_add_tablegen_target(tilelangir_tl_incgen)
