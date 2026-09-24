tilelangir_tablegen(tlir_dialect_decls_incgen
  TD_FILES TLIRDialect.td
  ARGS -gen-dialect-decls -dialect=tl
  OUT_SUFFIX .h.inc
)
tilelangir_tablegen(tlir_dialect_defs_incgen
  TD_FILES TLIRDialect.td
  ARGS -gen-dialect-defs -dialect=tl
  OUT_SUFFIX .cpp.inc
)
tilelangir_tablegen(tlir_enum_decls_incgen
  TD_FILES TLIRAttrs.td
  ARGS -gen-enum-decls -I ${CMAKE_CURRENT_SOURCE_DIR}
  OUT_SUFFIX .enum.h.inc
)
tilelangir_tablegen(tlir_enum_defs_incgen
  TD_FILES TLIRAttrs.td
  ARGS -gen-enum-defs -I ${CMAKE_CURRENT_SOURCE_DIR}
  OUT_SUFFIX .enum.cpp.inc
)
tilelangir_tablegen(tlir_attrdef_decls_incgen
  TD_FILES TLIRAttrs.td
  ARGS -gen-attrdef-decls -I ${CMAKE_CURRENT_SOURCE_DIR}
  OUT_SUFFIX .h.inc
)
tilelangir_tablegen(tlir_attrdef_defs_incgen
  TD_FILES TLIRAttrs.td
  ARGS -gen-attrdef-defs -I ${CMAKE_CURRENT_SOURCE_DIR}
  OUT_SUFFIX .cpp.inc
)
tilelangir_tablegen(tlir_op_decls_incgen
  TD_FILES TLIROps.td
  ARGS -gen-op-decls -I ${CMAKE_CURRENT_SOURCE_DIR}
  OUT_SUFFIX .h.inc
)
tilelangir_tablegen(tlir_op_defs_incgen
  TD_FILES TLIROps.td
  ARGS -gen-op-defs -I ${CMAKE_CURRENT_SOURCE_DIR}
  OUT_SUFFIX .cpp.inc
)

add_custom_target(tlir_dialect_tl_incgen
  DEPENDS
    tlir_dialect_decls_incgen
    tlir_dialect_defs_incgen
    tlir_enum_decls_incgen
    tlir_enum_defs_incgen
    tlir_attrdef_decls_incgen
    tlir_attrdef_defs_incgen
    tlir_op_decls_incgen
    tlir_op_defs_incgen
)
