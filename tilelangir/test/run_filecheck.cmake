if(NOT TL_OPT OR NOT FILECHECK OR NOT TEST_FILE)
  message(FATAL_ERROR "TL_OPT, FILECHECK, and TEST_FILE must be set")
endif()
execute_process(
  COMMAND ${TL_OPT} ${TEST_FILE}
  OUTPUT_VARIABLE _out
  ERROR_VARIABLE _err
  RESULT_VARIABLE _rc
)
if(NOT _rc EQUAL 0)
  message(FATAL_ERROR "tilelangir-opt failed:\n${_err}")
endif()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/tl_roundtrip.mlir" "${_out}")
execute_process(
  COMMAND ${FILECHECK} ${TEST_FILE}
  INPUT_FILE "${CMAKE_CURRENT_BINARY_DIR}/tl_roundtrip.mlir"
  ERROR_VARIABLE _fc_err
  RESULT_VARIABLE _fc_rc
)
if(NOT _fc_rc EQUAL 0)
  message(FATAL_ERROR "FileCheck failed:\n${_fc_err}\nIR:\n${_out}")
endif()
