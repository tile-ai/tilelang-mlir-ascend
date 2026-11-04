get_filename_component(_bisherir_root "${BISHENGIR_ROOT_PATH}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
set(_filecheck "${_bisherir_root}/bin/FileCheck")
if(NOT EXISTS "${_filecheck}")
  find_program(_filecheck NAMES FileCheck)
endif()

if(NOT _filecheck)
  message(STATUS "FileCheck not found; skipping TL dialect FileCheck test")
  return()
endif()

add_test(
  NAME tilelangir-tl-alloc-copy-add
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-add.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-add.mlir\""
)

add_test(
  NAME tilelangir-convert-tl-to-hivm-alloc-copy-add
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> --tilelangir-convert-tl-to-hivm \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-add.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-add.mlir\""
)
