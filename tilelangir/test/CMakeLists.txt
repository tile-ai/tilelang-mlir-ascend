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

add_test(
  NAME tilelangir-tl-alloc-copy-sub
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-sub.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-sub.mlir\""
)

add_test(
  NAME tilelangir-convert-tl-to-hivm-alloc-copy-sub
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> --tilelangir-convert-tl-to-hivm \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-sub.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-sub.mlir\""
)

add_test(
  NAME tilelangir-tl-alloc-copy-mul
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-mul.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-mul.mlir\""
)

add_test(
  NAME tilelangir-convert-tl-to-hivm-alloc-copy-mul
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> --tilelangir-convert-tl-to-hivm \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-mul.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-mul.mlir\""
)

add_test(
  NAME tilelangir-tl-alloc-copy-div
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-div.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-div.mlir\""
)

add_test(
  NAME tilelangir-convert-tl-to-hivm-alloc-copy-div
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> --tilelangir-convert-tl-to-hivm \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-div.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-div.mlir\""
)

add_test(
  NAME tilelangir-tl-alloc-copy-max
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-max.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-max.mlir\""
)

add_test(
  NAME tilelangir-convert-tl-to-hivm-alloc-copy-max
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> --tilelangir-convert-tl-to-hivm \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-max.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-max.mlir\""
)

add_test(
  NAME tilelangir-tl-alloc-copy-min
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-min.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-min.mlir\""
)

add_test(
  NAME tilelangir-convert-tl-to-hivm-alloc-copy-min
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> --tilelangir-convert-tl-to-hivm \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-min.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-min.mlir\""
)

add_test(
  NAME tilelangir-tl-alloc-copy-add-set-flag
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-add-set-flag.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-add-set-flag.mlir\""
)

add_test(
  NAME tilelangir-convert-tl-to-hivm-alloc-copy-add-set-flag
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> --tilelangir-convert-tl-to-hivm \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-add-set-flag.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-add-set-flag.mlir\""
)

add_test(
  NAME tilelangir-tl-alloc-copy-add-set-wait-flag
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-add-set-wait-flag.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-add-set-wait-flag.mlir\""
)

add_test(
  NAME tilelangir-convert-tl-to-hivm-alloc-copy-add-set-wait-flag
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> --tilelangir-convert-tl-to-hivm \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-add-set-wait-flag.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-add-set-wait-flag.mlir\""
)
