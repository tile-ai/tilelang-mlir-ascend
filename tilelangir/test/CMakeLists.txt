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
  NAME tilelangir-tl-alloc-copy-concat
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-concat.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-concat.mlir\""
)

add_test(
  NAME tilelangir-convert-tl-to-hivm-alloc-copy-concat
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> --tilelangir-convert-tl-to-hivm \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-concat.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-concat.mlir\""
)

add_test(
  NAME tilelangir-tl-alloc-copy-arange
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-arange.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-arange.mlir\""
)

add_test(
  NAME tilelangir-convert-tl-to-hivm-alloc-copy-arange
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> --tilelangir-convert-tl-to-hivm \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-arange.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-arange.mlir\""
)

add_test(
  NAME tilelangir-tl-sort
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/sort.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/sort.mlir\""
)

add_test(
  NAME tilelangir-convert-tl-to-hivm-sort
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> --tilelangir-convert-tl-to-hivm \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/sort.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/sort.mlir\""
 )
 
add_test(
 NAME tilelangir-tl-alloc-copy-cast
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-cast.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-cast.mlir\""
)

add_test(
  NAME tilelangir-convert-tl-to-hivm-alloc-copy-cast
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> --tilelangir-convert-tl-to-hivm \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-cast.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-cast.mlir\""
  )
add_test(
  NAME tilelangir-tl-alloc-copy-shr
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-shr.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-shr.mlir\""
)

add_test(
  NAME tilelangir-convert-tl-to-hivm-alloc-copy-shr
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> --tilelangir-convert-tl-to-hivm \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-shr.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-shr.mlir\""
  )
add_test(
  NAME tilelangir-tl-alloc-copy-shl
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-shl.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-shl.mlir\""
)

add_test(
  NAME tilelangir-convert-tl-to-hivm-alloc-copy-shl
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> --tilelangir-convert-tl-to-hivm \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-shl.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-shl.mlir\""
  )
  
add_test(
  NAME tilelangir-tl-print
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/print.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/print.mlir\""
)

add_test(
  NAME tilelangir-convert-tl-to-hivm-print
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> --tilelangir-convert-tl-to-hivm \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/print.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/print.mlir\""
)

add_test(
  NAME tilelangir-tl-print-invalid
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> --split-input-file --verify-diagnostics \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/print-invalid.mlir\""
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
 NAME tilelangir-tl-alloc-copy-exp
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-exp.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-exp.mlir\""
)

add_test(
  NAME tilelangir-convert-tl-to-hivm-alloc-copy-exp
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> --tilelangir-convert-tl-to-hivm \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-exp.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-exp.mlir\""
)

add_test(
  NAME tilelangir-tl-alloc-copy-rec
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-rec.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-rec.mlir\""
)

add_test(
  NAME tilelangir-convert-tl-to-hivm-alloc-copy-rec
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> --tilelangir-convert-tl-to-hivm \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-rec.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-rec.mlir\""
)

add_test(
  NAME tilelangir-tl-alloc-atomic-add
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-atomic-add.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-atomic-add.mlir\""
)

add_test(
  NAME tilelangir-convert-tl-to-hivm-alloc-atomic-add
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> --tilelangir-convert-tl-to-hivm \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-atomic-add.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-atomic-add.mlir\""
  )
  
add_test(
  NAME tilelangir-tl-alloc-copy-verf
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-verf.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-verf.mlir\""
)

add_test(
  NAME tilelangir-convert-tl-to-hivm-alloc-copy-verf
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> --tilelangir-convert-tl-to-hivm \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-verf.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-verf.mlir\""
 )
 
add_test(
  NAME tilelangir-tl-alloc-copy-pad
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-pad.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-pad.mlir\""
)

add_test(
  NAME tilelangir-convert-tl-to-hivm-alloc-copy-pad
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> --tilelangir-convert-tl-to-hivm \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-pad.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-pad.mlir\""
  )
add_test(
  NAME tilelangir-tl-alloc-copy-rsqrt
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-rsqrt.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-rsqrt.mlir\""
)

add_test(
  NAME tilelangir-convert-tl-to-hivm-alloc-copy-rsqrt
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> --tilelangir-convert-tl-to-hivm \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-rsqrt.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-rsqrt.mlir\""
)

add_test(
  NAME tilelangir-tl-alloc-copy-sqrt
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-sqrt.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-sqrt.mlir\""
)

add_test(
  NAME tilelangir-convert-tl-to-hivm-alloc-copy-sqrt
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> --tilelangir-convert-tl-to-hivm \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-sqrt.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-sqrt.mlir\""
)

add_test(
  NAME tilelangir-tl-alloc-copy-abs
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-abs.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-abs.mlir\""
)

add_test(
  NAME tilelangir-convert-tl-to-hivm-alloc-copy-abs
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> --tilelangir-convert-tl-to-hivm \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-abs.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-abs.mlir\""
)

add_test(
  NAME tilelangir-tl-alloc-copy-not
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-not.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-not.mlir\""
)

add_test(
  NAME tilelangir-convert-tl-to-hivm-alloc-copy-not
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> --tilelangir-convert-tl-to-hivm \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-not.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-not.mlir\""
)

add_test(
  NAME tilelangir-tl-alloc-copy-flip
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-flip.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-flip.mlir\""
)

add_test(
  NAME tilelangir-convert-tl-to-hivm-alloc-copy-flip
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> --tilelangir-convert-tl-to-hivm \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-flip.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-flip.mlir\""
  )
add_test(
  NAME tilelangir-tl-alloc-copy-reshape
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-reshape.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-reshape.mlir\""
)

add_test(
  NAME tilelangir-convert-tl-to-hivm-alloc-copy-reshape
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> --tilelangir-convert-tl-to-hivm \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-reshape.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-reshape.mlir\""
  )
add_test(
  NAME tilelangir-tl-alloc-copy-pow
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-pow.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-pow.mlir\""
)

add_test(
  NAME tilelangir-convert-tl-to-hivm-alloc-copy-pow
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> --tilelangir-convert-tl-to-hivm \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-pow.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-pow.mlir\""
)

add_test(
  NAME tilelangir-tl-alloc-copy-xor
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-xor.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-xor.mlir\""
)

add_test(
  NAME tilelangir-convert-tl-to-hivm-alloc-copy-xor
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> --tilelangir-convert-tl-to-hivm \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-xor.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-xor.mlir\""
)

add_test(
  NAME tilelangir-tl-alloc-copy-or
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-or.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-or.mlir\""
)

add_test(
  NAME tilelangir-convert-tl-to-hivm-alloc-copy-or
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> --tilelangir-convert-tl-to-hivm \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-or.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-or.mlir\""
)

add_test(
  NAME tilelangir-tl-alloc-copy-and
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-and.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Dialect/TL/alloc-copy-and.mlir\""
)

add_test(
  NAME tilelangir-convert-tl-to-hivm-alloc-copy-and
  COMMAND /bin/bash -c
    "$<TARGET_FILE:tilelangir-opt> --tilelangir-convert-tl-to-hivm \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-and.mlir\" | \"${_filecheck}\" \"${CMAKE_CURRENT_SOURCE_DIR}/Conversion/TLToHIVM/alloc-copy-and.mlir\""
)
