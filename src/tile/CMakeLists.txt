# Tile backend: lightweight .o load/launch runtime.
#
# The core launcher resolves ACL entry points through dlopen/dlsym. Optional
# launch profiling uses the CANN Ascend C runtime probe when its required
# libraries are available. The Ascend NPU runtime is Linux-only (matches the
# Ascend backend), so this backend is a no-op on non-UNIX hosts.
if(NOT UNIX)
  return()
endif()

set(TILELANG_CANN_PROFILER_AVAILABLE OFF)

if(DEFINED ENV{ASCEND_HOME_PATH}
   AND NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
  set(_tilelang_cann_lib_paths
    "$ENV{ASCEND_HOME_PATH}/lib64"
    "$ENV{ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64"
  )

  find_library(TILELANG_ASCENDC_RUNTIME_LIBRARY
    NAMES ascendc_runtime
    PATHS ${_tilelang_cann_lib_paths}
    NO_DEFAULT_PATH
    NO_CACHE
  )
  find_library(TILELANG_ASCENDCL_LIBRARY
    NAMES ascendcl
    PATHS ${_tilelang_cann_lib_paths}
    NO_DEFAULT_PATH
    NO_CACHE
  )
  find_library(TILELANG_PROFAPI_LIBRARY
    NAMES profapi
    PATHS ${_tilelang_cann_lib_paths}
    NO_DEFAULT_PATH
    NO_CACHE
  )
  find_library(TILELANG_RUNTIME_LIBRARY
    NAMES runtime
    PATHS ${_tilelang_cann_lib_paths}
    NO_DEFAULT_PATH
    NO_CACHE
  )
  find_library(TILELANG_UNIFIED_DLOG_LIBRARY
    NAMES unified_dlog
    PATHS ${_tilelang_cann_lib_paths}
    NO_DEFAULT_PATH
    NO_CACHE
  )
  find_library(TILELANG_MMPA_LIBRARY
    NAMES mmpa
    PATHS ${_tilelang_cann_lib_paths}
    NO_DEFAULT_PATH
    NO_CACHE
  )
  find_library(TILELANG_C_SEC_LIBRARY
    NAMES c_sec
    PATHS ${_tilelang_cann_lib_paths}
    NO_DEFAULT_PATH
    NO_CACHE
  )

  if(TILELANG_ASCENDC_RUNTIME_LIBRARY
     AND TILELANG_ASCENDCL_LIBRARY
     AND TILELANG_PROFAPI_LIBRARY
     AND TILELANG_RUNTIME_LIBRARY
     AND TILELANG_UNIFIED_DLOG_LIBRARY
     AND TILELANG_MMPA_LIBRARY
     AND TILELANG_C_SEC_LIBRARY)
    set(TILELANG_CANN_PROFILER_AVAILABLE ON)

    set(TILELANG_CANN_PROFILER_LIBRARIES
      ${TILELANG_ASCENDC_RUNTIME_LIBRARY}
      ${TILELANG_ASCENDCL_LIBRARY}
      ${TILELANG_PROFAPI_LIBRARY}
      ${TILELANG_RUNTIME_LIBRARY}
      ${TILELANG_UNIFIED_DLOG_LIBRARY}
      ${TILELANG_MMPA_LIBRARY}
      ${TILELANG_C_SEC_LIBRARY}
    )

    message(STATUS "Tile CANN profiler launch support enabled")
  else()
    message(STATUS "Tile CANN profiler launch support disabled")
  endif()
endif()

file(GLOB TILE_LANG_TILE_SRCS
  src/tile/runtime/tile_obj_launcher.cc
)

if(TILELANG_CANN_PROFILER_AVAILABLE)
  list(APPEND
    TILE_LANG_TILE_SRCS
    src/tile/runtime/tile_npu_profiler.cc
  )
endif()

list(APPEND TILE_LANG_SRCS ${TILE_LANG_TILE_SRCS})
