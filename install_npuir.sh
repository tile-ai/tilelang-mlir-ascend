#!/bin/bash
# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
#
# Build TileLang with the Ascend 910 (A3) NPUIR backend.
#
#   bash install_npuir.sh --bishengir-path=/path/to/AscendNPU-IR/build/install
#
# ``--bishengir-path`` points at an existing AscendNPU-IR install tree (the one
# produced by its own build-tools/build.sh).  Passing it is strongly
# recommended: without it the script has to build AscendNPU-IR from source,
# which takes a long time.
set -euo pipefail

PYTHON="$(command -v python3 2>/dev/null)" || PYTHON="$(command -v python 2>/dev/null)"
if [ -z "$PYTHON" ] || [ ! -x "$PYTHON" ]; then
  echo "Error: No python3/python found in PATH. Activate your venv/conda and re-run." >&2
  exit 1
fi
export PATH="$(dirname "$PYTHON"):$PATH"
echo "Using Python (current env): $PYTHON"
"$PYTHON" --version

BISHENGIR_PATH=""
ENABLE_TESTS=OFF
ENABLE_CUDA=OFF
BUILD_JOBS=""
CLEAN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bishengir-path=*) BISHENGIR_PATH="${1#*=}"; shift ;;
    --bishengir-path)
      if [ -n "${2:-}" ]; then BISHENGIR_PATH="$2"; shift 2; else
        echo "err: --bishengir-path needs the AscendNPU-IR install prefix" >&2; exit 1
      fi ;;
    --enable-tests) ENABLE_TESTS=ON; shift ;;
    --enable-cuda) ENABLE_CUDA=ON; shift ;;
    --jobs=*) BUILD_JOBS="${1#*=}"; shift ;;
    -j) BUILD_JOBS="$2"; shift 2 ;;
    --clean) CLEAN=1; shift ;;
    -h|--help)
      sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [ -z "$BUILD_JOBS" ]; then
  CORES="$(nproc)"
  BUILD_JOBS=$(( CORES * 75 / 100 ))
  [ "$BUILD_JOBS" -lt 1 ] && BUILD_JOBS=1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# The AscendNPU-IR install and the A3 code generators are built with clang.
if [ -z "${CC:-}" ] && command -v clang >/dev/null 2>&1; then
  export CC=clang
fi
if [ -z "${CXX:-}" ] && command -v clang++ >/dev/null 2>&1; then
  export CXX=clang++
fi
echo "Compilers: CC=${CC:-default} CXX=${CXX:-default}"

echo "Installing Python requirements..."
"$PYTHON" -m pip install -r requirements.txt

# TVM is a submodule; a source tarball or a fresh clone may not have it yet.
if [ ! -f 3rdparty/tvm/CMakeLists.txt ]; then
  echo "Initialising the TVM submodule..."
  git submodule update --init --recursive 3rdparty/tvm
fi

if [ -z "$BISHENGIR_PATH" ]; then
  echo "warning: no --bishengir-path given; building AscendNPU-IR from source (slow)"
  if [ ! -f 3rdparty/AscendNPU-IR/build-tools/build.sh ]; then
    echo "Error: 3rdparty/AscendNPU-IR is not initialised. Either pass" >&2
    echo "       --bishengir-path=<AscendNPU-IR install> or run" >&2
    echo "       git submodule update --init --recursive 3rdparty/AscendNPU-IR" >&2
    exit 1
  fi
  pushd 3rdparty/AscendNPU-IR >/dev/null
  bash ./build-tools/apply_patches.sh
  rm -rf ./build && mkdir build
  ./build-tools/build.sh -o ./build --python-binding \
    --c-compiler=clang --cxx-compiler=clang++ \
    --add-cmake-options="-DCMAKE_LINKER=lld -DLLVM_ENABLE_LLD=ON -DLLVM_ENABLE_RTTI=ON" \
    --apply-patches --bishengir-publish=off
  BISHENGIR_PATH="$REPO_ROOT/3rdparty/AscendNPU-IR/build/install"
  popd >/dev/null
fi

BISHENGIR_PATH="$(cd "$BISHENGIR_PATH" && pwd)"
echo "AscendNPU-IR install: $BISHENGIR_PATH"
for required in bin/bishengir-compile bin/bishengir-opt include/mlir/IR/BuiltinOps.h; do
  if [ ! -e "$BISHENGIR_PATH/$required" ]; then
    echo "Error: $BISHENGIR_PATH/$required is missing; is this a complete AscendNPU-IR install?" >&2
    exit 1
  fi
done

if [ "$CLEAN" = "1" ]; then
  rm -rf build
fi
mkdir -p build

echo "Configuring TileLang (Ascend 910 NPUIR backend)..."
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_ASCEND=ON \
  -DUSE_ASCEND910=ON \
  -DBISHENGIR_ROOT_PATH="$BISHENGIR_PATH" \
  -DUSE_CUDA="$ENABLE_CUDA" \
  -DMLIR_INCLUDE_TESTS="$ENABLE_TESTS" \
  -DPython3_EXECUTABLE="$PYTHON"

echo "Building TileLang with $BUILD_JOBS jobs..."
cmake --build build -j"$BUILD_JOBS"

if [ ! -f build/lib/libtilelang.so ] && [ ! -f build/libtilelang.so ]; then
  echo "Error: libtilelang.so was not produced." >&2
  exit 1
fi

cat <<EOF

TileLang (Ascend 910 NPUIR) build finished.

Use it from this checkout:
  export PYTHONPATH=$REPO_ROOT:\$PYTHONPATH
  source <CANN>/set_env.sh          # provides bishengir-compile / bishengir-opt

Then:
  python examples/ascend910/elementwise/vec_add_1d.py
EOF