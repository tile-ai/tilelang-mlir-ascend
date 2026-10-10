#!/bin/bash
# build_wheel.sh — Build TileLang NPUIR wheel from pre-compiled libraries.
#
# Prerequisites: Run ./install_npuir.sh to complete compilation. BISHENGIR_PATH
# can point to a separate NPUIR install prefix containing the compiler/bitcode.
# This script skips CMake and packages the existing build artifacts into a wheel.

set -euo pipefail

# --- Python detection (consistent with install_npuir.sh) ---
PYTHON="$(command -v python3 2>/dev/null)" || PYTHON="$(command -v python 2>/dev/null)"
if [ -z "$PYTHON" ] || [ ! -x "$PYTHON" ]; then
    echo "Error: No python3/python found in PATH. Activate your venv/conda and re-run." >&2
    exit 1
fi
echo "Using Python: $PYTHON"

# --- Pre-build verification ---
REQUIRED_LIBS=(
    "build/libtilelang.so"
    "build/libtilelang_module.so"
    "build/tvm/libtilelang_tvm.so"
    "build/tvm/libtilelang_tvm_runtime.so"
    "build/libtilelangir.so"
)

MISSING=()
for lib in "${REQUIRED_LIBS[@]}"; do
    if [ ! -f "$lib" ]; then
        MISSING+=("$lib")
    fi
done

if [ ${#MISSING[@]} -gt 0 ]; then
    echo "Error: Missing required build artifacts:" >&2
    for m in "${MISSING[@]}"; do
        echo "  - $m" >&2
    done
    echo "Run install_npuir.sh to complete compilation before packaging." >&2
    exit 1
fi
echo "All required libraries found."

# --- Stage the A2/A3 compiler and bitcode ---
export BISHENGIR_PATH="${BISHENGIR_PATH:-$(pwd)/3rdparty/AscendNPU-IR/build/install}"
mkdir -p 3rdparty/bin 3rdparty/lib
cp -L --preserve=mode "$BISHENGIR_PATH/bin/bishengir-compile" 3rdparty/bin/
for name in host.bc meta_op.aic.bc meta_op.aiv.bc meta_op.mix.aic.bc meta_op.mix.aiv.bc; do
    cp -L "$BISHENGIR_PATH/lib/$name" 3rdparty/lib/
done

# --- Build environment ---
export TILELANG_SKIP_BUILD=1
export USE_NPUIR=true

# --- Clean previous build artifacts ---
echo "Cleaning previous build artifacts..."
rm -rf build/lib build/bdist.* dist *.egg-info

# --- Build wheel ---
echo "Building wheel..."
"$PYTHON" setup.py bdist_wheel

# --- Result ---
echo ""
echo "========================================"
echo "Packaging completed successfully."
echo "Wheel location: $(pwd)/dist/"
ls -lh dist/*.whl 2>/dev/null || echo "(no .whl files found)"
echo "========================================"
