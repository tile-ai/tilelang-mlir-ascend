#!/bin/bash
# build_wheel.sh — Build TileLang NPUIR wheel from pre-compiled libraries.
#
# Prerequisites: Run ./install_npuir.sh first to complete compilation.
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
    "build/tvm/libtvm.so"
    "build/tvm/libtvm_runtime.so"
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
    echo "Run ./install_npuir.sh first to complete compilation." >&2
    exit 1
fi
echo "All required .so libraries found."

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
