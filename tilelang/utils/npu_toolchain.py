# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Locate NPUIR tools without importing the NPU runtime."""

import os
import shutil
from pathlib import Path


def find_npuir_tool(name):
    """Prefer an explicit override, then the bundled toolchain, then PATH."""
    override = os.environ.get("TILELANG_NPU_COMPILER_PATH")
    if override:
        root = Path(override)
        candidates = [root / name, root / "bin" / name]
        if name == "bishengir-compile":
            candidates.append(root / "npuc")  # Legacy compiler layout.
    else:
        candidates = [
            Path(__file__).resolve().parents[1] / "lib" / "npuir" / "bin" / name
        ]
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    if not override:
        path = shutil.which(name)
        if path:
            return path
    raise EnvironmentError(
        f"Cannot find executable {name}; checked "
        + ", ".join(str(path) for path in candidates)
        + (". Check TILELANG_NPU_COMPILER_PATH." if override else " and PATH.")
    )
