# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Load TileLang's private native dependency group without exporting it to CANN."""

import ctypes
import os
from pathlib import Path
import sys
import threading


_libraries = {}
_lock = threading.RLock()


def load_library(path):
    """Load a TileLang-owned DSO with a process-local, deep symbol scope."""
    if not sys.platform.startswith("linux") or not hasattr(os, "RTLD_DEEPBIND"):
        raise RuntimeError("Private TVM loading currently requires Linux with glibc")
    path = str(Path(path).resolve())
    with _lock:
        if path not in _libraries:
            _libraries[path] = ctypes.CDLL(
                path, mode=os.RTLD_NOW | os.RTLD_LOCAL | os.RTLD_DEEPBIND
            )
        return _libraries[path]


def library_directories():
    """Return only TileLang-owned locations, independent of CANN environment variables."""
    root = Path(__file__).resolve().parent
    return [root / "lib", root.parent / "build/tvm", root.parent / "build"]


def load_tvm():
    """Load TileLang's compiler library under its private SONAME."""
    for directory in library_directories():
        path = directory / "libtilelang_tvm.so"
        if path.is_file():
            return load_library(path)
    raise ImportError(
        "Missing libtilelang_tvm.so. Rebuild TileLang with its private TVM; "
        "legacy libtvm.so caches are incompatible with CANN coexistence."
    )
