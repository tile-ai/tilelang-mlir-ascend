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
    root = Path(__file__).resolve().parent
    # Only search TileLang-owned locations. CANN's TVM_LIBRARY_PATH/PYTHONPATH
    # and already imported public tvm modules must not affect this choice.
    return [root / "lib", root.parent / "build/tvm", root.parent / "build"]


def load_tvm():
    for directory in library_directories():
        path = directory / "libtilelang_tvm.so"
        if path.is_file():
            return load_library(path)
    raise ImportError(
        "Missing libtilelang_tvm.so. Rebuild TileLang with its private TVM prebuild; "
        "legacy libtvm.so caches are incompatible with CANN coexistence."
    )
