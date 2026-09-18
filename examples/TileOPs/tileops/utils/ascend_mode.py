"""Scoped ``TILELANG_ASCEND_MODE`` switching (CG-2026-0010 workaround).

The npuir programming mode (Expert / Developer) is a process-global env
var read at trace / lower / compile time by three tilelang call sites
(``tilelang/language/customize_npuir.py``, ``tilelang/engine/lower.py``,
``tilelang/jit/jit_npu.py``); there is no per-kernel parameter.  Kernel
modules declare their mode via ``os.environ.setdefault`` at import time
(first-import-wins), which breaks mixed-mode suites sharing one process
(e.g. ``pytest tests/ops/``): the losing side compiles under the wrong
pipeline and fails with ``hivm.hir.store`` constraint errors or crashes
natively in ``device_codegen``.

``Kernel.__call__`` wraps every kernel invocation with
``scoped_ascend_mode(kernel.ascend_mode)``: all three tilelang read points
execute inside the jitted kernel's first invocation, hence inside this
scope, so each kernel always compiles under its own mode regardless of
import order or ambient env.
"""

import os
from contextlib import contextmanager
from typing import Iterator

_ASCEND_MODE_KEY = "TILELANG_ASCEND_MODE"


@contextmanager
def scoped_ascend_mode(mode: str) -> Iterator[None]:
    """Run the block with ``TILELANG_ASCEND_MODE`` pinned to ``mode``.

    Restores the previous value (or removes the key) on exit, so nested /
    consecutive kernels of different modes in one process each compile
    under their own mode.
    """
    prev = os.environ.get(_ASCEND_MODE_KEY)
    os.environ[_ASCEND_MODE_KEY] = mode
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(_ASCEND_MODE_KEY, None)
        else:
            os.environ[_ASCEND_MODE_KEY] = prev
