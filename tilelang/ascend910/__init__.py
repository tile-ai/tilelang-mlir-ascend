"""Ascend 910 (A3) backend, parallel to the Ascend 950 backend.

Importing this package registers the ``ascend910`` target (also reachable as
``npuir``/``A2``/``A3``) and the backend manifest.  Everything under this
namespace is independent of ``tilelang.ascend`` (950).
"""

from __future__ import annotations

from . import backend  # noqa: F401  (registers the backend manifest)
from . import compiler
from . import jit
from . import runtime
from . import target  # noqa: F401  (registers the target normalizer)
from .codegen import ASCEND910_CODEGEN  # noqa: F401
from .compiler import (
    opt_command as bishengir_opt_command,
    compile_mlir,
    find_bishengir_compiler,
    find_bishengir_opt,
    normalize_command,
    optimize_mlir,
)
from .jit import Ascend910Compiler, Ascend910JitKernel  # noqa: F401
from .runtime import generate_npu_wrapper_src  # noqa: F401

__all__ = [
    "ASCEND910_CODEGEN",
    "Ascend910Compiler",
    "Ascend910JitKernel",
    "backend",
    "bishengir_opt_command",
    "compile_mlir",
    "compiler",
    "find_bishengir_compiler",
    "find_bishengir_opt",
    "generate_npu_wrapper_src",
    "jit",
    "normalize_command",
    "optimize_mlir",
    "runtime",
]