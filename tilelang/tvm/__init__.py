"""Compatibility shim for the legacy ``tilelang.tvm`` namespace.

Before the v0.1.15 rebase TileLang shipped its own TVM copy as
``tilelang.tvm``; the rebased tree builds against the upstream ``tvm``
package, whose TIR namespace is ``tvm.tirx``.  Legacy imports such as
``from tilelang.tvm import tir`` are forwarded here.
"""

import sys as _sys

import tvm as _tvm
from tvm import *  # noqa: F401,F403
from tvm import arith, ffi, ir, runtime, target, tirx  # noqa: F401
from tvm import tirx as tir  # noqa: F401  (legacy spelling)

_sys.modules[__name__ + ".tir"] = _tvm.tirx
_sys.modules[__name__ + ".tirx"] = _tvm.tirx
_sys.modules[__name__ + ".ir"] = _tvm.ir
_sys.modules[__name__ + ".ffi"] = _tvm.ffi
_sys.modules[__name__ + ".runtime"] = _tvm.runtime
_sys.modules[__name__ + ".target"] = _tvm.target

__all__ = ["arith", "ffi", "ir", "runtime", "target", "tir", "tirx"]
