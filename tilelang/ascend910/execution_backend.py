"""Execution backend policy for the A2 JIT."""

from __future__ import annotations

from tilelang.backend.execution_backend import ExecutionBackendSpec

ASCEND910_EXECUTION_BACKENDS = [
    ExecutionBackendSpec("cython"),
]
