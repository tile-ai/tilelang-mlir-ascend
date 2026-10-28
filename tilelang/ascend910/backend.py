"""Ascend 910 (A2) backend manifest."""

from __future__ import annotations

from tilelang.backend.module import BackendModule, register_backend

from . import codegen, execution_backend, pipeline
from .target import target_is_ascend910

BACKEND = register_backend(
    BackendModule(
        name="ascend910",
        target_kinds=("ascend910",),
        supports_target=target_is_ascend910,
        pipelines={"ascend910": pipeline.ascend910_pipeline},
        device_codegens={"ascend910": codegen.ASCEND910_CODEGEN},
        execution_backends=execution_backend.ASCEND910_EXECUTION_BACKENDS,
    )
)
