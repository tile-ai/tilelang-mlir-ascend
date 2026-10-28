"""Device-codegen entry points for the Ascend 910 (A2) backend."""

from __future__ import annotations

import os

from tvm import IRModule, ffi
from tvm.target import Target

from tilelang.backend.device_codegen import DeviceCodegen

_EXPERT_NAMES = {"", "expert", "exp", "e"}


def _mode_name() -> str:
    mode = os.environ.get("TILELANG_ASCEND_MODE", "expert").strip().lower()
    return "expert" if mode in _EXPERT_NAMES else "developer"


def build_ascend910(mod: IRModule, target: Target) -> IRModule:
    name = (
        "target.build.tilelang_ascend910"
        if _mode_name() == "expert"
        else "target.build.tilelang_ascend910_dev"
    )
    return ffi.get_global_func(name)(mod, target)


ASCEND910_CODEGEN = DeviceCodegen(
    "ascend910",
    build=build_ascend910,
    build_without_compile=build_ascend910,
)
