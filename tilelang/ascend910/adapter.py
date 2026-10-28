"""JIT adapter for the Ascend 910 (A3) backend.

The adapter is a thin bridge between the v0.1.15 ``JITKernel`` lifecycle and
:class:`tilelang.ascend910.jit.Ascend910Compiler`, which owns the legacy
``compiler_npu`` behaviour (bishengir-opt, bishengir-compile, launch/load
source generation and execution).
"""

from __future__ import annotations

from typing import Any
from collections.abc import Callable

from tilelang.jit.adapter.base import BaseKernelAdapter

from .jit import Ascend910Compiler


class Ascend910KernelAdapter(BaseKernelAdapter):
    """Compile the lowered A3 MLIR and expose it as a torch-callable."""

    def __init__(
        self,
        *,
        params=None,
        result_idx=None,
        target=None,
        func_or_mod=None,
        host_mod=None,
        device_mod=None,
        device_kernel_source: str = "",
        verbose: bool = False,
        pass_configs=None,
        compile_flags=None,
        **kwargs: Any,
    ) -> None:
        # Deliberately does not call BaseKernelAdapter.__init__: the A3 backend
        # has no host module / device module split, so the base __init__ inputs
        # (mod with imports) do not exist.
        self.mod = None
        self.params = params
        self.result_idx = result_idx
        self.target = target
        self.func_or_mod = func_or_mod
        self.host_mod = host_mod
        self.device_mod = device_mod
        self.device_kernel_source = device_kernel_source
        self.verbose = verbose
        self.pass_configs = pass_configs or {}
        self.compile_flags = compile_flags

        out_idx = result_idx
        self.kernel = Ascend910Compiler(verbose=verbose).compile(
            func_or_mod,
            device_kernel_source,
            out_idx=out_idx,
            pass_configs=self.pass_configs,
            device_mod=device_mod,
        )
        self.func = self.kernel

    # -- adapter API ---------------------------------------------------------
    def _convert_torch_func(self) -> Callable:
        return self.kernel

    def get_kernel_source(self, kernel_only: bool = True) -> str:
        return self.device_kernel_source

    def get_host_source(self) -> str:
        return self.kernel.get_launcher_source()

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.func(*args, **kwds)
