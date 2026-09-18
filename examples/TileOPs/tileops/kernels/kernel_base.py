"""Kernel base class — TileLang-based, shared by GPU and NPU backends.

Since TileLang supports both CUDA and NPU backends, this module preserves
the TileLang kernel integration. The ``supported_archs`` is typed as
``Optional[list]`` to accommodate both CUDA SM integers (``[80, 86, 89, 90]``)
and NPU architecture name strings (``["Ascend910B"]``); ``None`` means all
architectures are supported.

Autotune has been removed: NPU TileLang kernels use heuristic config
selection only. The ``init_config`` method resolves config from
``default_config`` or a user-provided override.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch

from tileops.utils import scoped_ascend_mode


class Kernel(ABC):
    """Abstract base class for TileLang-based operator kernels.

    Subclasses set:
    - ``supported_archs``: list of supported architectures (SM ints or
      NPU name strings); ``None`` means all supported.
    - ``default_config``: dict of default kernel parameters.
    - ``kernel``: the TileLang JIT-compiled kernel callable.
    """

    dtype: Optional[torch.dtype] = None
    config: Dict[str, Any]
    supported_archs: Optional[list] = None
    kernel: Optional[Any] = None
    # npuir programming mode this kernel's source form requires
    # ("Expert" / "Developer"; None = defer to the ambient env).  Every
    # invocation is scoped to this mode (see __call__), fixing the
    # process-global TILELANG_ASCEND_MODE battle between Expert and
    # Developer kernels sharing one pytest process (CG-2026-0010).
    ascend_mode: Optional[str] = None
    # Optional explicit msprof kernel-name override: the tilelang
    # ``@T.prim_func`` function name of the device kernel.  Empty string
    # (the default) means "not declared": the benchmark framework derives
    # the default ``msprof op --kernel-name`` filter from the
    # ``JitKernel_NPU`` object graph, which carries exactly that name.
    # Declare it only when auto-derivation would pick the wrong kernel
    # (e.g. several kernels instantiated at once in one Op).
    msprof_kernel_name: str = ""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.config = {}

    def init_config(self, config: Optional[Dict[str, Any]] = None) -> None:
        if config is not None:
            merged = dict(self.default_config)
            for k in merged:
                if config.get(k) is not None:
                    merged[k] = config[k]
            self.config = merged
        else:
            self.config = dict(self.default_config)

        print(f"{self.__class__.__name__} initialized with config: {self.config}")

    @property
    def dtype_str(self) -> str:
        """Convert dtype to str for tl kernels"""
        return self.dtype_to_str(self.dtype)

    @staticmethod
    def dtype_to_str(dtype: torch.dtype) -> str:
        """Convert a torch dtype to the TileLang dtype string."""
        return str(dtype).split(".")[-1]

    @property
    def default_config(self) -> Dict[str, Any]:
        """Return the default config for the kernel"""
        return {}

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run the kernel"""
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.ascend_mode is None:
            return self.forward(*args, **kwargs)
        # Scope TILELANG_ASCEND_MODE around forward: the jitted kernel's
        # first invocation (trace + lower + bishengir cmd, the three env
        # read points) happens inside, so this kernel always compiles
        # under its own mode regardless of import order (CG-2026-0010).
        with scoped_ascend_mode(self.ascend_mode):
            return self.forward(*args, **kwargs)
