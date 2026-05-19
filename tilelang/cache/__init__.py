# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The cache utils with class and database persistence - Init file"""

from typing import List, Union, Literal, Optional
from pathlib import Path
from tvm.target import Target
from tvm.tir import PrimFunc

from .kernel_cache import KernelCache
from tilelang import env

# Singleton instance of KernelCache
_kernel_cache_instance = KernelCache()


def cached_npu(
    func: PrimFunc = None,
    out_idx: List[int] = None,
    target: Union[str, Target] = "npuir",
    target_host: Union[str, Target] = None,
    execution_backend: Optional[Literal["cython"]] = "cython",
    verbose: Optional[bool] = True,
    pass_configs: Optional[dict] = None,
):
    """
    Caches and reuses compiled NPU kernels (using KernelCache class).
    """
    return _kernel_cache_instance.cached_npu(
        func,
        out_idx,
        target=target,
        target_host=target_host,
        execution_backend=execution_backend,
        verbose=verbose,
        pass_configs=pass_configs,
    )


def get_cache_dir() -> Path:
    """
    Gets the cache directory for the kernel cache.
    Example:
        >>> tilelang.cache.get_cache_dir()
        PosixPath('/Users/username/.tilelang/cache')
    """
    return _kernel_cache_instance.get_cache_dir()


def set_cache_dir(cache_dir: str):
    """
    Sets the cache directory for the kernel cache.
    Example:
        >>> tilelang.cache.set_cache_dir("/path/to/cache")
    """
    _kernel_cache_instance.set_cache_dir(cache_dir)


def clear_cache():
    """
    Clears the entire kernel cache (using KernelCache class).
    """
    _kernel_cache_instance.clear_cache()


if env.TILELANG_CLEAR_CACHE.lower() in ("1", "true", "yes", "on"):
    clear_cache()
