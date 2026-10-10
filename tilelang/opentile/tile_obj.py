"""Runtime binding and launch for compiler-owned Tile object bytes.

The compiler supplies the final device ABI through ``TileLaunchSpec``. This
module does not infer an ABI from TIR and does not accept object-file paths.
Object bytes are bound once to the C++ ACL runtime; steady-state launches pass
only a compact bound-kernel token and the current argument values.
"""

from __future__ import annotations

import struct
from typing import Any

from tilelang import tvm
from tilelang.env import env


__all__ = ["TileObjectKernel"]


def _current_npu_stream() -> int:
    """Return the raw handle of torch's current NPU stream."""
    from tilelang.jit.adapter.base import BaseKernelAdapter

    return int(BaseKernelAdapter.get_current_stream_functor()())


def _encode_scalar_arg(kind: str, value: Any) -> int:
    """Encode a scalar into the signed int64 transport used by the launcher."""
    if kind == "float32":
        return struct.unpack("<i", struct.pack("<f", value))[0]
    if kind == "float64":
        return struct.unpack("<q", struct.pack("<d", value))[0]
    return int(value)


class TileObjectKernel:
    """Callable ACL launcher for one compiler-owned Tile object.

    Static launch metadata is validated by ``TileLaunchSpec`` and runtime
    values by ``TileKernelAdapter``. This layer only encodes and forwards them.
    """

    def __init__(
        self,
        *,
        object_bytes: bytes,
        kernel_name: str,
        arg_types: list[str] | tuple[str, ...],
        block_count: int,
        dynamic_ubuf_bytes: int = 0,
    ):
        arg_types = list(arg_types)
        bind = tvm.ffi.get_global_func("tl.tile.BindKernel")
        launch = tvm.ffi.get_global_func("tl.tile.LaunchKernel")
        if bind is None or launch is None:
            raise RuntimeError(
                "Tile object runtime is not registered; rebuild tilelang with src/tile enabled "
                "(see src/tile/CMakeLists.txt)"
            )

        self.kernel_name = kernel_name
        self.arg_types = arg_types
        self.block_count = block_count
        self.dynamic_ubuf_bytes = dynamic_ubuf_bytes
        self._launch = launch
        self._kernel_token = int(bind(object_bytes, kernel_name, arg_types))

    def _compute_dynamic_ubuf_size(self) -> int:
        """Return launch-time UBUF bytes; retained for future dynamic expressions."""
        return self.dynamic_ubuf_bytes

    def _encode_args(self, args: list[Any]) -> list[int]:
        """Encode adapter-validated physical arguments for the C++ launcher."""
        encoded: list[int] = []
        for kind, value in zip(self.arg_types, args):
            if kind == "handle":
                encoded.append(int(value.data_ptr()))
            else:
                encoded.append(_encode_scalar_arg(kind, value))
        return encoded

    def __call__(self, *args: Any) -> None:
        encoded = self._encode_args(list(args))
        stream = _current_npu_stream()
        dynamic_ubuf_bytes = self._compute_dynamic_ubuf_size()
        profile_value = str(env.TILELANG_NPU_PROFILE).strip()

        if profile_value == "0":
            self._launch(
                self._kernel_token,
                self.block_count,
                dynamic_ubuf_bytes,
                stream,
                encoded,
            )
            return

        # Runtime-only import keeps compiler-side Tile object handling
        # free from profiler dependencies.
        from tilelang.profiler.npu import NPUProfileConfig, npu_annotation

        config = NPUProfileConfig.from_env()

        with npu_annotation(
            f"tilelang.launch::{self.kernel_name}",
            enabled=config.enabled,
            strict=config.strict,
        ):
            self._launch(
                self._kernel_token,
                self.block_count,
                dynamic_ubuf_bytes,
                stream,
                encoded,
            )
