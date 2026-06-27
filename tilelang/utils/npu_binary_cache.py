# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Runtime cache for Ascend NPU binary handles."""

from dataclasses import dataclass
from hashlib import sha256
import threading
from typing import Any, Callable, Dict


@dataclass(frozen=True)
class NPUBinaryCacheKey:
    name: str
    kernel_digest: str
    shared: int
    device: int
    mix_mode: str


class NPUBinaryHandleCache:
    """Process-local cache for loaded NPU runtime binary handles."""

    def __init__(self) -> None:
        self._entries: Dict[NPUBinaryCacheKey, Any] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _kernel_bytes(kernel: Any) -> bytes:
        if isinstance(kernel, bytes):
            return kernel
        if isinstance(kernel, bytearray):
            return bytes(kernel)
        if isinstance(kernel, memoryview):
            return kernel.tobytes()
        if isinstance(kernel, str):
            return kernel.encode("utf-8")
        try:
            return bytes(kernel)
        except TypeError as exc:
            raise TypeError(
                f"Unsupported NPU kernel binary type: {type(kernel).__name__}"
            ) from exc

    @staticmethod
    def _raise_if_load_failed(
        loaded: Any,
        name: str,
    ) -> None:
        if not isinstance(loaded, tuple) or len(loaded) < 2:
            return

        module_handle, function_handle = loaded[:2]
        if module_handle and function_handle:
            return

        raise RuntimeError(f"Failed to load NPU kernel binary: {name}")

    @classmethod
    def kernel_digest(cls, kernel: Any) -> str:
        return sha256(cls._kernel_bytes(kernel)).hexdigest()

    @classmethod
    def make_key(
        cls,
        name: str,
        kernel: Any,
        shared: int,
        device: int,
        mix_mode: str,
    ) -> NPUBinaryCacheKey:
        return NPUBinaryCacheKey(
            name=str(name),
            kernel_digest=cls.kernel_digest(kernel),
            shared=int(shared),
            device=int(device),
            mix_mode=str(mix_mode),
        )

    def get_or_load(
        self,
        loader: Callable[[str, Any, int, int, str], Any],
        name: str,
        kernel: Any,
        shared: int,
        device: int,
        mix_mode: str,
    ) -> Any:
        key = self.make_key(name, kernel, shared, device, mix_mode)
        with self._lock:
            if key not in self._entries:
                loaded = loader(name, kernel, shared, device, mix_mode)
                self._raise_if_load_failed(loaded, name)
                self._entries[key] = loaded
            return self._entries[key]

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
