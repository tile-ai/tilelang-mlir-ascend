"""A2-specific kernel launch surface (Ascend 910)."""

from __future__ import annotations

import threading
from typing import Any

from tilelang.language.kernel import launch_kernel

_A2_ACTIVE = threading.local()


def _set_active(value: bool):
    _A2_ACTIVE.value = value


def is_active() -> bool:
    return bool(getattr(_A2_ACTIVE, "value", False))


class _A2KernelFrame:
    """Kernel frame wrapper that mirrors the legacy NPU binding rules.

    ``T.Kernel(grid, is_npu=True)`` binds ``(cid, subid)`` (the legacy NPU
    kernel frame exposes the sub-block id as a second variable), while a plain
    ``T.Kernel(grid)`` keeps the regular tilelang semantics and yields whatever
    the underlying frame yields (a single Var for a 1-D grid).
    """

    def __init__(self, frame, npu: bool = False):
        self._frame = frame
        self._npu = npu

    def _sub_block_var(self):
        """Vector sub-block id (``blockIdx.z``) supplied by the launch frame."""
        try:
            return self._frame.get_thread_binding(2)
        except Exception:
            return None

    def __enter__(self):
        value = self._frame.__enter__()
        if not self._npu:
            return value
        _set_active(True)
        sub = self._sub_block_var()
        if isinstance(value, (list, tuple)):
            values = list(value)
            if len(values) == 1:
                return values[0], (sub if sub is not None else 0)
            # 2-D grid: the legacy NPU frame yielded ``(bx, by, subid)``.
            # ``launch_kernel`` may already expose the sub-block id, so only
            # append it when the frame stopped at the block coordinates.
            if len(values) == 2:
                return values[0], values[1], (sub if sub is not None else 0)
            return tuple(values)
        # A 1-D grid yields the block var directly (not a list).
        return value, (sub if sub is not None else 0)

    def __exit__(self, ptype, value, trace):
        try:
            return self._frame.__exit__(ptype, value, trace)
        finally:
            _set_active(False)


def Kernel(
    *blocks,
    threads: int | list[int] | tuple[int, ...] | None = None,
    is_cpu: bool = False,
    prelude: str | None = None,
    is_npu: bool = False,
    pipeline: bool = False,
    **kwargs: Any,
):
    attrs: dict[str, Any] = {}
    if is_npu:
        attrs["tilelang.is_npu_kernel_frame"] = True
    if prelude is not None:
        attrs["pragma_import_c"] = prelude
    attrs.update(kwargs)
    frame = launch_kernel(blocks, threads=threads, **attrs)
    return _A2KernelFrame(frame, npu=is_npu)
