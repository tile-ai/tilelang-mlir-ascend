"""Target registration for the Ascend 910 (A2) backend."""

from __future__ import annotations

from tvm.target import Target

from tilelang.backend.target import (
    TargetLike,
    register_target_normalizer,
)

_ALIASES = {
    "ascend910": "ascend910",
    "ascend-910": "ascend910",
    "a2": "ascend910",
    "a3": "ascend910",
    "npuir": "ascend910",
    "ascend910b": "ascend910",
}


def _make_ascend910_target(mcpu: str | None = None) -> Target:
    attrs: dict[str, object] = {"kind": "ascend910"}
    if mcpu:
        attrs["mcpu"] = mcpu
    return Target(attrs)


def normalize_ascend910_target(target: TargetLike) -> Target | None:
    """Normalize the public A2 target spellings.

    ``target="ascend910"`` is the canonical form.  Device names accepted by the
    old ``is_npu=True`` examples (e.g. ``Ascend910B``) are normalized to the same
    target so they do not leak into the codegen layer.
    """

    if not isinstance(target, str):
        return None
    key = target.strip().lower().replace("_", "").replace(" ", "")
    if key not in _ALIASES:
        return None
    mcpu = None
    if "910b" in key:
        mcpu = "ascend910b"
    return _make_ascend910_target(mcpu)


register_target_normalizer("ascend910", normalize_ascend910_target, override=True)


def target_is_ascend910(target: Target) -> bool:
    return target.kind.name == "ascend910"
