# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""TileLang's private TVM package; the public ``tvm`` name belongs to its caller."""

from pathlib import Path as _Path

_root = _Path(__file__).resolve().parents[1]
_candidates = [
    _root / "3rdparty/tvm/python/tvm",
    _root.parent / "build/tvm-python/tvm",
]
for _directory in _candidates:
    if (_directory / ".tilelang-vendor").is_file():
        break
else:
    raise ImportError(
        "TileLang's private TVM Python package is missing. Rebuild TileLang, or run "
        "python tools/prepare_tvm.py python 3rdparty/tvm/python/tvm build/tvm-python/tvm "
        "from the source checkout."
    )

__path__ = [str(_directory)]
__file__ = str(_directory / "__init__.py")
with open(__file__, "rb") as _source:
    exec(compile(_source.read(), __file__, "exec"), globals())
