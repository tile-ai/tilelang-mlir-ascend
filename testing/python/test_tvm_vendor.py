# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The vendor transformation must preserve import bindings and upstream source."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest

_spec = importlib.util.spec_from_file_location(
    "prepare_tvm", Path(__file__).resolve().parents[2] / "tools/prepare_tvm.py"
)
_helper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helper)


@pytest.mark.parametrize(
    "source",
    [
        "import tvm\nresult = tvm.marker\n",
        "import tvm.ir\nresult = tvm.ir.marker\n",
        "import tvm.ir as ir\nresult = ir.marker\n",
        "from tvm.ir import (\n    marker,  # comment stays\n)\nresult = marker\n",
        "# 注释\nimport os, tvm as alias\nresult = alias.marker\n",
    ],
)
def test_private_import_bindings(source, monkeypatch):
    root, private, ir, public = (
        ModuleType(name)
        for name in ["tilelang", "tilelang.tvm", "tilelang.tvm.ir", "tvm"]
    )
    root.__path__ = []
    private.__path__ = []
    root.tvm = private
    private.ir = ir
    private.marker = ir.marker = 123
    public.marker = -1
    for module in (root, private, ir, public):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    namespace = {}
    transformed = _helper.rewrite_imports(source)
    exec(transformed, namespace)
    assert namespace["result"] == 123
    assert sys.modules["tvm"] is public
    assert transformed.count("\n") == source.count("\n")
    if "# comment stays" in source:
        assert "# comment stays" in transformed


def test_registry_strings_are_not_python_imports():
    source = 'name = "tvm.contrib.random"\n# import tvm is documentation\n'
    assert _helper.rewrite_imports(source) == source


def test_vendor_preserves_upstream_and_is_repeatable(tmp_path):
    source = tmp_path / "upstream"
    (source / "_ffi").mkdir(parents=True)
    original = "def _load_lib():\n    pass\n\ntry:\n    import readline\nexcept ImportError:\n    pass\n"
    (source / "_ffi/base.py").write_text(original)
    (source / "__init__.py").write_text("# Licensed source\nimport tvm\n")
    output = tmp_path / "private"
    _helper.prepare_python(source, output)
    before = (output / "__init__.py").stat().st_mtime_ns
    _helper.prepare_python(source, output)
    assert (output / "__init__.py").stat().st_mtime_ns == before
    assert (source / "_ffi/base.py").read_text() == original
    assert (output / "__init__.py").read_text().startswith("# Licensed source\n")
