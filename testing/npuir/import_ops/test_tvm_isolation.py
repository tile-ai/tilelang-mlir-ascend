# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""CANN and TileLang must work in either import order in a fresh process."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.skipif(
    importlib.util.find_spec("tbe") is None, reason="CANN TBE is required"
)
@pytest.mark.parametrize("order", ["cann-first", "tilelang-first"])
def test_tvm_isolation(order, tmp_path):
    script = tmp_path / "check_isolation.py"
    script.write_text("""
from concurrent.futures import ThreadPoolExecutor
import importlib
import multiprocessing
import os
import pickle
import sys

def check_in_spawned_process(value):
    import te
    import tvm as cann
    from tilelang import tvm as private
    assert isinstance(value, private.tir.Var)
    assert value.name == "private_var"
    assert cann is not private
    return value.name

if __name__ == "__main__":
    original_path = os.environ.get("TVM_LIBRARY_PATH")
    if sys.argv[1] == "cann-first":
        import te
        import tvm as cann
        before = {k: v for k, v in sys.modules.items() if k == "tvm" or k.startswith("tvm.")}
        tilelang = importlib.import_module("tilelang")
        assert all(sys.modules[k] is v for k, v in before.items())
    else:
        tilelang = importlib.import_module("tilelang")
        assert "tvm" not in sys.modules
        import te
        import tvm as cann
    private = tilelang.tvm
    assert private.__name__ == "tilelang.tvm"
    assert private is not cann
    assert private._ffi.libinfo.find_lib_path()[0].endswith("/libtilelang_tvm.so")
    assert private._ffi.base._LIB._name.endswith("/libtilelang_tvm.so")
    assert os.environ.get("TVM_LIBRARY_PATH") == original_path
    assert cann.get_global_func("cce.product_init", allow_missing=True) is not None
    assert private.get_global_func("cce.product_init", allow_missing=True) is None
    assert cann.get_global_func("tl.transform.Simplify", allow_missing=True) is None
    assert private.get_global_func("tl.transform.Simplify", allow_missing=True) is not None
    cann.register_func("tilelang_isolation_probe", lambda x: x + 10, override=True)
    private.register_func("tilelang_isolation_probe", lambda x: x + 20, override=True)

    def check(index):
        for module, offset in [(cann, 10), (private, 20)]:
            assert module.get_global_func("tilelang_isolation_probe")(index) == index + offset
            value = module.tir.Var("x", "int32")
            assert int(module.arith.Analyzer().simplify(value - value)) == 0
            assert not hasattr(value, "isolation_missing_attribute")
        return index

    with ThreadPoolExecutor(max_workers=4) as executor:
        assert list(executor.map(check, range(20))) == list(range(20))
    value = private.tir.Var("private_var", "int32")
    restored = pickle.loads(pickle.dumps(value))
    assert isinstance(restored, private.tir.Var)
    assert restored.name == "private_var"
    with multiprocessing.get_context("spawn").Pool(1) as pool:
        assert pool.apply(check_in_spawned_process, (value,)) == "private_var"
    check(99)
    print("CANN / TileLang isolation passed", sys.argv[1], flush=True)
""")
    env = os.environ.copy()
    root = Path(__file__).resolve().parents[3]
    if (root / "tilelang/__init__.py").is_file():
        env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", str(script), order],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
