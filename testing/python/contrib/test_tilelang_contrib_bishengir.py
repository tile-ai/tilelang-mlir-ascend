# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Command-line tests for ``tilelang.contrib.bishengir``.

They need neither an Ascend device nor a real ``bishengir-compile`` binary::

    python testing/python/contrib/test_tilelang_contrib_bishengir.py
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

BASE_OPTIONS = [
    "--enable-auto-multi-buffer=true",
    "--disable-hivm-auto-inject-sync=false",
    "--enable-triton-kernel-compile=true",
    "--enable-hivm-compile=true",
]
EXPERT_FLAG = "--disable-hivm-tensor-compile=true"


def _load_bishengir():
    """Import the module; fall back to a file load for source-only checkouts."""
    try:
        return importlib.import_module("tilelang.contrib.bishengir")
    except ImportError:
        path = REPO_ROOT / "tilelang" / "contrib" / "bishengir.py"
        spec = importlib.util.spec_from_file_location("_tl_contrib_bishengir", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module


bishengir = _load_bishengir()


def _options(**kwargs):
    return bishengir.build_compile_options(**kwargs)


def _proc(returncode=0, stdout="", stderr=""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def test_ascend_mode_selects_expert_flag(monkeypatch):
    cases = [
        (None, True),
        ("expert", True),
        ("EXPERT", True),
        ("exp", True),
        ("e", True),
        (" E ", True),
        ("developer", False),
        ("Developer", False),
        ("dev", False),
        ("custom", False),
    ]
    for mode, expert in cases:
        if mode is None:
            monkeypatch.delenv(bishengir.ASCEND_MODE_ENV, raising=False)
        else:
            monkeypatch.setenv(bishengir.ASCEND_MODE_ENV, mode)
        assert bishengir.is_expert_mode() is expert
        assert _options() == BASE_OPTIONS + ([EXPERT_FLAG] if expert else [])


def test_explicit_mode_overrides_environment(monkeypatch):
    monkeypatch.setenv(bishengir.ASCEND_MODE_ENV, "developer")
    assert EXPERT_FLAG in _options(ascend_mode="expert")
    monkeypatch.setenv(bishengir.ASCEND_MODE_ENV, "expert")
    assert EXPERT_FLAG not in _options(ascend_mode="developer")


def test_three_variants_differ_only_by_expected_flags():
    expert = _options(ascend_mode="expert")
    developer = _options(ascend_mode="developer")
    hinted = _options(
        ascend_mode="expert",
        pass_configs={"npuir.enable_auto_multi_buffer": False},
    )
    assert developer == expert[:-1]  # developer only drops the expert-only flag
    assert set(expert) ^ set(hinted) == {
        "--enable-auto-multi-buffer=true",
        "--enable-auto-multi-buffer=false",
    }


class _Key(str, Enum):
    ENABLE = "npuir.enable_auto_multi_buffer"
    SYNC = "npuir.disable_hivm_auto_inject_sync"


def test_pass_hints_are_forwarded():
    assert "--enable-auto-multi-buffer=true" in _options()  # defaults
    for enable, sync in [
        ("npuir.enable_auto_multi_buffer", "npuir.disable_hivm_auto_inject_sync"),
        (_Key.ENABLE, _Key.SYNC),
    ]:
        options = _options(pass_configs={enable: False, sync: True})
        assert "--enable-auto-multi-buffer=false" in options
        assert "--disable-hivm-auto-inject-sync=true" in options


def test_pass_hints_with_tilelang_pass_config_key():
    key = pytest.importorskip("tilelang.transform.pass_config").PassConfigKey
    options = _options(pass_configs={key.NPUIR_ENABLE_AUTO_MULTI_BUFFER: False})
    assert "--enable-auto-multi-buffer=false" in options


def test_build_compile_command(tmp_path):
    src, dst = tmp_path / "kernel.npuir", tmp_path / "kernel"
    for mode, expected in [
        ("expert", BASE_OPTIONS + [EXPERT_FLAG]),
        ("developer", BASE_OPTIONS),
    ]:
        command = bishengir.build_compile_command(
            src, dst, ascend_mode=mode, compiler="/bin/bishengir-compile"
        )
        assert command == [
            "/bin/bishengir-compile",
            str(src),
            *expected,
            "-o",
            str(dst),
        ]


def test_build_opt_command(tmp_path):
    src = tmp_path / "kernel.npuir"
    dst = tmp_path / "kernel.opt.mlir"
    assert bishengir.build_opt_command(
        src, ["canonicalize", "cse"], compiler="/bin/bishengir-opt"
    ) == ["/bin/bishengir-opt", str(src), "canonicalize", "cse"]
    assert bishengir.build_opt_command(
        src, ["canonicalize", "cse"], output_path=dst, compiler="/bin/bishengir-opt"
    ) == ["/bin/bishengir-opt", str(src), "canonicalize", "cse", "-o", str(dst)]


def test_compile_mlir_success(tmp_path, monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        Path(command[command.index("-o") + 1]).write_text("binary")
        return _proc(stdout="ok")

    monkeypatch.setattr(bishengir.subprocess, "run", fake_run)
    result = bishengir.compile_mlir(
        tmp_path / "k.npuir", tmp_path / "k", compiler="/bin/bishengir-compile"
    )
    assert result.command[0] == "/bin/bishengir-compile"
    assert (result.returncode, result.stdout) == (0, "ok")
    assert (tmp_path / "k").read_text() == "binary"
    assert calls[0][1] == {"capture_output": True, "text": True, "check": False}


def test_compile_mlir_failure_details(tmp_path, monkeypatch):
    monkeypatch.setattr(
        bishengir.subprocess,
        "run",
        lambda command, **kwargs: _proc(1, "partial", "boom"),
    )
    with pytest.raises(bishengir.BishengIRError) as excinfo:
        bishengir.compile_mlir(
            tmp_path / "k.npuir", tmp_path / "k", compiler="/bin/bishengir-compile"
        )
    error = excinfo.value
    assert (error.command[0], error.returncode, error.stderr) == (
        "/bin/bishengir-compile",
        1,
        "boom",
    )
    assert "boom" in str(error)


def test_run_bishengir_opt_returns_transformed_ir(monkeypatch):
    def fake_run(command, **kwargs):
        Path(command[command.index("-o") + 1]).write_text("module attributes {}")
        return _proc()

    monkeypatch.setattr(bishengir.subprocess, "run", fake_run)
    transformed = bishengir.run_bishengir_opt(
        "module {}", ["adapt-triton-kernel"], compiler="/bin/bishengir-opt"
    )
    assert transformed == "module attributes {}"


def test_find_tools(monkeypatch):
    monkeypatch.setattr(bishengir.shutil, "which", lambda name: f"/fake/{name}")
    bishengir.clear_tool_cache()
    try:
        assert bishengir.find_bishengir_compile() == "/fake/bishengir-compile"
        assert bishengir.find_bishengir_opt() == "/fake/bishengir-opt"
    finally:
        bishengir.clear_tool_cache()


def test_find_compile_fallback_and_missing_toolchain(monkeypatch):
    monkeypatch.setattr(bishengir.shutil, "which", lambda name: None)
    monkeypatch.setenv(bishengir.NPU_COMPILER_ROOT_ENV, "/opt/npu")
    bishengir.clear_tool_cache()
    try:
        assert bishengir.find_bishengir_compile() == "/opt/npu/npuc"
        monkeypatch.delenv(bishengir.NPU_COMPILER_ROOT_ENV)
        bishengir.clear_tool_cache()
        for finder in (bishengir.find_bishengir_compile, bishengir.find_bishengir_opt):
            with pytest.raises(EnvironmentError):
                finder()
    finally:
        bishengir.clear_tool_cache()


def test_utils_compat_shims(monkeypatch):
    utils = pytest.importorskip("tilelang.utils")
    shims = (utils.get_npucompiler_path, utils.get_npucompiler_opt_path)
    monkeypatch.setattr(bishengir.shutil, "which", lambda name: f"/fake/{name}")
    bishengir.clear_tool_cache()
    for shim in shims:
        shim.cache_clear()
    try:
        assert shims[0]() == "/fake/bishengir-compile"
        assert shims[1]() == "/fake/bishengir-opt"
    finally:
        bishengir.clear_tool_cache()
        for shim in shims:
            shim.cache_clear()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
