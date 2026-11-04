# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Helpers for the BishengIR toolchain (``bishengir-compile`` / ``bishengir-opt``).

This module intentionally depends on the Python standard library only, so the
command lines can be built and asserted by unit tests without loading the
TileLang native runtime or requiring an Ascend device.
"""

from __future__ import annotations

import functools
import os
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

ASCEND_MODE_ENV = "TILELANG_ASCEND_MODE"
NPU_COMPILER_ROOT_ENV = "TILELANG_NPU_COMPILER_PATH"

COMPILE_BINARY = "bishengir-compile"
OPT_BINARY = "bishengir-opt"

DEFAULT_ASCEND_MODE = "expert"
#: ``TILELANG_ASCEND_MODE`` spellings that select expert-mode lowering. Every
#: other value (for example ``developer``) selects developer mode.
EXPERT_MODE_ALIASES = frozenset({"expert", "exp", "e"})

NPUIR_ENABLE_AUTO_MULTI_BUFFER = "npuir.enable_auto_multi_buffer"
NPUIR_DISABLE_HIVM_AUTO_INJECT_SYNC = "npuir.disable_hivm_auto_inject_sync"


class BishengIRError(RuntimeError):
    """Raised when a BishengIR tool exits with a non-zero status."""

    def __init__(
        self,
        stage: str,
        command: Sequence[str],
        returncode: int,
        stdout: str,
        stderr: str,
    ):
        self.stage = stage
        self.command = list(command)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        super().__init__(
            f"BishengIR {stage} failed (exit code {returncode}).\n"
            f"Command: {shlex.join(self.command)}\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}"
        )


@dataclass(frozen=True)
class CommandResult:
    """Result of a BishengIR tool invocation."""

    command: list[str]
    returncode: int
    stdout: str
    stderr: str


@functools.lru_cache()
def find_bishengir_compile() -> str:
    """Return the path of ``bishengir-compile``.

    ``PATH`` is searched first; ``TILELANG_NPU_COMPILER_PATH`` is used as a
    fallback root, with the executable expected at ``<root>/npuc``.
    """
    compiler_path = shutil.which(COMPILE_BINARY)
    if compiler_path is not None:
        return compiler_path
    compiler_root = os.getenv(NPU_COMPILER_ROOT_ENV, "")
    if compiler_root == "":
        raise EnvironmentError(
            "Couldn't find executable bishengir-compile or TILELANG_NPU_COMPILER_PATH."
        )
    return os.path.join(compiler_root, "npuc")


@functools.lru_cache()
def find_bishengir_opt() -> str:
    """Return the path of ``bishengir-opt``."""
    opt_path = shutil.which(OPT_BINARY)
    if opt_path is None:
        raise EnvironmentError("Couldn't find executable bishengir-opt.")
    return opt_path


def clear_tool_cache() -> None:
    """Drop the cached tool paths, e.g. after ``PATH`` changed."""
    find_bishengir_compile.cache_clear()
    find_bishengir_opt.cache_clear()


def _resolve_mode(ascend_mode: str | None) -> str:
    """Normalize an explicit mode or ``TILELANG_ASCEND_MODE``."""
    raw_mode = os.environ.get(ASCEND_MODE_ENV) if ascend_mode is None else ascend_mode
    if raw_mode is None:
        return DEFAULT_ASCEND_MODE
    return raw_mode.strip().lower()


def get_ascend_mode() -> str:
    """Return the normalized ``TILELANG_ASCEND_MODE`` (default ``expert``)."""
    return _resolve_mode(None)


def is_expert_mode() -> bool:
    """Whether expert-mode lowering is selected."""
    return _resolve_mode(None) in EXPERT_MODE_ALIASES


def is_developer_mode() -> bool:
    """Whether developer-mode lowering is selected."""
    return not is_expert_mode()


def _get_pass_config(
    pass_configs: Mapping[Any, Any] | None,
    key: str,
    default: Any,
) -> Any:
    """Read *key* from a pass-config mapping, accepting enum or string keys."""
    if not pass_configs:
        return default
    try:
        if key in pass_configs:
            return pass_configs[key]
    except TypeError:
        # Unhashable mapping keys: fall back to a linear scan below.
        pass
    for config_key, config_value in pass_configs.items():
        if getattr(config_key, "value", config_key) == key:
            return config_value
    return default


def build_compile_options(
    *,
    pass_configs: Mapping[Any, Any] | None = None,
    ascend_mode: str | None = None,
) -> list[str]:
    """Build the ``bishengir-compile`` options for the selected lowering mode.

    ``pass_configs`` carries TileLang pass hints: ``npuir.enable_auto_multi_buffer``
    and ``npuir.disable_hivm_auto_inject_sync`` are forwarded to the compiler.
    """
    enable_auto_multi_buffer = _get_pass_config(
        pass_configs,
        NPUIR_ENABLE_AUTO_MULTI_BUFFER,
        True,
    )
    disable_hivm_auto_inject_sync = _get_pass_config(
        pass_configs,
        NPUIR_DISABLE_HIVM_AUTO_INJECT_SYNC,
        False,
    )
    options = [
        f"--enable-auto-multi-buffer={str(enable_auto_multi_buffer).lower()}",
        f"--disable-hivm-auto-inject-sync={str(disable_hivm_auto_inject_sync).lower()}",
        "--enable-triton-kernel-compile=true",
        "--enable-hivm-compile=true",
    ]

    # Expert mode keeps the tensor compile path disabled; developer mode emits
    # the tensor IR that the BishengIR pipeline consumes directly.
    if _resolve_mode(ascend_mode) in EXPERT_MODE_ALIASES:
        options.append("--disable-hivm-tensor-compile=true")

    return options


def build_compile_command(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    pass_configs: Mapping[Any, Any] | None = None,
    ascend_mode: str | None = None,
    compiler: str | None = None,
) -> list[str]:
    """Build the full ``bishengir-compile`` argv for an npuir module."""
    compiler_path = compiler or find_bishengir_compile()
    options = build_compile_options(
        pass_configs=pass_configs,
        ascend_mode=ascend_mode,
    )
    return [
        compiler_path,
        os.fspath(input_path),
        *options,
        "-o",
        os.fspath(output_path),
    ]


def build_opt_command(
    input_path: str | os.PathLike[str],
    passes: Sequence[str],
    *,
    output_path: str | os.PathLike[str] | None = None,
    options: Sequence[str] | None = None,
    compiler: str | None = None,
) -> list[str]:
    """Build the ``bishengir-opt`` argv running *passes* over *input_path*."""
    opt_path = compiler or find_bishengir_opt()
    command = [opt_path, os.fspath(input_path), *passes]
    if options:
        command.extend(options)
    if output_path is not None:
        command.extend(["-o", os.fspath(output_path)])
    return command


def _run(command: Sequence[str], stage: str) -> CommandResult:
    process = subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        check=False,
    )
    result = CommandResult(
        command=list(command),
        returncode=process.returncode,
        stdout=process.stdout or "",
        stderr=process.stderr or "",
    )
    if result.returncode != 0:
        raise BishengIRError(
            stage,
            result.command,
            result.returncode,
            result.stdout,
            result.stderr,
        )
    return result


def compile_mlir(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    pass_configs: Mapping[Any, Any] | None = None,
    ascend_mode: str | None = None,
    compiler: str | None = None,
) -> CommandResult:
    """Run ``bishengir-compile`` over *input_path* and return the invocation."""
    command = build_compile_command(
        input_path,
        output_path,
        pass_configs=pass_configs,
        ascend_mode=ascend_mode,
        compiler=compiler,
    )
    return _run(command, "compile")


def run_bishengir_opt(
    mlir_text: str,
    passes: Sequence[str],
    *,
    options: Sequence[str] | None = None,
    compiler: str | None = None,
) -> str:
    """Run ``bishengir-opt`` over *mlir_text* and return the transformed IR."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.mlir"
        output_path = Path(tmpdir) / "output.mlir"
        input_path.write_text(mlir_text)
        command = build_opt_command(
            input_path,
            passes,
            output_path=output_path,
            options=options,
            compiler=compiler,
        )
        _run(command, "opt")
        return output_path.read_text()
