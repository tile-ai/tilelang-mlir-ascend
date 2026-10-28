"""bishengir-opt / bishengir-compile integration for Ascend 910 (A3).

This module is the A3 counterpart of ``tilelang.contrib.bisheng`` (Ascend 950).
It owns the *external* compile contract of the backend:

    codegen MLIR --bishengir-opt--> compile MLIR --bishengir-compile--> kernel.o

Every command line it produces is part of the golden corpus, so the option
order below must not be reordered.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping, Sequence

_ENV_COMPILER = "TILELANG_NPU_COMPILER_PATH"
_ENV_OPT = "TILELANG_NPU_OPT_PATH"

_CACHE_ENV = "TILELANG_CACHE_DIR"
_CACHE_SUBDIR = "ascend910"


# ---------------------------------------------------------------------------
# tool discovery
# ---------------------------------------------------------------------------
def _find_tool(name: str, env_var: str, *, required_by: str) -> str:
    explicit = os.environ.get(env_var)
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return str(path)
        raise FileNotFoundError(f"{env_var}={explicit!r} is not a file")

    # Prefer the compiler shipped with the active CANN toolkit: the standalone
    # AscendNPU-IR install may be first on PATH for its MLIR tools while its
    # bishengir-compile is not paired with the toolkit's hivmc.
    for env_name in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        root = os.environ.get(env_name)
        if not root:
            continue
        for rel in (f"tools/bishengir/bin/{name}", f"bin/{name}"):
            candidate = Path(root) / rel
            if candidate.is_file():
                return str(candidate)

    found = shutil.which(name)
    if found:
        return found

    raise RuntimeError(
        f"Cannot find {name} (needed for {required_by}). Set {env_var}, or source the CANN set_env.sh so tools/bishengir/bin is on PATH."
    )


def find_bishengir_compiler() -> str:
    return _find_tool("bishengir-compile", _ENV_COMPILER, required_by="device compile")


def find_bishengir_opt() -> str:
    return _find_tool("bishengir-opt", _ENV_OPT, required_by="MLIR canonicalization")


# ---------------------------------------------------------------------------
# options
# ---------------------------------------------------------------------------
def _bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    return bool(value)


def _pass_config(pass_configs: Mapping[str, object] | None, *keys: str, default=None):
    cfg = pass_configs or {}
    for key in keys:
        if key in cfg:
            return cfg[key]
    return default


def compiler_options(
    pass_configs: Mapping[str, object] | None = None,
    *,
    developer: bool = False,
) -> list[str]:
    """Return the exact option sequence used by the legacy A2 JIT."""

    auto_multi_buffer = _pass_config(
        pass_configs,
        "npuir.enable_auto_multi_buffer",
        "tl.npuir_enable_auto_multi_buffer",
        default=True,
    )
    disable_auto_sync = _pass_config(
        pass_configs,
        "npuir.disable_hivm_auto_inject_sync",
        "tl.npuir_disable_hivm_auto_inject_sync",
        default=False,
    )
    options = [
        f"--enable-auto-multi-buffer={str(_bool(auto_multi_buffer, True)).lower()}",
        f"--disable-hivm-auto-inject-sync={str(_bool(disable_auto_sync, False)).lower()}",
        "--enable-triton-kernel-compile=true",
        "--enable-hivm-compile=true",
    ]
    if not developer:
        options.append("--disable-hivm-tensor-compile=true")
    return options


def compile_command(
    compiler: str,
    mlir_path: str | os.PathLike[str],
    output_base: str | os.PathLike[str],
    pass_configs: Mapping[str, object] | None = None,
    *,
    developer: bool = False,
) -> list[str]:
    return [
        compiler,
        os.fspath(mlir_path),
        *compiler_options(pass_configs, developer=developer),
        "-o",
        os.fspath(output_base),
    ]


def opt_options() -> list[str]:
    """The exact MLIR pipeline applied between codegen and compile.

    ``--canonicalize`` matches the in-process canonicalizer of the legacy
    backend and ``--adapt-triton-kernel`` is the pass that adapts the kernel for
    ``--enable-triton-kernel-compile=true``.
    """

    # NOTE: ``--canonicalize`` used to run here as well, but its memref
    # canonicalization folds the ``copy(gm -> ub); copy(ub -> gm)`` staging pair
    # into a single ``memref.copy`` whose source and destination are both in the
    # GM address space.  HIVM supports only gm<->ub and ub<->ub copies, so
    # bishengir-compile then crashes (SIGSEGV) on the resulting module.
    # ``--adapt-triton-kernel`` is the pass the ``--enable-triton-kernel-compile``
    # option needs, so keep it alone.
    return ["--adapt-triton-kernel"]


def opt_command(opt: str, mlir_path: str | os.PathLike[str], out_path: str | os.PathLike[str]) -> list[str]:
    return [opt, os.fspath(mlir_path), *opt_options(), "-o", os.fspath(out_path)]


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OptimizeResult:
    mlir: str
    command: tuple[str, ...]
    path: Path


@dataclass(frozen=True)
class CompileResult:
    object_path: Path
    shared_object_path: Path | None
    command: tuple[str, ...]
    mlir_path: Path


def optimize_mlir(mlir: str, *, opt: str | None = None, timeout: int = 600, verbose: bool = False) -> OptimizeResult:
    """Run ``bishengir-opt`` over raw codegen output."""

    opt = opt or find_bishengir_opt()
    work = Path(tempfile.mkdtemp(prefix="tilelang-ascend910-opt-"))
    src = work / "kernel.codegen.mlir"
    dst = work / "kernel.npuir"
    src.write_text(mlir)
    command = opt_command(opt, src, dst)
    if verbose:
        print("bishengir-opt:", shlex.join(command))
    proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0 or not dst.is_file():
        raise RuntimeError(f"bishengir-opt failed.\nCommand: {shlex.join(command)}\nOutput:\n{proc.stderr or proc.stdout}")
    return OptimizeResult(mlir=dst.read_text(), command=tuple(command), path=dst)


def compile_mlir(
    mlir: str,
    output_dir: str | os.PathLike[str],
    *,
    pass_configs: Mapping[str, object] | None = None,
    developer: bool = False,
    compiler: str | None = None,
    timeout: int = 1800,
    verbose: bool = False,
) -> CompileResult:
    """Compile *mlir* into ``<output_dir>/kernel.o``."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    mlir_path = out / "kernel.npuir"
    mlir_path.write_text(mlir)
    compiler = compiler or find_bishengir_compiler()
    output_base = out / "kernel"
    command = compile_command(compiler, mlir_path, output_base, pass_configs, developer=developer)

    if verbose:
        print("bishengir-compile:", shlex.join(command))
    proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout)

    object_path = out / "kernel.o"
    shared_object_path = out / "libkernel.so"
    if proc.returncode != 0 or not object_path.is_file():
        raise RuntimeError(
            "AscendNPU IR compile failed.\n"
            f"Command: {shlex.join(command)}\n"
            f"Return code: {proc.returncode}\n"
            f"Output:\n{proc.stderr or proc.stdout}"
        )
    return CompileResult(
        object_path=object_path,
        shared_object_path=shared_object_path if shared_object_path.is_file() else None,
        command=tuple(command),
        mlir_path=mlir_path,
    )


# ---------------------------------------------------------------------------
# content-addressed cache
# ---------------------------------------------------------------------------
def _cache_root() -> Path:
    base = os.environ.get(_CACHE_ENV)
    root = Path(base).expanduser() if base else Path.home() / ".tilelang" / "cache"
    # pytest-xdist workers run with ``TILELANG_CLEAR_CACHE=1`` and each of them
    # wipes the *shared* cache on import, deleting artifacts another worker is
    # still writing or loading (intermittent precision failures / truncated
    # ``.so``).  Give every worker its own namespace so clearing is safe.
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    return root / _CACHE_SUBDIR / worker if worker else root / _CACHE_SUBDIR


def _compiler_stamp() -> str:
    compiler = find_bishengir_compiler()
    try:
        st = os.stat(compiler)
        return f"{compiler}:{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        return compiler


def make_key(
    mlir: str,
    *,
    pass_configs: Mapping[str, object] | None = None,
    developer: bool = False,
) -> str:
    payload = {
        "format": 1,
        "mlir": hashlib.sha256(mlir.encode()).hexdigest(),
        "pass_configs": dict(pass_configs or {}),
        "developer": bool(developer),
        "compiler": _compiler_stamp(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def cache_paths(key: str) -> tuple[Path, Path]:
    root = _cache_root() / key
    return root, root / "kernel.o"


def cache_load(key: str) -> Path | None:
    _, obj = cache_paths(key)
    return obj if obj.is_file() else None


def cache_shared_path(key: str) -> Path:
    """Path of the cached shared object (``libkernel.so``) for *key*."""

    root, _ = cache_paths(key)
    return root / "libkernel.so"


def cache_save(
    key: str,
    mlir: str,
    object_path: str | os.PathLike[str],
    shared_object_path: str | os.PathLike[str] | None = None,
) -> Path:
    root, obj = cache_paths(key)
    root.mkdir(parents=True, exist_ok=True)

    # Several processes (e.g. pytest-xdist workers) may compile the same kernel
    # concurrently into the shared cache directory; publish each artifact with an
    # atomic rename so a reader never observes a half-written file.
    def _publish(src, dst: Path) -> None:
        tmp = dst.with_name(dst.name + f".tmp{os.getpid()}")
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)

    _publish(object_path, obj)
    # ``libkernel.so`` carries the ``*_infer_workspace_shape_function`` callback
    # the launcher needs to size its per-block scratch buffer, so it has to
    # survive alongside the device object.
    if shared_object_path is not None and Path(shared_object_path).is_file():
        _publish(shared_object_path, cache_shared_path(key))
    tmp_mlir = (root / "kernel.npuir").with_name(f"kernel.npuir.tmp{os.getpid()}")
    tmp_mlir.write_text(mlir)
    os.replace(tmp_mlir, root / "kernel.npuir")
    (root / "metadata.json").write_text(
        json.dumps({"format": 1, "key": key, "compiler": _compiler_stamp()}, indent=2, sort_keys=True) + "\n"
    )
    return obj


def clear_cache() -> None:
    root = _cache_root()
    if root.exists():
        shutil.rmtree(root)


# ---------------------------------------------------------------------------
# normalisation helpers (golden comparison)
# ---------------------------------------------------------------------------
def normalize_command(command: Sequence[str]) -> list[str]:
    """Replace volatile paths so a command line can be compared with a golden."""

    out: list[str] = []
    for token in command:
        if token.endswith((".npuir", ".mlir")):
            out.append("{WORK}/" + Path(token).name)
        elif token.startswith("/tmp/") or token.startswith(tempfile.gettempdir()):
            out.append("{TMP}/" + Path(token).name)
        elif token.endswith((".o", ".so")) and os.sep in token:
            out.append("{WORK}/" + Path(token).name)
        else:
            out.append(token)
    return out
