# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Prepare TileLang's private TVM Python package and native libraries."""

import argparse
import ast
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


PRIVATE_PACKAGE = "tilelang.tvm"
PRIVATE_LIBRARY_NAMES = {
    "libtvm.so": "libtilelang_tvm.so",
    "libtvm_runtime.so": "libtilelang_tvm_runtime.so",
}


def _source_offsets(source):
    lines = source.encode("utf-8").splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    return offsets


def _node_range(node, offsets):
    return (
        offsets[node.lineno - 1] + node.col_offset,
        offsets[node.end_lineno - 1] + node.end_col_offset,
    )


def find_public_imports(source):
    """Return line numbers for imports that would bind the public ``tvm`` package."""
    findings = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            if any(
                name.name == "tvm" or name.name.startswith("tvm.")
                for name in node.names
            ):
                findings.append(node.lineno)
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module
            and (node.module == "tvm" or node.module.startswith("tvm."))
        ):
            findings.append(node.lineno)
    return sorted(findings)


def rewrite_imports(source):
    """Rewrite absolute TVM imports while preserving local import bindings.

    The rewrite is AST-guided so comments and multiline imports are retained.
    String constants such as TVM global-function registration names are not
    modified.
    """
    tree = ast.parse(source)
    offsets = _source_offsets(source)
    encoded = source.encode("utf-8")
    edits = []

    for node in ast.walk(tree):
        replacement = None
        if isinstance(node, ast.Import):
            imports = []
            changed = False
            for alias in node.names:
                if alias.name == "tvm" or alias.name.startswith("tvm."):
                    changed = True
                    if alias.asname:
                        imports.append(
                            f"import tilelang.{alias.name} as {alias.asname}"
                        )
                    elif alias.name == "tvm":
                        imports.append("from tilelang import tvm")
                    else:
                        imports.extend(
                            [
                                f"import tilelang.{alias.name}",
                                "from tilelang import tvm",
                            ]
                        )
                else:
                    imports.append(
                        "import "
                        + alias.name
                        + (f" as {alias.asname}" if alias.asname else "")
                    )
            if changed:
                replacement = "; ".join(imports)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module == "tvm" or node.module.startswith("tvm."):
                start, end = _node_range(node, offsets)
                original = encoded[start:end].decode("utf-8")
                replacement = original.replace(
                    node.module, "tilelang." + node.module, 1
                )

        if replacement is not None:
            start, end = _node_range(node, offsets)
            original = encoded[start:end]
            replacement += "\n" * max(
                0, original.count(b"\n") - replacement.count("\n")
            )
            edits.append((start, end, replacement.encode("utf-8")))

    result = encoded
    for start, end, replacement in sorted(edits, reverse=True):
        result = result[:start] + replacement + result[end:]
    rewritten = result.decode("utf-8")
    remaining = find_public_imports(rewritten)
    if remaining:
        raise RuntimeError(
            f"TVM import rewrite left public imports at lines {remaining}"
        )
    return rewritten


def _replace_required(text, old, new, relative_path):
    if old not in text:
        raise RuntimeError(
            f"Unsupported TVM source layout: expected marker not found in {relative_path}: {old!r}"
        )
    return text.replace(old, new, 1)


def _patch_ffi_source(text, relative_path):
    if relative_path == "_ffi/base.py":
        start = text.find("def _load_lib():")
        end = text.find("\n\ntry:", start)
        if start < 0 or end < 0:
            raise RuntimeError("Unsupported TVM _ffi/base.py: cannot locate _load_lib")
        text = (
            text[:start]
            + """def _load_lib():
    from tilelang._tvm_lib import load_tvm
    lib = load_tvm()
    lib.TVMGetLastError.restype = ctypes.c_char_p
    return lib, "libtilelang_tvm.so"
"""
            + text[end:]
        )
        text = _replace_required(
            text,
            '_FFI_MODE = os.environ.get("TVM_FFI", "auto")',
            '_FFI_MODE = "ctypes"  # Private bindings never use public TVM extensions.',
            relative_path,
        )
    elif relative_path == "_ffi/libinfo.py":
        start = text.find("def get_dll_directories():")
        end = text.find("\ndef find_lib_path(", start)
        if start < 0 or end < 0:
            raise RuntimeError(
                "Unsupported TVM _ffi/libinfo.py: cannot locate library search"
            )
        text = (
            text[:start]
            + """def get_dll_directories():
    from tilelang._tvm_lib import library_directories
    return [str(path) for path in library_directories() if path.is_dir()]

"""
            + text[end:]
        )
        text = text.replace('"libtvm.so"', '"libtilelang_tvm.so"')
        text = text.replace('"libtvm_runtime.so"', '"libtilelang_tvm_runtime.so"')
        text = text.replace('os.environ.get("TVM_USE_RUNTIME_LIB", False)', "False")
    elif relative_path == "_ffi/registry.py":
        text = _replace_required(
            text,
            "    module = sys.modules[module_name]",
            '    if module_name == "tvm" or module_name.startswith("tvm."):\n'
            '        module_name = "tilelang." + module_name\n'
            "    module = sys.modules[module_name]",
            relative_path,
        )
    return text


def _source_signature(source, files):
    digest = hashlib.sha256(Path(__file__).read_bytes())
    for path in files:
        digest.update(path.relative_to(source).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


@contextmanager
def _generation_lock(lock_path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as handle:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except ImportError:
            # The project only loads private libraries on Linux. This fallback
            # keeps source-transformation unit tests portable.
            pass
        yield


def prepare_python(source, output):
    """Copy and rewrite the vendored TVM package without modifying the submodule."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents:
        raise ValueError(
            "Private TVM output must be outside the upstream Python package"
        )
    required = [
        source / "_ffi/base.py",
        source / "_ffi/libinfo.py",
        source / "_ffi/registry.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"Missing or incompatible TVM Python source: {', '.join(missing)}"
        )

    files = sorted(
        path
        for path in source.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in (".pyc", ".so")
    )
    signature = _source_signature(source, files)
    stamp_name = ".tilelang-vendor"
    stamp = output / stamp_name
    lock = output.parent / ".tilelang-vendor.lock"

    with _generation_lock(lock):
        if stamp.is_file() and stamp.read_text(encoding="utf-8") == signature:
            return
        if output.exists() and not stamp.is_file():
            raise RuntimeError(f"Refusing to replace an unmarked directory: {output}")

        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
        try:
            for path in files:
                relative = path.relative_to(source)
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if path.suffix != ".py":
                    shutil.copy2(path, target)
                    continue
                text = path.read_text(encoding="utf-8")
                text = _patch_ffi_source(text, relative.as_posix())
                target.write_text(rewrite_imports(text), encoding="utf-8")
                shutil.copystat(path, target)
            (staging / stamp_name).write_text(signature, encoding="utf-8")
            if output.exists():
                shutil.rmtree(output)
            os.replace(staging, output)
        finally:
            if staging.exists():
                shutil.rmtree(staging)


def _tool(name):
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"Required tool is not installed: {name}")
    return path


def check_library(path):
    """Reject a library whose GNU-unique symbols would cross the TVM boundary."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    symbols = subprocess.check_output(
        [_tool("readelf"), "--dyn-syms", "--wide", str(path)], text=True
    )
    if any(" UNIQUE " in line for line in symbols.splitlines()):
        raise RuntimeError(
            f"{path} contains GNU-unique symbols; rebuild TVM and TileLang "
            "with -fno-gnu-unique (GCC). Renaming an old prebuild is not safe."
        )

    if path.name in PRIVATE_LIBRARY_NAMES.values():
        _check_private_metadata(path, path.name)


def _dynamic_metadata(path):
    return subprocess.check_output([_tool("readelf"), "-d", str(path)], text=True)


def _check_private_metadata(path, expected_name):
    dynamic = _dynamic_metadata(path)
    sonames = re.findall(r"\(SONAME\).*Library soname: \[([^]]+)\]", dynamic)
    if sonames != [expected_name]:
        raise RuntimeError(
            f"{path} must have SONAME {expected_name!r}, found {sonames or 'none'}; "
            "stage it with tools/prepare_tvm.py libraries instead of renaming it"
        )
    needed = set(re.findall(r"\(NEEDED\).*Shared library: \[([^]]+)\]", dynamic))
    public_needed = needed.intersection(PRIVATE_LIBRARY_NAMES)
    if public_needed:
        raise RuntimeError(
            f"{path} still depends on public TVM libraries: {sorted(public_needed)}"
        )


def stage_libraries(source, output):
    """Copy TVM DSOs under private names and repair their dynamic metadata."""
    source, output = Path(source), Path(output)
    originals = {name: source / name for name in PRIVATE_LIBRARY_NAMES}
    for original in originals.values():
        check_library(original)
    output.mkdir(parents=True, exist_ok=True)
    patchelf = _tool("patchelf")
    for public_name, private_name in PRIVATE_LIBRARY_NAMES.items():
        original = originals[public_name]
        fd, temporary_name = tempfile.mkstemp(prefix=f".{private_name}-", dir=output)
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            shutil.copy2(original, temporary)
            dynamic = _dynamic_metadata(temporary)
            needed = set(
                re.findall(r"\(NEEDED\).*Shared library: \[([^]]+)\]", dynamic)
            )
            if "libtvm_runtime.so" in needed:
                subprocess.run(
                    [
                        patchelf,
                        "--replace-needed",
                        "libtvm_runtime.so",
                        "libtilelang_tvm_runtime.so",
                        str(temporary),
                    ],
                    check=True,
                )
            subprocess.run(
                [patchelf, "--set-soname", private_name, str(temporary)], check=True
            )
            subprocess.run(
                [patchelf, "--set-rpath", "$ORIGIN", str(temporary)], check=True
            )
            _check_private_metadata(temporary, private_name)
            os.replace(temporary, output / private_name)
        finally:
            if temporary.exists():
                temporary.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("python", "libraries", "check"))
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    args = parser.parse_args()
    if args.mode == "check":
        check_library(args.source)
    else:
        if args.output is None:
            parser.error("output is required")
        (prepare_python if args.mode == "python" else stage_libraries)(
            args.source, args.output
        )
