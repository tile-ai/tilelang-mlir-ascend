#!/usr/bin/env python3
# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Prepare TileLang's private TVM Python package and prebuilt libraries."""

import argparse
import ast
import hashlib
from pathlib import Path
import shutil
import subprocess


def rewrite_imports(source):
    """Rewrite absolute TVM imports without changing unrelated source or strings."""
    tree = ast.parse(source)
    lines = source.encode("utf-8").splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
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
                # Only replace the module token, preserving multiline import lists/comments.
                start = offsets[node.lineno - 1] + node.col_offset
                end = offsets[node.end_lineno - 1] + node.end_col_offset
                original = source.encode("utf-8")[start:end].decode("utf-8")
                replacement = original.replace(
                    node.module, "tilelang." + node.module, 1
                )
        if replacement is not None:
            start = offsets[node.lineno - 1] + node.col_offset
            end = offsets[node.end_lineno - 1] + node.end_col_offset
            original = source.encode("utf-8")[start:end]
            replacement += "\n" * max(
                0, original.count(b"\n") - replacement.count("\n")
            )
            edits.append((start, end, replacement.encode("utf-8")))
    result = source.encode("utf-8")
    for start, end, replacement in sorted(edits, reverse=True):
        result = result[:start] + replacement + result[end:]
    return result.decode("utf-8")


def prepare_python(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents:
        raise ValueError(
            "Private TVM output must be outside the upstream Python package"
        )
    files = sorted(
        p
        for p in source.rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and p.suffix not in (".pyc", ".so")
    )
    digest = hashlib.sha256(Path(__file__).read_bytes())
    for path in files:
        digest.update(str(path.relative_to(source)).encode())
        digest.update(path.read_bytes())
    signature = digest.hexdigest()
    stamp = output / ".tilelang-vendor"
    if stamp.is_file() and stamp.read_text() == signature:
        return
    if not (source / "_ffi/base.py").is_file():
        raise RuntimeError(f"Missing TVM Python source at {source}")
    if output.exists():
        if not stamp.is_file():
            raise RuntimeError(f"Refusing to replace an unmarked directory: {output}")
        shutil.rmtree(output)
    for path in files:
        target = output / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix != ".py":
            shutil.copy2(path, target)
            continue
        text = path.read_text(encoding="utf-8")
        if path.relative_to(source).as_posix() == "_ffi/base.py":
            start = text.index("def _load_lib():")
            end = text.index("\n\ntry:", start)
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
            text = text.replace(
                '_FFI_MODE = os.environ.get("TVM_FFI", "auto")',
                '_FFI_MODE = "ctypes"  # Private bindings never use public TVM extensions.',
            )
        if path.relative_to(source).as_posix() == "_ffi/libinfo.py":
            start = text.index("def get_dll_directories():")
            end = text.index("\ndef find_lib_path(", start)
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
        if path.relative_to(source).as_posix() == "_ffi/registry.py":
            text = text.replace(
                "    module = sys.modules[module_name]",
                '    if module_name == "tvm" or module_name.startswith("tvm."):\n'
                '        module_name = "tilelang." + module_name\n'
                "    module = sys.modules[module_name]",
            )
        target.write_text(rewrite_imports(text), encoding="utf-8")
    stamp.write_text(signature)


def check_library(path):
    symbols = subprocess.check_output(
        ["readelf", "--dyn-syms", "--wide", str(path)], text=True
    )
    if any(" UNIQUE " in line for line in symbols.splitlines()):
        raise RuntimeError(
            f"{path} contains GNU-unique symbols; rebuild TVM and TileLang "
            "with -fno-gnu-unique (GCC). Old prebuilds cannot be used for isolation."
        )


def stage_libraries(source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("tvm", "tvm_runtime"):
        original = source / f"lib{name}.so"
        check_library(original)
        target = output / f"libtilelang_{name}.so"
        shutil.copy2(original, target)
        subprocess.run(
            ["patchelf", "--set-soname", target.name, str(target)], check=True
        )
        subprocess.run(["patchelf", "--set-rpath", "$ORIGIN", str(target)], check=True)


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
