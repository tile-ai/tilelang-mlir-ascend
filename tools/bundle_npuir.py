# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Stage an explicitly selected NPUIR install into a wheel build directory.

CANN and system runtime libraries remain external dependencies. This helper
does not copy files discovered through PATH or modify the input installation.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def bundle(install, source, destination):
    install, source, destination = map(Path, (install, source, destination))
    required = [
        install / "bin/bishengir-compile",
        install / "bin/bishengir-opt",
        install / "lib/bishengir_mrgsort.aiv.bc",
        source / "LICENSE",
        source / "NOTICE",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Incomplete NPUIR bundle: {path}")
    for name in ("mlir_core", "bishengir"):
        if not (install / "python_packages" / name).is_dir():
            raise FileNotFoundError(f"Missing NPUIR Python package: {name}")
    root = destination / "lib/npuir"
    if root.exists():
        raise FileExistsError(f"Use a fresh wheel staging directory: {root}")
    root.mkdir(parents=True)
    for name in ("bishengir-compile", "bishengir-opt"):
        (root / "bin").mkdir(exist_ok=True)
        shutil.copy2(install / "bin" / name, root / "bin" / name)
    (root / "lib").mkdir()
    for path in sorted((install / "lib").glob("*.bc")):
        shutil.copy2(path, root / "lib" / path.name)
    for name in ("LICENSE", "NOTICE"):
        shutil.copy2(source / name, root / name)
    for name in ("mlir_core", "bishengir"):
        target = destination / "lib/npuir_python" / name
        if target.exists():
            raise FileExistsError(f"Refusing mixed NPUIR Python packages: {target}")
        shutil.copytree(
            install / "python_packages" / name,
            target,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    entries = {}
    for directory in (root, destination / "lib/npuir_python"):
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                with path.open("rb") as stream:
                    digest = hashlib.sha256()
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                entries[str(path.relative_to(destination))] = digest.hexdigest()
    (root / "manifest.json").write_text(json.dumps(entries, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    bundle(args.install, args.source, args.destination)
