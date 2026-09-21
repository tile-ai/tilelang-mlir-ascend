# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Give a wheel's zstd dependency a private SONAME to avoid Python conflicts."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def isolate(package, library, license_file):
    package, library, license_file = map(Path, (package, library, license_file))
    patchelf = shutil.which("patchelf")
    if not patchelf:
        raise RuntimeError("patchelf is required for wheel dependency isolation")
    if not library.is_file() or not license_file.is_file():
        raise FileNotFoundError(
            "Both the selected zstd library and its license are required"
        )
    name = f"libtilelang-zstd-{digest(library)[:12]}.so.1"
    target = package / "lib" / name
    if target.exists():
        raise FileExistsError(f"Use a fresh wheel staging directory: {target}")
    shutil.copy2(library, target)
    subprocess.run([patchelf, "--set-soname", name, str(target)], check=True)
    licenses = package / "lib/licenses"
    licenses.mkdir(exist_ok=True)
    notice = licenses / "zstd-LICENSE"
    shutil.copy2(license_file, notice)
    changed = [target, notice]
    for path in sorted(package.rglob("*")):
        if not path.is_file():
            continue
        with path.open("rb") as stream:
            if stream.read(4) != b"\x7fELF":
                continue
        needed = subprocess.check_output(
            [patchelf, "--print-needed", str(path)], text=True
        )
        if "libzstd.so.1" not in needed.splitlines():
            continue
        old = subprocess.check_output(
            [patchelf, "--print-rpath", str(path)], text=True
        ).strip()
        relative = "$ORIGIN/" + os.path.relpath(target.parent, path.parent)
        subprocess.run(
            [
                patchelf,
                "--replace-needed",
                "libzstd.so.1",
                name,
                "--set-rpath",
                relative + (":" + old if old else ""),
                str(path),
            ],
            check=True,
        )
        changed.append(path)
    # Refresh hashes if dependency repair changed files in the NPUIR manifest.
    manifest = package / "lib/npuir/manifest.json"
    if manifest.is_file():
        entries = json.loads(manifest.read_text())
        for key in entries:
            entries[key] = digest(package / key)
        manifest.write_text(json.dumps(entries, indent=2) + "\n")
    entries = {str(path.relative_to(package)): digest(path) for path in changed}
    (package / "lib/dependency-manifest.json").write_text(
        json.dumps(entries, indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--license", required=True)
    args = parser.parse_args()
    isolate(args.package, args.library, args.license)
