# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Exercise wheel staging without importing TileLang or requiring NPU tools."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[2] / "tools/bundle_npuir.py"
SPEC = importlib.util.spec_from_file_location("bundle_npuir", SOURCE)
BUNDLER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUNDLER)


class BundleTest(unittest.TestCase):
    def test_missing_install_fails_before_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                BUNDLER.bundle(root / "install", root / "source", root / "stage")
            self.assertFalse((root / "stage").exists())

    def test_manifest_and_source_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            install, source, stage = root / "install", root / "source", root / "stage"
            files = [
                install / "bin/bishengir-compile",
                install / "bin/bishengir-opt",
                install / "lib/bishengir_mrgsort.aiv.bc",
                install / "python_packages/mlir_core/mlir/__init__.py",
                install / "python_packages/bishengir/bishengir/__init__.py",
                source / "LICENSE",
                source / "NOTICE",
            ]
            for path in files:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"test fixture\n")
            BUNDLER.bundle(install, source, stage)
            manifest = json.loads((stage / "lib/npuir/manifest.json").read_text())
            self.assertEqual(len(files), len(manifest))
            for name, digest in manifest.items():
                self.assertEqual(
                    hashlib.sha256((stage / name).read_bytes()).hexdigest(), digest
                )
            for path in files:
                self.assertEqual(b"test fixture\n", path.read_bytes())
            with self.assertRaises(FileExistsError):
                BUNDLER.bundle(install, source, stage)


if __name__ == "__main__":
    unittest.main()
