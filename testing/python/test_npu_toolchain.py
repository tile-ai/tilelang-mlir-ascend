# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Tool discovery tests requiring neither torch_npu nor a device."""

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[2] / "tilelang/utils/npu_toolchain.py"
SPEC = importlib.util.spec_from_file_location("npu_toolchain", SOURCE)
TOOLCHAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOLCHAIN)


class ToolchainTest(unittest.TestCase):
    def test_override_is_authoritative(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"TILELANG_NPU_COMPILER_PATH": directory}):
                with patch.object(
                    TOOLCHAIN.shutil, "which", return_value="/old/compiler"
                ):
                    with self.assertRaises(EnvironmentError):
                        TOOLCHAIN.find_npuir_tool("bishengir-compile")
                executable = Path(directory) / "bin/bishengir-compile"
                executable.parent.mkdir()
                executable.touch(mode=0o755)
                self.assertEqual(
                    str(executable), TOOLCHAIN.find_npuir_tool("bishengir-compile")
                )

    def test_bundle_precedes_path(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "tilelang"
            executable = package / "lib/npuir/bin/bishengir-compile"
            executable.parent.mkdir(parents=True)
            executable.touch(mode=0o755)
            with patch.dict(os.environ, {}, clear=True):
                with patch.object(
                    TOOLCHAIN, "__file__", str(package / "utils/npu_toolchain.py")
                ):
                    with patch.object(
                        TOOLCHAIN.shutil, "which", return_value="/old/compiler"
                    ):
                        self.assertEqual(
                            str(executable.resolve()),
                            TOOLCHAIN.find_npuir_tool("bishengir-compile"),
                        )

    def test_path_fallback(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(
                TOOLCHAIN.shutil, "which", return_value="/tools/bishengir-opt"
            ):
                self.assertEqual(
                    "/tools/bishengir-opt", TOOLCHAIN.find_npuir_tool("bishengir-opt")
                )


if __name__ == "__main__":
    unittest.main()
