"""Validation and installation tests for the three-layer TileLang Ascend knowledge base."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
KB_ROOT = ROOT / "assets" / "skills" / "tilelang-ascend-knowledge"
VALIDATOR = KB_ROOT / "scripts" / "validate_kb.py"

sys.path.insert(0, str(KB_ROOT / "scripts"))
import validate_kb as kb  # noqa: E402


class KnowledgeBaseTest(unittest.TestCase):
    def test_repo_knowledge_base_passes_validation(self):
        result = kb.validate_kb(KB_ROOT)
        self.assertTrue(result["ok"], result["errors"])
        self.assertGreaterEqual(result["cards"], 18)
        self.assertEqual(result["errors"], [])

    def test_validator_cli_runs_cleanly(self):
        proc = subprocess.run(
            [sys.executable, str(VALIDATOR), "--json"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
        payload = json.loads(proc.stdout)
        self.assertTrue(payload["ok"])
        self.assertGreaterEqual(payload["cards"], 18)

    def test_detects_orphan_card_and_broken_link(self):
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "kb"
            shutil.copytree(KB_ROOT, copy)
            orphan = copy / "patterns" / "unregistered-pattern.md"
            orphan.write_text(
                "---\n"
                "id: pattern/unregistered\n"
                "layer: pattern\n"
                "arch: [Ascend950]\n"
                "kind: hw-pattern\n"
                "status: active\n"
                "verified_scope: tilelang-ascend\n"
                "---\n\n"
                "See [missing](../foundations/does-not-exist.md).\n",
                encoding="utf-8",
            )
            result = kb.validate_kb(copy)
            self.assertFalse(result["ok"])
            joined = "\n".join(result["errors"])
            self.assertIn("orphan card", joined)
            self.assertIn("broken relative link", joined)

    def test_detects_private_path_leak_and_bad_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "kb"
            shutil.copytree(KB_ROOT, copy)
            card = copy / "patterns" / "softmax-col-fp32.md"
            card.write_text(
                "---\n"
                "id: pattern/softmax-col-fp32\n"
                "layer: family\n"
                "arch: []\n"
                "kind: invalid-kind\n"
                "status: active\n"
                "verified_scope: tilelang-ascend\n"
                "---\n\n"
                "Leaked /home/alice/private/run.log\n",
                encoding="utf-8",
            )
            result = kb.validate_kb(copy)
            self.assertFalse(result["ok"])
            joined = "\n".join(result["errors"])
            self.assertIn("private host path", joined)
            self.assertIn("does not match directory", joined)
            self.assertIn("invalid kind", joined)
            self.assertIn("arch list must not be empty", joined)

    def test_installed_copy_validates_standalone(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            proc = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "install_codex_skills.py"), "--workspace", str(workspace)],
                cwd=workspace,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            installed_kb = workspace / ".agents" / "skills" / "tilelang-ascend-knowledge"
            self.assertTrue((installed_kb / "INDEX.md").is_file())
            check = subprocess.run(
                [sys.executable, str(installed_kb / "scripts" / "validate_kb.py")],
                cwd=workspace,
                capture_output=True,
                text=True,
            )
            self.assertEqual(check.returncode, 0, check.stderr + check.stdout)


if __name__ == "__main__":
    unittest.main()
