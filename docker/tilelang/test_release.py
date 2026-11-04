"""Exercise release blocking and artifact integrity without network or CANN."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import release


def complete_config():
    config = json.loads(release.CONFIG.read_text())
    config["bisheng_home"] = "/usr/local/Ascend/test-compiler"
    config["cann_version_file"] = "/usr/local/Ascend/test-version.info"
    for package in config["cann_packages"]:
        package.update(
            url=f"https://example.invalid/Ascend-cann-{package['name']}_{config['cann_package_version']}_linux-x86_64.run",
            sha256="a" * 64,
        )
    for name in ("torch", "torch_npu"):
        config[name].update(
            version="1.0.0", url=f"https://example.invalid/{name}.whl", sha256="b" * 64
        )
    return config


class ReleaseTest(unittest.TestCase):
    def test_complete_config(self):
        release.validate(complete_config())

    def test_incomplete_config_cannot_publish(self):
        config = complete_config()
        config["torch_npu"]["url"] = ""
        with self.assertRaisesRegex(ValueError, "torch_npu.url"):
            release.validate(config)

    def test_reject_wrong_versions_insecure_urls_and_invalid_hashes(self):
        baseline = complete_config()
        for key, value in (("cann_version", "9.2.0"), ("tilelang_commit", "main")):
            with self.subTest(key=key):
                config = copy.deepcopy(baseline)
                config[key] = value
                with self.assertRaises(ValueError):
                    release.validate(config)
        for key, value in (
            ("url", "http://example.invalid/toolkit.run"),
            ("sha256", "unknown"),
        ):
            with self.subTest(key=key):
                config = copy.deepcopy(baseline)
                config["cann_packages"][0][key] = value
                with self.assertRaises(ValueError):
                    release.validate(config)

    def test_reject_ops_before_toolkit(self):
        config = complete_config()
        config["cann_packages"].reverse()
        with self.assertRaisesRegex(ValueError, "toolkit and 950-ops"):
            release.validate(config)

    def test_reject_arm_package_and_other_weekly_build(self):
        for old, new in (("x86_64", "aarch64"), ("20260916.01", "20260917.01")):
            with self.subTest(replacement=new):
                config = complete_config()
                config["cann_packages"][0]["url"] = config["cann_packages"][0][
                    "url"
                ].replace(old, new)
                with self.assertRaisesRegex(
                    ValueError, "must identify the x86_64 package"
                ):
                    release.validate(config)

    def test_download_integrity(self):
        payload = b"test installer bytes"
        package = {
            "url": "https://example.invalid/toolkit.run",
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

        def fake_curl(command, **kwargs):
            Path(command[command.index("--output") + 1]).write_bytes(payload)

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("release.subprocess.run", side_effect=fake_curl),
        ):
            self.assertEqual(
                release.download(package, Path(directory)).read_bytes(), payload
            )
            package["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                release.download(package, Path(directory))


if __name__ == "__main__":
    unittest.main()
