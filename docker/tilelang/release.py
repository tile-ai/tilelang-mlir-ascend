"""Fixed release inputs and installation helpers; no dispatch-time overrides."""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlsplit

CONFIG = Path(__file__).with_name("release.json")


def validate(config):
    errors = []
    for key, expected in {
        "tilelang_version": "0.1.15",
        "tilelang_commit": "a35f8ddf45eba16c21211ec8822d56ce5363036f",
        "cann_version": "9.3.0",
        "cann_package_version": "9.3.0~weekly.20260916.01",
    }.items():
        if config.get(key) != expected:
            errors.append(f"{key} must be {expected}")
    for key in ("cann_home", "cann_set_env", "bisheng_home", "cann_version_file"):
        if not config.get(key, "").startswith("/usr/local/Ascend/"):
            errors.append(f"{key}: supply the verified CANN 9.3.0 installation path")
    packages = config.get("cann_packages", [])
    names = [package["name"] for package in packages]
    if names[:2] != ["toolkit", "950-ops"] or len(names) != len(set(names)):
        errors.append(
            "cann_packages must begin with toolkit and 950-ops, with unique names"
        )
    for name in ("torch", "torch_npu"):
        if not re.fullmatch(
            r"[0-9][a-zA-Z0-9.+]*", config.get(name, {}).get("version", "")
        ):
            errors.append(f"{name}.version: supply the verified companion version")
    for package in [
        *packages,
        {"name": "torch", **config["torch"]},
        {"name": "torch_npu", **config["torch_npu"]},
    ]:
        name = package["name"]
        url = urlsplit(package.get("url", ""))
        if url.scheme != "https" or not url.netloc or url.username or url.password:
            errors.append(f"{name}.url: supply a CI-accessible HTTPS download URL")
        if not re.fullmatch(r"[0-9a-f]{64}", package.get("sha256", "")):
            errors.append(f"{name}.sha256: supply the package SHA256")
        suffix = ".whl" if name in ("torch", "torch_npu") else ".run"
        if not unquote(url.path).endswith(suffix):
            errors.append(f"{name}.url must identify a {suffix} package")
        if name in ("toolkit", "950-ops"):
            expected_name = (
                f"Ascend-cann-{name}_{config['cann_package_version']}_linux-x86_64.run"
            )
            if Path(unquote(url.path)).name != expected_name:
                errors.append(
                    f"{name}.url must identify the x86_64 package {expected_name}"
                )
    if errors:
        raise ValueError(
            "Release configuration is incomplete or invalid:\n- " + "\n- ".join(errors)
        )


def download(package, directory):
    filename = Path(unquote(urlsplit(package["url"]).path)).name
    destination = directory / filename
    subprocess.run(
        [
            "curl",
            "--fail",
            "--location",
            "--show-error",
            "--silent",
            "--retry",
            "3",
            "--connect-timeout",
            "30",
            "--max-time",
            "1800",
            "--proto",
            "=https",
            "--proto-redir",
            "=https",
            "--referer",
            "https://www.hiascend.com/",
            "--output",
            str(destination),
            package["url"],
        ],
        check=True,
    )
    digest = hashlib.sha256()
    with destination.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != package["sha256"]:
        raise ValueError(f"SHA256 mismatch: {filename}")
    return destination


def check_cann(config):
    metadata = Path(config["cann_version_file"]).read_text()
    version = (
        "(?:"
        + "|".join(
            re.escape(config[key]) for key in ("cann_version", "cann_package_version")
        )
        + ")"
    )
    if not re.search(
        rf"(?m)^\s*(?:version|Version|CANN_VERSION)\s*[:=]\s*{version}\s*$", metadata
    ):
        raise ValueError(
            "CANN installation metadata does not identify the configured 9.3.0 release"
        )
    for path in (
        config["cann_set_env"],
        f"{config['bisheng_home']}/bin/bisheng",
        f"{config['bisheng_home']}/bin/ld.lld",
    ):
        if not Path(path).is_file():
            raise ValueError(f"Missing CANN toolchain file: {path}")


def install_cann(config):
    # List order is installation order. Append required companion .run packages
    # after the toolkit and 950 ops once the release bundle is available.
    with tempfile.TemporaryDirectory() as directory:
        for index, package in enumerate(config["cann_packages"]):
            installer = download(package, Path(directory))
            if index == 0:
                command = [
                    "bash",
                    str(installer),
                    "--quiet",
                    "--install",
                    "--install-for-all",
                ]
            else:
                command = [
                    "bash",
                    "-c",
                    'source "$1" && bash "$2" --quiet --install --install-for-all',
                    "--",
                    config["cann_set_env"],
                    str(installer),
                ]
            subprocess.run(command, check=True)
            installer.unlink()
    check_cann(config)
    Path("/opt/cann").symlink_to(config["cann_home"], target_is_directory=True)
    Path("/opt/bisheng").symlink_to(config["bisheng_home"], target_is_directory=True)
    Path("/opt/tilelang-image/cann-env.sh").symlink_to(config["cann_set_env"])


def install_torch(config):
    constraints = Path("/opt/tilelang-image/constraints.txt")
    constraints.write_text(
        "".join(
            f"{name}=={config[name]['version']}\n" for name in ("torch", "torch_npu")
        )
    )
    with tempfile.TemporaryDirectory() as directory:
        wheels = [
            str(download(config[name], Path(directory)))
            for name in ("torch", "torch_npu")
        ]
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--constraint",
                str(constraints),
                *wheels,
            ],
            check=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("validate", "install-cann", "install-torch")
    )
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    try:
        validate(config)
        if args.command == "install-cann":
            install_cann(config)
        elif args.command == "install-torch":
            install_torch(config)
        else:
            print("Release configuration is complete")
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
