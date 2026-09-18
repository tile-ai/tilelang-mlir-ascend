"""Collect report metadata and the runtime environment without extra dependencies."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _npu_info() -> str | None:
    try:
        import torch
        import torch_npu  # noqa: F401

        count = torch.npu.device_count()
        if count <= 0:
            return None
        return f"{torch.npu.get_device_name(0)} x {count}"
    except Exception:
        return None


def _version_from_file(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    for pattern in (
        r"(?im)^\s*(?:Version|version)\s*=\s*([^\s]+)",
        r"(?im)^\s*(?:package_version|Driver_Version)\s*=\s*([^\s]+)",
    ):
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return None


def _cann_version() -> str | None:
    toolkit = os.environ.get("ASCEND_TOOLKIT_HOME")
    paths = []
    if toolkit:
        root = Path(toolkit)
        paths.extend(
            (
                root / "compiler/version.info",
                root / "version.info",
                root / "version.cfg",
            )
        )
    paths.extend(
        (
            Path("/usr/local/Ascend/ascend-toolkit/latest/version.cfg"),
            Path("/usr/local/Ascend/ascend-toolkit/version.cfg"),
            Path("/usr/local/Ascend/latest/version.cfg"),
        )
    )
    for path in paths:
        version = _version_from_file(path)
        if version:
            return version
    if toolkit:
        match = re.search(r"cann-([\w.-]+)", toolkit)
        if match:
            return match.group(1)
    return None


def _driver_version() -> str | None:
    explicit = os.environ.get("ASCEND_DRIVER_VERSION")
    if explicit:
        return explicit
    for path in (
        Path("/usr/local/Ascend/driver/version.info"),
        Path("/etc/ascend_install.info"),
    ):
        version = _version_from_file(path)
        if version:
            return version
    return None


def _container_info() -> str | None:
    image = os.environ.get("TILEOPS_IMAGE") or os.environ.get("CANN_BENCH_IMAGE")
    if image:
        return image
    if Path("/.dockerenv").exists():
        return "container"
    try:
        cgroup = Path("/proc/1/cgroup").read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    markers = ("docker", "containerd", "kubepods")
    return "container" if any(key in cgroup for key in markers) else None


def collect_setup_info(
    *,
    scope: str,
    run_id: str,
    generated_at: str,
    profiler: str,
    git_commit: str | None,
    test_target: str,
    benchmark_target: str,
) -> dict[str, dict[str, Any]]:
    """Return the CANN-Bench-style metadata and environment sections."""
    return {
        "metadata": {
            "framework": f"TileOPs Reporting {_package_version('TileOPs') or '0.1.0'}",
            "date": generated_at or datetime.now().astimezone().isoformat(),
            "evaluation_scope": scope,
            "benchmark": "TileOPs pytest operator suite",
            "profiler": profiler,
            "run_id": run_id,
            "git_commit": git_commit,
            "correctness_target": test_target,
            "benchmark_target": benchmark_target,
        },
        "environment": {
            "npu": _npu_info(),
            "cpu": platform.machine() or "unknown",
            "cann": _cann_version(),
            "driver": _driver_version(),
            "pytorch": _package_version("torch"),
            "pytorch_npu": _package_version("torch-npu"),
            "tilelang": _package_version("tilelang"),
            "python": sys.version.split()[0] if sys.version else "unknown",
            "os": platform.platform() or f"{platform.system()} {platform.release()}",
            "container": _container_info(),
        },
    }


__all__ = ["collect_setup_info"]
