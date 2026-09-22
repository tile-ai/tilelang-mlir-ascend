"""Mode synchronization tests for the TileOPs Stage 5 integration script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "examples/TileOPs/.agents/skills/add-npu-op/scripts/integrate_kernel.py"
)


def _run(
    tmp_path: Path, mode: str
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    tileops_root = tmp_path / "TileOPs"
    wrapper = tileops_root / "tileops/kernels/reduction/mop/mop.py"
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    if not wrapper.exists():
        wrapper.write_text(
            "from ._f import f\n\n"
            "class MopKernel(Kernel):\n"
            '    """Test wrapper."""\n'
            "    pass\n",
            encoding="utf-8",
        )
    source = tmp_path / "examples/mop/f/f.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        f'ASCEND_MODE = "{mode}"\n\ndef f():\n    pass\n', encoding="utf-8"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--op-slug",
            "mop",
            "--family",
            "reduction",
            "--functions",
            "f",
            "--extracted-module",
            "_f",
            "--wrapper",
            str(wrapper),
            "--tileops-root",
            str(tileops_root),
            "--examples-root",
            str(tmp_path / "examples"),
            "--skip-smoke",
        ],
        capture_output=True,
        text=True,
    )
    return result, wrapper, source


def test_mode_is_inserted_and_updated_idempotently(tmp_path):
    result, wrapper, _ = _run(tmp_path, "Expert")
    assert result.returncode == 0, result.stderr
    assert 'ascend_mode = "Expert"' in wrapper.read_text(encoding="utf-8")

    result, wrapper, _ = _run(tmp_path, "Expert")
    assert result.returncode == 0, result.stderr
    assert wrapper.read_text(encoding="utf-8").count("ascend_mode =") == 1
    assert "[mode]" in result.stdout and "unchanged" in result.stdout

    result, wrapper, _ = _run(tmp_path, "Developer")
    assert result.returncode == 0, result.stderr
    text = wrapper.read_text(encoding="utf-8")
    assert 'ascend_mode = "Developer"' in text
    assert text.count("ascend_mode =") == 1


def test_unknown_mode_fails_before_integration(tmp_path):
    result, wrapper, source = _run(tmp_path, "Unknown")
    assert result.returncode != 0
    assert "missing or conflicting ASCEND_MODE" in result.stderr
    assert "ascend_mode" not in wrapper.read_text(encoding="utf-8")
    assert not (wrapper.parent / "mop_kernel" / source.name).exists()


def test_existing_literal_setdefault_is_accepted_but_conflict_is_rejected(tmp_path):
    result, wrapper, source = _run(tmp_path, "Expert")
    assert result.returncode == 0, result.stderr
    source.write_text(
        'import os\nos.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")\n'
        "\ndef f():\n    pass\n",
        encoding="utf-8",
    )
    result, wrapper, _ = _run_with_existing_source(tmp_path, wrapper)
    assert result.returncode == 0, result.stderr
    assert 'ascend_mode = "Developer"' in wrapper.read_text(encoding="utf-8")

    source.write_text(
        'ASCEND_MODE = "Expert"\nimport os\n'
        'os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")\n'
        "\ndef f():\n    pass\n",
        encoding="utf-8",
    )
    result, _, _ = _run_with_existing_source(tmp_path, wrapper)
    assert result.returncode != 0
    assert "missing or conflicting ASCEND_MODE" in result.stderr


def _run_with_existing_source(tmp_path: Path, wrapper: Path):
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--op-slug",
            "mop",
            "--family",
            "reduction",
            "--functions",
            "f",
            "--extracted-module",
            "_f",
            "--wrapper",
            str(wrapper),
            "--tileops-root",
            str(tmp_path / "TileOPs"),
            "--examples-root",
            str(tmp_path / "examples"),
            "--skip-smoke",
        ],
        capture_output=True,
        text=True,
    )
    return result, wrapper, tmp_path / "examples/mop/f/f.py"
