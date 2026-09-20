"""Non-NPU checks for the Stage 4 experiment runner template."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


TEMPLATE = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "tilelang-op-optimize"
    / "references"
    / "run_experiments_template.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location("stage4_runner_template", TEMPLATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_uses_only_inventory_tune_workloads(tmp_path, monkeypatch):
    runner = _load_runner()
    inventory = {
        "workloads": [
            {"kernel_id": "k1", "workload_id": "large", "kind": "tune"},
            {"kernel_id": "k1", "workload_id": "smoke-1m", "kind": "smoke"},
            {"kernel_id": "k2", "workload_id": "other", "kind": "tune"},
        ],
    }
    path = tmp_path / "workload_inventory.json"
    path.write_text(json.dumps(inventory), encoding="utf-8")
    monkeypatch.setattr(runner, "KERNEL_ID", "k1")
    monkeypatch.setattr(runner, "WORKLOAD_INVENTORY", str(path))
    assert runner.tune_workload_ids() == ["large"]
    assert "/" not in runner.safe_component("../smoke;rm -rf")
    assert runner.build_bench_cmd("kernel.py", "shape with spaces")[-1] == "shape with spaces"
    monkeypatch.setattr(sys, "argv", ["run_experiments.py", "--workload", "smoke-1m"])
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 2


def test_candidate_cannot_run_all_workloads(tmp_path, monkeypatch):
    runner = _load_runner()
    path = tmp_path / "workload_inventory.json"
    path.write_text(json.dumps({"workloads": [
        {"kernel_id": "k1", "workload_id": "shape-a", "kind": "tune"},
    ]}), encoding="utf-8")
    monkeypatch.setattr(runner, "KERNEL_ID", "k1")
    monkeypatch.setattr(runner, "WORKLOAD_INVENTORY", str(path))
    monkeypatch.setattr(sys, "argv", ["run_experiments.py", "--phase", "candidate", "--all"])
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 2
