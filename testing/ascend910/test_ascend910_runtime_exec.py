"""End-to-end A2 execution test (requires torch_npu and an NPU)."""

import sys
from pathlib import Path

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
if not torch.npu.is_available() or torch.npu.device_count() < 1:
    pytest.skip("no Ascend NPU available", allow_module_level=True)

import tilelang

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "elementwise"))
from vec_add_1d import vec_add  # noqa: E402


def test_vec_add_executes_on_npu():
    n = 4096
    func = vec_add(n, n)
    kernel = tilelang.compile(func, target="ascend910")

    a = torch.randn(n, dtype=torch.float32).npu()
    b = torch.randn(n, dtype=torch.float32).npu()
    c = torch.zeros(n, dtype=torch.float32).npu()

    kernel(a, b, c, n)
    torch.npu.synchronize()

    torch.testing.assert_close(a + b, c, rtol=1e-2, atol=1e-2)


def test_vec_add_profiler():
    n = 4096
    func = vec_add(n, n)
    kernel = tilelang.compile(func, target="ascend910")
    profiler = kernel.get_profiler()
    latency = profiler.do_bench(warmup=10, rep=20, backend="event")
    assert latency >= 0
