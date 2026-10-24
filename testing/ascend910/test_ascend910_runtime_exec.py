"""End-to-end A2 execution test (requires torch_npu and an NPU)."""

import sys
from pathlib import Path

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
if not torch.npu.is_available() or torch.npu.device_count() < 1:
    pytest.skip("no Ascend NPU available", allow_module_level=True)

import tilelang
import tilelang.language as T

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



_AUTOTUNE_N = 4096


def _autotune_ref(x, y):
    return x + y


def _autotune_supply(params):
    torch.manual_seed(0)
    return [torch.randn(_AUTOTUNE_N).npu(), torch.randn(_AUTOTUNE_N).npu()]


@tilelang.autotune(
    configs=[{"block": 1024}, {"block": 2048}],
    ref_prog=_autotune_ref,
    supply_prog=_autotune_supply,
    atol=1e-2,
    rtol=1e-2,
)
@tilelang.jit(out_idx=[-1], target="ascend910")
def _autotune_add(n, block):

    @T.prim_func
    def main(
        A: T.Tensor((n,), "float32"),
        B: T.Tensor((n,), "float32"),
        C: T.Tensor((n,), "float32"),
    ):
        with T.Kernel(n // block) as (cid, _):
            a = T.alloc_ub((block,), "float32")
            b = T.alloc_ub((block,), "float32")
            c = T.alloc_ub((block,), "float32")
            T.copy(A[cid * block : cid * block + block], a)
            T.copy(B[cid * block : cid * block + block], b)
            T.vadd(a, b, c)
            T.copy(c, C[cid * block : cid * block + block])

    return main


def test_autotune_vec_add():
    kernel = _autotune_add(_AUTOTUNE_N)
    result = kernel.get_tuner_result()
    assert result["config"]["block"] in (1024, 2048)
    assert result["latency"] > 0
