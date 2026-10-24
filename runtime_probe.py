import sys
from pathlib import Path

import torch
import torch_npu

sys.path.insert(0, str(Path(__file__).resolve().parent / "examples" / "elementwise"))
from vec_add_1d import vec_add  # noqa: E402

import tilelang  # noqa: E402

N = 4096
func = vec_add(N, N)
kernel = tilelang.compile(func, target="ascend910")

a = torch.randn(N, dtype=torch.float32).npu()
b = torch.randn(N, dtype=torch.float32).npu()
c = torch.zeros(N, dtype=torch.float32).npu()

kernel(a, b, c, N)
torch.npu.synchronize()
torch.testing.assert_close(a + b, c, rtol=1e-2, atol=1e-2)
print("RUNTIME_OK", float((a + b - c).abs().max()))
