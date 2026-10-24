import torch
import torch_npu

import tilelang
import tilelang.language as T

N = 4096


def ref_prog(x, y):
    return x + y


def supply_prog(params):
    torch.manual_seed(0)
    return [torch.randn(N).npu(), torch.randn(N).npu()]


@tilelang.autotune(
    configs=[{"block": 1024}, {"block": 2048}],
    ref_prog=ref_prog,
    supply_prog=supply_prog,
    atol=1e-2,
    rtol=1e-2,
)
@tilelang.jit(out_idx=[-1], target="ascend910")
def add_kernel(N, block):
    @T.prim_func
    def main(
        A: T.Tensor((N,), "float32"),
        B: T.Tensor((N,), "float32"),
        C: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(N // block) as (cid, _):
            a = T.alloc_ub((block,), "float32")
            b = T.alloc_ub((block,), "float32")
            c = T.alloc_ub((block,), "float32")
            T.copy(A[cid * block : cid * block + block], a)
            T.copy(B[cid * block : cid * block + block], b)
            T.vadd(a, b, c)
            T.copy(c, C[cid * block : cid * block + block])

    return main


kernel = add_kernel(N)
print("AUTOTUNE_OK", kernel.get_tuner_result())
