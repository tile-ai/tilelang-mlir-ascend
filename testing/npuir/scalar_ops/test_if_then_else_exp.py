# Copyright (c) Huawei Technologies Co., Ltd. 2026.
import pytest
import torch

import tilelang
import tilelang.language as T

pytestmark = [pytest.mark.op("if_then_else"), pytest.mark.mode("Expert")]


@tilelang.jit(target="npuir")
def conditional_scalar():
    @T.prim_func
    def main(A: T.Tensor((2,), "int32"), B: T.Tensor((2, 8), "int32")):
        with T.Kernel(2, is_npu=True) as (cid, _):
            out = T.alloc_ub((8,), "int32")
            value = T.if_then_else(cid == 0, A[0], A[1] + 7)
            T.vbrc(value, out)
            T.copy(out, B[cid, :])

    return main


def test_conditional_scalar():
    kernel = conditional_scalar()
    a = torch.tensor([11, 23], dtype=torch.int32, device="npu")
    b = torch.empty((2, 8), dtype=torch.int32, device="npu")
    kernel(a, b)
    expected = torch.tensor([[11] * 8, [30] * 8], dtype=torch.int32)
    torch.testing.assert_close(b.cpu(), expected, rtol=0, atol=0)
