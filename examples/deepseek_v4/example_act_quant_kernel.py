# Copyright (c) Huawei Technologies Co., Ltd. 2026.
import os
import tilelang
import tilelang.language as T
import torch
from typing import Tuple

tilelang.cache.clear_cache()

FP16 = "float16"
BF16 = "bfloat16"
FP32 = "float32"
INT8 = "int8"


@tilelang.jit(target="npuir")
def act_quant_kernel(
    N: int,
    block_M: int = 32,
    block_N: int = 32,
    inplace: bool = False,
):
    M = T.symbolic("M")

    int8_abs_max = 127.0
    # inplace True  => output bf16 (fused dequant)
    # inplace False => output int8
    out_dtype = BF16 if inplace else INT8

    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        X: T.Tensor([M, N], BF16),
        Y: T.Tensor([M, N], out_dtype),
        S: T.Tensor([M, 1], FP32),
    ):

        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bm = cid // n_num
            bn = cid % n_num

            x_ub = T.alloc_shared([block_M, block_N], BF16)
            x_ub_fp = T.alloc_shared([block_M, block_N], FP32)
            x_ub_fp_abs = T.alloc_shared([block_M, block_N], FP32)

            # Intermediate buffers for fp32 -> int8 conversion
            x_ub_half = T.alloc_shared([block_M, block_N], FP16)
            q_ub = T.alloc_shared([block_M, block_N], INT8)

            # When inplace is True, the output is a dequantized bf16 value
            y_ub = T.alloc_shared([block_M, block_N], out_dtype)

            max_ub = T.alloc_shared([block_M, 1], FP32)
            scale_ub = T.alloc_shared([block_M, 1], FP32)

            T.copy(X[bm * block_M, bn * block_N], x_ub)

            T.vcast(x_ub, x_ub_fp)
            T.vabs(x_ub_fp, x_ub_fp_abs)
            T.reduce_max(x_ub_fp_abs, max_ub, dim=1)

            for i in T.Parallel(block_M):
                # Guard against all‑zero rows causing scale = 0
                max_ub[i, 0] = T.max(max_ub[i, 0], 1e-4)
                scale_ub[i, 0] = max_ub[i, 0] / int8_abs_max

            for i, j in T.Parallel(block_M, block_N):
                x_ub_fp[i, j] = x_ub_fp[i, j] / scale_ub[i, 0]

            T.vclamp(x_ub_fp, x_ub_fp, -127.0, 127.0)
            T.vcast(x_ub_fp, x_ub_fp, round_mode="round")

            # Quantize: fp32 rounded -> fp16 -> int8
            T.vcast(x_ub_fp, x_ub_half)
            T.vcast(x_ub_half, q_ub)

            if inplace:
                # Fused quant + dequant:
                # q = int8(round(clamp(x / scale)))
                # y = bf16(float(q) * scale)
                T.vcast(q_ub, x_ub_half)
                T.vcast(x_ub_half, x_ub_fp)

                for i, j in T.Parallel(block_M, block_N):
                    x_ub_fp[i, j] = x_ub_fp[i, j] * scale_ub[i, 0]

                T.vcast(x_ub_fp, y_ub)
            else:
                T.copy(q_ub, y_ub)

            T.copy(y_ub, Y[bm * block_M, bn * block_N], size=[block_M, block_N])
            T.copy(scale_ub, S[bm * block_M, 0], size=[block_M, 1])

    return main


def act_quant(
    x: torch.Tensor,
    inplace: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous(), "Input tensor must be contiguous"

    N = x.size(-1)

    # The current kernel expects S as [M, 1], so N must equal block_N.
    # This assertion can be relaxed once the kernel supports grouped scales.
    assert N == 32, (
        "Current S shape is [M, 1]; this test version assumes N == block_N == 32"
    )

    if inplace:
        y = torch.empty_like(x, dtype=torch.bfloat16)
    else:
        y = torch.empty_like(x, dtype=torch.int8)

    s = x.new_empty(x.size(0), 1, dtype=torch.float32)

    kernel = act_quant_kernel(N, inplace=inplace)
    kernel(x.view(-1, N), y.view(-1, N), s)

    if inplace:
        x.copy_(y)
        return x, s

    return y, s


def act_quant_torch(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch reference for normal INT8 quantization.

    y = round(clamp(x / scale, -127, 127)).to(int8)
    scale = max(abs(x)) / 127
    """
    x_f32 = x.float()

    abs_max = x_f32.abs().max(dim=1, keepdim=True)[0]
    abs_max = torch.clamp(abs_max, min=1e-4)

    scale = abs_max / 127.0

    x_scaled = x_f32 / scale
    x_clamped = torch.clamp(x_scaled, -127.0, 127.0)
    x_rounded = torch.round(x_clamped)

    y = x_rounded.to(torch.int8)
    s = scale

    return y, s


def act_quant_inplace_torch(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch reference for fused quant + dequant.

    q = round(clamp(x / scale, -127, 127)).to(int8)
    y = (q.float() * scale).to(bfloat16)
    """
    q, scale = act_quant_torch(x)

    y = q.float() * scale
    y = y.to(torch.bfloat16)

    return y, scale


def test_act_quant():
    M = 64
    N = 32
    dtype = torch.bfloat16

    x = torch.randn(size=[M, N], dtype=dtype).npu()

    y, s = act_quant(x, inplace=False)
    y_ref, s_ref = act_quant_torch(x)

    print("Start Testing normal quant: M = 64, N = 32")
    print("y:", y)
    print("y_ref:", y_ref)
    print("s:", s)
    print("s_ref:", s_ref)

    torch.testing.assert_close(s, s_ref, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(y, y_ref, atol=1, rtol=0)

    print("Normal quant comparison passed.")


def test_act_quant_inplace():
    M = 64
    N = 32
    dtype = torch.bfloat16

    x = torch.randn(size=[M, N], dtype=dtype).npu()
    x_ref_input = x.clone()

    y, s = act_quant(x, inplace=True)
    y_ref, s_ref = act_quant_inplace_torch(x_ref_input)

    print("Start Testing inplace quant-dequant: M = 64, N = 32")
    print("y:", y)
    print("y_ref:", y_ref)
    print("s:", s)
    print("s_ref:", s_ref)

    torch.testing.assert_close(s, s_ref, atol=1e-3, rtol=1e-3)

    # inplace output is bf16; quantisation then dequantisation introduces small errors
    torch.testing.assert_close(y, y_ref, atol=1e-2, rtol=1e-2)

    # Verify that x has been overwritten
    torch.testing.assert_close(x, y_ref, atol=1e-2, rtol=1e-2)

    print("Inplace quant-dequant comparison passed.")


if __name__ == "__main__":
    os.environ["TILELANG_ASCEND_MODE"] = "Developer"
    torch.manual_seed(888)
    test_act_quant()
    test_act_quant_inplace()
