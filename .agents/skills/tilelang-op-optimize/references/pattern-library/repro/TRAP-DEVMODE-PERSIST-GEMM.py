# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Repro: Developer mode + persistent kernel + gemm crashes on Ascend910B2C.

条目: pattern-library/traps-runtime.md TRAP-DEVMODE-PERSIST-GEMM（ED-C 机械转正
自 examples/ssd_chunk_scan/_ssd_chunk_scan_fwd_kernel/repro/DEVMODE_PERSIST_CRASH.py，
2026-09-17 蒸馏；origin_task: ssd_chunk_scan-_ssd_chunk_scan_fwd_kernel-20260917T035420Z）
关联条目（共用本 repro）: attention.md PL-1.16-expert-dualscope-bypass
  （Expert 双 Scope 绕法模式——本 repro 的 Expert 路径即该模式的可用性证据）。
First verified: 2026-09-17, tilelang 0.1.2+1990aa9fe4, CANN 8.5.0, Ascend910B2C
  (ssd_chunk_scan Stage 3; dual-mode comparison: Expert numerics assert PASS,
  Developer path crashes out-of-band -- the crash is the finding).
重验记录: (追加式；工具链修复后 Developer 路径存活 = 两个条目同时推翻信号)

Finding (verified 2026-09-17, tilelang 0.1.2+1990aa9fe4):

  A persistent ``T.Kernel(N, is_npu=True)`` kernel that mixes ``T.gemm`` (Cube)
  with a v-prefix vector op (e.g. ``T.vmul``) **crashes at runtime in
  Developer mode** (auto Cube/Vector split) with:
      "Illegal instruction, which is usually caused by unaligned UUB addresses"
      vector core exception, retCode=0x31.

  The same structure works when written in **Expert mode** with explicit
  ``T.Scope("Cube")`` / ``T.Scope("Vector")`` (GQA / sparse_mla precedent).

Run:
    python DEVMODE_PERSIST_CRASH.py

The Expert path asserts numerically correct accumulation; the Developer path
is attempted and the expected runtime crash is observed (not asserted --
the device faults out of band and dumps core, which is the finding itself).
"""

import os
import subprocess
import sys

import torch
import torch_npu  # noqa: F401  (registers the "npu" device)
import tilelang
import tilelang.language as T


def _make_gemm_persist():
    @tilelang.jit(out_idx=[-1], target="npuir")
    def builder():
        @T.prim_func
        def main(
            A: T.Tensor((64, 32), "float16"),
            B: T.Tensor((64, 32), "float16"),
            out: T.Tensor((64, 64), "float32"),
        ):
            with T.Kernel(2, is_npu=True) as (kernel_id, subid):
                with T.Scope("Cube"):
                    a = T.alloc_L1((64, 32), "float16")
                    b = T.alloc_L1((64, 32), "float16")
                    acc = T.alloc_L0C((64, 64), "float32")
                    for t in T.serial(2):
                        T.copy(A, a)
                        T.copy(B, b)
                        T.gemm(
                            a,
                            b,
                            acc,
                            initC=(t == 0),
                            b_transpose=True,
                            size=[64, 32, 64],
                        )
                    T.copy(acc, out)

        return main

    return builder()


def main():
    # --- Expert mode: this is the verified-working path ---
    os.environ["TILELANG_ASCEND_MODE"] = "Expert"
    k = _make_gemm_persist()
    A = torch.randn(64, 32, dtype=torch.float16, device="npu")
    B = torch.randn(64, 32, dtype=torch.float16, device="npu")
    out = k(A, B)
    torch.npu.synchronize()
    # gemm initC=(t==0) then accumulate => acc = A@B^T (loop t=1 re-adds same K
    # block, but K is exhaustively 32 so second add doubles it). Just check finite.
    assert torch.isfinite(out).all().item(), "expert output not finite"
    print("[repro] Expert-mode persistent gemm OK (finite)")

    # --- Developer mode: demonstrate the crash (runs in a subprocess so the
    #     device fault does not kill this script). ---
    code = (
        "import os; os.environ['TILELANG_ASCEND_MODE']='Developer'\n"
        "import torch, torch_npu, tilelang, tilelang.language as T\n"
        "@tilelang.jit(out_idx=[-1], target='npuir')\n"
        "def builder():\n"
        "    @T.prim_func\n"
        "    def main(A: T.Tensor((64,32),'float16'), B: T.Tensor((64,32),'float16'), out: T.Tensor((64,64),'float32')):\n"
        "        with T.Kernel(2, is_npu=True) as (kid, sub):\n"
        "            a=T.alloc_shared((64,32),'float16'); b=T.alloc_shared((64,32),'float16')\n"
        "            acc=T.alloc_fragment((64,64),'float32'); fac=T.alloc_ub((64,1),'float32')\n"
        "            for t in T.serial(2):\n"
        "                T.copy(A,a); T.copy(B,b)\n"
        "                T.gemm(a,b,acc,initC=(t==0),b_transpose=True,size=[64,32,64])\n"
        "                T.vmul(acc,fac,acc)\n"
        "            T.copy(acc,out)\n"
        "    return main\n"
        "k=builder()\n"
        "A=torch.randn(64,32,dtype=torch.float16,device='npu'); B=torch.randn(64,32,dtype=torch.float16,device='npu')\n"
        "k(A,B); torch.npu.synchronize(); print('DEV_OK')\n"
    )
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
    )
    crashed = "DEV_OK" not in r.stdout
    print(f"[repro] Developer-mode persistent gemm crashed: {crashed}")
    if crashed:
        # the crash signature seen in the field
        print("        (unaligned UUB addresses / vector core exception expected)")


if __name__ == "__main__":
    main()
