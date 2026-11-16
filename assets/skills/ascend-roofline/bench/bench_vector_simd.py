"""TileLang Ascend SIMD (SimdVF) FP32 op probe for vexpdif / vexp.

The SIMD block is entered once per large UB tile: inside it, a TIR loop walks
256-element chunks (four 64-lane registers) and a second TIR loop repeats the
probed op on register-resident data. Registers are updated through
``T.simd.alloc_local`` because the eager builder rejects SSA values carried
across loop regions.
"""

import argparse

import tilelang
import tilelang.ascend.language as T
import torch

LANES = 64
CHAINS = 4
CHUNK = LANES * CHAINS
TILE = 8192
CHUNKS = TILE // CHUNK
OPS = ("vexpdif", "vexp", "control")


def build(op, blocks, tiles, rounds):
    n = blocks * tiles * TILE

    @T.prim_func
    def main(X: T.Tensor((n,), "float32"), Y: T.Tensor((n,), "float32")):
        with T.Kernel(blocks) as bx:
            buf = T.alloc_shared((TILE,), "float32")
            for tile_id in T.serial(tiles):
                start = (tile_id * blocks + bx) * TILE
                T.copy(X[start : start + TILE], buf)
                with T.SimdVF():
                    full = T.simd.pset(32, "PAT_ALL")
                    v = T.simd.alloc_local(CHAINS, "float32")
                    a = T.simd.alloc_local(CHAINS, "float32")
                    for chunk in T.serial(CHUNKS):
                        base = chunk * CHUNK
                        mx = T.simd.vld(buf[base], "BRC_B32")
                        for i in range(CHAINS):
                            v[i] = T.simd.vld(buf[base + i * LANES], "NORM")
                            a[i] = T.simd.vmuls(v[i], T.float32(0.0), full)
                        for _round in T.serial(rounds):
                            for i in range(CHAINS):
                                v[i] = T.simd.vmuls(v[i], T.float32(0.999), full)
                                if op == "vexpdif":
                                    e = T.simd.vexpdif(v[i], mx, full)
                                elif op == "vexp":
                                    e = T.simd.vexp(v[i], full)
                                else:
                                    e = v[i]
                                a[i] = T.simd.vadd(a[i], e, full)
                        for i in range(CHAINS):
                            T.simd.vsts(buf[base + i * LANES], a[i], full, "NORM_B32")
                T.copy(buf, Y[start : start + TILE])

    return main


def reference(op, x, tiles, blocks):
    rows = x.reshape(tiles, blocks, CHUNKS, CHUNK)
    decayed = rows * 0.999
    if op == "vexp":
        out = torch.exp(decayed)
    elif op == "vexpdif":
        out = torch.exp(decayed - rows[..., :1])
    else:
        out = decayed
    return out.reshape(-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("op", choices=OPS)
    parser.add_argument("--rounds", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=72, help="AIV count; use 56 for 28 AIC")
    parser.add_argument("--tiles", type=int, default=2)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if min(args.rounds, args.blocks, args.tiles) <= 0:
        parser.error("sizes must be positive")
    if args.verify and args.rounds != 1:
        parser.error("--verify needs --rounds 1")

    n = args.blocks * args.tiles * TILE
    torch.manual_seed(0)
    x_cpu = torch.rand(n, dtype=torch.float32).add_(0.5)
    x = x_cpu.npu()
    kernel = tilelang.compile(build(args.op, args.blocks, args.tiles, args.rounds), out_idx=-1)
    torch.npu.synchronize()
    y = kernel(x)
    torch.npu.synchronize()

    if args.verify:
        expected = reference(args.op, x_cpu, args.tiles, args.blocks)
        torch.testing.assert_close(y.cpu(), expected, rtol=2e-6, atol=1e-6)
        print(f"PASS op={args.op} rounds=1 N={n}", flush=True)
    else:
        if not torch.isfinite(y).all():
            raise SystemExit("non-finite output")
        print(f"PASS op={args.op} rounds={args.rounds} N={n} logical_bytes={8 * n} "
              f"semantic_ops={n * args.rounds}", flush=True)
