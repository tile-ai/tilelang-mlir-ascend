"""TileLang Ascend SIMT FP32 op-throughput probe: recurrence plus accumulator.

Each element is updated by a cheap FMA recurrence that keeps it in [0.75, 1.25],
and the probed op accumulates into a live register. The accumulator keeps every
round observable, so the compiler cannot delete or hoist the op, and the control
op (no function) has the identical instruction skeleton.
"""

import argparse

import tilelang
import tilelang.ascend.language as T
import torch

TILE = 4096
LANES = TILE // 4
SCALE = 0.5
BIAS = 0.5
OPS = ("fma", "control", "exp", "expm", "exps", "expdif", "ln", "sqrt", "rsqrt", "rcp", "sigmoid")


def apply_op(name, value):
    if name == "exp":
        return T.call_pure_extern("float32", "expf", value)
    if name == "expm":
        return T.call_pure_extern("float32", "expf", value - T.float32(1.0))
    if name == "exps":
        return T.call_pure_extern("float32", "expf", value) - T.float32(0.5)
    if name == "expdif":
        return T.call_pure_extern("float32", "expf", value - T.float32(0.5))
    if name == "ln":
        return T.call_pure_extern("float32", "logf", value)
    if name == "sqrt":
        return T.call_pure_extern("float32", "sqrtf", value)
    if name == "rsqrt":
        return T.call_pure_extern("float32", "rsqrtf", value)
    if name == "rcp":
        return T.float32(1.0) / value
    if name == "sigmoid":
        return T.float32(1.0) / (T.float32(1.0) + T.call_pure_extern("float32", "expf", T.float32(0.0) - value))
    return value


def build(op, blocks, tiles, rounds):
    n = blocks * tiles * TILE
    unroll = min(rounds, 16)

    @T.prim_func
    def main(X: T.Tensor((n,), "float32"), Y: T.Tensor((n,), "float32")):
        with T.Kernel(blocks) as bx:
            buf = T.alloc_shared((TILE,), "float32")
            for tile_id in T.serial(tiles):
                start = (tile_id * blocks + bx) * TILE
                T.copy(X[start : start + TILE], buf)
                with T.SimtVF(threads=LANES):
                    for i in T.Parallel(LANES):
                        a = T.alloc_var("float32", init=buf[i])
                        b = T.alloc_var("float32", init=buf[i + LANES])
                        c = T.alloc_var("float32", init=buf[i + 2 * LANES])
                        d = T.alloc_var("float32", init=buf[i + 3 * LANES])
                        sa = T.alloc_var("float32", init=T.float32(0.0))
                        sb = T.alloc_var("float32", init=T.float32(0.0))
                        sc = T.alloc_var("float32", init=T.float32(0.0))
                        sd = T.alloc_var("float32", init=T.float32(0.0))
                        for _group in T.serial(rounds // unroll):
                            for _step in T.unroll(unroll):
                                a = T.call_pure_extern("float32", "fmaf", a, T.float32(SCALE), T.float32(BIAS))
                                b = T.call_pure_extern("float32", "fmaf", b, T.float32(SCALE), T.float32(BIAS))
                                c = T.call_pure_extern("float32", "fmaf", c, T.float32(SCALE), T.float32(BIAS))
                                d = T.call_pure_extern("float32", "fmaf", d, T.float32(SCALE), T.float32(BIAS))
                                if op != "fma":
                                    sa = sa + apply_op(op, a)
                                    sb = sb + apply_op(op, b)
                                    sc = sc + apply_op(op, c)
                                    sd = sd + apply_op(op, d)
                        if op == "fma":
                            buf[i] = a
                            buf[i + LANES] = b
                            buf[i + 2 * LANES] = c
                            buf[i + 3 * LANES] = d
                        else:
                            buf[i] = sa
                            buf[i + LANES] = sb
                            buf[i + 2 * LANES] = sc
                            buf[i + 3 * LANES] = sd
                T.copy(buf, Y[start : start + TILE])

    return main


def reference(op, x):
    v = x.mul(SCALE).add(BIAS)
    return {
        "fma": lambda t: t,
        "control": lambda t: t,
        "exp": torch.exp,
        "expdif": lambda t: torch.exp(t - 0.5),
        "expm": lambda t: torch.exp(t - 1.0),
        "exps": lambda t: torch.exp(t) - 0.5,
        "ln": torch.log,
        "sqrt": torch.sqrt,
        "rsqrt": torch.rsqrt,
        "rcp": torch.reciprocal,
        "sigmoid": torch.sigmoid,
    }[op](v)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("op", choices=OPS)
    parser.add_argument("--rounds", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=72, help="AIV count; use 56 for 28 AIC")
    parser.add_argument("--tiles", type=int, default=4)
    parser.add_argument("--verify", action="store_true", help="check one round against torch instead of timing")
    args = parser.parse_args()
    if min(args.rounds, args.blocks, args.tiles) <= 0 or (args.rounds > 16 and args.rounds % 16):
        parser.error("use positive sizes and a multiple of 16 above 16 rounds")
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
        actual = y.cpu().reshape(args.tiles * args.blocks, TILE)
        source = x_cpu.reshape(args.tiles * args.blocks, TILE)
        expected = reference(args.op, source)
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-6)
        print(f"PASS op={args.op} rounds=1 N={n}", flush=True)
    else:
        if not torch.isfinite(y).all():
            raise SystemExit("non-finite output; check the recurrence domain")
        print(f"PASS op={args.op} rounds={args.rounds} N={n} logical_bytes={8 * n} "
              f"semantic_ops={n * args.rounds}", flush=True)
