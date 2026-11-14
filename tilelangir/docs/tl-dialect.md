# TL dialect (TLIR) v0.1

Restricted TileLang tile-level IR between TVM TIR and HIVM. v0.1 covers
expert-mode elementwise add/sub/mul/div/max/min and the `set_flag` cross-pipe
sync primitive: parse/print, TIR → TLIR import, and
`tilelangir-convert-tl-to-hivm`.

## Attributes

| Attr | Cases |
|------|--------|
| `#tl.address_space<...>` | `gm`, `ub`, `l1`, `l0` |
| `#tl.layout<...>` | `nd`, `nz` |
| `#tl.core_type<...>` | `aic`, `aiv`, `mix` |
| `#tl.pipe<...>` | `pipe_s`, `pipe_v`, `pipe_m`, `pipe_mte1`, `pipe_mte2`, `pipe_mte3`, `pipe_all`, `pipe_mte4`, `pipe_mte5`, `pipe_v2`, `pipe_fix` |

`#tl.address_space` is intended as a memref memory space.
`#tl.core_type` is a function attribute (`tl.core_type = #tl.core_type<aiv>`).
`#tl.pipe` names the HIVM-style execution pipes used by `tl.set_flag`.

## Operations

| Op | Meaning |
|----|---------|
| `tl.alloc` | Allocate a tile buffer |
| `tl.copy` | Same-shape DMA-style copy (spaces may differ) |
| `tl.add` | Destination-style elementwise add |
| `tl.sub` | Destination-style elementwise subtract (`out = lhs - rhs`) |
| `tl.mul` | Destination-style elementwise multiply |
| `tl.div` | Destination-style elementwise divide (`out = lhs / rhs`) |
| `tl.max` | Destination-style elementwise maximum |
| `tl.min` | Destination-style elementwise minimum |
| `tl.set_flag` | Set a cross-pipe sync flag/event (`set_pipe`, `wait_pipe`, `event_id`) |

```mlir
func.func @elemwise_add(%gm_a: memref<64xf32, #tl.address_space<gm>>,
                        %gm_b: memref<64xf32, #tl.address_space<gm>>,
                        %gm_c: memref<64xf32, #tl.address_space<gm>>)
    attributes {tl.core_type = #tl.core_type<aiv>} {
  %ub_a = tl.alloc : memref<64xf32, #tl.address_space<ub>>
  %ub_b = tl.alloc : memref<64xf32, #tl.address_space<ub>>
  %ub_c = tl.alloc : memref<64xf32, #tl.address_space<ub>>
  tl.copy %gm_a -> %ub_a : memref<64xf32, #tl.address_space<gm>>, memref<64xf32, #tl.address_space<ub>>
  tl.copy %gm_b -> %ub_b : memref<64xf32, #tl.address_space<gm>>, memref<64xf32, #tl.address_space<ub>>
  tl.add ins(%ub_a, %ub_b) outs(%ub_c) : memref<64xf32, #tl.address_space<ub>>
  tl.copy %ub_c -> %gm_c : memref<64xf32, #tl.address_space<ub>>, memref<64xf32, #tl.address_space<gm>>
  return
}
```

Round-trip:

```bash
tilelangir-opt tilelangir/test/Dialect/TL/alloc-copy-add.mlir
```

Lower to HIVM (expert-mode add subset: `memref.alloc` / `memref.copy` / `hivm.hir.vadd`):

```bash
tilelangir-opt --tilelangir-convert-tl-to-hivm \
  tilelangir/test/Dialect/TL/alloc-copy-add.mlir
```

This pass does not inject HACC / FFTS / workspace ABI.

### `tl.set_flag`

`tl.set_flag` mirrors `hivm.hir.set_flag`: it signals an event on `set_pipe`
that a consumer on `wait_pipe` can later wait on. `event_id` is any
signless integer value; it is widened to `i64` during `TIR -> TLIR` lowering
the same way the legacy NPUIR path widens `tl.npuir_set_flag`'s event id.

```mlir
%c0_i64 = arith.constant 0 : i64
tl.set_flag[#tl.pipe<pipe_mte2>, #tl.pipe<pipe_v>, %c0_i64] : i64
```

lowers to:

```mlir
hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, %c0_i64]
```

Round-trip and conversion tests:

```bash
tilelangir-opt tilelangir/test/Dialect/TL/alloc-copy-add-set-flag.mlir
tilelangir-opt --tilelangir-convert-tl-to-hivm \
  tilelangir/test/Conversion/TLToHIVM/alloc-copy-add-set-flag.mlir
```

`tl.wait_flag` is not part of this subset yet; it can follow the same
pattern (`TL_PipeAttr` × 2 + an integer operand, lowering to
`hivm.hir.wait_flag`) in a follow-up change.
