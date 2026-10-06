# TL dialect (TLIR) v0.1

Restricted TileLang tile-level IR between TVM TIR and HIVM. v0.1 covers
expert-mode elementwise add: parse/print, TIR → TLIR import, and
`tilelangir-convert-tl-to-hivm`.

## Attributes

| Attr | Cases |
|------|--------|
| `#tl.address_space<...>` | `gm`, `ub`, `l1`, `l0` |
| `#tl.layout<...>` | `nd`, `nz` |
| `#tl.core_type<...>` | `aic`, `aiv`, `mix` |

`#tl.address_space` is intended as a memref memory space.
`#tl.core_type` is a function attribute (`tl.core_type = #tl.core_type<aiv>`).

## Operations

| Op | Meaning |
|----|---------|
| `tl.alloc` | Allocate a tile buffer |
| `tl.copy` | Same-shape DMA-style copy (spaces may differ) |
| `tl.add` | Destination-style elementwise add |

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
