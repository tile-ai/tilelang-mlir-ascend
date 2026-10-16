# CANN 9.1.0 NPUIR patches

The A2/A3 dependency `3rdparty/AscendNPU-IR` is pinned by its Git submodule
entry to `8796a8ac1508380d78128427c61ab47d03f05739` (upstream `v1.2.0`,
the compiler revision shipped with CANN 9.1.0). Do not advance it to master
to pick up fixes: keep the release baseline and carry backports here.

`3rdparty/AscendNPU-IR-Dev` and the `--build-a5` path remain unchanged.
These patches do not apply to that dependency. The future A5 source migration
is separate work. Packaging the A2/A3 compiler is documented in
[`docs/developer/NPUIR-toolchain-packaging.md`](../../docs/developer/NPUIR-toolchain-packaging.md).

## Patch contents

`0001-HFusionToHIVM-skip-resultless-mmadl1.patch` backports the guard from
upstream `f9f4507c4187fe05035a52f1094a6ae602f6bb71` and adds a regression test.
The HFusion-to-HIVM pass walks existing `hivm.hir.mmadL1` operations to propagate
`TileMixCubeNum`. Buffer-form GEMM has no SSA results; accessing `getResult(0)`
crashes. The guard skips annotation propagation for this form while preserving
the existing tensor-form path and the GEMM itself.

## Apply

From the TileLang repository root:

```bash
git submodule update --init --recursive 3rdparty/AscendNPU-IR
bash 3rdparty/patch/apply_npuir_patches.sh
```

Both the A2/A3 source-build path in `install_npuir.sh` and the Docker compiler
build invoke this script before building NPUIR. An externally supplied
`--bishengir-path` is not modified. The script checks the exact release revision,
skips patches already applied, and stops on conflicts. It does not create
commits or modify the submodule's HEAD; applied files appear as local changes
inside the submodule. NPUIR's own `build-tools/apply_patches.sh` still handles
its LLVM/torch-mlir patches separately.

To check an isolated checkout without changing the default dependency:

```bash
bash 3rdparty/patch/apply_npuir_patches.sh /path/to/AscendNPU-IR
```

## Regression test

The patch adds `bishengir/test/Conversion/HFusionToHIVM/mmad-l1-buffer.mlir`.
After building the patched compiler, run from the NPUIR source directory:

```bash
set -o pipefail
build/bin/bishengir-opt \
  bishengir/test/Conversion/HFusionToHIVM/mmad-l1-buffer.mlir \
  -convert-hfusion-to-hivm="mm-map-mode=macro_instr" | \
  build/bin/FileCheck bishengir/test/Conversion/HFusionToHIVM/mmad-l1-buffer.mlir
```

The unfixed compiler crashes in the annotation propagation walk. The fixed
compiler must finish successfully and retain the buffer-form `mmadL1` and its
output. This pass-level test needs no NPU device. It does not replace later
wheel-build and real-device validation.
