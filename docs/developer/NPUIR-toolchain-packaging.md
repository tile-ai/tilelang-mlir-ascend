# TileLang-managed NPUIR compiler

## Layout and lookup

Stage release artifacts in the source checkout before building an NPUIR wheel:

```text
3rdparty/
  bin/bishengir-compile
  lib/host.bc
  lib/meta_op.aic.bc
  lib/meta_op.aiv.bc
  lib/meta_op.mix.aic.bc
  lib/meta_op.mix.aiv.bc
```

The complete `bin` and `lib` directories are copied into the wheel as
`tilelang/3rdparty/bin` and `tilelang/3rdparty/lib`. Executable permissions are
preserved, and bitcode remains next to the executable's parent directory, where
the CANN 9.1.0 compiler resolves its resources. These generated source directories
are Git-ignored but included in source distributions through `MANIFEST.in`.

For A2/A3, JIT compiler selection is:

1. Executable `bishengir-compile` in TileLang's `3rdparty/bin`.
2. `bishengir-compile` found through `PATH`.
3. The existing `${TILELANG_NPU_COMPILER_PATH}/npuc` fallback.
4. The existing missing-compiler error.

`tilelang.env.THIRD_PARTY_ROOT` resolves the source-checkout or installed-package
layout. A missing/non-executable bundled file falls back; a selected compiler's
compilation failure does not silently retry with a different compiler. Automatic
target detection also recognizes the bundled compiler. Lookup remains cached
per process, so restart Python after changing the compiler payload.

The packaged release compiler is currently for A2/A3. A5 JIT explicitly retains
the legacy compiler lookup until its source migration is completed. This change
does not package or replace `hivmc`, `bisheng`, Toolkit headers, or the NPU runtime;
source the CANN environment before use.

## Local packaging

Use a native-architecture NPUIR build containing the patches in `3rdparty/patch`.
Use CMake >= 3.28 (and Ninja >= 1.12 if using Ninja). Configure
`BISHENGIR_PUBLISH=ON` to avoid passing internal options to the public CANN
`hivmc`. When building its bitcode, source CANN 9.1.0 and configure
`BISHENGIR_BUILD_TEMPLATE=ON` and `BISHENG_COMPILER_PATH` to the directory
containing CANN's `ccec` and `llvm-link`. The CI workflow shows the complete
configuration. The staged executable must match the wheel's architecture and
the destination system's runtime libraries.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Run from the repository root after compiling TileLang and TVM.
# The install prefix must contain bin/bishengir-compile and the five lib/*.bc.
BISHENGIR_PATH=/path/to/patched-npuir/install bash build_wheel.sh
```

`build_wheel.sh` copies the compiler and bitcode into `3rdparty/bin` and
`3rdparty/lib`, then packages the existing TileLang/TVM libraries and MLIR Python
bindings. Without `BISHENGIR_PATH`, it uses
`3rdparty/AscendNPU-IR/build/install`.

To package manually supplied `3rdparty/bin` and `3rdparty/lib` directories,
run `USE_NPUIR=true TILELANG_SKIP_BUILD=1 python setup.py bdist_wheel` with the
prebuilt libraries and bindings available. `setup.py` rejects an NPUIR wheel
with no executable compiler or missing/empty required bitcode. It copies the
staged directories instead of downloading a compiler or selecting one from
the build machine's `PATH`.

## CI

NPUIR prebuild, TileLang wheel build and the wheel test job use
`quay.io/ascend/cann:9.1.0-910b-ubuntu22.04-py3.11` and the corresponding
`py3.12` image. Both x64 and arm64 builders produce Python 3.11 and 3.12 wheels
using the matching image's native Python. NPUIR Python extensions are compiled
separately for each Python version; the TVM prebuild is shared across Python
versions on the same architecture. Run steps use
`bash -l -e -o pipefail {0}`: login Bash reads `/etc/profile`, where the image
already sources Toolkit, AscendNPU-IR and NNAL. This avoids repeating `source`
commands in individual steps. GitHub Actions overrides the job container's
entrypoint, so the entrypoint alone cannot initialize these steps.
See the [runner implementation](https://github.com/actions/runner/blob/main/src/Runner.Worker/ContainerOperationProvider.cs)
and the [CANN image Dockerfile](https://github.com/Ascend/cann-container-image/blob/main/cann/9.1.0-910b-ubuntu22.04-py3.11/Dockerfile).

- The NPUIR prebuild applies the release backport before upstream dependency
  patches, builds the compiler and bitcode using CANN's BiSheng, and caches the
  installation including `bin` and `lib`.
- Before starting any NPUIR build container, a host job checks all four caches
  with `lookup-only`. Only missing architecture/Python combinations enter the
  container build matrix. If all caches exist, the build job is skipped and the
  precheck still supplies the NPUIR commit to downstream wheel jobs.
- Cache keys include architecture, Python version, the pinned NPUIR commit and
  a hash of the patch directory and prebuild workflow. The precheck, build and
  wheel jobs all use zstd and matching cache keys. The `toolchain-v2` namespace
  requires one initial rebuild; old gzip/Python 3.11 caches are not reused.
- The wheel job copies the cached compiler and bitcode into `3rdparty` directly
  in its existing packaging step, then builds and uploads the wheel.
- The NPU wheel test uses matched `torch==2.10.0+cpu` / `torch-npu==2.10.0`
  and runs the existing examples and operator tests for both Python versions,
  sequentially to avoid competing for the same hardware. Reports include the
  Python version in their artifact name. Release and pre-release uploads collect
  all four wheels after both test jobs pass.
