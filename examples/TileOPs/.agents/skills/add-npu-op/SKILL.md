---
name: add-npu-op
description: Add a new operator to the TileOPs project (NPU backend). Caller provides the op name, the GPU TileOPs repo root (source of truth for porting); skill creates all 7 files (manifest entry, workload, kernel, op, test, benchmark, conftest wiring) and verifies on NPU. Idempotent.
---
# add-npu-op

Add a new operator to the `TileOPs/` standalone NPU project.

## Prerequisites

- **NPU project root** (working directory): the standalone NPU `TileOPs/`
  project — this is where the 7 output files are created.
- **GPU project root** (`gpu_repo_root`, caller-provided): the independent GPU
  `TileOPs/` project repo. The GPU op's end-to-end implementation lives here
  and is the **source of truth** for porting. The NPU project no longer
  bundles GPU sources, so the caller must supply this path.
- NPU environment: `torch_npu` installed, `torch.npu.is_available() == True`
- The NPU `TileOPs/` package is `pip install -e`'d or on `PYTHONPATH`

## Path notation

- `{gpu_repo_root}` = the caller-provided GPU `TileOPs/` repo root.
  All GPU-side source paths in this skill are relative to `{gpu_repo_root}`.
- Unprefixed `tileops/`, `tests/`, `benchmarks/`, `docs/` paths are relative
  to the NPU project root (the working directory) — these are NPU-side
  targets/references.

## Inputs

The caller provides:

1. **`op_name`** (required) — PascalCase manifest key, e.g. `SoftmaxFwdOp`
2. **`gpu_repo_root`** (required) — absolute path to the GPU `TileOPs/`
   project repo root, e.g. `/home/user/TileOPs`. Used as the source for
   Phase 0 and all "Port from GPU" steps.
3. **`family`** (optional) — op family directory, e.g. `reduction` (default: `reduction`)

## Output

7 files created/updated (all under the NPU project root `TileOPs/`):

| #  | File                                           | Purpose                                                 |
| -- | ---------------------------------------------- | ------------------------------------------------------- |
| S1 | `tileops/manifest/{family}.yaml`             | Manifest entry (signature, workloads, roofline, source) |
| S2 | `tileops/workloads/{family}.py`              | Workload class (input generation)                       |
| S3 | `tileops/kernels/{family}/{op_slug}/` (`{op_slug}.py` + `__init__.py` + extracted kernels) | NPU kernel package (TileLang-based, NPUIR target)      |
| S4 | `tileops/ops/{family}/{op_slug}.py`          | Op class (validation, reshape, dispatch, roofline)      |
| S5 | `tests/ops/test_{op_slug}.py`                | Correctness tests vs PyTorch reference                  |
| S6 | `benchmarks/ops/bench_{op_slug}.py`          | Performance benchmarks (manifest-driven)                |
| S7 | `tileops/{kernels,ops}/{family}/__init__.py` + `tileops/kernels/{family}/{op_slug}/__init__.py` | Package exports (updated/created) |

## Methodology

**The GPU op's end-to-end flow is the source of truth.** Each op has its own
flow — input count, reshape strategy, kernel internal structure, validation
logic, roofline formula — dictated by the GPU implementation in
`{gpu_repo_root}/tileops/`.
Do **not** force every op into a single template (e.g. LogSumExp's reduction
flow). Instead:

1. **Phase 0**: Locate and study the GPU op's full end-to-end implementation.
2. **Port** each of the 7 files from GPU, preserving the op-specific flow.
3. **Apply** the standard NPU adaptations (cross-cutting, listed once below).
4. **Verify** structure (Tier 1); runtime verification (Tier 2) happens
   after the NPU kernel component rewrites the extracted kernels for
   `target="npuir"`.

**Scope boundary**: This skill ports the op's end-to-end structure
(manifest, workload, Op class, Kernel class, custom_op wrapper, tests,
benchmark) and extracts the GPU TileLang kernel functions via the
extraction script, importing them into the kernel file. The NPU TileLang
kernel *re-implementation* (adapting `@tilelang.jit` to `target="npuir"`

- `@T.prim_func` body) is authored by a separate NPU kernel component.

LogSumExpFwdOp is just reference example (reduction family), **not** universal templates.

## Phase 0 — Locate and study the GPU op

Before writing any NPU code, locate and read the GPU op's full end-to-end
implementation under `{gpu_repo_root}/`:

| Artifact       | Where to find it                                                                         |
| -------------- | ---------------------------------------------------------------------------------------- |
| Manifest entry | `{gpu_repo_root}/tileops/manifest/<family>*.yaml` — grep for `{op_name}`            |
| Op class       | `{gpu_repo_root}/tileops/ops/<family>/*.py` — grep for `class {op_name}`            |
| Kernel class   | `{gpu_repo_root}/tileops/kernels/<family>/*.py` — from manifest `source.kernel_map` |
| Workload       | `{gpu_repo_root}/workloads/<family>.py` — grep for the op's workload class            |
| Test           | `{gpu_repo_root}/tests/ops/test_*.py` — from manifest `source.test`                 |
| Benchmark      | `{gpu_repo_root}/benchmarks/ops/bench_*.py` — from manifest `source.bench`          |

Read all 6 GPU files before starting. Understand and note:

- **Input count**: How many tensor inputs? (1 for unary/reduction, 2 for
  binary, 3 for lerp-tensor/where, etc.)
- **Reshape strategy**: Does the Op flatten to 1-D `(N_total,)`, reshape to
  2-D `(M, N)`, use multi-dim helpers, or pass shapes through?
- **Kernel structure**: Reduction loop? Elementwise map? Online recurrence?
  What buffers does it allocate?
- **GPU execution params**: Does the factory/kernel use `threads` (CUDA
  threads-per-block) and/or `npt` (num_per_thread)? Reduction ops use
  `threads` (removed on NPU — K3/K9); elementwise ops use
  `threads * npt` (collapsed into `block_size` on NPU — K11). Note which
  applies to this op.
- **Constructor params**: What does the Op `__init__` take? (shapes, dtype,
  op-specific params like `dim`, `weight`, `rounding_mode`, ...)
- **Roofline formula**: `flops` and `bytes` expressions — copy verbatim
  (roofline is device-agnostic).

Also read the existing NPU adaptation guide (in the NPU project root):
`docs/gpu_to_npu_adaptation.md` — the full catalogue of
GPU-to-NPU adaptation points (common mechanism + per-op).

## Standard NPU Adaptations

These cross-cutting adaptations apply to **every** op, regardless of family.
They are listed once here; each step below references this section. Full
details and rationale: `docs/gpu_to_npu_adaptation.md` (NPU project root).

### TileLang kernel functions (applies to S3)

The TileLang kernel functions (e.g. `_{op_slug}_kernel_single`,
`_{op_slug}_kernel_tiled`, `_{op_slug}_kernel`) are **extracted** from
the GPU repo via the extraction script and **imported** into the kernel
file — the full GPU implementation (function signature, docstring, and
`@T.prim_func` body) is pulled in as-is. The NPU TileLang re-implementation
(adapting `@tilelang.jit` to `target="npuir"` and rewriting the body for
NPUIR) is authored by a separate NPU kernel component, not by this skill.

| #  | GPU original                                                      | NPU handling (this skill)                                                                                                                                                      |
| -- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| K1 | `@tilelang.jit` decorator + `@T.prim_func` body (CUDA target) | **Extracted as-is** — the GPU implementation is imported directly; the NPU component rewrites the decorator to `target="npuir"` and adapts the body                   |
| K2 | `T.Kernel(grid, threads=threads)` inside the prim_func body     | **Extracted as-is** — the NPU component rewrites for NPUIR grid/sync semantics                                                                                          |
| K3 | `threads` param in factory callable signature                   | **Preserved in extraction** — the NPU component removes `threads` (reduction ops' callable takes `block_m`; elementwise ops' callable takes `block_size` per K11) |
| K4 | Pad N to`DEFAULT_ALIGNMENT` + masked loads (inside the body)    | **Extracted as-is** — the NPU component decides padding strategy during re-implementation                                                                               |

### Kernel class (applies to S3)

The **Kernel class** (e.g. `LogSumExpKernel`) is ported in full — config
selection, tiling heuristics, and `forward` dispatch are preserved from the
GPU version with GPU-specific parts adapted. The class calls the imported
kernel functions (GPU implementations extracted in Part A); these will not
run on NPU until the NPU component rewrites them for `target="npuir"`.

| #   | GPU original                                                                                                                          | NPU adaptation                                                                                                                                                                                                                                                                                                                                                                                               |
| --- | ------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| K5  | `supported_archs = [80, 86, 89, 90]` (CUDA SM ints)                                                                                 | `supported_archs = None` (all archs)                                                                                                                                                                                                                                                                                                                                                                       |
| K6  | `device_smem_budget` via `torch.cuda.get_device_properties()`                                                                     | Import from`tileops.kernels.reduction._primitives` (already backend-agnostic)                                                                                                                                                                                                                                                                                                                              |
| K7  | `@torch.library.custom_op("top::...")`                                                                                              | `@torch.library.custom_op("npub::...")`                                                                                                                                                                                                                                                                                                                                                                    |
| K8  | `autotune_configs`, `autotune()`, `_tile_n_candidates`, `_MAX_TILE_N_CANDIDATES`, `tune` param, `tune_by_forward`         | **All removed** — heuristic config selection only; `init_config(config)` (no `tune` arg)                                                                                                                                                                                                                                                                                                          |
| K9  | `threads` in `default_config` dict and `forward` call                                                                           | **Removed** — reduction ops: `default_config` returns `{"block_m", "tile_n"}`; elementwise ops: `default_config` returns `{"block_size"}` (see K11); `forward` does not pass `threads`                                                                                                                                                                                                    |
| K10 | T.macro factories (`make_reduce_epilogue`, `make_softmax_epilogue`, etc.) in `_primitives.py`                                   | **Preserved unchanged** (used by the NPU component later)                                                                                                                                                                                                                                                                                                                                              |
| K11 | `npt` (num_per_thread) in factory signature, `default_config`, and `T.Parallel(threads, npt)` — **elementwise ops only** | **Collapsed into `block_size`** — the GPU `threads * npt` product (per-block element count) becomes a single `block_size` param; factory signature drops `npt`; callable takes `block_size`; `default_config` returns `{"block_size": threads * npt}`; `T.Parallel(block_size)` replaces `T.Parallel(threads, npt)`. **Reduction ops are unaffected** (they have no `npt`). |

### Op class (applies to S4)

| #  | GPU original                                                 | NPU adaptation                                                                                           |
| -- | ------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| O1 | `x.is_cuda` device check                                   | `backend.is_device_tensor(x)` via `get_device_backend()`                                             |
| O2 | `device="cuda"` hard-coded                                 | `backend.name` via `get_device_backend()`                                                            |
| O3 | `tune` param in `__init__` and `_get_or_create_kernel` | **Removed** — kernel constructor takes `(M, N, op_kind, dtype, config=None, device_index=None)` |
| O4 | `from tileops.utils import get_sm_version`                 | `from tileops.device import get_device_backend`                                                        |
| O5 | `from .compile_boundary import register_instance`          | Removed (not used on NPU)                                                                                |
| O6 | Op-specific flow (reshape, validate, dispatch, roofline)     | **Preserved unchanged** — only device-specific calls adapted                                      |

### Workload (applies to S2)

| #  | GPU original                                                    | NPU adaptation                                       |
| -- | --------------------------------------------------------------- | ---------------------------------------------------- |
| W1 | `device="cuda"` in `torch.randn(...)` / `torch.rand(...)` | `device=backend.name` via `get_device_backend()` |

### Test / Benchmark (applies to S5, S6)

| #  | GPU original                                                 | NPU adaptation                                |
| -- | ------------------------------------------------------------ | --------------------------------------------- |
| T1 | `device="cuda"` in manual tensor creation                  | `device=get_device_backend().name`          |
| T2 | `@pytest.mark.skipif(not torch.cuda.is_available(), ...)`  | Remove or replace with NPU availability check |
| T3 | `tune=True` / `tune` param in test/bench parametrization | **Removed**                             |
| T4 | try/except for "No configurations to tune"                   | **Removed** (autotune-only)             |

## Playbook

Execute S1-S7 in order. For each step:

1. **Port from GPU**: copy the op-specific flow from the GPU file identified
   in Phase 0.
2. **Apply NPU adaptations**: apply the standard adaptations from the
   section above.

### Step S1 — Manifest entry

Edit `tileops/manifest/{family}.yaml` (create if absent). Add a top-level
key `{op_name}:`.

**Port from GPU**: Copy the manifest entry from
`{gpu_repo_root}/tileops/manifest/<family>*.yaml`.

- **Preserve verbatim**: `signature` (inputs, outputs, params, shape_rules),
  `roofline` formula (flops, bytes — device-agnostic).
- **Paths**: `source` section paths use `tileops/` (same as the package
  directory); `tests/ops/` and `benchmarks/ops/` paths stay the same pattern.
- **Workloads**: at least 4 entries — 1 smoke (small) + 3 full (realistic
  shapes). First workload is the smoke case. May reuse GPU shapes or adjust
  for NPU memory.

**Naming rules**:

- `op_slug` = snake_case of `op_name` without `Fwd`/`Op` suffix when the
  family groups multiple ops in one file (e.g. `LogSumExpFwdOp` → `softmax.py`
  because it shares the softmax family file). Otherwise use the op name in
  snake_case.
- `kernel_key` = `{op_slug}_fwd` (or as appropriate to match the GPU
  `source.kernel_map` key).

**Multi-input note**: If the op has more than 1 tensor input (e.g.
LerpTensor with `input`, `end`, `weight`), the manifest `signature.inputs`
will have multiple keys. The workload shape key becomes `{input_name}_shape`
for each input. This affects S6 (benchmark parametrization) — see below.

### Step S2 — Workload class

Edit `tileops/workloads/{family}.py` (create if absent).

**Port from GPU**: Copy the workload class from `{gpu_repo_root}/workloads/<family>.py`.

- **Preserve**: input count, generation logic (randn vs rand vs bool vs
  small-range), shape fields, any op-specific input constraints.
- **Apply W1**: `device="cuda"` → `device=backend.name`.

If the GPU workload inherits a GPU base class (e.g. `RandnWorkload`), the
NPU version should inherit the NPU-adapted equivalent
(`tileops.workloads.workload_base.RandnWorkload`). If the GPU workload
is a standalone class with custom `gen_inputs()`, port the class and apply
W1 to every `torch.*` call that hard-codes `device="cuda"`.

### Step S3 — NPU kernel (TileLang-based)

Create `tileops/kernels/{family}/{op_slug}/{op_slug}.py` inside a new
`{op_slug}/` subfolder under the family directory. Each op gets its own
subfolder so that the extracted TileLang kernel file and the kernel class
file live together under `tileops/kernels/{family}/{op_slug}/`. The
subfolder is a Python package (requires an `__init__.py`, created in
Part B below). The file has two parts with different treatment:

#### Part A — TileLang kernel functions (extracted + imported)

The TileLang kernel functions are **extracted** from the GPU repo via the
extraction script and **imported** into the kernel file. No manual stubbing
is needed — the GPU kernel function implementations (signature, docstring,
and `@T.prim_func` body) are pulled in directly as the reference for the
NPU kernel component to reimplement for `target="npuir"`.

**Run the extraction script** (from the NPU project root):

```bash
python scripts/extract_tl_kernel.py \
  --op-name {op_name} \
  --gpu-repo-root {gpu_repo_root} \
  --out tileops/kernels/{family}/{op_slug}
```

This extracts the GPU TileLang kernel functions (the `@tilelang.jit` +
`@T.prim_func` implementations) from
`{gpu_repo_root}/tileops/kernels/<family>/*.py` and writes them to a
single file inside the `{op_slug}/` subfolder. When `--out` is a
directory, the script auto-generates the filename inside it; when
`--out` is a file path, it is used directly. The script prints the list
of extracted function names **and** the output file path (via
`[ok] wrote <path>`) to stdout — **capture both** for the import
statement (below) and the per-kernel output prompts (end of this skill).

**Import the extracted functions** at the top of
`tileops/kernels/{family}/{op_slug}/{op_slug}.py`:

```python
from .{extracted_module} import (
    _{op_slug}_kernel_single,
    _{op_slug}_kernel_tiled,
    _{op_slug}_kernel,
)
```

Where `{extracted_module}` is the module name (filename without `.py`
suffix) of the file written by the extraction script — derived from the
actual output file path printed by the script (the `[ok] wrote <path>`
line). The relative import (`from .{extracted_module}`) works because
both `{op_slug}.py` and the extracted file live in the same `{op_slug}/`
package. The exact set of imported names is the list printed by the
script.

- **Preserve**: the extracted function names, parameter lists, and
  docstrings — these document the algorithm the NPU component will
  reimplement.
- **Apply K1-K4 and K11** from the Standard NPU Adaptations section —
  these adaptations are handled by the NPU kernel component during
  re-implementation, not by this skill.
- **No stubbing**: the imported functions are the full GPU implementations;
  they serve as reference. The NPU component rewrites them for
  `target="npuir"`.
- **Helper function extraction (mandatory)**: If the GPU TileLang kernel
  functions reference helper functions (e.g. shared math helpers, utility
  wrappers, macro factories, or internal dispatchers defined in the GPU
  repo — whether in the same kernel file or in a sibling module such as
  `_primitives.py`), those helper functions **must also be extracted into
  the same file** (`{extracted_file}`) as the kernel functions. The
  `{op_slug}/` package must be self-contained: every name called by the
  extracted kernel functions must be defined within `{extracted_file}`.
  This keeps the implementation flow complete so the NPU kernel component
  can trace the full call chain when reimplementing for `target="npuir"`.
  If the extraction script does not automatically pull in a referenced
  helper, append it manually to `{extracted_file}` and add its name to the
  import statement in `{op_slug}.py`. When manually appending, copy the
  helper verbatim from the GPU repo (signature + body), preserve any
  `@T.macro` / `@T.prim_func` decorators, and follow the GPU import order
  so dependencies resolve top-down.

**Structural guidance** (op-specific — follow the GPU factory structure):

- The extracted function set depends on the op. A reduction op may have
  single-tile + tiled + dispatcher (3 functions); an elementwise op may
  have a single flat variant; a conv op has its own tiling variants.
  **The extraction script pulls all of them into a single file.**
- The extracted callable signature is op-family-specific:
  - **Reduction ops**: `block_m` (single-tile) or `(block_m, tile_n)`
    (tiled) — the NPU component removes `threads` (K3).
  - **Elementwise ops**: `block_size` — the NPU component removes `threads`
    and `npt`, collapsing the GPU `threads * npt` product into
    `block_size` (K3 + K11).
- The dispatcher function (e.g. `_{op_slug}_kernel`) selects between
  variants and **is fully implemented** in the GPU source — it is
  extracted as-is.

#### Part B — custom_op wrapper + Kernel class (fully ported)

The `@torch.library.custom_op` wrapper and the **Kernel class** are ported
in full — they call the imported kernel functions (GPU implementations
extracted in Part A), which the NPU component will reimplement for
`target="npuir"`.

**Port from GPU**: Copy the `custom_op` wrapper, `register_fake`, and the
Kernel class from `{gpu_repo_root}/tileops/kernels/<family>/*.py`.

- **Preserve**: `op_kind` validation, config selection logic, tiling
  heuristics (`_tile_n_for_block_m`, `default_config`), `forward` dispatch
  signature, `_elem_bytes`, `init_config` flow.
- **Apply K5-K11** from the Standard NPU Adaptations section. For
  elementwise ops, K11 applies: replace GPU `{"threads", "num_per_thread"}`
  config with `{"block_size"}` (the `threads * npt` product).

**Structural guidance** (op-specific — follow the GPU kernel, not a fixed
template):

- The `custom_op` wrapper signature must match the kernel's `forward`
  arguments. A single-input kernel wraps `(M, N, dtype_str, block_m, tile_n, x)`; a multi-input kernel wraps `(M, N, dtype_str, block_m, tile_n, a, b, w)`. **Follow the GPU wrapper, removing `threads`.**
- `register_fake` must return a tensor with the correct output shape for
  this op (e.g. `(M,)` for reduction, `(M, N)` for elementwise). **Follow
  the GPU `register_fake`, not a fixed shape.**
- The Kernel class `__init__` constructs and caches the factory callable
  via `_{op_slug}_kernel(...)`. Since the imported functions are GPU
  implementations (CUDA target), they will not run on NPU until the NPU
  component reimplements them for `target="npuir"` — this is expected.
  The class structure (config selection, cache logic) is ported so it is
  ready once the kernel functions are rewritten for NPUIR.

Create `tileops/kernels/{family}/{op_slug}/__init__.py` to make the
`{op_slug}/` subfolder a proper Python package and re-export the kernel
class so that the family-level `__init__.py` can import it by package
name:

```python
from tileops.kernels.{family}.{op_slug}.{op_slug} import {KernelClassName}

__all__ = ["{KernelClassName}"]
```

Then update `tileops/kernels/{family}/__init__.py` to export the kernel
class from the op package:

```python
from tileops.kernels.{family}.{op_slug} import {KernelClassName}
```

This import path is the same as the flat layout would use, because the
op package's `__init__.py` re-exports the class.

### Step S4 — Op class

Create `tileops/ops/{family}/{op_slug}.py`.

**Port from GPU**: Copy the Op class from `{gpu_repo_root}/tileops/ops/<family>/*.py`.

- **Preserve**: constructor signature (params, shapes, dtype), validation
  logic (`_validate`), reshape/transpose strategy, `forward` flow (input
  → reshape → kernel dispatch → output reshape), output shape computation,
  `eval_roofline()` formula, `_get_or_create_kernel` cache logic.
- **Apply O1-O6** from the Standard NPU Adaptations section.

**Structural guidance** (op-specific — follow the GPU Op, not a fixed
template):

- A single-input reduction Op validates one tensor, reshapes to `(M, N)`,
  dispatches a reduction kernel, and reshapes the `(M,)` output back. It
  may use `_multidim.py` helpers for multi-dim reduction.
- A binary Op validates two tensors, broadcasts them, flattens to
  `(N_total,)`, dispatches a 1-D kernel, and reshapes back.
- A multi-input Op (e.g. LerpTensor) validates 3 tensors, broadcasts to a
  common shape, reshapes for the kernel, and restores the broadcast shape.
- **Do not assume any of these structures.** Read the GPU Op and port its
  actual flow.

Update `tileops/ops/{family}/__init__.py` to export the op class.

### Step S5 — Tests

Create `tests/ops/test_{op_slug}.py`.

**Port from GPU**: Copy the test cases from the GPU test file (identified
in Phase 0 via `source.test`, under `{gpu_repo_root}/tests/ops/`).

- **Preserve**: test structure (Fixture params, TestBase subclass,
  `ref_program` logic), param values (shapes, dtypes, op params), test
  case coverage (smoke, full, broadcast, edge cases, dtype rejection,
  roofline check).
- **Apply T1-T4** from the Standard NPU Adaptations section.

**Structural guidance**:

- First 3 params must be `smoke` (fp32, fp16, bf16) — one also `packaging`.
- `ref_program` uses PyTorch as ground truth (upcast to fp32, downcast back)
  — port the GPU `ref_program` logic.
- The test param structure must match the op's input count. A single-input
  op uses `("shape, dtype", ...)`; a multi-input op may use
  `("shape, dtype", ...)` (same shape for all inputs) or
  `("a_shape, b_shape, w_shape, dtype", ...)` (broadcast test). **Follow
  the GPU test structure.**
- Use `_device()` (returns `get_device_backend().name`) for any manual
  tensor creation.

### Step S6 — Benchmark

Create `benchmarks/ops/bench_{op_slug}.py`.

**Port from GPU**: Copy the benchmark function from the GPU bench file
(identified in Phase 0 via `source.bench`, under `{gpu_repo_root}/benchmarks/ops/`).

- **Preserve**: parametrization logic, baseline function, recording calls.
- **Apply T3-T4** from the Standard NPU Adaptations section.

**Single-input vs multi-input parametrization**:

- **Single-input ops** (1 tensor input): use
  `workloads_to_params({op_name}, include_extra=True)` from
  `tileops.benchmark.benchmark_base`. This reads the manifest and
  generates `(shape, dtype, op_params)` params. The manifest's
  `single_input_workload_contract` validates that the op has exactly 1
  tensor input.
- **Multi-input ops** (2+ tensor inputs): `workloads_to_params` raises
  `KeyError` because the contract requires exactly 1 input. Instead, write
  a custom parametrization helper (e.g. `_shape_dtype_params`) that reads
  `load_workloads({op_name})` and generates params manually. See
  `benchmarks/ops/bench_lerp_tensor.py` for the multi-input pattern.

### Step S7 — Verify

Since the kernel functions are extracted GPU implementations (Part A of
S3), verification is split into two tiers:

#### Tier 1 — Structural verification (this skill's responsibility)

Run these commands from the NPU project root:

```bash
# 1. Import check (verifies all imports + class structure are valid)
python -c "
from tileops.kernels.{family}.{op_slug} import {KernelClassName}
from tileops.ops.{family}.{op_slug} import {OpName}
from tileops.workloads.{family} import {OpName}Workload
from tileops.manifest import load_manifest
assert '{OpName}' in load_manifest()
print('OK: imports + structure valid')
"

# 2. Lint (if ruff available)
ruff check tileops/kernels/{family}/ tileops/ops/{family}/ \
  tileops/workloads/{family}.py \
  tests/ops/test_{op_slug}.py benchmarks/ops/bench_{op_slug}.py

# 3. Manifest validation (verifies workload contract)
python -c "
from tileops.manifest import load_workloads
wl = load_workloads('{OpName}')
assert len(wl) >= 4, 'need >= 4 workloads'
print(f'OK: {len(wl)} workloads')
"

# 4. Test collection (verifies test parametrization is valid — does NOT run)
python -m pytest tests/ops/test_{op_slug}.py --collect-only -q
python -m pytest benchmarks/ops/bench_{op_slug}.py --collect-only -q
```

All Tier 1 checks must pass. This confirms the 7 files are structurally
correct and the extracted kernel functions are properly imported — ready
for the NPU kernel component to reimplement for `target="npuir"`.

**Helper completeness check** (mandatory, Part A continuation): After
Tier 1 passes, scan `{extracted_file}` for every callable name invoked
by the extracted kernel functions (any `name(...)` call whose `name` is
not a Python builtin, a TileLang `T.*` API, or a `torch.*`/`tl.*` API).
For each such name, verify it is defined as a `def` (or `@T.macro` /
`@T.prim_func` decorated function) inside `{extracted_file}`. If any
referenced helper is missing, re-run the extraction script with the
broader scope or append the helper manually (see "Helper function
extraction (mandatory)" in S3 Part A), then re-run Tier 1. The
`{op_slug}/` package must be self-contained before the per-kernel
prompts are emitted in the Output Prompt step.

#### Tier 2 — Runtime verification (after NPU kernel component rewrites kernels for NPUIR)

Once the kernel functions are reimplemented for `target="npuir"` by the
NPU kernel component, run:

```bash
# 1. Correctness tests (smoke only first)
python -m pytest tests/ops/test_{op_slug}.py -v -m smoke --tb=short

# 2. Full test suite
python -m pytest tests/ops/test_{op_slug}.py -v --tb=short

# 3. Benchmarks
python -m pytest benchmarks/ops/bench_{op_slug}.py -v --tb=short
```

All tests must pass. Common runtime failure causes (for the NPU kernel
component to address):

- **fp16/bf16 tolerance**: if the kernel computes in the input dtype and
  the reference upcasts to fp32, precision may differ. Fix by adding fp32
  intermediates in the TileLang kernel (cast inputs to fp32, compute,
  cast back).
- **NPUIR vsel shape mismatch**: if the kernel pads N and uses
  `T.if_then_else` for boundary handling, switch to operating on raw N
  (no padding) or select a `tile_n` that divides N evenly.
- **Multi-input `workloads_to_params` KeyError**: multi-input ops cannot
  use `workloads_to_params`; write a custom parametrization helper.

## Reference

All paths below are relative to the NPU project root (working directory)
unless prefixed with `{gpu_repo_root}/`.

### Primary

- **Adaptation guide**: `docs/gpu_to_npu_adaptation.md` (NPU project) —
  the full catalogue of GPU-to-NPU adaptation points (common mechanism
  C1-C28 + per-op O1-O24 for LogSumExp + E1-E15 for LerpTensor). Read this
  before starting.
- **GPU source of truth**: `{gpu_repo_root}/tileops/` — the GPU op's
  end-to-end implementation (manifest, ops, kernels, workloads, tests,
  benchmarks). Consult Phase 0 for the per-artifact path map.
- **Device backend**: `tileops/device.py` — the single device adaptation
  surface; all device-specific code goes through `get_device_backend()`.
- **Kernel base**: `tileops/kernels/kernel_base.py` — NPU-adapted
  `Kernel` base (autotune removed, `init_config(config)` without `tune`).
- **Op base**: `tileops/ops/op_base.py` — NPU-adapted `Op` base
  (arch check via backend name, no `tune`).
- **Reduction primitives**: `tileops/kernels/reduction/_primitives.py` —
  `device_smem_budget`, `align_up`, `compute_tile_n`, T.macro factories
  (all NPU-compatible).

### Reference examples (NPU project, not universal templates)

Study the example closest to your op's family for the end-to-end pattern,
then adapt to your op's specific flow:

- **Reduction family** (single-input, 2-D `(M,N)` reshape, reduction kernel):
  - `tileops/kernels/reduction/logsumexp/` (op package: `logsumexp.py` + `_log_sum_exp_fwd_kernels.py` + `__init__.py`)
  - `tileops/ops/reduction/softmax.py`
  - `tests/ops/test_softmax.py`
  - `benchmarks/ops/bench_softmax.py`

## Output Prompt

After all 7 files are created and **Tier 1 structural verification passes**,
this skill emits **one migration prompt per extracted TileLang kernel
function** for the **TileLang-NPUIR 算子端到端开发编排 Agent**
(`tilelang-op-conductor`). If the op has N kernel functions extracted in
S3 Part A, the skill emits N prompts — one per function.

### Purpose

Each prompt hands off the NPU TileLang kernel *re-implementation* work
for a **single** extracted kernel function (adapting it to
`target="npuir"`) to the conductor agent, which then drives design →
develop → review → optimize stages end-to-end. Each prompt conveys:

- Which op is being migrated and its manifest identity.
- The **single** TileLang kernel function name to reimplement.
- Where the GPU reference implementations were extracted to (the
  single output file from the extraction script, shared across all
  per-kernel prompts for this op).
- Where the op spec (workloads), correctness tests, and benchmarks live
  in the NPU project.
- The programming mode to use (`developer`).

### When to emit

Emit the prompts **once**, immediately after Tier 1 (S7) passes. Emit
**one prompt per extracted kernel function** — if the script extracted N
functions (reported in its stdout), emit N prompts. Do **not** emit if
Tier 1 fails — fix structural issues first. The prompts are not needed
for Tier 2 (runtime verification), which is the conductor's
responsibility after it rewrites the kernel functions for NPUIR.

**Mandatory constraint**: Regardless of whether the TileLang kernel
functions in S3 Part A have already been reimplemented for `target="npuir"`
(by a prior conductor run) or are still GPU implementations, **every
execution of this skill must emit the full set of per-kernel prompts for
the op**. Implementation status never suppresses the prompts — the skill
always outputs them as the final messages so the conductor can (re-)run
the design → develop → review → optimize pipeline on demand for each
kernel independently.

### Template

Fill the placeholders below and output **one text block per extracted
kernel function** as the final messages of the skill invocation (one
block per function, in the order the script reported them). The prompt is
written in Chinese to match the conductor agent's working language.

```
迁移 {op} 算子 {op_name}。 其 TileLang 实现函数 {kernel_function_name} 在 {extracted_file}。
{kernel_function_name} 是对外接口，不要改变接口的参数(除非该参数跟后端实现相关，在迁移时需要适配)。
{op_name} 的规格见 examples/TileOPs/tileops/manifest/{family}.yaml中 {op_name} 的 workloads 部分。
精度用例见 examples/TileOPs/tests/ops/test_{test_slug}.py，
性能用例见 examples/TileOPs/benchmarks/ops/bench_{bench_slug}.py
用 developer 模式实现。
```

### Placeholder resolution

Each placeholder is resolved from values gathered during Phase 0 and S1-S7.
Do not ask the caller for additional input — all values are already known by
the end of S7. The `{kernel_function_name}` placeholder is **per-kernel** —
it takes a different value in each emitted prompt block (one per extracted
function). All other placeholders, including `{extracted_file}`, are
shared across all prompts for the same op.

| Placeholder                | Scope                | Source                                                                                                                                                                                                                                                                                     | Example                                                   |
| -------------------------- | -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------- |
| `{op}`                   | shared               | `op_slug` derived in S1 (snake_case of the op)                                                                                                                                                                                                                                           | `logsumexp`                                             |
| `{op_name}`              | shared               | PascalCase manifest key from Inputs                                                                                                                                                                                                                                                        | `LogSumExpFwdOp`                                        |
| `{kernel_function_name}` | **per-kernel** | A single function name from the extraction script's reported function list in S3 Part A                                                                                                                                                                                                    | `_logsumexp_kernel_single`                              |
| `{extracted_file}`       | shared               | The output file path printed by the extraction script in S3 Part A (the`[ok] wrote <path>` line). When `--out` is a directory, the script auto-generates the filename inside it; when `--out` is a file path, it is used directly. All extracted functions live in this single file. | `tileops/kernels/reduction/logsumexp/_log_sum_exp_fwd_kernels.py` |
| `{family}`               | shared               | Op family directory from Inputs                                                                                                                                                                                                                                                            | `reduction`                                             |
| `{test_slug}`            | shared               | Test file slug extracted from manifest`source.test` (strip `tests/ops/test_` prefix and `.py` suffix)                                                                                                                                                                                | `softmax`                                               |
| `{bench_slug}`           | shared               | Bench file slug extracted from manifest`source.bench` (strip `benchmarks/ops/bench_` prefix and `.py` suffix)                                                                                                                                                                        | `softmax`                                               |

**Notes on resolution**:

- `{kernel_function_name}` is a **single** function name — one of the
  names reported by the extraction script. Each emitted prompt block
  uses exactly one function name. The full set of prompts covers
  **all** variants (e.g. `_{op_slug}_kernel_single`,
  `_{op_slug}_kernel_tiled`) and the dispatcher (e.g. `_{op_slug}_kernel`)
  if it is among the extracted names — each gets its own prompt.
- `{extracted_file}` is the **shared** output file path printed by the
  extraction script (the `[ok] wrote <path>` line). When `--out` is a
  directory (e.g. `tileops/kernels/{family}/{op_slug}`), the script
  auto-generates the filename inside it; when `--out` is a file path, it
  is used directly. This is the single file where all GPU kernel
  functions now live in the NPU project (extracted from the GPU repo by
  the script). Every per-kernel prompt for the same op references this
  same file.
- `{test_slug}` and `{bench_slug}` come from the **NPU** manifest's
  `source.test` / `source.bench` fields (written in S1). They may differ
  from `{op}` when the family groups multiple ops in one test/bench file
  (e.g. `logsumexp` op → `softmax` test file). Always use the manifest
  value, not the op slug.

### Reference example

For `LogSumExpFwdOp` (family `reduction`, GPU repo root
`/home/tilelang/zuochuanuong/TileOPs-fork`), after running:

```bash
python scripts/extract_tl_kernel.py \
  --op-name LogSumExpFwdOp \
  --gpu-repo-root /home/tilelang/zuochuanuong/TileOPs-fork \
  --out tileops/kernels/reduction/logsumexp
```

the script auto-generates the filename and prints
`[ok] wrote tileops/kernels/reduction/logsumexp/_log_sum_exp_fwd_kernels.py`.
The script extracts `_logsumexp_kernel_single` and
`_logsumexp_kernel_tiled`. Two prompts are emitted — one per kernel
function, both referencing the same extracted file:

**Prompt 1** (for `_logsumexp_kernel_single`):

```
迁移 logsumexp 算子 LogSumExpFwdOp。 其 TileLang 实现函数 _logsumexp_kernel_single 在 tileops/kernels/reduction/logsumexp/_log_sum_exp_fwd_kernels.py。
_logsumexp_kernel_single 是对外接口，不要改变接口的参数(除非该参数跟后端实现相关，在迁移时需要适配)。
LogSumExpFwdOp 的规格见 examples/TileOPs/tileops/manifest/reduction.yaml中 LogSumExpFwdOp 的 workloads 部分。
精度用例见 examples/TileOPs/tests/ops/test_softmax.py，
性能用例见 examples/TileOPs/benchmarks/ops/bench_softmax.py
用 developer 模式实现。
```

**Prompt 2** (for `_logsumexp_kernel_tiled`):

```
迁移 logsumexp 算子 LogSumExpFwdOp。 其 TileLang 实现函数 _logsumexp_kernel_tiled 在 tileops/kernels/reduction/logsumexp/_log_sum_exp_fwd_kernels.py。
_logsumexp_kernel_tiled 是对外接口，不要改变接口的参数(除非该参数跟后端实现相关，在迁移时需要适配)。
LogSumExpFwdOp 的规格见 examples/TileOPs/tileops/manifest/reduction.yaml中 LogSumExpFwdOp 的 workloads 部分。
精度用例见 examples/TileOPs/tests/ops/test_softmax.py，
性能用例见 examples/TileOPs/benchmarks/ops/bench_softmax.py
用 developer 模式实现。
```

### Verification of the prompts

Before emitting, sanity-check **each** prompt against the NPU files just
created:

1. `{kernel_function_name}` exists as a `def` statement in the
   extracted file (`{extracted_file}`).
2. `{extracted_file}` exists and is the extracted kernel file containing
   all extracted functions.
3. The number of emitted prompts equals the number of TileLang kernel
   functions reported by the extraction script.
4. `examples/TileOPs/tileops/manifest/{family}.yaml` contains a
   `{op_name}:` key with a `workloads:` section.
5. `examples/TileOPs/tests/ops/test_{test_slug}.py` exists.
6. `examples/TileOPs/benchmarks/ops/bench_{bench_slug}.py` exists.

If any check fails, correct the placeholder value (or the underlying file)
before emitting — the conductor agent will act on each prompt verbatim.

## Machine Mode (migration pipeline)

When this skill is invoked by an automated agent (e.g. `tileops-scaffolder`)
rather than a human caller, the invocation prompt says so explicitly. Machine
mode changes only the **output bookkeeping** — all Phases 0-S7 (including
Tier 1 and the helper completeness check) run unchanged, and the per-kernel
prompts are still emitted as the final messages.

After Tier 1 and the helper completeness check pass, **write the migration
metadata file** (in addition to emitting the prompts):

```
tileops/kernels/{family}/{op_slug}/.migration_meta.json
```

Schema (all fields required unless noted; consumed by
`.agents/skills/add-npu-op/scripts/integrate_kernel.py` and the conductor):

```json
{
  "op_name": "<PascalCase manifest key, e.g. SSDChunkScanFwdOp>",
  "op_slug": "<snake_case op slug, e.g. ssd_chunk_scan>",
  "family": "<family directory, e.g. mamba>",
  "gpu_repo_root": "<caller-provided GPU repo root, absolute>",
  "kernel_class_name": "<Kernel class, e.g. SSDChunkScanFwdKernel>",
  "op_class_name": "<Op class, e.g. SSDChunkScanFwdOp>",
  "test_path": "tests/ops/test_{test_slug}.py",
  "test_slug": "<test slug>",
  "bench_path": "benchmarks/ops/bench_{bench_slug}.py",
  "bench_slug": "<bench slug>",
  "wrapper_path": "tileops/kernels/{family}/{op_slug}/{op_slug}.py",
  "extracted_file": "tileops/kernels/{family}/{op_slug}/{extracted_file_name}",
  "extracted_functions": ["<function names reported by the extraction script>"],
  "created_at": "<ISO 8601 UTC>"
}
```

Rules:

- Paths are relative to the NPU project root (`examples/TileOPs/`).
- `extracted_functions` is exactly the function list printed by the
  extraction script in S3 Part A.
- The meta file is overwritten on each (idempotent) re-run.
- Re-emitting prompts is still mandatory in machine mode (see the
  "Mandatory constraint" above); the meta file supplements, not replaces,
  the prompts.
