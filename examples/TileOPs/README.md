# TileOPs

Standalone NPU benchmark framework, extracted from [TileOPs](https://github.com/tile-ai/TileOPs)
(GPU/TileLang-based) and adapted for NPU backends (Ascend via `torch_npu`).

**No dependency on the TileOPs repository.** All code is self-contained.

## Structure

```
TileOPs/
├── tileops/              # Main package
│   ├── device.py           # Device backend abstraction (GPU→NPU adaptation surface)
│   ├── utils/              # Utilities (str2dtype, etc.)
│   ├── manifest/           # Op manifest (standalone YAML spec)
│   ├── workloads/          # Workload definitions (input generation)
│   ├── ops/                # Op layer (validation, reshape, kernel dispatch, roofline)
│   ├── kernels/            # Kernel layer (NPU implementation)
│   ├── testing/            # Test base (correctness vs PyTorch reference)
│   └── benchmark/          # Benchmark base (latency / TFLOPS / bandwidth)
├── tests/                  # Correctness tests
├── benchmarks/             # Performance benchmarks
├── .agents/
│   └── skills/add-npu-op/  # add-npu-op skill (7-file porting guide)
│       └── scripts/        # extract_tl_kernel.py, integrate_kernel.py
├── docs/
│   ├── gpu_to_npu_adaptation.md       # GPU→NPU adaptation points
│   ├── roofline_metrics_analysis.md   # Roofline metric analysis
│   └── roofline_types_and_latency_analysis.md
└── pyproject.toml
```

## Quick Start

```bash
cd TileOPs
pip install -e .[dev]

# Run correctness tests
pytest tests/ -v

# Run benchmarks
pytest benchmarks/ -v
```

## Unified Evaluation Reports

The existing correctness tests and benchmarks remain the execution source of
truth.  ``tileops-report`` only orchestrates them: correctness runs first, the
benchmark runs only after it passes, and the existing pytest/msprof results are
collected into JSON, Markdown, and HTML reports.

```bash
# List manifest operators and their pytest entry points
tileops-report list

# Run the complete correctness suite, then the complete benchmark suite
tileops-report run --all --prof-mode msprof

# Correctness gate -> msprof benchmark -> structured report
tileops-report run --op MishFwdOp --prof-mode msprof --kernel-name main

# Device-event mode (useful when msprof is unavailable)
tileops-report run --op MishFwdOp --prof-mode events

# An operator not yet registered in the manifest can provide both pytest paths
tileops-report run --op PoolFwdOp \
  --test-file tests/ops/test_pool.py \
  --benchmark-file benchmarks/ops/bench_pool.py \
  --prof-mode msprof --kernel-name main
```

Reports are written under ``reports/tileops/<run-id>_<operator>/``.  Each run
contains the canonical ``run.json``, human-readable ``report.md`` and
``report.html``, pytest JUnit/log files, the structured benchmark JSON, and
uniquely named retained msprof artifacts.  Performance benchmarks profile only
the TileOps implementation.  PyTorch reference implementations remain in
``tests/ops`` for correctness validation and are not profiled for performance.

The HTML view follows the CANN-Bench report hierarchy and is filled from
``tileops/reporting/report_template.html`` without an additional template-engine
dependency.  It contains experiment metadata and the detected runtime environment,
an overall pass-rate summary, a per-operator analysis table, and per-operator shape
details.  The per-operator ``Correctness`` cell uses ``Pass Rate (Passed/Total)``,
for example ``100.0% (22/22)``.  ``Ratio Range`` shows the minimum and maximum of
the individual shapes; it is not an average across unrelated shapes.  Correctness
and performance counts are shown separately because pytest cases and benchmark
shapes do not necessarily match.  ``Avg Max Abs Error`` is calculated from the
``max_abs_err`` values emitted by numerical correctness tests.

Performance tables begin with ``Label``, ``Latency``, and ``Ratio``.  The label is
captured from the workload's pytest parameter ID; old records without a label fall
back to the real workload shape.  Shapes and dtypes remain visible alongside it,
while stable hashed ``case_id`` values and full artifact paths remain available in
the expandable details and ``run.json``.  Existing results can be rendered again
without rerunning NPU tests:

```bash
tileops-report render reports/tileops/<run-id>/run.json
```

Report runs must be serial.  Do not pass pytest-xdist ``-n`` or
``--numprocesses`` options: benchmark records are process-local and cannot yet be
merged safely across workers.  The CLI rejects these options instead of producing an
incomplete report.

## Adding a New Op

See `.agents/skills/add-npu-op/SKILL.md` for the step-by-step guide. The skill ports an
op end-to-end from a GPU `TileOPs` repo (caller provides `gpu_repo_root`) and creates 7
files: manifest entry (S1), workload (S2), kernel package (S3), Op class (S4), tests
(S5), benchmark (S6), and package exports (S7).

Machine mode also writes `tileops/kernels/{family}/{op_slug}/.migration_meta.json`
(extracted `@tilelang.jit` functions, wrapper/test/bench slugs) for downstream
agent-driven migration.

## Agent-Driven Migration

The repo-level agent system (`.opencode/agents/`) automates GPU→NPU migration in two
stages:

- **Stage 0 (`tileops-scaffolder`)**: executes the `add-npu-op` skill in machine mode —
  produces the 7-file scaffold plus `.migration_meta.json` with per-kernel migration
  prompts.
- **Stage 1-3 (designer / reviewer / developer)**: each extracted kernel function is
  independently designed, reviewed, and implemented under `examples/{op_slug}/{func}/`
  with embedded L0/L1 precision gates.
- **Stage 5 (`tilelang-op-integrator`)**: runs
  `.agents/skills/add-npu-op/scripts/integrate_kernel.py` to copy verified kernels into
  `tileops/kernels/{family}/{op_slug}/{op_slug}_kernel/` — each function's Stage 1 design
  doc (`DESIGN.md`) is copied alongside the integrated kernel as `{func}_DESIGN.md` —
  rewrite the wrapper import into a baseline/perf_opt kernel-source selection block,
  run pytest (smoke → full), and report benchmarks.

## Baseline vs perf_opt Kernel Selection

Each kernel wrapper (`tileops/kernels/{family}/{op_slug}/{op_slug}.py`) carries a
**kernel source selection block**: two import paths side by side — the Stage 3
baseline kernel and the Stage 4 tuned `perf_opt` kernel — with exactly one active.
Selection is done by swapping the comment:

```python
# --- baseline (Stage 3) ---
# from .{op_slug}_kernel import {func}
# --- perf_opt (Stage 4 tuned) ---
from .{op_slug}_kernel.perf_opt.{func} import {func}
```

- Default policy: the `perf_opt` source is active once the tuned kernel (a drop-in
  replacement with the same factory signature, produced by the optimize stage under
  `{op_slug}_kernel/perf_opt/`) has passed its L0/L1 regression; the baseline source
  is active otherwise.
- `pytest tests/ops/` and `pytest benchmarks/ops/` dispatch through whichever source
  is active, so benchmarks measure the adopted kernel without any other code change.
- When the two kernel versions ship different tuned default parameters (e.g.
  `block_size`), the paired default assignment inside the block is toggled together
  with the import (see `tileops/kernels/elementwise/mish/mish.py` for an example).
- The baseline kernel files are never modified; rolling back means flipping the
  comment back to the baseline import.

See `.opencode/agents/tilelang-op-conductor.md` for the full stage-gate orchestration.
