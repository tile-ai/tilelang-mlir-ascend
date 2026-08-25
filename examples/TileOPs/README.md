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
  rewrite wrapper imports, run pytest (smoke → full), and report benchmarks.

See `.opencode/agents/tilelang-op-conductor.md` for the full stage-gate orchestration.
