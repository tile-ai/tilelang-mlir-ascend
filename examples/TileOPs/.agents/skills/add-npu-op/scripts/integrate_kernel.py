"""Integrate conductor-produced NPU kernels into the TileOPs package.

Copies the per-function kernel files produced by the tilelang-op-conductor
pipeline (``examples/{project}/{func}/{func}.py``) into the TileOPs kernel
package (``tileops/kernels/{family}/{op_slug}/{op_slug}_kernel/``), copies
each function's Stage 1 design doc (``DESIGN.md`` co-located with the kernel
product) into the same package as ``{func}_DESIGN.md``, generates
the aggregation ``__init__.py``, rewrites the wrapper import to point at the
integrated package, and runs an import smoke test. Idempotent: re-running
overwrites integrated copies and leaves an already-rewritten wrapper intact.
A function without a co-located ``DESIGN.md`` is warned about (not fatal)
and its kernel is still integrated.

Usage:
    python integrate_kernel.py --meta .migration_meta.json
    python integrate_kernel.py --op-slug ssd_chunk_scan --family mamba \\
        --functions _ssd_chunk_scan_fwd_kernel
    python integrate_kernel.py --meta .migration_meta.json \\
        --func-dir _kernel_single=examples/myproj/_kernel_single
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


def find_tileops_root() -> Path:
    """This script lives at <tileops_root>/.agents/skills/add-npu-op/scripts/."""
    return Path(__file__).resolve().parents[4]


def load_meta(path: Path) -> dict:
    meta = json.loads(path.read_text(encoding="utf-8"))
    for key in ("op_slug", "family", "wrapper_path"):
        if key not in meta:
            raise SystemExit(f"[error] meta file {path} missing required key '{key}'")
    return meta


def find_conductor_file(
    func: str, examples_root: Path, op_slug: str, overrides: dict[str, Path]
) -> Path:
    """Locate the conductor product file for one kernel function.

    Search order: explicit --func-dir override, the canonical migration slot
    ``examples/{op_slug}/{func}/{func}.py``, then a global glob
    ``examples/*/{func}/{func}.py``.
    """
    if func in overrides:
        p = overrides[func]
        if p.is_dir():
            exact = p / f"{func}.py"
            if exact.is_file():
                return exact.resolve()
            import ast as _ast

            matches = []
            for cand in sorted(p.glob("*.py")):
                if cand.name == "__init__.py":
                    continue
                try:
                    tree = _ast.parse(cand.read_text(encoding="utf-8"))
                except SyntaxError:
                    continue
                if any(isinstance(n, _ast.FunctionDef) and n.name == func for n in tree.body):
                    matches.append(cand)
            if len(matches) == 1:
                return matches[0].resolve()
            raise SystemExit(f"[error] --func-dir dir for {func} has no unique defining file: {p}")
        if not p.is_file():
            raise SystemExit(f"[error] --func-dir override for {func} has no file: {p}")
        return p.resolve()

    candidate = examples_root / op_slug / func / f"{func}.py"
    if candidate.is_file():
        return candidate.resolve()

    matches = sorted(examples_root.glob(f"*/{func}/{func}.py"))
    if len(matches) == 1:
        return matches[0].resolve()
    if not matches:
        raise SystemExit(
            f"[error] no conductor product for '{func}': tried {candidate} and "
            f"glob {examples_root}/*/{func}/{func}.py; pass --func-dir {func}=<path>"
        )
    raise SystemExit(
        f"[error] ambiguous conductor product for '{func}': {matches}; "
        f"pass --func-dir {func}=<path> to disambiguate"
    )


def find_design_doc(kernel_src: Path) -> Path | None:
    """Locate the Stage 1 DESIGN.md for one kernel function.

    The conductor pipeline writes ``DESIGN.md`` into the same directory as
    the kernel product (``examples/{op_slug}/{func}/DESIGN.md``), so the
    kernel source's parent directory is the canonical location. Returns None
    when no design doc is co-located (warned, not fatal).
    """
    candidate = kernel_src.parent / "DESIGN.md"
    return candidate if candidate.is_file() else None


def parse_wrapper_imports(wrapper_path: Path, extracted_module: str) -> list[str]:
    """Return the names the wrapper imports from .{extracted_module}."""
    tree = ast.parse(wrapper_path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level == 1 and module == extracted_module:
                names.extend(a.name for a in node.names)
    return names


def defining_module(name: str, integrated_files: dict[str, Path]) -> str | None:
    """Find the integrated module (file stem) that top-level-defines *name*."""
    for stem, path in integrated_files.items():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in tree.body:
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and node.name == name
            ):
                return stem
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == name:
                        return stem
    return None


def gen_init_py(names: list[str], integrated_files: dict[str, Path], op_slug: str) -> str:
    """Generate the {op_slug}_kernel/__init__.py re-export block."""
    lines = [
        '"""Aggregated NPU kernels migrated by the conductor pipeline.',
        "",
        f"Re-exports the integrated per-function kernel modules produced for {op_slug}.",
        "Generated by integrate_kernel.py; edit the per-function modules instead.",
        '"""',
        "",
    ]
    all_names: list[str] = []
    for name in names:
        stem = defining_module(name, integrated_files)
        if stem is not None:
            lines.append(f"from .{stem} import {name}")
        else:
            lines.append(f"# [warn] {name}: not found in integrated modules")
        all_names.append(name)
    lines.append("")
    lines.append("__all__ = [")
    lines.extend(f'    "{n}",' for n in all_names)
    lines.append("]")
    lines.append("")
    return "\n".join(lines)


def rewrite_wrapper_import(wrapper_path: Path, extracted_module: str, op_slug: str) -> bool:
    """Point the wrapper's .{extracted_module} import at .{op_slug}_kernel.

    Returns True when a rewrite happened, False when already integrated.
    """
    text = wrapper_path.read_text(encoding="utf-8")
    if re.search(rf"from \.{op_slug}_kernel(?:\.\w+)?\s+import", text):
        return False
    pattern = rf"from \.{re.escape(extracted_module)}\s+import"
    if not re.search(pattern, text):
        raise SystemExit(
            f"[error] wrapper {wrapper_path} has no 'from .{extracted_module} import'; "
            f"check --extracted-module or edit the wrapper manually"
        )
    text = re.sub(pattern, f"from .{op_slug}_kernel import", text)
    wrapper_path.write_text(text, encoding="utf-8")
    return True


def smoke_test(
    tileops_root: Path, family: str, op_slug: str, names: list[str], kernel_class: str | None
) -> tuple[bool, str]:
    pkg = f"tileops.kernels.{family}.{op_slug}.{op_slug}_kernel"
    wrapper_mod = f"tileops.kernels.{family}.{op_slug}.{op_slug}"
    cls_check = (
        (
            f"import {wrapper_mod} as w\n"
            f"assert hasattr(w, {kernel_class!r}), 'missing {kernel_class}'\n"
        )
        if kernel_class
        else ""
    )
    code = (
        f"import {pkg} as m\n"
        f"missing = [n for n in {names!r} if not hasattr(m, n)]\n"
        "assert not missing, f'missing exports: {missing}'\n"
        f"{cls_check}"
        "print('smoke-ok', m.__all__)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=tileops_root, capture_output=True, text=True, timeout=300
    )
    out = (proc.stdout + proc.stderr).strip()
    return proc.returncode == 0, out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--meta",
        type=Path,
        default=None,
        help=".migration_meta.json written by the scaffolder step",
    )
    ap.add_argument("--op-slug", default=None)
    ap.add_argument("--family", default=None)
    ap.add_argument(
        "--functions", nargs="+", default=None, help="kernel function names to integrate"
    )
    ap.add_argument(
        "--extracted-module",
        default=None,
        help="stem of the GPU extracted file the wrapper imports from",
    )
    ap.add_argument(
        "--wrapper", type=Path, default=None, help="wrapper file (default: from meta.wrapper_path)"
    )
    ap.add_argument(
        "--func-dir",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="explicit conductor product dir for one function",
    )
    ap.add_argument("--tileops-root", type=Path, default=None)
    ap.add_argument("--examples-root", type=Path, default=None)
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    meta = load_meta(args.meta) if args.meta else {}
    op_slug = args.op_slug or meta.get("op_slug")
    family = args.family or meta.get("family")
    functions = args.functions or meta.get("extracted_functions")
    if not (op_slug and family and functions):
        raise SystemExit("[error] need --meta or (--op-slug --family --functions)")
    tileops_root = (args.tileops_root or find_tileops_root()).resolve()
    examples_root = (args.examples_root or tileops_root.parent).resolve()

    wrapper_path = args.wrapper
    if wrapper_path is None:
        rel = meta.get("wrapper_path")
        if not rel:
            raise SystemExit("[error] need --wrapper or meta.wrapper_path")
        wrapper_path = tileops_root / rel
    wrapper_path = wrapper_path.resolve()
    if not wrapper_path.is_file():
        raise SystemExit(f"[error] wrapper not found: {wrapper_path}")

    extracted_module = args.extracted_module
    if extracted_module is None:
        ext_file = meta.get("extracted_file")
        if ext_file:
            extracted_module = Path(ext_file).stem
        else:
            guess = sorted(wrapper_path.parent.glob("_*_kernels.py"))
            if len(guess) == 1:
                extracted_module = guess[0].stem
            else:
                raise SystemExit("[error] cannot infer extracted module; pass --extracted-module")

    overrides: dict[str, Path] = {}
    for item in args.func_dir:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"[error] bad --func-dir {item!r}, expected NAME=PATH")
        overrides[name] = Path(path)

    wrapper_names = parse_wrapper_imports(wrapper_path, extracted_module)
    if not wrapper_names:
        wrapper_names = list(functions)

    sources = {f: find_conductor_file(f, examples_root, op_slug, overrides) for f in functions}

    target_dir = tileops_root / "tileops" / "kernels" / family / op_slug / f"{op_slug}_kernel"
    integrated: dict[str, Path] = {}
    design_integrated: dict[str, Path] = {}
    print(f"[plan] op_slug={op_slug} family={family} target={target_dir}")
    for func, src in sources.items():
        dst = target_dir / src.name
        integrated[src.stem] = dst
        print(f"[copy] {src} -> {dst}")
        design_src = find_design_doc(src)
        design_dst: Path | None = None
        if design_src is not None:
            design_dst = target_dir / f"{func}_DESIGN.md"
            design_integrated[func] = design_dst
            print(f"[copy] {design_src} -> {design_dst}")
        else:
            print(
                f"[warn] no DESIGN.md co-located with '{func}' product "
                f"(looked in {src.parent}); design doc not integrated"
            )
        if not args.dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)
            for cache in target_dir.glob("__pycache__"):
                shutil.rmtree(cache)
            shutil.copy2(src, dst)
            if design_src is not None and design_dst is not None:
                shutil.copy2(design_src, design_dst)

    init_path = target_dir / "__init__.py"
    init_text = gen_init_py(wrapper_names, integrated, op_slug)
    print(f"[init] {init_path}")
    if not args.dry_run:
        init_path.write_text(init_text, encoding="utf-8")

    if args.dry_run:
        print("[dry-run] wrapper rewrite + smoke skipped")
        print("[ok] dry run complete")
        return

    rewritten = rewrite_wrapper_import(wrapper_path, extracted_module, op_slug)
    print(
        f"[wrapper] {wrapper_path} "
        f"{'rewritten -> .' + op_slug + '_kernel' if rewritten else 'already integrated'}"
    )

    ok, out = (
        (True, "skipped")
        if args.skip_smoke
        else smoke_test(tileops_root, family, op_slug, wrapper_names, meta.get("kernel_class_name"))
    )
    if not ok:
        print(out)
        raise SystemExit("[error] import smoke test FAILED")
    if out != "skipped":
        print(f"[smoke] {out.splitlines()[-1]}")

    summary = {
        "op_slug": op_slug,
        "family": family,
        "functions": functions,
        "sources": {f: str(p) for f, p in sources.items()},
        "design_docs": {
            f: (str(design_integrated[f]) if f in design_integrated else None) for f in functions
        },
        "target_dir": str(target_dir),
        "wrapper": str(wrapper_path),
        "wrapper_rewritten": rewritten,
        "reexported": wrapper_names,
        "smoke": "pass" if out != "skipped" else "skipped",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    print("[json] " + json.dumps(summary))
    print("[ok] integration complete")


if __name__ == "__main__":
    main()
