"""msprof op-based kernel profiling for NPU.

Provides :func:`bench_kernel_msprof` — an alternative to the events-based
:func:`bench_kernel` that uses ``msprof op`` to obtain pure kernel latency
(no launch overhead).

Workflow:
  1. Generate a standalone Python script that reconstructs the functor
     and runs it ``warm_up + launch_count`` times.
  2. Execute the script under::

     msprof op --kernel-name=xxx --output=xxx \
               --launch-count=10 --warm-up=5 python xxx.py

  3. Parse ``OpBasicInfo*.csv`` in the output directory to extract
     ``Task Duration(us)``.
  4. Return the median latency across measured launches in **milliseconds**.

Configuration (environment variables):
  TILEOPS_PROF_MODE           — "msprof" (default) or "events"
  TILEOPS_MSPROF_KERNEL_NAME  — explicit kernel name filter for
                                --kernel-name (overrides the default
                                derived from @T.prim_func via the
                                JitKernel_NPU object graph)
  TILEOPS_MSPROF_LAUNCH_COUNT — number of measured launches (default 10)
  TILEOPS_MSPROF_WARM_UP      — number of warm-up iterations (default 5)
  TILEOPS_MSPROF_OUTPUT_DIR   — persistent output directory (optional)
  TILEOPS_MSPROF_KEEP_OUTPUT  — "1" to keep msprof output (default "0")
"""

from __future__ import annotations

import csv
import glob
import inspect
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
from typing import Any, Callable, Optional

import torch

from tileops.device import get_device_backend

__all__ = ["bench_kernel_msprof"]

_logger = logging.getLogger("tileops.benchmark.msprof")


# ---------------------------------------------------------------------------
# Functor introspection helpers
# ---------------------------------------------------------------------------


def _is_serializable(val: Any) -> bool:
    """Check if *val* can be serialized as a Python literal in generated code."""
    if val is None:
        return True
    if isinstance(val, bool):
        return True
    if isinstance(val, (int, float, str)):
        return True
    if isinstance(val, torch.dtype):
        return True
    if isinstance(val, (list, tuple)):
        return all(_is_serializable(x) for x in val)
    return False


def _val_repr(val: Any) -> str:
    """Convert *val* to its Python source repr for code generation."""
    if isinstance(val, torch.dtype):
        return repr(val)
    if isinstance(val, str):
        return repr(val)
    if isinstance(val, (list, tuple)):
        return "[" + ", ".join(_val_repr(x) for x in val) + "]"
    return repr(val)


def _is_op_instance(functor: Any) -> bool:
    """Check if *functor* is an Op-like instance (has ``forward`` and is not a function)."""
    if inspect.isfunction(functor) or inspect.ismethod(functor):
        return False
    return (
        hasattr(functor, "forward") and callable(functor.forward) and hasattr(functor, "__class__")
    )


def _is_importable_callable(functor: Callable) -> bool:
    """Check if *functor* is a module-level callable that can be imported by name."""
    if not callable(functor):
        return False
    mod = getattr(functor, "__module__", None)
    name = getattr(functor, "__name__", None)
    if not mod or not name:
        return False
    qual = getattr(functor, "__qualname__", "")
    return "<locals>" not in qual and "<lambda>" not in qual


def _extract_op_init_args(op: Any) -> dict:
    """Extract serializable constructor args from an Op instance.

    Introspects ``type(op).__init__`` and reads the corresponding attributes
    from ``op``.  Only simple serializable types are included; complex objects
    (kernels, kernel maps, caches) are skipped.
    """
    cls = type(op)
    try:
        sig = inspect.signature(cls.__init__)
    except (ValueError, TypeError):
        return {}

    init_args: dict = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if hasattr(op, name):
            val = getattr(op, name)
            if _is_serializable(val):
                init_args[name] = val
    return init_args


# ---------------------------------------------------------------------------
# Script generation
# ---------------------------------------------------------------------------


def _generate_op_script(
    op: Any,
    inputs_path: str,
    warm_up: int,
    launch_count: int,
    device_str: str,
    device_index: int,
) -> str:
    """Generate a Python script that reconstructs an Op and runs it."""
    cls = type(op)
    module = cls.__module__
    class_name = cls.__name__
    init_args = _extract_op_init_args(op)

    args_src = ", ".join(f"{k}={_val_repr(v)}" for k, v in init_args.items())

    lines = [
        "# Auto-generated msprof profiling script (Op reconstruction)",
        "import os",
        "import warnings",
        'warnings.filterwarnings("ignore")',
        "",
        "import torch",
    ]
    if device_str == "npu":
        lines.append("import torch_npu  # noqa: F401")
        lines.append(f"torch.npu.set_device({device_index})")
    lines.extend(
        [
            "",
            f"from {module} import {class_name}",
            "",
            f"inputs = torch.load({repr(inputs_path)})",
            "if not isinstance(inputs, (list, tuple)):",
            "    inputs = (inputs,)",
            f"inputs = [x.to({repr(device_str)}) if isinstance(x, torch.Tensor) else x for x in inputs]",
            "",
            f"op = {class_name}({args_src})",
            "",
            "# Ensure all JIT compilation / setup kernels are flushed",
            "torch.npu.synchronize()",
            "",
            "# Warm up",
            f"for _ in range({warm_up}):",
            "    op(*inputs)",
            "torch.npu.synchronize()",
            "",
            "# Measured launches",
            f"for _ in range({launch_count}):",
            "    op(*inputs)",
            "torch.npu.synchronize()",
        ]
    )
    return "\n".join(lines) + "\n"


def _generate_callable_script(
    functor: Callable,
    inputs_path: str,
    warm_up: int,
    launch_count: int,
    device_str: str,
    device_index: int,
) -> str:
    """Generate a Python script that imports a callable and runs it."""
    module = functor.__module__
    func_name = functor.__name__

    lines = [
        "# Auto-generated msprof profiling script (callable import)",
        "import os",
        "import warnings",
        'warnings.filterwarnings("ignore")',
        "",
        "import torch",
    ]
    if device_str == "npu":
        lines.append("import torch_npu  # noqa: F401")
        lines.append(f"torch.npu.set_device({device_index})")
    lines.extend(
        [
            "",
            f"import {module} as _mod",
            f"_fn = getattr(_mod, {repr(func_name)})",
            "",
            f"inputs = torch.load({repr(inputs_path)})",
            "if not isinstance(inputs, (list, tuple)):",
            "    inputs = (inputs,)",
            f"inputs = [x.to({repr(device_str)}) if isinstance(x, torch.Tensor) else x for x in inputs]",
            "",
            "# Warm up",
            f"for _ in range({warm_up}):",
            "    _fn(*inputs)",
            "torch.npu.synchronize()",
            "",
            "# Measured launches",
            f"for _ in range({launch_count}):",
            "    _fn(*inputs)",
            "torch.npu.synchronize()",
        ]
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------


def _parse_op_basic_info(output_dir: str) -> tuple[list[float], str, list[str]]:
    """Return durations, the first op name, and all captured op names.

    The msprof output directory has the structure::

        <output_dir>/
          └── OPPROF_<timestamp>_<random>/
              └── <OpName>/
                  ├── 0/OpBasicInfo_<ts>.csv
                  ├── 1/OpBasicInfo_<ts>.csv
                  ...
                  └── N/OpBasicInfo_<ts>.csv

    Each CSV has one row with a ``Task Duration(us)`` column.
    """
    csv_files = sorted(
        glob.glob(
            os.path.join(output_dir, "**", "OpBasicInfo*.csv"),
            recursive=True,
        )
    )
    if not csv_files:
        raise FileNotFoundError(f"No OpBasicInfo*.csv found under {output_dir}")

    durations: list[float] = []
    op_name = ""
    op_names: set[str] = set()
    for csv_path in csv_files:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                val = row.get("Task Duration(us)")
                if val is not None and val.strip():
                    try:
                        durations.append(float(val))
                    except ValueError:
                        _logger.warning(
                            "Non-numeric Task Duration(us) in %s: %r",
                            csv_path,
                            val,
                        )
                name_val = row.get("Op Name", "")
                if name_val and name_val.strip():
                    captured_name = name_val.strip()
                    op_names.add(captured_name)
                    if not op_name:
                        op_name = captured_name

    if not durations:
        raise ValueError(
            f"No Task Duration(us) values found in OpBasicInfo CSVs under {output_dir}"
        )

    return durations, op_name, sorted(op_names)


# ---------------------------------------------------------------------------
# Kernel-name resolution (explicit first, @T.prim_func-derived default last)
# ---------------------------------------------------------------------------


def _derive_kernel_name_from_prim_func(
    functor: Any,
    max_depth: int = 5,
    max_nodes: int = 128,
) -> Optional[str]:
    """Derive the default msprof kernel name from the functor's kernel graph.

    TileLang names the compiled device kernel after the ``@T.prim_func``
    function, so the default ``msprof op --kernel-name`` filter is exactly
    that name.  Two complementary passes recover it:

    **Pass 1 (runtime object graph)**: bounded BFS over instance
    attributes and closure cells looking for an object exposing a string
    ``kernel_name`` (``tilelang.jit.jit_npu.JitKernel_NPU``).  This hits
    kernels that are already materialized and reachable -- Developer-mode
    ops whose ``Kernel.kernel`` *is* the JIT kernel, or any state that
    keeps the built kernel alive.

    **Pass 2 (static source scan)**: during the same BFS, record the
    module file of every visited function (``func.__globals__['__file__']``
    -- the tilelang kernel factory chain lives in the kernel module), then
    regex-scan those files for ``@T.prim_func``-decorated defs.  This hits
    Expert-mode ops whose ``Kernel.kernel`` is only a *middle factory*
    (the ``JitKernel_NPU`` is created and discarded inside ``forward`` per
    config, so no runtime object carries it at bench time).

    A single distinct name across all candidate modules is returned;
    ambiguity (or nothing found) returns ``None`` -- the caller then runs
    msprof unfiltered with a loud warning in ``benchmark_base``.
    """
    _PRIM_FUNC_DEF_RE = re.compile(
        r"@T\.prim_func[^\n]*\n(?:[ \t]*@[\w.]+[^\n]*\n)*[ \t]*def\s+([A-Za-z_]\w*)"
    )

    def _scan_prim_func_names(module_file: Any) -> list[str]:
        if not isinstance(module_file, str) or not module_file.endswith(".py"):
            return []
        try:
            with open(module_file, "r", encoding="utf-8") as f:
                source = f.read()
        except (OSError, UnicodeDecodeError):
            return []
        names: list[str] = []
        for match in _PRIM_FUNC_DEF_RE.finditer(source):
            if match.group(1) not in names:
                names.append(match.group(1))
        return names

    def _bfs(root: Any) -> tuple[Optional[str], list[str]]:
        seen: set[int] = set()
        module_files: list[str] = []
        queue: list[tuple[Any, int]] = [(root, 0)]
        while queue and len(seen) < max_nodes:
            obj, depth = queue.pop(0)
            if depth > max_depth or id(obj) in seen:
                continue
            seen.add(id(obj))
            name = getattr(obj, "kernel_name", None)
            if isinstance(name, str) and name:
                return name, module_files
            if isinstance(obj, types.FunctionType):
                module_file = obj.__globals__.get("__file__")
                if isinstance(module_file, str) and module_file not in module_files:
                    module_files.append(module_file)
            if isinstance(obj, dict):
                # e.g. Op._kernel_cache {key: Kernel instance} (lazy-commit
                # ops keep built kernels here instead of self.kernel)
                for value in obj.values():
                    queue.append((value, depth + 1))
            elif isinstance(obj, (list, tuple, set, frozenset)):
                for value in obj:
                    queue.append((value, depth + 1))
            elif hasattr(obj, "__dict__"):
                for value in vars(obj).values():
                    queue.append((value, depth + 1))
            for cell in getattr(obj, "__closure__", None) or ():
                try:
                    queue.append((cell.cell_contents, depth + 1))
                except ValueError:
                    # empty closure cell
                    continue
        return None, module_files

    candidate_files: list[str] = []
    for root in (getattr(functor, "kernel", None), functor):
        if root is None:
            continue
        name, files = _bfs(root)
        if name:
            return name
        candidate_files.extend(f for f in files if f not in candidate_files)

    distinct_names: list[str] = []
    for module_file in candidate_files:
        for name in _scan_prim_func_names(module_file):
            if name not in distinct_names:
                distinct_names.append(name)
    if len(distinct_names) == 1:
        return distinct_names[0]
    return None


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------


def bench_kernel_msprof(
    functor: Callable,
    args: tuple = (),
    *,
    kernel_name: Optional[str] = None,
    launch_count: Optional[int] = None,
    warm_up: Optional[int] = None,
    output_dir: Optional[str] = None,
    keep_output: bool = False,
) -> tuple[float, str]:
    """Benchmark a kernel using ``msprof op``.

    Generates a standalone Python script that reconstructs *functor* and
    runs it ``warm_up + launch_count`` times, executes the script under
    ``msprof op``, then parses ``OpBasicInfo*.csv`` to extract
    ``Task Duration(us)``.

    Args:
        functor: An Op instance (reconstructed via constructor introspection)
            or a module-level callable (imported by ``__module__`` + ``__name__``).
        args: Tuple of input arguments (tensors saved via ``torch.save``).
        kernel_name: Kernel name filter for ``--kernel-name``.  If ``None``,
            resolved in order (explicit values win, derived default last):

            1. this argument (programmatic, per-call);
            2. ``functor.msprof_kernel_name`` (Op-level declaration, e.g.
               multi-kernel dispatch pointing at the kernel its forward
               actually invokes);
            3. ``functor.kernel.msprof_kernel_name`` (Kernel-level
               declaration);
            4. the ``TILEOPS_MSPROF_KERNEL_NAME`` env var (manual session
               override);
            5. the derived default: the ``@T.prim_func`` function name,
               read from the ``JitKernel_NPU`` object in the functor's
               kernel object graph.

            ``None`` after all of the above means no filter (legacy
            behavior; ``benchmark_base`` warns if multiple ops are
            captured).
        launch_count: Number of measured kernel launches (default from
            ``TILEOPS_MSPROF_LAUNCH_COUNT`` or 10).
        warm_up: Number of warm-up iterations (default from
            ``TILEOPS_MSPROF_WARM_UP`` or 5).
        output_dir: msprof output directory.  If ``None``, a temp dir is used.
        keep_output: If ``True``, keep the msprof output directory after
            parsing (copied to ``./msprof_output`` if *output_dir* is temp).

    Returns:
        ``(latency_ms, prof_output_dir)`` where *latency_ms* is the median
        kernel latency across measured launches in milliseconds and
        *prof_output_dir* is the path
        to the msprof output directory (containing ``OPPROF_*`` subdirs).
        On success the temp workspace is kept so the caller can parse
        ``visualize_data.bin``; the caller is responsible for cleanup via
        :func:`_cleanup_msprof_output`.

    Raises:
        RuntimeError: If ``msprof`` is not found or the profiling fails.
        FileNotFoundError: If no ``OpBasicInfo`` CSV is found.
        TypeError: If *functor* is not an Op instance or importable callable.
    """
    if not isinstance(args, tuple):
        raise TypeError(f"bench_kernel_msprof expects a tuple of args, got {type(args).__name__}.")

    launch_count = (
        launch_count
        if launch_count is not None
        else int(os.environ.get("TILEOPS_MSPROF_LAUNCH_COUNT", "10"))
    )
    warm_up = warm_up if warm_up is not None else int(os.environ.get("TILEOPS_MSPROF_WARM_UP", "5"))

    msprof_bin = shutil.which("msprof")
    if msprof_bin is None:
        raise RuntimeError(
            "msprof not found in PATH. Install Ascend profiler tools "
            "(usually under /usr/local/Ascend/.../tools/profiler/bin/)."
        )

    backend = get_device_backend()
    device_str = backend.name
    try:
        device_index = backend.current_device()
    except Exception:
        device_index = 0

    # --- Determine functor type and generate script -----------------------
    is_op = _is_op_instance(functor)
    is_callable = _is_importable_callable(functor) if not is_op else False

    if not is_op and not is_callable:
        raise TypeError(
            f"bench_kernel_msprof: functor must be an Op instance or a "
            f"module-level callable; got {type(functor).__name__} "
            f"(module={getattr(functor, '__module__', '?')}, "
            f"qualname={getattr(functor, '__qualname__', '?')}). "
            f"Closures and lambdas are not supported — use events mode."
        )

    # Kernel-name resolution: explicit values win, derived default last
    # (see the bench_kernel_msprof docstring for the full chain).  NOT
    # from _op_name: tilelang compiled kernels are named after the
    # @T.prim_func function, not the op name.  Empty-string declarations
    # (the Kernel base-class default) are normalized to None.
    if kernel_name is None and is_op:
        kernel_name = getattr(functor, "msprof_kernel_name", None) or None
    if kernel_name is None and is_op:
        kernel = getattr(functor, "kernel", None)
        kernel_name = getattr(kernel, "msprof_kernel_name", None) or None
    if kernel_name is None:
        kernel_name = os.environ.get("TILEOPS_MSPROF_KERNEL_NAME") or None
    if kernel_name is None and is_op:
        kernel_name = _derive_kernel_name_from_prim_func(functor)

    # --- Create temp workspace --------------------------------------------
    tmp_dir = tempfile.mkdtemp(prefix="tileops_msprof_")
    _success = False
    user_output = None
    copied_to_persistent = False
    try:
        inputs_path = os.path.join(tmp_dir, "inputs.pt")
        script_path = os.path.join(tmp_dir, "prof_script.py")

        # Save inputs (move tensors to CPU for portability)
        saved_args = [a.cpu() if isinstance(a, torch.Tensor) else a for a in args]
        torch.save(saved_args, inputs_path)

        # Generate the profiling script
        if is_op:
            script = _generate_op_script(
                functor,
                inputs_path,
                0,
                1,
                device_str,
                device_index,
            )
        else:
            script = _generate_callable_script(
                functor,
                inputs_path,
                0,
                1,
                device_str,
                device_index,
            )

        with open(script_path, "w") as f:
            f.write(script)

        # --- Set up output directory --------------------------------------
        user_output = output_dir or os.environ.get("TILEOPS_MSPROF_OUTPUT_DIR")
        if user_output:
            os.makedirs(user_output, exist_ok=True)
            prof_output_dir = user_output
        else:
            prof_output_dir = os.path.join(tmp_dir, "msprof_output")
            os.makedirs(prof_output_dir, exist_ok=True)
        # msprof refuses to profile into a group/world-writable directory
        # (and still exits 0, producing no CSV); force owner-only permissions
        # regardless of umask (e.g. the default 0002 yields 775).
        os.chmod(prof_output_dir, 0o700)

        # --- Prepare subprocess environment --------------------------------
        env = dict(os.environ)
        cwd = os.getcwd()
        pypath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{cwd}:{pypath}" if pypath else cwd

        # --- Run msprof (with retry if --kernel-name yields no CSV) -------
        durations_us: list[float] = []
        op_name = ""
        captured_op_names: list[str] = []
        tried_kernel_names: list[Optional[str]] = []

        # Build the list of kernel-name candidates to try:
        # 1. User-provided / auto-detected kernel_name
        # 2. None (no --kernel-name filter, profile all kernels)
        candidates = [kernel_name]
        if kernel_name is not None:
            candidates.append(None)

        for kn in candidates:
            tried_kernel_names.append(kn)
            # Clean output dir for retry
            if os.path.exists(prof_output_dir):
                for entry in os.listdir(prof_output_dir):
                    entry_path = os.path.join(prof_output_dir, entry)
                    if os.path.isdir(entry_path):
                        shutil.rmtree(entry_path, ignore_errors=True)

            cmd = [msprof_bin, "op"]
            if kn is not None:
                cmd.append(f"--kernel-name={kn}")
            cmd.append(f"--output={prof_output_dir}")
            cmd.append(f"--launch-count={launch_count}")
            cmd.append(f"--warm-up={warm_up}")
            # These two lines below avoid excessive profiling time caused by parsing
            # too much data; here we only care about latency, not the detailed specifics.
            cmd.append("--aic-metrics=Roofline")
            cmd.append("--dump=off")
            cmd.extend([sys.executable, script_path])

            _logger.info("msprof command: %s", " ".join(cmd))

            print("msprof command: ", " ".join(cmd))

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                env=env,
            )

            if result.returncode != 0:
                _logger.warning(
                    "msprof op returned exit %d with --kernel-name=%s; stderr: %s",
                    result.returncode,
                    repr(kn),
                    result.stderr.strip()[:500],
                )
                continue

            try:
                durations_us, op_name, captured_op_names = _parse_op_basic_info(prof_output_dir)
            except (FileNotFoundError, ValueError) as e:
                _logger.warning(
                    "No OpBasicInfo CSV found with --kernel-name=%s: %s",
                    repr(kn),
                    e,
                )
                continue

            break  # success

        if not durations_us:
            raise RuntimeError(
                f"msprof op failed to produce OpBasicInfo CSV data.\n"
                f"Tried kernel names: {tried_kernel_names}\n"
                f"--- last stdout ---\n{result.stdout}\n"
                f"--- last stderr ---\n{result.stderr}\n"
                f"--- script ---\n{script}"
            )

        # --- Compute latency ----------------------------------------------
        durations_us.sort()
        median_us = durations_us[len(durations_us) // 2]
        latency_ms = median_us / 1000.0

        _logger.info(
            "msprof: op=%s kernel_name=%s samples=%d "
            "min=%.3f median=%.3f max=%.3f us -> latency_ms=%.4f",
            op_name or "(unknown)",
            kn if durations_us else "(none)",
            len(durations_us),
            durations_us[0],
            median_us,
            durations_us[-1],
            latency_ms,
        )

        # --- Optionally persist output -----------------------------------
        if (keep_output or os.environ.get("TILEOPS_MSPROF_KEEP_OUTPUT") == "1") and not user_output:
            artifact_root = os.environ.get(
                "TILEOPS_MSPROF_ARTIFACT_ROOT", os.path.join(cwd, "msprof_output")
            )
            os.makedirs(artifact_root, exist_ok=True)
            artifact_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", op_name or kernel_name or "kernel")
            persistent = tempfile.mkdtemp(prefix=f"{artifact_name}_", dir=artifact_root)
            try:
                shutil.copytree(prof_output_dir, persistent, dirs_exist_ok=True)
                _logger.info("msprof output copied to %s", persistent)
                prof_output_dir = persistent
                copied_to_persistent = True
            except Exception as e:
                _logger.warning("Failed to copy msprof output: %s", e)

        metadata_path = os.path.join(prof_output_dir, "tileops_msprof_metadata.json")
        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "kernel_name_requested": kernel_name,
                        "kernel_name_resolved": kn,
                        "kernel_name_attempts": tried_kernel_names,
                        "op_name": op_name or None,
                        "captured_op_names": captured_op_names,
                        "captured_op_count": len(captured_op_names),
                        "sample_count": len(durations_us),
                        "latency_us": median_us,
                        "min_latency_us": durations_us[0],
                        "max_latency_us": durations_us[-1],
                    },
                    f,
                    indent=2,
                )
        except OSError as e:
            _logger.warning("Failed to write msprof metadata: %s", e)

        _success = True
        return latency_ms, prof_output_dir

    finally:
        should_keep = keep_output or os.environ.get("TILEOPS_MSPROF_KEEP_OUTPUT") == "1"
        if user_output or copied_to_persistent or not should_keep and not _success:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        elif not should_keep:
            _logger.debug(
                "msprof temp files kept at %s for roofline parsing (caller cleanup)",
                tmp_dir,
            )
        else:
            _logger.info("msprof temp files kept at %s", tmp_dir)
