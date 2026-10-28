"""Ascend 910 (A3) JIT: kernel metadata, launch/load artifacts and execution.

This module is the v0.1.15 counterpart of the legacy ``tilelang/jit/jit_npu.py``.
The legacy class pair ``compiler_npu`` + ``JitKernel_NPU`` is preserved as
``Ascend910Compiler`` + ``Ascend910JitKernel`` so the generated launch/load
sources and the compiler command lines stay byte-compatible with the golden
corpus, while the TIR side uses ``tvm.tirx`` (v0.1.15) instead of ``tvm.tir``.
"""

from __future__ import annotations

import ctypes
import importlib.util
import contextlib
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import torch

from tvm import tirx

from tilelang.utils.npu_utils import (
    NPUUtils,
    build_npu_ext,
    get_npu_launcher_header,
    get_runtime_file_cache,
    last_build_command,
    precompile_npu_ext,
    safe_copy,
)

from . import compiler as bishengir
from .runtime import generate_npu_wrapper_src

__all__ = [
    "Ascend910Compiler",
    "Ascend910JitKernel",
    "check_debug_op",
    "parse_metadata",
    "parse_signature",
]

_EXPERT_MODES = {"", "expert", "exp", "e"}

#: Suffix of the callback bishengir-compile emits for kernels that need scratch
#: memory.  Its return value is the workspace size in bytes *per block*.
_WORKSPACE_SUFFIX = "_infer_workspace_shape_function"


def _find_infer_symbols(lib_path: str, suffix: str) -> list[str]:
    """Symbols in *lib_path* ending with *suffix* (dynamic table first)."""

    found: list[str] = []
    for nm_args in (["-D"], []):
        try:
            result = subprocess.run(
                ["nm", *nm_args, lib_path],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            continue
        if result.returncode != 0:
            continue
        for line in result.stdout.splitlines():
            parts = line.strip().split()
            if len(parts) >= 3 and parts[2].endswith(suffix) and parts[2] not in found:
                found.append(parts[2])
        if found:
            break
    return found


def infer_workspace_size(lib_path: str | os.PathLike[str] | None, suffix: str = _WORKSPACE_SUFFIX) -> int:
    """Workspace bytes per block declared by a compiled kernel's shared object.

    ``0`` means the kernel needs no scratch memory, in which case the launcher
    passes ``nullptr`` instead of allocating.  The callback is only generated for
    kernels that actually need a workspace, so its absence is meaningful and must
    not be papered over with a fixed size.
    """

    if lib_path is None:
        return 0
    lib_path = os.fspath(lib_path)
    if not os.path.exists(lib_path):
        return 0
    symbols = _find_infer_symbols(lib_path, suffix)
    if not symbols:
        return 0
    try:
        lib = ctypes.CDLL(lib_path)
    except OSError:
        return 0
    for func_name in symbols:
        try:
            func = getattr(lib, func_name)
            func.restype = ctypes.c_int64
            func.argtypes = []
            size = int(func())
        except (AttributeError, OSError, TypeError):
            continue
        return size if size > 0 else 0
    return 0


def ascend910_mode() -> str:
    """Return ``expert`` or ``developer`` for the current process."""

    mode = os.environ.get("TILELANG_ASCEND_MODE", "expert").strip().lower()
    return "expert" if mode in _EXPERT_MODES else "developer"


def normalize_out_idx(out_idx, total_params: int):
    if out_idx is None:
        return None
    if isinstance(out_idx, int):
        out_idx = [out_idx]
    elif isinstance(out_idx, (list, tuple)):
        out_idx = list(out_idx)
    else:
        raise TypeError(f"out_idx must be int, list, or tuple; got {type(out_idx).__name__}")
    normalized = []
    for i in out_idx:
        idx = i if i >= 0 else total_params + i
        if idx < 0 or idx >= total_params:
            raise ValueError(f"out_idx {i} is out of bounds for kernel with {total_params} parameters")
        normalized.append(idx)
    return normalized


# ---------------------------------------------------------------------------
# MLIR metadata
# ---------------------------------------------------------------------------
_KERNEL_NAME_RE = re.compile(r"func\.func\s+@(\w+)")
_MIX_MODE_RE = re.compile(r"#hivm\.module_core_type<([^>]+)>")
_MIX_SUFFIX_RE = re.compile(r"_(mix_aic|mix_aiv)$")
_SIG_RE = re.compile(r"func\.func\s*@[^(]*\(([^)]*)\)")

_SIG_TYPES = (
    "i1",
    "i8",
    "i16",
    "i32",
    "i64",
    "u32",
    "u64",
    "fp16",
    "bf16",
    "fp32",
    "f32",
    "fp64",
    "f16",
)


def parse_metadata(mlir: str) -> dict[str, Any]:
    """Extract kernel name, runtime name and core type from the MLIR text."""

    name_match = _KERNEL_NAME_RE.search(mlir)
    if name_match is None:
        raise ValueError("cannot find the kernel entry point in the generated MLIR")
    kernel_name = name_match.group(1)
    mix_match = _MIX_MODE_RE.search(mlir)
    mix_mode = mix_match.group(1).lower() if mix_match else "aiv"
    return {
        "kernel_name": kernel_name,
        "name": _MIX_SUFFIX_RE.sub("", kernel_name),
        "mix_mode": mix_mode,
        "shared": 1,
        "tensor_kinds": [],
    }


def splice_signature_params(params_str: str) -> list[str]:
    params: list[str] = []
    current = ""
    brace = 0
    angle = 0
    for char in params_str:
        if char == "," and brace == 0 and angle == 0:
            params.append(current.strip())
            current = ""
            continue
        current += char
        if char == "{":
            brace += 1
        elif char == "}":
            brace -= 1
        elif char == "<":
            angle += 1
        elif char == ">":
            angle -= 1
    if current:
        params.append(current.strip())
    return params


def parse_signature(mlir: str, *, developer: bool = False) -> dict[int, str]:
    """Parse the user-visible kernel signature out of the MLIR function header.

    The compiler inserts three leading arguments (ffts_addr, syncBlockLock,
    workspace) and six trailing ones; the A3 path keeps the legacy slicing.
    """

    match = _SIG_RE.search(mlir)
    if not match:
        return {}
    params = splice_signature_params(match.group(1))
    compiler_inserted_params = 3
    result: dict[int, str] = {}
    index = 0
    for param in params[compiler_inserted_params:-6]:
        found_type = None
        for t_type in _SIG_TYPES:
            if re.search(r"\bx" + t_type + r"\b", param):
                found_type = "*" + t_type
                break
            if re.search(r"\b" + t_type + r"\b", param):
                found_type = t_type
                break
        if found_type is None:
            continue
        if found_type == "f16":
            found_type = "fp16"
        elif found_type == "*f16":
            found_type = "*fp16"
        elif found_type == "f32":
            found_type = "fp32"
        elif found_type == "*f32":
            found_type = "*fp32"
        result[index] = found_type
        index += 1
    return result


# ---------------------------------------------------------------------------
# TIR helpers (tirx)
# ---------------------------------------------------------------------------
def check_debug_op(func) -> bool:
    """Whether the kernel contains a device-side print/debug intrinsic."""

    found = False

    def visit(node):
        nonlocal found
        if isinstance(node, tirx.Call) and "debug" in str(node.op):
            found = True

    if func is not None:
        tirx.stmt_functor.post_order_visit(func.body, fvisit=visit)
    return found


class LaunchThreadExtractor:
    """Collect the grid extent bound to ``threadIdx.x`` (``blockIdx.x``)."""

    def __init__(self) -> None:
        self.expressions: list[Any] = []

    def visit(self, node):
        for attr in ("body", "then_case", "else_case", "block"):
            child = getattr(node, attr, None)
            if child is not None:
                self.visit(child)
        attr_key = getattr(node, "attr_key", None)
        if attr_key == "thread_extent":
            thread_tag = getattr(getattr(node, "node", None), "thread_tag", None)
            if thread_tag == self.thread:
                self.expressions.append(node.value)

    def extract(self, node, thread: str):
        self.thread = thread
        self.visit(node)
        return self.expressions[0] if self.expressions else None


def _collect_tir_vars(expr):
    vars_found = []
    seen = set()

    def visit(node):
        if isinstance(node, tirx.Var) and node.name not in seen:
            seen.add(node.name)
            vars_found.append(node)

    tirx.stmt_functor.post_order_visit(expr, fvisit=visit)
    return vars_found


def _detect_affine(expr, var):
    if isinstance(expr, tirx.Var):
        if expr.same_as(var) or expr.name == var.name:
            return (1, 0)
        return None
    if isinstance(expr, tirx.IntImm):
        return (0, int(expr))
    for node_cls, op in (
        (tirx.Add, lambda a, b: (a[0] + b[0], a[1] + b[1])),
        (tirx.Sub, lambda a, b: (a[0] - b[0], a[1] - b[1])),
    ):
        if isinstance(expr, node_cls):
            left = _detect_affine(expr.a, var)
            right = _detect_affine(expr.b, var)
            if left is not None and right is not None:
                return op(left, right)
            return None
    if isinstance(expr, tirx.Mul):
        left = _detect_affine(expr.a, var)
        right = _detect_affine(expr.b, var)
        if left is None or right is None:
            return None
        if left[0] == 0:
            return (right[0] * left[1], right[1] * left[1])
        if right[0] == 0:
            return (left[0] * right[1], left[1] * right[1])
        return None
    if isinstance(expr, tirx.FloorDiv):
        left = _detect_affine(expr.a, var)
        if left is not None and isinstance(expr.b, tirx.IntImm) and left[0] == 0:
            return (0, left[1] // int(expr.b))
        return None
    return None


def dynamic_symbolic_map(func) -> dict:
    """Map dynamic shape vars to the (buffer, dim[, scale, bias]) that binds them."""

    params = func.params
    buffer_map = func.buffer_map
    all_vars = []
    seen = set()
    for param in params:
        if param not in buffer_map:
            continue
        for shape in buffer_map[param].shape:
            for var in _collect_tir_vars(shape):
                if var.name not in seen:
                    seen.add(var.name)
                    all_vars.append(var)

    result: dict = {}
    for var in all_vars:
        if var in params:
            continue
        for i, param in enumerate(params):
            if param not in buffer_map:
                continue
            for j, shape in enumerate(buffer_map[param].shape):
                if isinstance(shape, tirx.Var) and shape.same_as(var):
                    result[var] = (i, j)
                    break
            if var in result:
                break
        if var in result:
            continue
        for i, param in enumerate(params):
            if param not in buffer_map:
                continue
            for j, shape in enumerate(buffer_map[param].shape):
                if isinstance(shape, tirx.Var):
                    continue
                affine = _detect_affine(shape, var)
                if affine is not None and affine[0] != 0:
                    result[var] = (i, j, affine[0], affine[1])
                    break
            if var in result:
                break
    return result


def _eval_tir_expr(expr, dynamic_val):
    if isinstance(expr, tirx.Var):
        value = dynamic_val.get(str(expr))
        if value is None:
            raise ValueError(f"Missing runtime value for symbolic var '{expr}'")
        return value
    if isinstance(expr, tirx.IntImm):
        return int(expr)
    if isinstance(expr, tirx.Add):
        return _eval_tir_expr(expr.a, dynamic_val) + _eval_tir_expr(expr.b, dynamic_val)
    if isinstance(expr, tirx.Sub):
        return _eval_tir_expr(expr.a, dynamic_val) - _eval_tir_expr(expr.b, dynamic_val)
    if isinstance(expr, tirx.Mul):
        return _eval_tir_expr(expr.a, dynamic_val) * _eval_tir_expr(expr.b, dynamic_val)
    if isinstance(expr, tirx.FloorDiv):
        return _eval_tir_expr(expr.a, dynamic_val) // _eval_tir_expr(expr.b, dynamic_val)
    if isinstance(expr, tirx.Mod):
        return _eval_tir_expr(expr.a, dynamic_val) % _eval_tir_expr(expr.b, dynamic_val)
    if isinstance(expr, tirx.Max):
        return max(_eval_tir_expr(expr.a, dynamic_val), _eval_tir_expr(expr.b, dynamic_val))
    if isinstance(expr, tirx.Min):
        return min(_eval_tir_expr(expr.a, dynamic_val), _eval_tir_expr(expr.b, dynamic_val))
    raise ValueError(f"Cannot evaluate shape expression '{expr}'")


def replace_by_longest_key(text: str, replace_map) -> str:
    for key in sorted(replace_map, key=lambda x: (-len(x), x)):
        text = text.replace(key, str(replace_map[key]))
    return text


# ---------------------------------------------------------------------------
# compiler
# ---------------------------------------------------------------------------
class Ascend910Compiler:
    """Turn codegen MLIR + the source PrimFunc into a launchable kernel."""

    def __init__(self, *, verbose: bool = False) -> None:
        self.verbose = verbose

    # -- artifact helpers -----------------------------------------------------
    def _optimize(self, raw_mlir: str) -> tuple[str, tuple[str, ...]]:
        result = bishengir.optimize_mlir(raw_mlir, verbose=self.verbose)
        return result.mlir, result.command

    def _compile(self, mlir: str, pass_configs, developer: bool):
        work = Path(tempfile.mkdtemp(prefix="tilelang-ascend910-"))
        try:
            result = bishengir.compile_mlir(
                mlir,
                work,
                pass_configs=pass_configs,
                developer=developer,
                verbose=self.verbose,
            )
        except Exception as exc:
            # Report the command even when the compiler rejects the MLIR (the
            # legacy backend failed on the same inputs).
            command = tuple(
                bishengir.compile_command(
                    bishengir.find_bishengir_compiler(),
                    work / "kernel.npuir",
                    work / "kernel",
                    pass_configs,
                    developer=developer,
                )
            )
            return None, command, exc, None
        return result.object_path, result.command, None, result.shared_object_path

    def _compile_or_cache(self, raw_mlir: str, pass_configs, developer: bool):
        key = bishengir.make_key(raw_mlir, pass_configs=pass_configs, developer=developer)
        root, cached_object = bishengir.cache_paths(key)
        cached_mlir = root / "kernel.npuir"
        if cached_object.is_file() and cached_mlir.is_file():
            shared = bishengir.cache_shared_path(key)
            return (
                cached_mlir.read_text(),
                cached_object,
                shared if shared.is_file() else None,
                None,
                None,
            )
        opt_mlir, opt_command = self._optimize(raw_mlir)
        object_path, compile_command, error, shared_object_path = self._compile(opt_mlir, pass_configs, developer)
        if error is not None:
            # Golden comparison harnesses need the MLIR of kernels that the
            # installed bishengir-compile rejects (the legacy backend fails on
            # the same input).  Production keeps the error.
            if os.environ.get("TILELANG_ASCEND910_ALLOW_COMPILE_FAILURE") != "1":
                raise error
            return opt_mlir, None, None, opt_command, compile_command
        stored = bishengir.cache_save(key, opt_mlir, object_path, shared_object_path)
        shared = bishengir.cache_shared_path(key)
        return (
            opt_mlir,
            stored,
            shared if shared.is_file() else None,
            opt_command,
            compile_command,
        )

    def _precompile_header(self, header_src: str) -> str:
        """Build (and cache) the precompiled launcher header."""

        cache_dir = get_runtime_file_cache(header_src)
        header_path = os.path.join(cache_dir, "npu_launcher.h")
        gch_path = os.path.join(cache_dir, "npu_launcher.h.gch")
        if not (os.path.exists(gch_path) and os.path.getsize(gch_path) > 0):
            with tempfile.TemporaryDirectory() as tmpdir:
                safe_copy(header_src, header_path)
                tmp_gch = os.path.join(tmpdir, "npu_launcher.h.gch")
                precompile_npu_ext(header_path, tmp_gch)
                safe_copy(tmp_gch, gch_path)
        return header_path

    def _build_launcher(self, name: str, header_src: str, wrapper_src: str) -> tuple[str, tuple[str, ...]]:
        header_path = self._precompile_header(header_src)
        cache_key = wrapper_src.encode("utf-8") + Path(header_src).read_bytes()
        cache_dir = get_runtime_file_cache(cache_key)
        launcher_so = os.path.join(cache_dir, f"{name}.so")
        if os.path.exists(launcher_so) and os.path.getsize(launcher_so) > 0:
            # The launcher is already cached: report the command that produced
            # it instead of re-running the compiler.
            from tilelang.utils.npu_utils import npu_ext_command

            return launcher_so, tuple(
                npu_ext_command(
                    name,
                    header_path,
                    os.path.join(cache_dir, f"{name}.cxx"),
                    kernel_launcher="torch",
                    precompile=True,
                    output_path=launcher_so,
                )
            )
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, f"{name}.cxx")
            tmp_so = os.path.join(tmpdir, f"{name}.so")
            with open(src_path, "w") as handle:
                handle.write(wrapper_src)
            so = build_npu_ext(
                name,
                header_path,
                src_path,
                kernel_launcher="torch",
                precompile=True,
                output_path=tmp_so,
            )
            command = tuple(last_build_command() or ())
            safe_copy(so, launcher_so)
        return launcher_so, command

    # -- public API -----------------------------------------------------------
    def compile(
        self,
        func,
        raw_mlir: str,
        out_idx=None,
        pass_configs=None,
        device_mod=None,
    ) -> Ascend910JitKernel:
        developer = ascend910_mode() == "developer"
        mlir, object_path, shared_object_path, opt_command, compile_command = self._compile_or_cache(raw_mlir, pass_configs, developer)
        metadata = parse_metadata(mlir)
        signature = parse_signature(mlir, developer=developer)
        need_debug = check_debug_op(func)
        # The launcher allocates ``workspace_size * blockNum`` bytes of scratch on
        # the device; a kernel that declares a callback but is launched with
        # nullptr writes through a null pointer and aborts the runtime.
        workspace_size = infer_workspace_size(shared_object_path)
        override = os.environ.get("TILELANG_ASCEND_WORKSPACE_SIZE")
        if override is not None:
            with contextlib.suppress(ValueError):
                workspace_size = int(override)
        wrapper_src = generate_npu_wrapper_src(
            {},
            signature,
            workspace_size,
            metadata["mix_mode"],
            -1,
            0,
            need_debug,
            False,
            0,
            False,
            True,
        )
        header_src = get_npu_launcher_header()
        launcher_path, launcher_command = self._build_launcher(metadata["kernel_name"], header_src, wrapper_src)
        return Ascend910JitKernel(
            func=func,
            device_mod=device_mod,
            raw_mlir=raw_mlir,
            mlir=mlir,
            object_path=str(object_path) if object_path is not None else "",
            launcher_path=launcher_path,
            wrapper_src=wrapper_src,
            metadata=metadata,
            signature=signature,
            out_idx=out_idx,
            need_debug=need_debug,
            opt_command=opt_command,
            compile_command=compile_command,
            launcher_command=launcher_command,
        )


# ---------------------------------------------------------------------------
# compiled kernel
# ---------------------------------------------------------------------------
class Ascend910JitKernel:
    """Compiled A3 kernel: owns the launch wrapper and the device object."""

    def __init__(
        self,
        *,
        func,
        device_mod=None,
        raw_mlir: str,
        mlir: str,
        object_path: str,
        launcher_path: str,
        wrapper_src: str,
        metadata: dict,
        signature: dict,
        out_idx,
        need_debug: bool,
        opt_command,
        compile_command,
        launcher_command,
    ) -> None:
        self.func = func
        self.device_mod = device_mod
        self.raw_mlir = raw_mlir
        self.mlir = mlir
        self.object_path = object_path
        self.launcher_path = launcher_path
        self.wrapper_src = wrapper_src
        self.metadata = metadata
        self.signature = signature
        self.need_debug = need_debug
        self.opt_command = opt_command
        self.compile_command = compile_command
        self.launcher_command = launcher_command

        params = list(getattr(func, "params", []) or [])
        self.params = params
        self.param_info = self._extract_param_info(func, out_idx)
        self.out_idx = normalize_out_idx(out_idx, len(self.param_info))
        self.symbolic = dynamic_symbolic_map(func) if func is not None else {}
        self.kernel_name = metadata["kernel_name"]
        self.utils_name = metadata["name"]
        self.mix_mode = metadata["mix_mode"]
        self.launch_packed_metadata = {
            "kernel_name": metadata["name"],
            "tensor_kinds": metadata["tensor_kinds"],
        }
        extractor = LaunchThreadExtractor()
        # The grid extent only exists after lowering: the pre-lowering PrimFunc
        # spells the launch as a ``T.thread_binding`` For, while the lowered
        # device module carries the ``thread_extent`` AttrStmt the extractor
        # looks for.  Prefer the lowered module so the grid is not lost.
        grid_node = func
        if device_mod is not None:
            functions = getattr(device_mod, "functions", None)
            if functions is not None:
                grid_node = next(iter(functions.values()), func)
            else:
                grid_node = device_mod
        grid_expr = extractor.extract(grid_node, "blockIdx.x") if grid_node is not None else None
        self.gridfunc = str(grid_expr) if grid_expr is not None else "1"
        self.launch_grid = [1, 1, 1]
        self._extra_args: list = []
        self._launcher = None
        self._loaded = None

    # -- metadata -------------------------------------------------------------
    @staticmethod
    def _tvm_dtype_to_torch(dtype_str: str):
        mapping = {
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
            "bfloat16": torch.bfloat16,
            "int8": torch.int8,
            "int16": torch.int16,
            "int32": torch.int32,
            "int64": torch.int64,
            "bool": torch.bool,
        }
        return mapping.get(dtype_str, torch.float32)

    def _extract_param_info(self, func, out_idx):
        if func is None:
            return []
        buffer_map = func.buffer_map
        params = func.params
        total = len(params)
        pos_out = None
        if out_idx is not None:
            raw = [out_idx] if isinstance(out_idx, int) else list(out_idx)
            pos_out = {i if i >= 0 else total + i for i in raw}
        infos = []
        for i, param in enumerate(params):
            is_output = pos_out is not None and i in pos_out
            if param in buffer_map:
                buffer = buffer_map[param]
                infos.append(
                    {
                        "dtype": self._tvm_dtype_to_torch(str(buffer.dtype)),
                        "shape": list(buffer.shape),
                        "is_output": is_output,
                    }
                )
            else:
                infos.append(
                    {
                        "dtype": self._tvm_dtype_to_torch(str(param.dtype)),
                        "shape": [],
                        "is_output": False,
                    }
                )
        return infos

    # -- runtime --------------------------------------------------------------
    def _get_launcher(self):
        if self._launcher is None:
            spec = importlib.util.spec_from_file_location("__tilelang_launcher", self.launcher_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._launcher = module
        return self._launcher

    def _load(self):
        if self._loaded is None:
            device = torch.npu.current_device()
            with open(self.object_path, "rb") as handle:
                binary = handle.read()
            self._loaded = NPUUtils.get().load_binary(self.utils_name, binary, self.metadata["shared"], device, self.mix_mode)
        return self._loaded

    def _calcu_grid(self, orig_to_input, *args):
        dynamic_val = {}
        extra_args: list = []
        for key, pos in self.symbolic.items():
            if not isinstance(pos, (tuple, list)) or len(pos) < 2:
                continue
            tensor_idx, dim_idx = pos[0], pos[1]
            if tensor_idx not in orig_to_input:
                raise ValueError(f"Symbolic {key} depends on an output buffer")
            arg = args[orig_to_input[tensor_idx]]
            if not isinstance(arg, torch.Tensor) or dim_idx >= len(arg.shape):
                raise ValueError(f"Cannot resolve symbolic {key}")
            actual = arg.shape[dim_idx]
            if len(pos) == 4:
                scale, bias = pos[2], pos[3]
                value = (actual - bias) // scale
            else:
                value = actual
            dynamic_val[str(key)] = value
            extra_args.append(value)
        self._extra_args = extra_args
        expression = replace_by_longest_key(self.gridfunc, dynamic_val)
        value = eval(expression, {"__builtins__": {}}, {"math": __import__("math"), **dynamic_val})
        if hasattr(value, "__iter__"):
            self.launch_grid = [int(v) for v in value]
        else:
            self.launch_grid = [int(value), 1, 1]
        return dynamic_val

    def _current_stream(self) -> int:
        import torch_npu

        device = torch.npu.current_device()
        if hasattr(torch_npu._C, "_npu_getCurrentRawStreamNoWait"):
            return torch_npu._C._npu_getCurrentRawStreamNoWait(device)
        if hasattr(torch_npu._C, "_npu_getCurrentRawStream"):
            return torch_npu._C._npu_getCurrentRawStream(device)
        return torch.npu.current_stream(device).npu_stream

    def __call__(self, *args):
        total_params = len(self.param_info)
        num_inputs = total_params - (len(self.out_idx) if self.out_idx is not None else 0)
        if len(args) != num_inputs:
            raise ValueError(f"Expected {num_inputs} inputs, got {len(args)}")

        orig_to_input = {}
        input_pos = 0
        for i, info in enumerate(self.param_info):
            if not info["is_output"]:
                orig_to_input[i] = input_pos
                input_pos += 1

        dynamic_val = self._calcu_grid(orig_to_input, *args)

        full_args = [None] * total_params
        input_ptr = 0
        for i, info in enumerate(self.param_info):
            if info["is_output"]:
                shape = []
                for dim in info["shape"]:
                    if isinstance(dim, tirx.Var):
                        value = dynamic_val.get(str(dim))
                        if value is None:
                            raise ValueError(f"Missing value for {dim}")
                        shape.append(value)
                    elif isinstance(dim, tirx.IntImm):
                        shape.append(int(dim))
                    elif isinstance(dim, tirx.PrimExpr):
                        shape.append(_eval_tir_expr(dim, dynamic_val))
                    else:
                        shape.append(int(dim))
                device = args[0].device if args else torch.device("cpu")
                full_args[i] = torch.empty(shape, dtype=info["dtype"], device=device)
            else:
                full_args[i] = args[input_ptr]
                input_ptr += 1

        full_args.extend(self._extra_args)

        _, npu_function, _, _ = self._load()
        try:
            self._launch_with(npu_function, full_args)
        except TypeError:
            # A stale cached launcher reports a wrong argument count through
            # the C entry point.  Rebuild it once (bypassing the cache) and
            # retry before surfacing the error.
            self._rebuild_launcher()
            self._launch_with(npu_function, full_args)

        if self.out_idx is None:
            return None
        if len(self.out_idx) == 1:
            return full_args[self.out_idx[0]]
        return [full_args[i] for i in self.out_idx]

    def _launch_with(self, npu_function, full_args) -> None:
        self._get_launcher().launch(
            self.launch_grid[0],
            self.launch_grid[1],
            self.launch_grid[2],
            self._current_stream(),
            npu_function,
            self.launch_packed_metadata,
            {},
            None,
            None,
            *full_args,
        )

    def _rebuild_launcher(self) -> None:
        """Rebuild the launch wrapper in a private directory.

        The shared launcher cache is content addressed, but a reused entry can
        still describe a different argument count (for example when a kernel is
        compiled with and without ``out_idx``).  Building into a fresh directory
        guarantees the wrapper matches ``self.wrapper_src``.
        """

        import tempfile
        from pathlib import Path

        from tilelang.utils.npu_utils import build_npu_ext

        tmp_dir = Path(tempfile.mkdtemp(prefix="a910-launcher-"))
        cxx_path = tmp_dir / "launcher.cxx"
        cxx_path.write_text(self.wrapper_src)
        so_path = tmp_dir / "launcher.so"
        build_npu_ext(
            "launcher",
            get_npu_launcher_header(),
            str(cxx_path),
            kernel_launcher="torch",
            precompile=True,
            output_path=str(so_path),
        )
        self.launcher_path = str(so_path)
        self._launcher = None

    # -- introspection --------------------------------------------------------
    def get_kernel_source(self) -> str:
        return self.mlir

    def get_raw_kernel_source(self) -> str:
        return self.raw_mlir

    def get_launcher_source(self) -> str:
        return self.wrapper_src

    def get_profiler(self, *args, **kwargs):
        from tilelang.profiler import Profiler

        return Profiler(self.params, self.out_idx or [], *args, **kwargs).with_direct_func(self)

    def benchmark(self, warmup: int = 25, rep: int = 100, n_warmup: int = 1, n_repeat: int = 1) -> float:
        return self.get_profiler().do_bench(func=self, warmup=warmup, rep=rep, n_warmup=n_warmup, n_repeat=n_repeat)
