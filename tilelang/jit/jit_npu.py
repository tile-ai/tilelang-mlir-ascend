# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import ctypes
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Union
import torch
from mako.template import Template
from ..engine import lower
from ..utils import (
    NPUUtils,
    get_npucompiler_path,
    get_bisheng_path,
    build_npu_ext,
    precompile_npu_ext,
    get_runtime_file_cache,
    get_npu_launcher_header,
    safe_copy,
)

from tvm import tir
from tvm.tir import PrimFunc
from tvm import transform

from tilelang.profiler import Profiler, TensorSupplyType
from tilelang.transform.pass_config import normalize_pass_configs


class LaunchThreadExtractor:
    def __init__(self) -> None:
        self.expressions = []

    def visit_thread_extent(self, node):
        if hasattr(node, "body"):
            self.visit_thread_extent(node.body)
        if hasattr(node, "then_case"):
            self.visit_thread_extent(node.then_case)
        if hasattr(node, "else_case"):
            self.visit_thread_extent(node.else_case)
        if hasattr(node, "block"):
            self.visit_thread_extent(node.block)

        if (
            hasattr(node, "attr_key")
            and node.attr_key == "thread_extent"
            and node.node.thread_tag == self.thread
        ):
            self.expressions.append(node.value)

    def extract(self, node: PrimFunc, thread: str):
        self.thread = thread
        self.visit_thread_extent(node)
        if self.expressions is None:
            return None
        return self.expressions[0]


def replace_by_longest_key(calculate_str, replace_dict):
    sorted_keys = sorted(replace_dict.keys(), key=lambda x: (-len(x), x))
    result = calculate_str
    for key in sorted_keys:
        result = result.replace(key, str(replace_dict[key]))
    return result


def _transform_stmt(stmt, symbolic_var_names):
    """Convert statements, remove symbolic variable definitions"""
    if stmt is None:
        return stmt

    # Handling Block Nodes
    if isinstance(stmt, tir.Block):
        new_body = _transform_stmt(stmt.body, symbolic_var_names)
        return tir.Block(
            iter_vars=stmt.iter_vars,
            reads=stmt.reads,
            writes=stmt.writes,
            name_hint=stmt.name_hint,
            body=new_body,
            init=stmt.init,
            alloc_buffers=stmt.alloc_buffers,
            annotations=stmt.annotations,
            span=stmt.span,
        )

    # Process the SeqStmt node.
    elif isinstance(stmt, tir.SeqStmt):
        new_seqs = []
        for seq_stmt in stmt.seq:
            transformed = _transform_stmt(seq_stmt, symbolic_var_names)
            if transformed is not None:
                new_seqs.append(transformed)
        return tir.SeqStmt(new_seqs, stmt.span) if new_seqs else None

    # Processing LetStmt Node - Removing Symbol Variable Definitions
    elif isinstance(stmt, tir.LetStmt):
        if stmt.var.name in symbolic_var_names:
            # Skip symbol variable definitions, only transform the body.
            return _transform_stmt(stmt.body, symbolic_var_names)
        else:
            # Convert a regular LetStmt
            new_body = _transform_stmt(stmt.body, symbolic_var_names)
            return tir.LetStmt(
                var=stmt.var, value=stmt.value, body=new_body, span=stmt.span
            )

    # Other types of statements remain unchanged.
    else:
        return stmt


# Collect all variables used in the buffer shape
def _process_dynamic_symbolic(func):
    params = func.params
    buffer_map = func.buffer_map
    dynamic_symbolic_map = {}
    for i, param in enumerate(params):
        if param not in buffer_map:
            continue
        buffer = buffer_map[param]
        for j, shape in enumerate(buffer.shape):
            if isinstance(shape, tir.Var) and (shape not in dynamic_symbolic_map):
                dynamic_symbolic_map[shape] = (i, j)
    return dynamic_symbolic_map


# Creating a Pass for Symbolic Variable Promotion
def _symbolic_var_promoter_pass(func: PrimFunc):
    dynamic_symbolic_map = _process_dynamic_symbolic(func)
    symbolic_vars = list(dynamic_symbolic_map.keys())

    if len(symbolic_vars) == 0:
        return func, {}

    # Create new parameter list: symbolic variables + original parameters
    new_params = list(func.params) + symbolic_vars

    # Transform function body, remove symbolic variable definitions
    new_body = _transform_stmt(func.body, symbolic_vars)

    # Create a new PrimFunc
    new_primfunc = tir.PrimFunc(
        params=new_params,
        body=new_body,
        ret_type=func.ret_type,
        buffer_map=func.buffer_map,
        attrs=func.attrs,
        span=func.span,
    )
    return new_primfunc, dynamic_symbolic_map


def convert_sigtype_to_int(sigty: str):
    MAP_SIGTYPE_TO_INT = {
        # Boolean
        "i1": 12,  # BOOL
        # Integer types
        "i8": 2,  # INT8
        "i16": 6,  # INT16
        "i32": 3,  # INT32
        "i64": 9,  # INT64
        # Unsigned integer types
        "u32": 8,  # UINT32
        "u64": 10,  # UINT64
        # Floating point types
        "fp16": 1,  # FLOAT16
        "bf16": 27,  # DT_BF16
        "fp32": 0,  # FLOAT
        "fp64": 11,  # DOUBLE
    }
    if sigty not in MAP_SIGTYPE_TO_INT:
        raise ValueError(f"Unsupported data type: {sigty}")

    return MAP_SIGTYPE_TO_INT[sigty]


def extract_device_print_code_from_cann():
    ccec_compiler_bin_folder, _ = os.path.split(os.path.realpath(get_bisheng_path()))
    ccec_compiler_folder, _ = os.path.split(ccec_compiler_bin_folder)
    clang_version = os.listdir(os.path.join(ccec_compiler_folder, "lib/clang/"))[0]
    ccelib_path = os.path.join(
        ccec_compiler_folder, f"lib/clang/{clang_version}/include/ccelib"
    )

    def read_header(header_path):
        with open(os.path.join(ccelib_path, header_path), "r") as f:
            code = f.read()

        # remove all #include "..."
        lines = code.splitlines()
        purged_lines = []
        for line in lines:
            normalized_line = " ".join(line.split())
            if not normalized_line.startswith('#include "'):
                purged_lines.append(line)
        code = "\n".join(purged_lines)

        # remove [aicore] functions
        aicore_positions = []
        for m in re.finditer(r"\[aicore\]", code):
            aicore_positions.append(m.start())

        def find_aicore_function_span(src, pos):
            for i in range(pos - 1, -1, -1):
                if (
                    src[i] == "}"
                ):  # this relies on that all [aicore] functions come after normal functions
                    left = i + 1
                    break
            n = len(src)
            brace_nest = 0
            for j in range(pos, n, 1):
                if src[j] == "{":
                    brace_nest += 1
                elif src[j] == "}":
                    brace_nest -= 1
                    if brace_nest == 0:
                        right = j
                        break
            return left, right

        new_code = ""
        segment_start = 0
        for pos in aicore_positions:
            left, right = find_aicore_function_span(code, pos)
            new_code += code[segment_start:left]
            segment_start = right + 1
        new_code += code[segment_start:]

        # remove __gm__ and rename macros
        new_code = new_code.replace("__gm__", " ")
        new_code = new_code.replace("__CCELIB_RT_ERROR_NONE", "RT_ERROR_NONE")
        new_code = new_code.replace("__CCELIB_RT_MEMORY_HBM", "RT_MEMORY_HBM")
        new_code = new_code.replace(
            "__CCELIB_RT_MEMCPY_HOST_TO_DEVICE", "RT_MEMCPY_HOST_TO_DEVICE"
        )
        new_code = new_code.replace(
            "__CCELIB_RT_MEMCPY_DEVICE_TO_HOST", "RT_MEMCPY_DEVICE_TO_HOST"
        )
        return new_code

    # the following headers should be included in this order
    return "\n".join(
        [
            read_header("common/common_impl.h"),
            read_header("internal/debug_tunnel/payload.h"),
            read_header("internal/debug_tunnel/payload_impl.h"),
            read_header("internal/debug_tunnel/tunnel.h"),
            read_header("internal/debug_tunnel/tunnel_impl.h"),
        ]
    )


_NPU_LAUNCHER_TEMPLATE = None


def _get_npu_launcher_template():
    global _NPU_LAUNCHER_TEMPLATE
    if _NPU_LAUNCHER_TEMPLATE is None:
        template_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "utils",
            "npu_launcher.mako",
        )
        _NPU_LAUNCHER_TEMPLATE = Template(filename=template_path)
    return _NPU_LAUNCHER_TEMPLATE


_TY_TO_CPP = {
    "i1": "int32_t",
    "i8": "int8_t",
    "i16": "int16_t",
    "i32": "int32_t",
    "i64": "int64_t",
    "u1": "uint32_t",
    "u8": "uint8_t",
    "u16": "uint16_t",
    "u32": "uint32_t",
    "u64": "uint64_t",
    "fp16": "float",
    "bf16": "float",
    "fp32": "float",
    "f32": "float",
    "fp64": "double",
}

_EXTRACTED_TY = {
    "i1": "int32_t",
    "i8": "int8_t",
    "i16": "int16_t",
    "i32": "int32_t",
    "i64": "int64_t",
    "u1": "uint32_t",
    "u8": "uint8_t",
    "u16": "uint16_t",
    "u32": "uint32_t",
    "u64": "uint64_t",
    "fp16": "float",
    "bf16": "float",
    "fp32": "float",
    "f32": "float",
    "fp64": "double",
}

_FORMAT_OF = {
    "PyObject*": "O",
    "float": "f",
    "double": "d",
    "long": "l",
    "int8_t": "b",
    "int16_t": "h",
    "int32_t": "i",
    "int64_t": "L",
    "uint8_t": "B",
    "uint16_t": "H",
    "uint32_t": "I",
    "uint64_t": "K",
}

_SIGTYPE_TO_INT = {
    "i1": 12,
    "i4": 29,
    "i8": 2,
    "i16": 6,
    "i32": 3,
    "i64": 9,
    "u1": 30,
    "u8": 4,
    "u16": 7,
    "u32": 8,
    "u64": 10,
    "fp16": 1,
    "bf16": 27,
    "fp32": 0,
    "fp64": 11,
    "fp8e5": 35,
    "fp8e4nv": 36,
}


def _format_of_msprof_task_type_ratio(bs_task_type, mix_mode):
    default_task_type = (
        "MSPROF_GE_TASK_TYPE_AIV"
        if mix_mode == "aiv"
        else "MSPROF_GE_TASK_TYPE_AI_CORE"
    )
    if not bs_task_type:
        return default_task_type, 0
    task_type_num, mix_block_dim_ratio = divmod(int(bs_task_type), 10)
    task_type_map = {
        1: "MSPROF_GE_TASK_TYPE_AIV",
        2: "MSPROF_GE_TASK_TYPE_AI_CORE",
        3: "MSPROF_GE_TASK_TYPE_MIX_AIC",
        4: "MSPROF_GE_TASK_TYPE_MIX_AIV",
    }
    task_type = task_type_map.get(task_type_num, default_task_type)
    return task_type, mix_block_dim_ratio


def generate_npu_wrapper_src(
    constants,
    signature,
    workspace_size,
    mix_mode,
    lock_num,
    lock_ini_val,
    need_debug,
    bs_task_type=0,
    compile_on_910_95=False,
    force_simt_only=False,
    shared_mem_dynamic_size=0,
):
    from ..utils.npu_arch import is_ffts_supported, force_disable_ffts, get_ascend_device_name

    ty_to_cpp = {}
    extracted_ty = {}
    for ty in signature.values():
        if ty not in ty_to_cpp:
            ty_to_cpp[ty] = "void*" if ty[0] == "*" else _TY_TO_CPP[ty]
        if ty not in extracted_ty:
            extracted_ty[ty] = "PyObject*" if ty[0] == "*" else _EXTRACTED_TY[ty]

    sigtype_to_int = {}
    for ty in signature.values():
        if ty.startswith("*") and ty[1:] not in sigtype_to_int:
            sigtype_to_int[ty[1:]] = _SIGTYPE_TO_INT.get(ty[1:], 0)

    arg_decls = ", ".join(
        f"{ty_to_cpp[ty]} arg{i}" for i, ty in signature.items()
    )
    format_str = "iiiKKOOOO" + "".join(
        [_FORMAT_OF[extracted_ty[ty]] for ty in signature.values()]
    )
    launch_args = ", ".join(
        f"ptr_info{i}.dev_ptr" if ty[0] == "*" else f"_arg{i}"
        for i, ty in signature.items()
    )

    grid_info = {"X": "i32", "Y": "i32", "Z": "i32"}
    for ty in grid_info.values():
        if ty not in ty_to_cpp:
            ty_to_cpp[ty] = _TY_TO_CPP[ty]

    enable_taskqueue = os.getenv("TILELANG_ENABLE_TASKQUEUE", "true").lower() in (
        "true",
        "1",
    )
    enable_auto_map_parallel_blocks = os.getenv(
        "TILELANG_ALL_BLOCKS_PARALLEL", "false"
    ).lower() in ("true", "1")
    enable_grid_warn_print = os.getenv(
        "TILELANG_GRID_WARN_PRINT", "false"
    ).lower() in ("true", "1")
    npu_utils = NPUUtils.get()
    num_physical_blocks = (
        npu_utils.get_aivector_core_num()
        if mix_mode == "aiv"
        else npu_utils.get_aicore_num()
    )

    task_type, mix_block_dim_ratio = _format_of_msprof_task_type_ratio(
        bs_task_type, mix_mode
    )
    is_mix_task_type = "MIX" in task_type

    arch = get_ascend_device_name()
    target_support_ffts = is_ffts_supported(arch) and (not force_disable_ffts())

    enable_simt = force_simt_only

    debug_print_code = ""
    if need_debug:
        debug_print_code = extract_device_print_code_from_cann()

    template = _get_npu_launcher_template()
    return template.render(
        arg_decls=arg_decls,
        format_str=format_str,
        need_debug=need_debug,
        debug_print_code=debug_print_code,
        enable_taskqueue=enable_taskqueue,
        enable_auto_map_parallel_blocks=enable_auto_map_parallel_blocks,
        enable_grid_warn_print=enable_grid_warn_print,
        num_physical_blocks=num_physical_blocks,
        lock_num=lock_num,
        lock_ini_val=lock_ini_val,
        workspace_size=workspace_size,
        task_type=task_type,
        mix_block_dim_ratio=mix_block_dim_ratio,
        is_mix_task_type=is_mix_task_type,
        target_support_ffts=target_support_ffts,
        compile_on_910_95=compile_on_910_95,
        enable_simt=enable_simt,
        force_simt_only=force_simt_only,
        shared_mem_dynamic_size=shared_mem_dynamic_size,
        signature=signature,
        constants=constants,
        grid_info=grid_info,
        ty_to_cpp=ty_to_cpp,
        extracted_ty=extracted_ty,
        sigtype_to_int=sigtype_to_int,
        launch_args=launch_args,
    )


def read_binary_file(file_path, mode="rb", chunk_size=None, return_type="bytes"):
    """
    Function to read a binary file

    Parameters:
        file_path (str): Path to the file to be read
        mode (str): File opening mode, defaults to 'rb' (read binary)
        chunk_size (int): If specified, reads the file in chunks of the given size; otherwise reads the entire file
        return_type (str): Return data type, can be 'bytes' or 'bytearray'

    Returns:
        Returns bytes or bytearray object according to return_type parameter
        If chunk_size is specified, returns a generator that yields data chunk by chunk

    Raises:
        FileNotFoundError: When the file does not exist
        IOError: When an error occurs during file reading
    """
    try:
        with open(file_path, mode) as file:
            if chunk_size:
                # Read file in chunks
                def chunk_reader():
                    while True:
                        chunk = file.read(chunk_size)
                        if not chunk:
                            break
                        if return_type == "bytearray":
                            yield bytearray(chunk)
                        else:
                            yield chunk

                return chunk_reader()
            else:
                # Read the entire file in one go
                data = file.read()
                if return_type == "bytearray":
                    return bytearray(data)
                else:
                    return data
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {file_path}") from e
    except IOError as e:
        raise IOError(f"Error occurred while reading the file: {e}") from e


class JitKernel_NPU:
    def __init__(self, metadata: dict, out_idx=None) -> None:
        self.params = metadata["params"]
        self.signature = metadata.get("signature", {})
        self.out_idx = out_idx
        self.param_info = metadata.get("param_info", [])
        # 1 launch path
        self.so_launcher_path = metadata.get(
            "so_launcher_path", f"{metadata['kernel_name']}.so"
        )
        self.so_utils_path = "npu_utils.so"
        self.utils_name = f"{metadata['name']}"
        # 2 kernel path
        self.utils_kernel_src = metadata["kernel_src"]
        self.utils_shared = metadata[
            "shared"
        ]  # Retain the interface, temporarily not in effect.
        self.mlir_content = metadata["mlir_content"]
        self.mix_mode = metadata["mix_mode"]
        self.utils_device = torch.npu.current_device()
        self.launch_stream = torch.npu.current_stream(
            torch.npu.current_device()
        ).npu_stream
        self.launch_packedMetadata = {
            "kernel_name": f"{metadata['name']}",
            "tensor_kinds": metadata["tensor_kinds"],
        }
        self.kernel_name = f"{metadata['name']}"
        self.tensor_kinds = metadata["tensor_kinds"]
        self.launch_metadata = {}
        self.launch_enter_hook = None
        self.launch_exit_hook = None
        self.gridfunc = metadata["gridfunc"]
        self.symbolic = metadata["symbolic"]
        self.prim_func = metadata["primfunc"]
        self.out_idx = metadata["out_idx"]
        self._launch()

    @classmethod
    def from_database(
        cls,
        mod: PrimFunc,
        kernel_source: str,
        kernel_launcher_path: str,
        kernel_utils_path: str,
        metadata: str,
        out_idx: Union[List[int], int],
    ):
        if isinstance(out_idx, int):
            out_idx = [out_idx]
        metadata["so_launcher_path"] = kernel_launcher_path
        metadata["out_idx"] = out_idx
        instance = cls(metadata)
        instance.so_launcher_path = kernel_launcher_path
        instance.so_utils_path = kernel_utils_path
        return instance

    def _launch(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "__tilelang_launcher", self.so_launcher_path
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.launch_npu = mod.launch

    def _calcu_grid(self, orig_to_input, *args: Any):
        """
        Calculate grid dimensions based on symbolic variables and input tensors.

        Args:
            tensor_args: List of input tensor arguments
            orig_to_tensor_pos: Mapping from original parameter indices to tensor_args positions

        Returns:
            Dictionary of dynamic values extracted from tensors
        """
        dynamic_val = {}
        extra_args = []
        for key, pos in self.symbolic.items():
            # Ensure that pos is a tuple with two elements.
            if isinstance(pos, (tuple, list)) and len(pos) >= 2:
                tensor_idx, dim_idx = pos[0], pos[1]
                if tensor_idx in orig_to_input:
                    pos = orig_to_input[tensor_idx]
                    arg = args[pos]
                    if isinstance(arg, torch.Tensor) and dim_idx < len(arg.shape):
                        value = arg.shape[dim_idx]
                        dynamic_val[str(key)] = value
                        extra_args.append(value)
                    else:
                        raise ValueError(f"Cannot resolve symbolic {key}")
                else:
                    raise ValueError(f"Symbolic {key} depends on output")

        self.extra_args = extra_args
        result = replace_by_longest_key(self.gridfunc, dynamic_val)

        try:
            # If the result is a number, use it directly.
            if isinstance(result, (int, float)):
                grid_value = result
            # If the result is a string, evaluate it safely.
            elif isinstance(result, str):
                # Only mathematical expressions are allowed to avoid security risks.
                grid_value = eval(
                    result,
                    {"__builtins__": {}},
                    {"math": __import__("math"), **dynamic_val},
                )
            else:
                grid_value = result

            # Ensure the result is an integer
            if hasattr(grid_value, "__iter__"):
                self.launch_grid = [int(x) for x in grid_value]
            else:
                self.launch_grid = [int(grid_value), 1, 1]

        except Exception as e:
            raise ValueError(
                f"Failed to evaluate grid expression '{result}': {e}"
            ) from e

        return dynamic_val

    def __call__(self, *args: Any) -> Any:
        # Calculate the input params：total_params - out_params
        total_params = len(self.param_info)
        num_inputs = total_params - (
            len(self.out_idx) if self.out_idx is not None else 0
        )

        if len(args) != num_inputs:
            raise ValueError(f"Expected {num_inputs} inputs, got {len(args)}")

        # Build the mapping from original inputs to the args position
        orig_to_input = {}
        input_pos = 0
        for i, info in enumerate(self.param_info):
            if not info["is_output"]:
                orig_to_input[i] = input_pos
                input_pos += 1

        # Calculate grid and get dynamic values
        dynamic_val = self._calcu_grid(orig_to_input, *args)

        # Build full argument list
        full_args = [None] * total_params
        input_ptr = 0

        for i, info in enumerate(self.param_info):
            if info["is_output"]:  # Output parameter (must be tensor)
                dtype = info["dtype"]
                shape = []
                for dim in info["shape"]:
                    if isinstance(dim, tir.Var):
                        val = dynamic_val.get(str(dim))
                        if val is None:
                            raise ValueError(f"Missing value for {dim}")
                        shape.append(val)
                    else:
                        shape.append(int(dim))

                # Shape should not be empty (validated in _extract_param_info)
                device = args[0].device if args else torch.device("cpu")
                full_args[i] = torch.empty(shape, dtype=dtype, device=device)
            else:
                # Input parameter (tensor or scalar)
                full_args[i] = args[input_ptr]
                input_ptr += 1

        # Append extra_args
        full_args.extend(self.extra_args)

        # Run kernel
        npu_utils = NPUUtils.get()
        t_module, t_function, t_n_regs, t_n_spills = npu_utils.load_binary(
            self.utils_name,
            self.utils_kernel_src,
            self.utils_shared,
            self.utils_device,
            self.mix_mode,
        )
        self.launch_npu(
            self.launch_grid[0],
            self.launch_grid[1],
            self.launch_grid[2],
            self.launch_stream,
            t_function,
            self.launch_packedMetadata,
            self.launch_metadata,
            self.launch_enter_hook,
            self.launch_exit_hook,
            *full_args,
        )

        # Return outputs
        if self.out_idx is None:
            return None
        if len(self.out_idx) == 1:
            return full_args[self.out_idx[0]]
        else:
            return [full_args[i] for i in self.out_idx]
        # return args[idx]

    def get_profiler(
        self, tensor_supply_type: TensorSupplyType = TensorSupplyType.Auto
    ) -> Profiler:
        """
        Creates a profiler to benchmark the compiled runtime module.

        Parameters
        ----------
        tensor_supply_type : TensorSupplyType, optional
            The type of input tensors to supply for profiling (default: TensorSupplyType.Auto).

        Returns
        -------
        Profiler
            A Profiler instance for benchmarking the runtime module.
        """
        return Profiler(
            self.params, self.out_idx[0], tensor_supply_type
        ).with_direct_func(self)

    def benchmark(
        self, warmup: int = 25, rep: int = 100, n_warmup: int = 1, n_repeat: int = 1
    ) -> float:
        profiler = self.get_profiler()
        return profiler.do_bench(
            func=self, warmup=warmup, rep=rep, n_warmup=n_warmup, n_repeat=n_repeat
        )

    def update_tuner_result(
        self, latency: float, config: Dict[str, Any], ref_latency: float
    ) -> "JitKernel_NPU":
        """
        Updates the tuning results for this kernel.

        Parameters
        ----------
        latency : float
            The measured latency of this kernel configuration.
        config : Dict[str, Any]
            The configuration parameters used for this kernel.
        ref_latency : float
            The reference latency to compare against.

        Returns
        -------
        None
        """
        self.latency = latency
        self.config = config
        self.ref_latency = ref_latency

        return self

    def get_kernel_source(self) -> str:
        """
        Returns the source code of the compiled kernel function.

        Returns
        -------
        str
            The source code of the compiled kernel function.
        """

        return self.utils_kernel_src

    def get_tuner_result(self) -> Dict[str, Any]:
        """
        Gets the tuning results for this kernel.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing:
            - latency: The measured latency of this kernel
            - config: The configuration parameters used
            - ref_latency: The reference latency for comparison
        """
        if self.latency is None:
            raise ValueError(
                "Tuning results are not available. Please tune the kernel first."
            )

        return {
            "latency": self.latency,
            "config": self.config,
            "ref_latency": self.ref_latency,
        }


class compiler_npu:
    def __init__(self) -> None:
        pass

    def _get_workspace_size(self, lib_path, suffix, default=0):
        # Try to get the infer_workspace_shape_function in the kernel, then use the return value as workspace_size
        # Use default to avoid except
        # If you have set the os env "TILELANG_ASCEND_WORKSPACE_SIZE", "TILELANG_ASCEND_WORKSPACE_SIZE" has a higher priority
        if not os.path.exists(lib_path):
            return default
        symbols = []
        # Try to get the kernel symbol table and match function name "***_infer_workspace_shape_function"
        try:
            result = subprocess.run(
                ["nm", "-D", lib_path], capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        sym_name = parts[2]
                        if sym_name.endswith(suffix):
                            symbols.append(sym_name)
        except (subprocess.SubprocessError, FileNotFoundError, OSError, TimeoutError):
            pass

        if not symbols:
            return default
        # Load the lib
        try:
            lib = ctypes.CDLL(lib_path)
        except OSError:
            return default
        # Get the return value
        for func_name in symbols:
            try:
                func = getattr(lib, func_name)
                func.restype = ctypes.c_int
                return func()
            except (AttributeError, OSError, TypeError):
                continue
        return default

    def compile(
        self, mod: PrimFunc, out_idx=None, pass_configs: dict[str, Any] = None
    ) -> JitKernel_NPU:
        self.original_mod = mod
        # extract_param_info
        param_info = self._extract_param_info(mod, out_idx)
        # process negative out_idx
        if out_idx is not None:
            total_params = len(param_info)
            out_idx = [i if i >= 0 else total_params + i for i in out_idx]
        self.metadata = {}
        self.metadata["out_idx"] = out_idx
        self.metadata["param_info"] = param_info
        self.mod, self.metadata["symbolic"] = _symbolic_var_promoter_pass(mod)
        self.need_debug = self.check_debug_op(self.mod)
        # get grid message
        self._parse_grid()
        self.metadata["params"] = self.mod.params
        self.out_idx = out_idx
        self.metadata["out_idx"] = self.out_idx

        # Apply pass_configs using TVM PassContext
        if pass_configs is None:
            pass_configs = {}
        # Store pass_configs for later use in NPU compilation
        self.pass_configs = normalize_pass_configs(pass_configs)
        with transform.PassContext(opt_level=3, config=pass_configs):
            mlir_path = lower(self.mod)
        if mlir_path.endswith(".mlir"):
            self.mlir_content = self._read_mlir_file(mlir_path)
        else:
            self.mlir_content = mlir_path
        self.constants = {}
        # get signature information
        self.signature = self._parse_signature()

        self.metadata["signature"] = self.signature
        self.metadata["primfunc"] = self.mod
        self.metadata["mlir_content"] = self.mlir_content

        self.lock_num = -1
        self.lock_ini_val = 0
        self._parse_npuir_metadata()
        self.metadata["kernel_src"] = self._npuir_to_bin_enable_npu_compile()
        self.header_path = get_npu_launcher_header()

        TILELANG_ASCEND_WORKSPACE_SIZE = os.environ.get(
            "TILELANG_ASCEND_WORKSPACE_SIZE"
        )
        if TILELANG_ASCEND_WORKSPACE_SIZE is not None:
            try:
                self.workspace_size = int(TILELANG_ASCEND_WORKSPACE_SIZE)
            except ValueError:
                print(
                    f"Warning: TILELANG_ASCEND_WORKSPACE_SIZE must be integer, "
                    f"got '{TILELANG_ASCEND_WORKSPACE_SIZE}', using {self.workspace_size}"
                )

        from ..utils.npu_arch import is_compile_on_910_95, get_ascend_device_name

        arch = get_ascend_device_name()
        compile_on_910_95 = is_compile_on_910_95(arch)

        bs_task_type = self.metadata.get("bs_task_type", 0)
        force_simt_only = self.metadata.get("force_simt_only", False)
        shared_mem_dynamic_size = self.metadata.get("shared_mem_dynamic_size", 0)

        if force_simt_only and shared_mem_dynamic_size == 0:
            shared_mem_dynamic_size = 122880
        elif not force_simt_only and shared_mem_dynamic_size == 0:
            shared_mem_dynamic_size = 221184

        self.wrapper_src = generate_npu_wrapper_src(
            self.constants,
            self.signature,
            self.workspace_size,
            self.metadata["mix_mode"],
            self.lock_num,
            self.lock_ini_val,
            self.need_debug,
            bs_task_type=bs_task_type,
            compile_on_910_95=compile_on_910_95,
            force_simt_only=force_simt_only,
            shared_mem_dynamic_size=shared_mem_dynamic_size,
        )
        self.so_launcher_path = self.make_npu_launcher_stub(
            self.metadata["kernel_name"], self.header_path, self.wrapper_src
        )
        self.metadata["so_launcher_path"] = self.so_launcher_path

        return JitKernel_NPU(metadata=self.metadata, out_idx=out_idx)

    def _extract_param_info(self, func: PrimFunc, out_idx):
        """
        Extract parameter information from PrimFunc.

        Returns a list of dicts, each containing:
            - dtype: torch.dtype of the parameter
            - shape: list of dimensions (may contain tir.Var for dynamic shapes)
            - is_output: bool indicating if this parameter is an output tensor
        """
        buffer_map = func.buffer_map
        params = func.params
        info_list = []

        # Convert out_idx to positive indices and validate
        total_params = len(params)
        pos_out_idx = None
        if out_idx is not None:
            pos_out_idx = {i if i >= 0 else total_params + i for i in out_idx}

        for i, param in enumerate(params):
            is_output = pos_out_idx is not None and i in pos_out_idx
            if param in buffer_map:
                # Tensor parameter (has buffer)
                buffer = buffer_map[param]
                dtype_str = str(buffer.dtype)
                torch_dtype = self._tvm_dtype_to_torch(dtype_str)
                shape_expr = list(buffer.shape)
                # Check for zero-dimensional tensor (not supported as output)
                if is_output and len(shape_expr) == 0:
                    raise ValueError(
                        f"Output parameter at index {i} has zero-dimensional shape. "
                        f"TileLang does not support scalar outputs. "
                        f"Please use 1D tensor with shape (1,) instead."
                    )
                info_list.append(
                    {
                        "dtype": torch_dtype,
                        "shape": shape_expr,
                        "is_output": is_output,
                    }
                )
            else:
                # Scalar parameter - cannot be output
                if is_output:
                    raise ValueError(
                        f"Parameter at index {i} is a scalar (T.{param.dtype}) but marked as output. "
                        f"TileLang does not support scalar outputs. "
                        f"Please use 1D tensor with shape (1,) instead."
                    )

                # Get dtype from param
                dtype_str = str(param.dtype)
                torch_dtype = self._tvm_dtype_to_torch(dtype_str)
                info_list.append(
                    {
                        "dtype": torch_dtype,
                        "shape": [],  # scalar has empty shape
                        "is_output": False,
                    }
                )

        return info_list

    def _tvm_dtype_to_torch(self, dtype_str):
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

    def _parse_grid(self):
        launcher = LaunchThreadExtractor()
        expr = launcher.extract(self.mod, "blockIdx.x")
        self.metadata["gridfunc"] = str(expr)

    def _read_mlir_file(self, file_path) -> str:
        """
        Read the content of the MLIR file and return it as a string.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
            return content
        except FileNotFoundError:
            print(f"Error: File '{file_path}' does not exist")
            return None
        except Exception as e:
            print(f"Error occurred while reading the file: {e}")
            return None

    def _parse_npuir_metadata(self) -> None:
        """
        Parse NPU IR to extract metadata required for NPU compilation.
        Extracts and updates the following fields in metadata:
          - mix_mode
          - kernel_name
          - tensor_kinds (currently hardcoded)
          - shared (currently hardcoded)
          - name (combined kernel_name and mix_mode)

        Additionally, removes the mix_mode attribute from the IR.
        """
        # --- Regular expressions and examples ---
        # Example: func.func @gather_sorted_kernel(%arg0: ...) -> gather_sorted_kernel
        KERNEL_NAME_REGEX = r"func\.func\s+@(\w+)"

        # Example：hivm.module_core_type<MIX> -> MIX
        MIX_MODE_REGEX = r"#hivm\.module_core_type<([^>]+)>"

        # Example: test_mix_aic -> test
        MIX_SUFFIX_REGEX = r"_(mix_aic|mix_aiv)$"

        # Note: Compiled Kernel requires to estimate size of shared memory to occupy
        # Currently, NPU backend does not limit on shared memory
        self.metadata["shared"] = 1
        # the mix mode is also encoded into metadata['name'] for runtime to distinguish
        kernel_name = re.search(KERNEL_NAME_REGEX, self.mlir_content).group(1)
        self.metadata["kernel_name"] = kernel_name
        # matching the end of the _mix_aic or _mix_aiv
        self.metadata["name"] = re.sub(MIX_SUFFIX_REGEX, "", kernel_name)
        self.metadata["tensor_kinds"] = []
        self.metadata["mix_mode"] = (
            re.search(MIX_MODE_REGEX, self.mlir_content).group(1).lower()
        )

    def _parse_signature(self) -> dict:
        """
        Parse parameter types from MLIR text and return a dictionary.
        """
        # Define the data types of concern
        target_types = {
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
        }

        # Extract the function signature part (the content within the parentheses)
        pattern = r"func\.func\s*@[^(]*\(([^)]*)\)"
        match = re.search(pattern, self.mlir_content)

        if not match:
            return {}

        params_str = match.group(1)

        # Segmentation parameters
        params = []
        current_param = ""
        brace_count = 0
        angle_count = 0

        for char in params_str:
            if char == "," and brace_count == 0 and angle_count == 0:
                params.append(current_param.strip())
                current_param = ""
            else:
                current_param += char
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                elif char == "<":
                    angle_count += 1
                elif char == ">":
                    angle_count -= 1

        if current_param:
            params.append(current_param.strip())

        result = {}
        index = 0

        # Skip parameters insert by compiler
        for param in params[3:-6]:
            # Check if the type includes the target type
            found_type = None
            for t_type in target_types:
                # Check for types with an x prefix (e.g., xf16)
                x_pattern = r"\bx" + t_type + r"\b"
                if re.search(x_pattern, param):
                    found_type = "*" + t_type
                    break
                # Check the common type (such as i32)
                elif re.search(r"\b" + t_type + r"\b", param):
                    found_type = t_type
                    break

            if found_type:
                # Special handling: f16 should be mapped to fp16,
                # and f32 should be mapped to fp32.
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

    def _npuir_to_bin_enable_npu_compile(self):
        linalg = self.mlir_content
        with tempfile.TemporaryDirectory() as tmpdir:
            ttadapter_path = os.path.join(tmpdir, "kernel.npuir")
            Path(ttadapter_path).write_text(linalg)
            bin_file = os.path.join(tmpdir, "kernel")
            bin_path = os.path.join(tmpdir, "kernel.o")
            so_path = os.path.join(tmpdir, "libkernel.so")

            npu_compiler_path = get_npucompiler_path()
            # TileLang Ascend JIT Runtime now follows Triton JIT style.
            # bishengir-compile --enable-triton-kernel-compile=true make sure the way.

            # Build compile options with pass_configs support
            _compile_option_list = []

            # Get configuration from pass_configs with default values
            pass_configs = getattr(self, "pass_configs", {})

            # Handle --enable-auto-multi-buffer option
            enable_auto_multi_buffer = pass_configs.get(
                "npuir.enable_auto_multi_buffer",
                True,  # Default: enabled
            )
            _compile_option_list.append(
                f"--enable-auto-multi-buffer={str(enable_auto_multi_buffer).lower()}"
            )

            # Handle --disable-hivm-auto-inject-sync option
            disable_hivm_auto_inject_sync = pass_configs.get(
                "npuir.disable_hivm_auto_inject_sync",
                False,  # Default: enabled
            )
            _compile_option_list.append(
                f"--disable-hivm-auto-inject-sync={str(disable_hivm_auto_inject_sync).lower()}"
            )

            _compile_option_list.append("--enable-triton-kernel-compile=true")

            _compile_option_list.append("--enable-hivm-compile=true")

            TILELANG_ASCEND_MODE = os.environ.get("TILELANG_ASCEND_MODE")
            if TILELANG_ASCEND_MODE is None or TILELANG_ASCEND_MODE.lower().strip() in [
                "expert",
                "exp",
                "e",
            ]:
                _compile_option_list.append("--disable-hivm-tensor-compile=true")

            cmd_list = (
                [npu_compiler_path, ttadapter_path]
                + _compile_option_list
                + ["-o", bin_file]
            )
            try:
                ret = subprocess.run(
                    cmd_list, capture_output=True, check=True, text=True
                )
                print("AscendNPU IR compile success:", ret.stdout)
            except subprocess.CalledProcessError as e:
                # print ir
                print("AscendNPU IR:\n")
                print(self.mlir_content)
                # print error info
                print("err cmd:", " ".join(cmd_list))
                print(f"err code: {e.returncode}")
                raise RuntimeError(f"NPU IR opt failed: {e.stderr}") from e
            except Exception as e:
                raise RuntimeError(f"NPU IR opt failed: {e.stderr}") from e
            result = self._get_workspace_size(
                so_path, "_infer_workspace_shape_function"
            )
            self.workspace_size = result
            if not Path(bin_path).exists():
                err_lines = [
                    "AscendNPU IR compile reported success but output object was not generated.",
                    f"Expected output: {bin_path}",
                    f"cmd: {' '.join(cmd_list)}",
                ]
                if ret.stdout:
                    err_lines.append(f"stdout:\n{ret.stdout}")
                if ret.stderr:
                    err_lines.append(f"stderr:\n{ret.stderr}")
                raise RuntimeError("\n".join(err_lines))

            return Path(bin_path).read_bytes()

    def make_npu_launcher_stub(self, name, header_src, wrapper_src, debug=False):
        """
        Generate the launcher stub to launch the kernel
        """
        precompile_cache_path = get_runtime_file_cache(header_src)
        header_path = os.path.join(precompile_cache_path, "npu_launcher.h")
        precompile_header_path = os.path.join(
            precompile_cache_path, "npu_launcher.h.gch"
        )
        if not (
            os.path.exists(precompile_header_path)
            and os.path.getsize(precompile_header_path) > 0
        ):
            print("Precompiling NPU launcher header...")
            with tempfile.TemporaryDirectory() as tmpdir:
                safe_copy(header_src, header_path)
                tmp_header_gch_path = os.path.join(tmpdir, "npu_launcher.h.gch")
                precompile_npu_ext(header_path, tmp_header_gch_path)
                safe_copy(tmp_header_gch_path, precompile_header_path)

        cache_key = wrapper_src.encode("utf-8") + Path(header_src).read_bytes()
        wrapper_cache_path = get_runtime_file_cache(cache_key)
        launcher_so_path = os.path.join(wrapper_cache_path, f"{name}.so")
        if os.path.exists(launcher_so_path) and os.path.getsize(launcher_so_path) > 0:
            return launcher_so_path

        with tempfile.TemporaryDirectory() as tmpdir:
            dst_path = os.path.join(tmpdir, f"{name}.cxx")
            tmp_so_path = os.path.join(tmpdir, f"{name}.so")
            with open(dst_path, "w") as f:
                f.write(wrapper_src)
            so = build_npu_ext(
                name,
                header_path,
                dst_path,
                kernel_launcher="torch",
                precompile=True,
                output_path=tmp_so_path,
            )
            safe_copy(so, launcher_so_path)
            return launcher_so_path

    def check_debug_op(self, func) -> bool:
        """
        Check if there are debug operations in the func, we only need to print debug info
        in the device side when there are debug operations in the func, otherwise it may cause performance loss.
        """
        assert isinstance(func, PrimFunc), "Expected func to be a PrimFunc"

        found = False

        def visit(node):
            nonlocal found
            if isinstance(node, tir.Call) and "debug" in node.op.name:
                found = True

        tir.stmt_functor.post_order_visit(func.body, visit)
        return found
