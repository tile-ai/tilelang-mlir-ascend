"""Ascend 910 (A2) JIT adapter.

Compilation is fully supported: the adapter can emit MLIR, run
``bishengir-compile`` and build the host launcher.  Launching still requires the
runtime bridge (``NPUUtils.load_binary``), which is only available on an NPU
host, so a call raises a clear error instead of silently doing nothing.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path


_MLIR_DTYPE = {
    "float16": "fp16",
    "bfloat16": "bf16",
    "float32": "fp32",
    "float": "fp32",
    "float64": "fp64",
    "int8": "i8",
    "int16": "i16",
    "int32": "i32",
    "int64": "i64",
    "uint32": "u32",
    "uint64": "u64",
    "bool": "i1",
}


class Ascend910KernelAdapter:
    def __init__(
        self,
        *,
        params=None,
        result_idx=None,
        target=None,
        func_or_mod=None,
        host_mod=None,
        device_mod=None,
        device_kernel_source: str = "",
        verbose: bool = False,
        pass_configs=None,
        compile_flags=None,
        **kwargs,
    ):
        self.params = params or []
        self.result_idx = result_idx or []
        self.target = target
        self.func_or_mod = func_or_mod
        self.host_mod = host_mod
        self.device_mod = device_mod
        self.device_kernel_source = device_kernel_source
        self.verbose = verbose
        self.pass_configs = pass_configs or {}
        self.compile_flags = compile_flags
        self.func = self._call
        self.object_path: str | None = None
        self.launcher_path: str | None = None
        self.wrapper_src: str | None = None
        self._compiled = False

    # -- signature / metadata -------------------------------------------------
    def _signature(self):
        if self.params:
            sig = {}
            for i, p in enumerate(self.params):
                try:
                    dtype = str(p.dtype)
                except Exception:
                    dtype = "float32"
                sig[i] = "*" + _MLIR_DTYPE.get(dtype, "fp32")
            return sig
        sig = {}
        f = self.func_or_mod
        if f is not None and hasattr(f, "params") and hasattr(f, "buffer_map"):
            for i, p in enumerate(f.params):
                if p in f.buffer_map:
                    sig[i] = "*" + _MLIR_DTYPE.get(
                        str(f.buffer_map[p].dtype), "fp32"
                    )
                else:
                    sig[i] = _MLIR_DTYPE.get(str(p.dtype), "i32")
        return sig

    def _metadata(self):
        src = self.device_kernel_source
        name = re.search(r"func\.func\s+@(\w+)", src)
        mix = re.search(r"#hivm\.module_core_type<([^>]+)>", src)
        return (
            name.group(1) if name else "main",
            (mix.group(1).lower() if mix else "aiv"),
        )

    # -- compilation ----------------------------------------------------------
    def compile_artifacts(self, output_dir: str | None = None):
        """Compile MLIR to an object and build the host launcher."""
        if self._compiled:
            return self.object_path, self.launcher_path
        from .bishengir import compile_mlir
        from .launcher_compile import compile_launcher
        from .runtime import (
            generate_npu_wrapper_src,
            get_npu_launcher_header,
        )

        from . import cache

        developer = "dev" in str(self.target)
        hit = cache.load(
            self.device_kernel_source,
            pass_configs=self.pass_configs,
            developer=developer,
        )
        if hit is not None and output_dir is None:
            out = hit.root
            self.object_path = str(hit.object_path)
        else:
            out = Path(
                output_dir
                or os.environ.get("TILELANG_A2_BUILD_DIR")
                or tempfile.mkdtemp(prefix="tilelang_a2_")
            )
            out.mkdir(parents=True, exist_ok=True)
            result = compile_mlir(
                self.device_kernel_source,
                out,
                developer=developer,
                verbose=self.verbose,
            )
            self.object_path = str(result.object_path)
            if output_dir is None:
                cached = cache.save(
                    cache.make_key(
                        self.device_kernel_source,
                        pass_configs=self.pass_configs,
                        developer=developer,
                    ),
                    self.device_kernel_source,
                    result.object_path,
                    result.shared_object_path,
                    pass_configs=self.pass_configs,
                    developer=developer,
                )
                out = cached.root
                self.object_path = str(cached.object_path)

        _, mix_mode = self._metadata()
        workspace = int(os.environ.get("TILELANG_ASCEND_WORKSPACE_SIZE", "0"))
        self.wrapper_src = generate_npu_wrapper_src(
            {},
            self._signature(),
            workspace,
            mix_mode,
            -1,
            0,
            False,
        )
        launcher = out / "libkernel_launcher.so"
        self.launcher_path, _ = compile_launcher(
            self.wrapper_src,
            get_npu_launcher_header(),
            launcher,
            verbose=self.verbose,
        )
        self._compiled = True
        return self.object_path, self.launcher_path

    # -- execution ------------------------------------------------------------
    def _grid(self):
        """Best-effort static launch grid from the lowered TIR."""
        text = ""
        for obj in (self.func_or_mod, self.device_mod):
            try:
                if obj is not None:
                    candidate = obj.script()
                    if "thread_binding" in candidate:
                        text = candidate
                        break
            except Exception:
                continue
        grid = []
        for axis in ("x", "y", "z"):
            match = re.search(
                rf'thread_binding\(([^,]+),\s*thread="blockIdx\.{axis}"', text
            )
            value = 1
            if match:
                try:
                    value = int(match.group(1).strip())
                except Exception:
                    value = 1
            grid.append(max(1, value))
        return tuple(grid)

    def __call__(self, *args, **kwargs):
        return self._call(*args, **kwargs)

    def _call(self, *args, **kwargs):
        import importlib.util

        import torch

        try:
            import torch_npu  # noqa: F401
        except Exception as exc:  # pragma: no cover - runtime only
            raise RuntimeError("Ascend 910 execution requires torch_npu") from exc

        from tilelang.utils.npu_utils import NPUUtils

        if not self._compiled:
            self.compile_artifacts()
        if not self.launcher_path or not self.object_path:
            raise RuntimeError("Ascend 910 launcher was not built")

        device = torch.npu.current_device()
        npu = NPUUtils.get()
        name, mix_mode = self._metadata()
        with open(self.object_path, "rb") as handle:
            binary = handle.read()
        _, function, _, _ = npu.load_binary(name, binary, 0, device, mix_mode)

        spec = importlib.util.spec_from_file_location(
            "__tilelang_launcher", self.launcher_path
        )
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)

        if hasattr(torch_npu._C, "_npu_getCurrentRawStreamNoWait"):
            stream = torch_npu._C._npu_getCurrentRawStreamNoWait(device)
        elif hasattr(torch_npu._C, "_npu_getCurrentRawStream"):
            stream = torch_npu._C._npu_getCurrentRawStream(device)
        else:
            stream = torch.npu.current_stream(device).npu_stream

        args = list(args)
        outputs = []
        if self.result_idx and args:
            buffers = []
            if self.func_or_mod is not None and hasattr(self.func_or_mod, "params"):
                buffers = [
                    self.func_or_mod.buffer_map[p]
                    for p in self.func_or_mod.params
                    if p in self.func_or_mod.buffer_map
                ]
            count = len(buffers) or (len(args) + len(self.result_idx))
            for idx in self.result_idx:
                pos = idx if idx >= 0 else count + idx
                shape = [int(dim) for dim in buffers[pos].shape] if buffers else None
                dtype = getattr(torch, str(buffers[pos].dtype)) if buffers else None
                out = torch.empty(
                    shape, dtype=dtype, device=args[0].device
                )
                while len(args) <= pos:
                    args.append(None)
                args[pos] = out
                outputs.append(out)

        signature = self._signature()
        tensors = []
        for index, value in enumerate(args):
            kind = signature.get(index, "")
            if (
                not kind.startswith("*")
                and isinstance(value, torch.Tensor)
                and value.numel() == 1
            ):
                value = value.item()
            elif isinstance(value, torch.Tensor) and not getattr(
                value, "is_npu", False
            ):
                value = value.npu()
            tensors.append(value)

        grid = self._grid()
        launcher.launch(
            grid[0],
            grid[1],
            grid[2],
            stream,
            function,
            {"kernel_name": name, "tensor_kinds": []},
            {},
            None,
            None,
            *tensors,
        )
        if kwargs.get("synchronize", True):
            torch.npu.synchronize()
        if outputs:
            return outputs[0] if len(outputs) == 1 else outputs
        return None

    def get_kernel_source(self, kernel_only: bool = True) -> str:
        return self.device_kernel_source

    def get_profiler(self, *args, **kwargs):
        raise RuntimeError("Ascend 910 profiling requires the device runtime bridge")
