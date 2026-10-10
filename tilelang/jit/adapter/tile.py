"""PyTorch adapter for compiler-owned Tile object bytes and final launch ABI."""

import torch

from .base import BaseKernelAdapter
from tilelang.env import env
from tilelang.opentile.compiler import TileCompilationResult
from tilelang.opentile.manifest import ParameterBinding, validate_scalar_value
from tilelang.opentile.tile_obj import TileObjectKernel


class TileKernelAdapter(BaseKernelAdapter):
    def __init__(self, *, artifact, compilation_result: TileCompilationResult,
                 result_idx=None):
        if not isinstance(compilation_result, TileCompilationResult):
            raise TypeError("compilation_result must be TileCompilationResult")
        self.compilation_result = compilation_result
        launch_spec = compilation_result.launch_spec
        if artifact.params is None:
            raise ValueError("artifact.params is required; do not lower with runtime_only=True")

        self.artifact = artifact
        self.mod = artifact.device_mod
        self.params = list(artifact.params)
        self.launch_spec = launch_spec
        self.result_idx = self._legalize_result_idx(result_idx)
        count = len(self.params)
        self.input_idx = [i for i in range(count) if i not in self.result_idx]

        # KernelParam uses empty shape for scalars. Explicit handle bindings
        # also permit rank-zero tensors without mistaking then for scalars.
        mapped_kinds = {}
        for arg in launch_spec.arguments:
            if isinstance(arg.source, ParameterBinding):
                i = arg.source.index
                if i >= count:
                    raise ValueError(f"device argument references missing logical parameter {i}")
                if i in mapped_kinds and mapped_kinds[i] != arg.kind:
                    raise ValueError(f"logical parameter {i} has conflicting ABI kinds")
                mapped_kinds[i] = arg.kind
        if any(i not in mapped_kinds for i in self.result_idx):
            raise ValueError("every allocated output must be bound to a device argument")

        self.tensor_params = set()
        self.shapes = {}
        self.dtypes = {}
        self.scalar_kinds = {}
        for i, param in enumerate(self.params):
            is_tensor = bool(param.shape) or mapped_kinds.get(i) == "handle"
            if is_tensor:
                if mapped_kinds.get(i, "handle") != "handle":
                    raise ValueError(f"tensor parameter {i} requires ABI kind handle")
                self.tensor_params.add(i)
                self.shapes[i] = tuple(int(dim) for dim in param.shape)
                self.dtypes[i] = param.torch_dtype()
            else:
                dtype = str(param.dtype)
                kind = "int32" if dtype == "bool" else dtype
                if mapped_kinds.get(i, kind) != kind:
                    raise ValueError(f"scalar parameter {i}: ABI kind must match {kind}")
                self.scalar_kinds[i] = kind
        if any(i not in self.tensor_params for i in self.result_idx):
            raise ValueError("out_idx may only select tensor parameters")
        if not any(i in self.tensor_params for i in self.input_idx):
            raise NotImplementedError("at least one input tensor is required to select the NPU device")

        self.runtime_kernel = TileObjectKernel(
            object_bytes=compilation_result.object_bytes,
            kernel_name=launch_spec.kernel_name,
            arg_types=[arg.kind for arg in launch_spec.arguments],
            block_count=launch_spec.block_count,
            dynamic_ubuf_bytes=launch_spec.dynamic_ubuf_bytes,
        )
        self._post_init()

    def _invoke_validated(self, logical, device):
        with torch.npu.device(device):
            for i in self.result_idx:
                logical[i] = torch.empty(
                    self.shapes[i],
                    dtype=self.dtypes[i],
                    device=device,
                )

            physical = [
                logical[arg.source.index]
                if isinstance(arg.source, ParameterBinding)
                else arg.source.value
                for arg in self.launch_spec.arguments
            ]

            self.runtime_kernel(*physical)

        outputs = [logical[i] for i in self.result_idx]
        return outputs[0] if len(outputs) == 1 else outputs

    def _convert_torch_func(self):
        def invoke(*args):
            if len(args) != len(self.input_idx):
                raise TypeError(f"expected {len(self.input_idx)} inputs, got {len(args)}")
            logical = [None] * len(self.params)
            device = None
            for i, value in zip(self.input_idx, args):
                if i in self.tensor_params:
                    if not isinstance(value, torch.Tensor):
                        raise TypeError(f"parameter {i} must be a tensor")
                    if value.device.type != "npu":
                        raise ValueError(f"parameter {i} must be on NPU")
                    if value.shape != self.shapes[i] or value.dtype != self.dtypes[i]:
                        raise ValueError(f"parameter {i} requires shape={self.shapes[i]}, dtype={self.dtypes[i]}")
                    if not value.is_contiguous():
                        raise ValueError(f"parameter {i} must be contiguous")
                    if device is not None and value.device != device:
                        raise ValueError("all input tensors must be on the same NPU")
                    device = value.device
                else:
                    validate_scalar_value(self.scalar_kinds[i], value)
                logical[i] = value
            profile_value = str(env.TILELANG_NPU_PROFILE).strip()
            if profile_value == "0":
                return self._invoke_validated(logical, device)

            from tilelang.profiler.npu import (
                NPUProfileConfig,
                get_npu_profiler_controller,
                npu_annotation,
            )

            config = NPUProfileConfig.from_env()
            controller = get_npu_profiler_controller()

            def profiled_invoke():
                with npu_annotation(
                    f"tilelang::{self.launch_spec.kernel_name}",
                    enabled=config.enabled,
                    strict=config.strict,
                ):
                    return self._invoke_validated(logical, device)

            return controller.run(config, profiled_invoke)
        return invoke


    def get_kernel_source(self, kernel_only=True):
        return self.artifact.kernel_source

    def get_host_source(self):
        return ""   # Python adapter; no generated host source.

    def get_exportable_executable(self):
        raise NotImplementedError("Tile object persistence is not integrated with JIT cache yet")
