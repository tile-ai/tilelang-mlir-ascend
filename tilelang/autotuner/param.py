"""The auto-tune parameters."""

from __future__ import annotations

from tilelang import tvm as tvm
from tvm.tir import PrimFunc
from tvm.target import Target
from typing import Callable, Literal, Any
from dataclasses import dataclass

import json
import hashlib


@dataclass(frozen=True)
class CompileArgs:
    """Compile arguments for the auto-tuner. Detailed description can be found in `tilelang.jit.compile`.
    Attributes:
        out_idx: List of output tensor indices.
        execution_backend: Execution backend to use for kernel execution (default: "cython").
        target: Compilation target, either as a string or a TVM Target object (default: "npuir").
        target_host: Target host for cross-compilation (default: None).
        verbose: Whether to enable verbose output (default: False).
        pass_configs: Additional keyword arguments to pass to the Compiler PassContext.
        Refer to `tilelang.PassConfigKey` for supported options.
    """

    out_idx: list[int] | int | None = None
    execution_backend: Literal["cython"] = "cython"
    target: Literal["npuir"] = "npuir"
    target_host: str | Target = None
    verbose: bool = False
    pass_configs: dict[str, Any] | None = None

    def compile_program(self, program: PrimFunc):
        import tilelang

        return tilelang.compile(
            program,
            out_idx=self.out_idx,
            target=self.target,
            target_host=self.target_host,
            verbose=self.verbose,
            pass_configs=self.pass_configs,
        )

    def __hash__(self):
        data = {
            "execution_backend": self.execution_backend,
            "target": str(self.target),
            "target_host": str(self.target_host) if self.target_host else None,
            "verbose": self.verbose,
            "pass_configs": json.dumps(self.pass_configs, sort_keys=True)
            if self.pass_configs
            else None,
        }

        hash_obj = hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8"))
        return int.from_bytes(hash_obj.digest(), byteorder="big")


@dataclass(frozen=True)
class ProfileArgs:
    """Profile arguments for the auto-tuner.

    Attributes:
        warmup: Number of warmup iterations.
        rep: Number of repetitions for timing.
        timeout: Maximum time per configuration.
        supply_type: Type of tensor supply mechanism.
        ref_prog: Reference program for correctness validation.
        supply_prog: Supply program for input tensors.
        out_idx: Union[List[int], int] = -1
        supply_type: tilelang.TensorSupplyType = tilelang.TensorSupplyType.Auto
        ref_prog: Callable = None
        supply_prog: Callable = None
        rtol: float = 1e-2
        atol: float = 1e-2
        max_mismatched_ratio: float = 0.01
        skip_check: bool = False
        manual_check_prog: Callable = None
        cache_input_tensors: bool = True
    """

    warmup: int = 25
    rep: int = 100
    timeout: int = 30
    supply_type: Any = None
    ref_prog: Callable | None = None
    supply_prog: Callable | None = None
    rtol: float = 1e-2
    atol: float = 1e-2
    max_mismatched_ratio: float = 0.01
    skip_check: bool = False
    manual_check_prog: Callable | None = None
    cache_input_tensors: bool = True

    def __hash__(self):
        data = {
            "warmup": self.warmup,
            "rep": self.rep,
            "timeout": self.timeout,
            "supply_type": str(self.supply_type),
            "rtol": self.rtol,
            "atol": self.atol,
            "max_mismatched_ratio": self.max_mismatched_ratio,
        }
        hash_obj = hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8"))
        return int.from_bytes(hash_obj.digest(), byteorder="big")


@dataclass(frozen=True)
class AutotuneResult:
    """Results from auto-tuning process.

    Attributes:
        latency: Best achieved execution latency.
        config: Configuration that produced the best result.
        ref_latency: Reference implementation latency.
        libcode: Generated library code.
        func: Optimized function.
        kernel: Compiled kernel function.
    """

    latency: float | None = None
    config: dict | None = None
    ref_latency: float | None = None
    libcode: str | None = None
    func: Callable | None = None
    kernel: Callable | None = None
