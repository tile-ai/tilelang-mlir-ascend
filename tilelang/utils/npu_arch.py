# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import os
import logging
from typing import List


CHIP_SPECS = {
    "Ascend910A": {
        "cores": 32,
        "UB": 256 * 1024,
        "L1": 1024 * 1024,
        "L0A": 64 * 1024,
        "L0B": 64 * 1024,
        "L0C": 256 * 1024,
        "L2": 16 * 1024 * 1024,
        "cube": [16, 16, 16],
    },
    "Ascend910B": {
        "cores": 30,
        "UB": 192 * 1024,
        "L1": 1024 * 1024,
        "L0A": 64 * 1024,
        "L0B": 64 * 1024,
        "L0C": 512 * 1024,
        "L2": 16 * 1024 * 1024,
        "cube": [16, 16, 16],
    },
    "Ascend950": {
        "cores": 24,
        "UB": 248 * 1024,
        "L1": 512 * 1024,
        "L0A": 64 * 1024,
        "L0B": 64 * 1024,
        "L0C": 256 * 1024,
        "L2": 112 * 1024 * 1024,
        "cube": [16, 16, 16],
    },
}
DEFAULT_CHIP = "Ascend910B"


class CubeInstruction:
    def __init__(self, name: str, shape: List[int]):
        self.name = name
        self.shape = shape


class AscendArch:
    """Ascend NPU architecture capabilities."""

    def __init__(self, chip_name: str = None):
        if chip_name is None:
            chip_name = get_ascend_device_name()

        valid_chips = list(CHIP_SPECS.keys())
        if chip_name not in valid_chips:
            chip_name = DEFAULT_CHIP

        self.name = chip_name
        self.chip_name = chip_name
        self.platform = "ascend"

        spec = CHIP_SPECS.get(chip_name, CHIP_SPECS[DEFAULT_CHIP]).copy()

        try:
            from tilelang.utils import NPUUtils

            npuutils = NPUUtils()
            self.compute_max_core = npuutils.get_aicube_core_num()
            self.aicube_core_num = self.compute_max_core
            self.aivector_core_num = npuutils.get_aivector_core_num()
        except Exception as e:
            logging.getLogger(__name__).warning(
                f"Failed to get Ascend arch from NPUUtils: {e}. Using fallback specs."
            )
            self.compute_max_core = spec["cores"]
            self.aicube_core_num = spec["cores"]
            self.aivector_core_num = spec["cores"]

        self.ub_cap = spec["UB"]
        self.l1_cap = spec["L1"]
        self.l0a_cap = spec["L0A"]
        self.l0b_cap = spec["L0B"]
        self.l0c_cap = spec["L0C"]
        self.l2_cache_size_bytes = spec["L2"]
        self.cube_spec = spec.get("cube", [16, 16, 16])

        self.smem_cap = self.ub_cap
        self.max_smem_usage = self.smem_cap
        self.reg_cap = 0
        self.transaction_size = [32, 32]
        self.bandwidth = [900000, 900000]
        self.warp_size = 1
        self.sm_partition = 1

        try:
            from tvm.target import Target

            self.target = Target("llvm -keys=ascend")
        except ImportError:
            self.target = None

    @property
    def cube_dim(self) -> int:
        return self.cube_spec[-1]

    @property
    def cube_shape(self) -> List[int]:
        return self.cube_spec

    @property
    def fractal_shape(self) -> tuple:
        return (self.cube_spec[0], self.cube_spec[1])

    @property
    def mem_cap(self) -> dict:
        return {
            "UB": self.ub_cap,
            "L1": self.l1_cap,
            "L0A": self.l0a_cap,
            "L0B": self.l0b_cap,
            "L0C": self.l0c_cap,
            "L2": self.l2_cache_size_bytes,
        }

    @property
    def supports_native_bf16(self) -> bool:
        # Currently, all supported chips require legalization for BF16.
        return False

    def get_avaliable_tensorintrin_shapes(self):
        self.available_cube_instructions = (CubeInstruction("Davich", [16, 16]),)
        return [t.shape for t in self.available_cube_instructions]


def is_ascend_arch(arch) -> bool:
    return isinstance(arch, AscendArch)


def is_cube_supported_precision(in_dtype: str, accum_dtype: str, arch) -> bool:
    if not isinstance(arch, AscendArch):
        return False
    if arch.chip_name in ["Ascend910A", "Ascend910B", "Ascend310P"]:
        return in_dtype in ["float16", "bfloat16"] and accum_dtype in [
            "float16",
            "bfloat16",
            "float32",
        ]
    return False


class AscendArch910B(AscendArch):
    """Specific properties for Ascend 910B series."""

    pass


class AscendArch910_95(AscendArch):
    """Specific properties for Ascend 910_95 series."""

    pass


# Map device name prefixes to their corresponding architecture classes.
# Order matters if prefixes overlap.
ARCH_MAP = {
    "Ascend910B": AscendArch910B,
    "Ascend910_95": AscendArch910_95,
}


def get_arch_obj(device_name: str = None) -> AscendArch:
    """Identify the architecture type based on the device name prefix."""
    if device_name is None:
        device_name = get_ascend_device_name()
    for prefix, arch_cls in ARCH_MAP.items():
        if device_name.startswith(prefix):
            return arch_cls(device_name)
    # Default fallback for unknown architectures.
    return AscendArch(device_name)


def get_ascend_device_name() -> str:
    # 1. Highest priority: User-specified environment variable
    #    Useful for cross-compilation or overriding runtime detection.
    device_name = os.environ.get("TILELANG_ASCEND_DEVICE_NAME")
    if device_name:
        return device_name.strip()

    # 2. Secondary priority: Runtime capability detection
    try:
        from tilelang.utils import NPUUtils

        return NPUUtils.get().get_arch()
    except Exception as e:
        # We don't want to crash on non-Ascend machines, but silent pass is bad for debugging
        logging.getLogger(__name__).warning(
            f"Failed to get Ascend arch from NPUUtils: {e}. "
            "Please set TILELANG_ASCEND_DEVICE_NAME environment variable."
            "Otherwise we will fallback to Ascend910B."
        )

    # 3. Fallback to DEFAULT_CHIP if runtime detection fails
    return DEFAULT_CHIP


def supports_native_bf16(device_name: str = None) -> bool:
    """Check if the given device natively supports BF16 instructions."""
    arch = get_arch_obj(device_name)
    return arch.supports_native_bf16


def get_arch(target=None) -> AscendArch:
    """Get AscendArch from TVM Target or return default AscendArch.

    Args:
        target: TVM Target object (ignored, always returns AscendArch)

    Returns:
        AscendArch instance
    """
    return get_arch_obj()


def is_tensorcore_supported_precision(
    in_dtype: str, accum_dtype: str, arch=None
) -> bool:
    """Check if tensorcore supports the precision (Ascend uses cube, not tensorcore).

    Returns False for Ascend as tensorcore is NVIDIA-specific.
    """
    return False


__all__ = [
    "AscendArch",
    "is_ascend_arch",
    "is_cube_supported_precision",
    "get_arch_obj",
    "get_ascend_device_name",
    "supports_native_bf16",
    "get_arch",
    "is_tensorcore_supported_precision",
]
