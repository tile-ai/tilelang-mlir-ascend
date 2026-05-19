# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

# Import necessary modules and classes
from abc import ABC, abstractmethod  # For defining abstract base classes
from dataclasses import dataclass, field  # For defining data classes
from tilelang.utils.npu_arch import (
    AscendArch,
    get_arch_obj,
)  # Import Ascend architecture
from ..roller.hint import Hint  # Import the Hint class
from ..roller.node import OutputNode  # Import the OutputNode class
from typing import List  # For type hinting
from tvm.tir import PrimFunc  # Import PrimFunc for handling tensor IR functions


def auto_infer_current_arch() -> AscendArch:
    """Auto-infer the current Ascend architecture."""
    return get_arch_obj()


@dataclass
class BaseTemplate(ABC):
    """
    Base class template for hardware-aware configurations.
    This serves as an abstract base class (ABC) that defines the structure
    for subclasses implementing hardware-specific optimizations.
    """

    # The architecture of the device, inferred automatically unless explicitly set
    _arch: AscendArch = field(
        default_factory=auto_infer_current_arch, init=False, repr=False
    )

    # The function associated with this template, initially None
    _func: PrimFunc = field(default=None, init=False, repr=False)

    # The outputs nodes associated with this template, initially None
    _output_nodes: List[OutputNode] = field(default=None, init=False, repr=False)

    @abstractmethod
    def get_hardware_aware_configs(
        self, arch: AscendArch = None, topk: int = 10
    ) -> List[Hint]:
        """
        Abstract method that must be implemented by subclasses.
        It should return a list of hardware-aware configurations (hints)
        based on the specified architecture.

        Args:
            arch (AscendArch, optional): The target architecture. Defaults to None.
            topk (int, optional): Number of top configurations to return. Defaults to 10.

        Returns:
            List[Hint]: A list of recommended hardware-aware configurations.
        """
        pass

    def with_arch(self, arch: AscendArch) -> "BaseTemplate":
        """
        Sets the architecture for this template and returns itself.

        Args:
            arch (AscendArch): The architecture to set.

        Returns:
            BaseTemplate: The instance with the updated architecture.
        """
        self._arch = arch
        return self

    def has_arch(self) -> bool:
        """
        Checks whether the architecture is set.

        Returns:
            bool: True if the architecture is set, False otherwise.
        """
        return self._arch is not None

    def equivalent_function(self) -> PrimFunc:
        """
        Returns the function associated with this template.

        Returns:
            PrimFunc: The stored function.
        """
        return self._func

    def initialize_function(self) -> None:
        """
        Placeholder method that should be implemented by subclasses.
        This method is responsible for initializing the function.

        Raises:
            NotImplementedError: If not implemented in the subclass.
        """
        raise NotImplementedError("initialize_function is not implemented")

    def set_function(self, func: PrimFunc) -> "BaseTemplate":
        """
        Sets the function for this template and returns itself.

        Args:
            func (PrimFunc): The function to associate with this template.

        Returns:
            BaseTemplate: The instance with the updated function.
        """
        self._func = func
        return self

    def set_output_nodes(self, output_nodes: List[OutputNode]) -> "BaseTemplate":
        """
        Sets the output nodes for this template and returns itself.

        Args:
            output_nodes (List[OutputNode]): The output nodes to associate with this template.

        Returns:
            BaseTemplate: The instance with the updated output nodes.
        """
        self._output_nodes = output_nodes
        return self

    def recommend_hints(self, topk: int = 10) -> List[Hint]:
        """
        Provides a list of recommended hardware-aware configurations.

        Args:
            topk (int, optional): Number of top configurations to return. Defaults to 10.

        Returns:
            List[Hint]: A list of recommended configurations.
        """
        return self.get_hardware_aware_configs(self._arch, topk)

    @property
    def arch(self) -> AscendArch:
        """
        Returns the current architecture.

        Returns:
            AscendArch: The architecture of this template.
        """
        return self._arch

    @property
    def output_nodes(self) -> List[OutputNode]:
        """
        Returns the output nodes associated with this template.

        Returns:
            List[OutputNode]: The output nodes.
        """
        return self._output_nodes

    def __post_init__(self):
        """
        Post-initialization method that is called after the data class is created.
        Ensures that the function is initialized.
        """
        self.initialize_function()
