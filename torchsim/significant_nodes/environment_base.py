from abc import ABC, abstractmethod
from typing import Tuple

from torchsim.core.graph.node_base import EmptyInputs
from torchsim.core.graph.node_group import NodeGroupBase, GroupOutputs
from torchsim.core.models.expert_params import ParamsBase


class CommonEnvironmentOutputs(GroupOutputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.data = self.create("Data")
        self.label = self.create("Label")


class EnvironmentParamsBase(ParamsBase, ABC):
    env_size: Tuple[int, int, int]
    n_shapes: int

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()


class EnvironmentBase(NodeGroupBase[EmptyInputs, CommonEnvironmentOutputs], ABC):

    def __init__(self, params: EnvironmentParamsBase, name: str):
        super().__init__(name, outputs=CommonEnvironmentOutputs(self))

        self.validate_params(params)

        self.params = params

    def __str__(self):
        return self._name

    @abstractmethod
    def get_correct_label_memory_block(self):
        pass

    @staticmethod
    @abstractmethod
    def validate_params(params: EnvironmentParamsBase):
        """Check if the params are valid."""
