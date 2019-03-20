from dataclasses import dataclass

import torch
from torchsim.core.graph.slot_container import Inputs, MemoryBlocks
from torchsim.core.graph.unit import Unit
from torchsim.core.graph.worker_node_base import WorkerNodeBase
from torchsim.core.memory.tensor_creator import TensorCreator
from torchsim.core.models.expert_params import ParamsBase
from torchsim.gui.observables import ObserverPropertiesItem, enable_on_runtime
from torchsim.gui.validators import *
from torchsim.gui.validators import validate_predicate


@dataclass
class VisitedAreaParams(ParamsBase):
    fading_factor: float = 0.9


class VisitedAreaUnit(Unit):
    _creator: TensorCreator
    _params: VisitedAreaParams

    def __init__(self, creator: TensorCreator,
                 output_height: int, output_width: int, num_channels: int, params: VisitedAreaParams):
        super().__init__(creator.device)

        self._creator = creator
        self._params = params

        self.last_visited_area = creator.zeros([output_height, output_width, num_channels], dtype=creator.float,
                                               device=creator.device)

    def step(self, node_input: torch.Tensor):
        current = self.last_visited_area * self._params.fading_factor
        self.last_visited_area.copy_(torch.add(current, node_input))


class VisitedAreaInputs(Inputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.node_input = self.create("Visited_area_mask")


class VisitedAreaOutputs(MemoryBlocks):
    def __init__(self, owner):
        super().__init__(owner)
        self.visited_area = self.create("Visited area")

    def prepare_slots(self, unit: VisitedAreaUnit):
        self.visited_area.tensor = unit.last_visited_area


class VisitedAreaNode(WorkerNodeBase):
    _unit: VisitedAreaUnit
    _params: VisitedAreaParams
    inputs: VisitedAreaInputs
    outputs: VisitedAreaOutputs

    @property
    def fading_factor(self) -> float:
        return self._params.fading_factor

    @fading_factor.setter
    def fading_factor(self, value: float):
        validate_positive_float(value)
        self._params.fading_factor = value

    def __init__(self, name="VisitedAreaNode"):
        super().__init__(name=name,
                         inputs=VisitedAreaInputs(self),
                         outputs=VisitedAreaOutputs(self))

        self._params = VisitedAreaParams()

    def _create_unit(self, creator: TensorCreator):
        input_height, input_width, n_input_channels = self.inputs.node_input.tensor.shape

        return VisitedAreaUnit(creator, input_height, input_width, n_input_channels, self._params)

    def _step(self):
        self._unit.step(self.inputs.node_input.tensor)

    def validate(self):
        """Checks that the input has correct dimensions."""
        super().validate()

        node_input = self.inputs.node_input.tensor

        validate_predicate(lambda: node_input.dim() == 3,
                           f"The input should be 3D (y, x, channels) but has shape {node_input.shape}")

    def get_properties(self) -> List[ObserverPropertiesItem]:
        """Define which properties can be changed from GUI and how."""

        return [self._prop_builder.auto('Number of inputs', type(self).fading_factor, edit_strategy=enable_on_runtime)]
