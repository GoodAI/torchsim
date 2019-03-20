from dataclasses import dataclass, field

import numpy as np
import torch

from torchsim.core.graph.slot_container import MemoryBlocks, Inputs
from torchsim.core.graph.unit import Unit
from torchsim.core.graph.worker_node_base import WorkerNodeBase
from torchsim.core.memory.tensor_creator import TensorCreator
from torchsim.core.models.expert_params import ParamsBase
from torch import FloatTensor

from torchsim.gui.observables import ObserverPropertiesItem, disable_on_runtime, enable_on_runtime
from torchsim.gui.validators import *


@dataclass
class WeightedAvgNodeParams(ParamsBase):
    num_inputs: int = 2  # number of inputs, also expected size of 3rd dimension of the input
    weights: List[float] = field(default_factory=list)
    weights_on_input: bool = False  # if True, the weights are read from the second input of the Node


class WeightedAvgNodeUnit(Unit):
    _creator: TensorCreator
    _params: WeightedAvgNodeParams

    def __init__(self, creator: TensorCreator,
                 output_height: int,
                 output_width: int,
                 params: WeightedAvgNodeParams,
                 num_weights: int = 0):

        super().__init__(creator.device)

        self._params = params
        self._creator = creator

        if self._params.weights_on_input:
            self.num_weights = num_weights

            weights = np.ones((output_height, output_width, self.num_weights))
        else:
            assert len(self._params.weights) != 0, 'Parametric weights not set!'
            assert self._params.num_inputs == len(self._params.weights)
            self.num_weights = len(self._params.weights)

            weights = np.ones((output_height, output_width, self.num_weights))
            for i in range(self.num_weights):
                weights[i] = np.dot(weights[i], params.weights[i])

        self.node_weights = creator.tensor(weights, dtype=creator.float, device=creator.device)
        self.last_output = creator.zeros([output_width, output_height], dtype=creator.float, device=creator.device)

    def step(self, inputs: FloatTensor, input_weights: FloatTensor = None):
        if self._params.weights_on_input:
            for i in range(self.num_weights):
                self.node_weights[i] *= input_weights[i]

        self.last_output.copy_(torch.sum(inputs * self.node_weights, dim=0))


class WeightedAvgNodeInputs(Inputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.node_inputs = self.create("Inputs")
        self.input_weights = self.create("Input weights")


class WeightedAvgNodeOutputs(MemoryBlocks):
    def __init__(self, owner):
        super().__init__(owner)
        self.output = self.create("Output")

    def prepare_slots(self, unit: WeightedAvgNodeUnit):
        self.output.tensor = unit.last_output


class WeightedAvgNode(WorkerNodeBase):
    """ Node that computes weighted average of the input of shape [Y, Xm, n_inputs] across the last dimension.

    The `n_inputs` has to correspond to `params.num_inputs`.
    The given list of weights defines weight with which the input dimensions are weighted.

    """
    _unit: WeightedAvgNodeUnit
    _params: WeightedAvgNodeParams
    inputs: WeightedAvgNodeInputs
    outputs: WeightedAvgNodeOutputs

    @property
    def num_inputs(self) -> int:
        return self._params.num_inputs

    @num_inputs.setter
    def num_inputs(self, value: int):
        element = 1.0 / value

        self._params.num_inputs = value
        self._params.weights = [element] * value

    @property
    def weights_on_input(self) -> bool:
        return self._params.weights_on_input

    @weights_on_input.setter
    def weights_on_input(self, value: bool):
        self._params.weights_on_input = value

    @property
    def input_weights(self) -> List[float]:
        return self._params.weights

    @input_weights.setter
    def input_weights(self, value: List[int]):
        self._params.num_inputs = len(value)
        validate_list_of_size(value=value, size=self._params.num_inputs)

        self._params.weights = value

    def __init__(self, name="WeightedAvgNode"):
        super().__init__(name=name,
                         inputs=WeightedAvgNodeInputs(self),
                         outputs=WeightedAvgNodeOutputs(self))

        self._params = WeightedAvgNodeParams()

    def _create_unit(self, creator: TensorCreator) -> Unit:
        height, width, num_inputs = self.inputs.node_inputs.tensor.shape

        if self._params.weights_on_input:
            num_weights = self.inputs.input_weights.tensor.shape[0]
            return WeightedAvgNodeUnit(creator=creator,
                                       output_height=height,
                                       output_width=width,
                                       params=self._params,
                                       num_weights=num_weights)
        else:
            return WeightedAvgNodeUnit(creator=creator,
                                       output_height=height,
                                       output_width=width,
                                       params=self._params)

    def _step(self):
        if self._params.weights_on_input:
            self._unit.step(self.inputs.node_inputs.tensor, self.inputs.input_weights.tensor)
        else:
            self._unit.step(self.inputs.node_inputs.tensor)

    def validate(self):
        """Checks that the input has correct dimensions."""
        super().validate()

        node_inputs = self.inputs.node_inputs.tensor

        validate_predicate(lambda: node_inputs.dim() == 3,
                           f"The input should be 3D (y, x, input_idx) but has shape {node_inputs.shape}")

        if self._params.weights_on_input:
            validate_predicate(lambda: self.inputs.input_weights.tensor is not None,
                               f"if the weights_on_input is enabled, the input_weights has to be connected")

    def get_properties(self) -> List[ObserverPropertiesItem]:
        """Define which properties can be changed from GUI and how."""

        return [self._prop_builder.auto('Number of inputs', type(self).num_inputs, edit_strategy=disable_on_runtime),
                self._prop_builder.auto('Input weights', type(self).input_weights, edit_strategy=enable_on_runtime),
                self._prop_builder.auto('Weights on input', type(self).weights_on_input,
                                        edit_strategy=disable_on_runtime)]
