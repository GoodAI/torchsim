import logging
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import torch
from torchsim.core.graph.inverse_pass_packet import InversePassOutputPacket, InversePassInputPacket
from torchsim.core.graph.slot_container import Inputs, MemoryBlocks
from torchsim.core.graph.unit import InvertibleUnit
from torchsim.core.graph.worker_node_base import InvertibleWorkerNodeBase
from torchsim.core.memory.tensor_creator import TensorCreator
from torchsim.core.models.expert_params import ParamsBase
from torchsim.gui.observables import disable_on_runtime
from torchsim.gui.observer_system import ObserverPropertiesItem
from torchsim.gui.validators import *

logger = logging.getLogger(__name__)


class Distribution(Enum):
    """Enumeration of the probability distributions available for generating random noise."""
    Uniform = 1
    Normal = 2


class RandomNoiseUnit(InvertibleUnit):
    """Generates random noise and (optionally) adds it to the input from another node.

    See RandomNoiseNode for further details.
    """
    distribution: Distribution
    amplitude: float
    _image_buffer: torch.Tensor = None
    _random_numbers: torch.Tensor = None

    def __init__(self,
                 creator: TensorCreator,
                 shape: List[int],
                 distribution: Distribution = Distribution.Uniform,
                 amplitude: float = 1.,
                 has_input: bool = True):
        super().__init__(creator.device)
        self.distribution = distribution
        self.amplitude = amplitude
        if has_input:
            self._image_buffer = creator.zeros(*shape, dtype=self._float_dtype, device=self._device)
            self._random_numbers = creator.zeros(*shape, dtype=self._float_dtype, device=self._device)
        self.output = creator.zeros(*shape, dtype=self._float_dtype, device=self._device)

    def step(self, tensors: List[torch.Tensor]):
        n_inputs = len(tensors)
        assert n_inputs <= 1, "The number of inputs needs to be zero or one"
        if n_inputs == 0:
            self._sample(self.output)
        else:
            self._add_in_buffer(tensors[0])

    def _sample(self, tensor: torch.Tensor):
        if self.distribution == Distribution.Uniform:
            tensor.uniform_(to=self.amplitude)
        else:
            tensor.normal_(std=self.amplitude)

    def _add_in_buffer(self, input_tensor: torch.Tensor):
        self._image_buffer.copy_(input_tensor)
        self._sample(self._random_numbers)
        self._image_buffer.add_(self._random_numbers)
        self.output.copy_(self._image_buffer)

    def inverse_projection(self, output_tensor: torch.Tensor) -> torch.Tensor:
        return output_tensor


class RandomNoiseInputs(Inputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.input = self.create("Input")


class RandomNoiseOutputs(MemoryBlocks):
    def __init__(self, owner):
        super().__init__(owner)
        self.output = self.create("Output")

    def prepare_slots(self, unit: RandomNoiseUnit):
        self.output.tensor = unit.output


@dataclass
class RandomNoiseParams(ParamsBase):
    shape: Optional[Sequence[int]] = None
    distribution: str = 'Uniform'  # Use str for JSON serialization
    amplitude: float = 1.0


class RandomNoiseNode(InvertibleWorkerNodeBase[RandomNoiseInputs, RandomNoiseOutputs]):
    """Generates random noise and (optionally) adds it to the input from another node.

    The node generates random noise with the following parameters:
        RandomNoiseParams.distribution: The probability distribution from which to generate noise.
        RandomNoiseParams.amplitude: A factor that multiplies the noise.

    The node has two modes: It can add noise to the output of another node or generate noise without any input.
    In the latter case, an additional parameter is required:
        RandomNoiseParams.shape: The shape of the noise tensor.
    """
    inputs: RandomNoiseInputs
    outputs: RandomNoiseOutputs
    _params: RandomNoiseParams
    _unit: RandomNoiseUnit

    def __init__(self,
                 params: RandomNoiseParams = None, name="RandomNoise"):
        super().__init__(name=name,
                         inputs=RandomNoiseInputs(self),
                         outputs=RandomNoiseOutputs(self))
        if params is None:
            self._params = RandomNoiseParams()
        else:
            self._params = params.clone()

    def _create_unit(self, creator: TensorCreator):

        if self.has_input:
            self._params.shape = list(self.inputs.input.tensor.shape)

        return RandomNoiseUnit(creator,
                               self._params.shape,
                               Distribution[self._params.distribution],
                               self._params.amplitude,
                               self.has_input)

    def _inverse_projection(self, data: InversePassOutputPacket) -> List[InversePassInputPacket]:
        result = self._unit.inverse_projection(data.tensor)

        return [InversePassInputPacket(result, self.inputs.input)]

    @property
    def distribution(self) -> str:
        return self._params.distribution

    @distribution.setter
    def distribution(self, value: str):
        proposed_value = value.capitalize()
        if proposed_value not in Distribution.__members__:
            raise FailedValidationException('Distribution must be Uniform or Normal')
        self._params.distribution = proposed_value
        if self.is_initialized():
            self._unit.distribution = proposed_value

    @property
    def amplitude(self) -> float:
        return self._params.amplitude

    @amplitude.setter
    def amplitude(self, value: float):
        validate_positive_float(value)
        self._params.amplitude = value
        if self.is_initialized():           # runtime property
            self._unit.amplitude = value

    @property
    def shape(self) -> Optional[List[int]]:
        return self._params.shape

    @shape.setter
    def shape(self, value: Optional[List[int]]):
        self._params.shape = value

    def get_properties(self) -> List[ObserverPropertiesItem]:
        return [
            self._prop_builder.auto('Shape', type(self).shape, enabled=not self.has_input, edit_strategy=disable_on_runtime),
            self._prop_builder.auto('Amplitude', type(self).amplitude),
            self._prop_builder.auto('Distribution', type(self).distribution)
        ]

    def _step(self):
        step_parameters = [self.inputs.input.tensor] if self.has_input else []
        self._unit.step(step_parameters)

    @property
    def has_input(self):
        return self.inputs.input.connection is not None
