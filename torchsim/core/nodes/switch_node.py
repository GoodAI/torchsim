import logging
from dataclasses import dataclass

import torch
from torchsim.core.graph.node_base import NodeValidationException
from torchsim.core.graph.slot_container import MemoryBlocks, Inputs
from torchsim.core.graph.unit import Unit
from torchsim.core.graph.worker_node_base import WorkerNodeBase
from torchsim.core.memory.tensor_creator import TensorCreator
from torchsim.gui.observer_system import ObserverPropertiesItem
from torchsim.gui.validators import *

logger = logging.getLogger(__name__)


class SwitchUnit(Unit):
    get_index_from_input: bool
    active_input_index: int

    def __init__(self,
                 shape,
                 creator: TensorCreator,
                 get_index_from_input: bool,
                 active_input_index: int):
        super().__init__(creator.device)
        self.get_index_from_input = get_index_from_input
        self.active_input_index = active_input_index
        self.output = creator.zeros(*shape, dtype=self._float_dtype, device=self._device)

    def step(self, tensors: List[torch.Tensor]):
        if self.get_index_from_input:
            if tensors[-1].numel() == 1:
                self.active_input_index = int(tensors[-1].item())
            else:
                self.active_input_index = torch.argmax(tensors[-1]).item()
        self.output.copy_(tensors[self.active_input_index])


class SwitchInputs(Inputs):
    def __init__(self, owner, n_inputs, get_index_from_input):
        super().__init__(owner)
        for i in range(n_inputs):
            self.create(f"Input {i}")

        if get_index_from_input:
            self.switch_signal = self.create(f"Switch input")


class SwitchOutputs(MemoryBlocks):
    def __init__(self, owner):
        super().__init__(owner)
        self.output = self.create("Output")

    def prepare_slots(self, unit: SwitchUnit):
        self.output.tensor = unit.output


@dataclass
class SwitchNodeParams:
    get_index_from_input: bool = False
    active_input_index: int = 0


class SwitchNode(WorkerNodeBase[SwitchInputs, SwitchOutputs]):
    """Routes one of multiple inputs to a single output.

    The node takes an index and outputs the input corresponding to the index. It can operate in two modes:
        1. Get the index from a user-editable property (index in [0, number_of_inputs - 1])
        2. Get the index from the last input (index in [0, number_of_inputs - 1])
        3. Get the index from argmax of input (argmax in tensor [number_of_inputs])

    A checkbox in the GUI determines which mode to be in. The mode does not change after the unit has been instantiated.
    If the index comes from the user property, it can be changed when the simulation is running and input will be
    rerouted accordingly.
    """
    inputs: SwitchInputs
    outputs: SwitchOutputs
    _n_inputs: int
    _device: str
    _unit: SwitchUnit
    _params: SwitchNodeParams

    def __init__(self,
                 n_inputs: int,
                 get_index_from_input: bool=False,
                 active_input_index: int=0,
                 name="Switch"):
        super().__init__(name=name,
                         inputs=SwitchInputs(self, n_inputs, get_index_from_input),
                         outputs=SwitchOutputs(self))
        self._params = SwitchNodeParams(get_index_from_input, active_input_index)
        self._n_inputs = n_inputs

    def _create_unit(self, creator: TensorCreator) -> Unit:
        assert self._n_inputs >= 2, "The switch node requires at least two inputs"
        shape = self.inputs[0].tensor.shape
        switchable_inputs = self.inputs[:-1] if self._params.get_index_from_input else self.inputs
        assert all([i.tensor.shape == shape for i in switchable_inputs]), "All input tensors must have the same shape"
        return SwitchUnit(shape,
                          creator,
                          self._params.get_index_from_input,
                          self._params.active_input_index)

    def _step(self):
        self._unit.step([i.tensor for i in self.inputs])

    @property
    def get_index_from_input(self) -> bool:
        return self._params.get_index_from_input

    @get_index_from_input.setter
    def get_index_from_input(self, value: bool):
        self._params.get_index_from_input = value
        if self.is_initialized():
            self._unit.get_index_from_input = value

    @property
    def active_input_index(self) -> int:
        if self._unit:
            return self._unit.active_input_index
        return self._params.active_input_index

    @active_input_index.setter
    def active_input_index(self, value: int):
        validate_positive_with_zero_int(value)
        if value >= self._n_inputs:
            raise FailedValidationException(f"Input index must be lesser than {self._n_inputs} (number of inputs)")
        self._params.active_input_index = value
        if self.is_initialized():
            self._unit.active_input_index = value

    def get_properties(self) -> List[ObserverPropertiesItem]:
        """Returns the user-settable properties for the switch node.

        Properties:
            Index from input: If True, get the input index from the 0th node input, otherwise from the user.
            Input index: Allows the user to change the input index on the fly.
        """
        return [
            self._prop_builder.auto('Index from input', type(self).get_index_from_input),
            self._prop_builder.auto('Input index', type(self).active_input_index),
        ]

    def validate(self):
        super().validate()
        if self.get_index_from_input:
            if self.inputs.switch_signal.tensor.numel() != 1 \
                    and tuple(self.inputs.switch_signal.tensor.shape) != (self._n_inputs,):
                raise NodeValidationException(f"The switch signal input vector is of wrong dimension "
                                              f"{self.inputs[0].tensor.shape}, it should be either one integer or a one"
                                              f" hot vector of the size equal to the `n_inputs` (self._n_inputs).")

    def change_input(self, index: int):
        if not 0 <= index < self._n_inputs:
            raise ValueError(f"Switch node '{self.name}' switched to {index} outside of range [0, {self._n_inputs})")
        else:
            self._unit.active_input_index = index
