import logging
from dataclasses import dataclass
from typing import Dict

import torch
from torchsim.core import FLOAT_NAN, SMALL_CONSTANT
from torchsim.core.graph.slot_container import Inputs, MemoryBlocks
from torchsim.core.graph.unit import Unit
from torchsim.core.graph.worker_node_base import WorkerNodeWithInternalsBase
from torchsim.core.memory.tensor_creator import TensorCreator
from torchsim.core.persistence.loader import Loader
from torchsim.core.persistence.saver import Saver
from torchsim.gui.observables import Observable
from torchsim.gui.observer_system import ObserverPropertiesItem
from torchsim.gui.observers.plot_observers import StepLinePlotObserver
from torchsim.gui.validators import *
from torch import Tensor

logger = logging.getLogger(__name__)


class AccuracyUnit(Unit):
    _buffer_size: int

    _accuracy_output: torch.Tensor
    _accuracy_output_per_flock: Tensor

    _last_pos: int
    _elements_in_buffer: int
    _buffer_filled: bool

    def __init__(self,
                 creator: TensorCreator,
                 input_shape: torch.Size,
                 buffer_size: int):
        super().__init__(creator.device)

        self._buffer_size = buffer_size
        # TODO: They need to be tensors to ensure serialization persistence, because currently just tensors are
        # TODO: serialized. once this will change, we can change this back to fields.
        self._last_pos = self._buffer_size - 1
        self._elements_in_buffer = 0
        self._flock_size = input_shape[0]

        self._buffer = self._create_tensor((self._buffer_size, self._flock_size), creator)

        self._accuracy_output = self._create_tensor([1], creator)
        self._accuracy_output_per_flock = self._create_tensor([self._flock_size], creator)

    def _create_tensor(self, sizes, creator: TensorCreator):
        return creator.full(sizes,
                            fill_value=FLOAT_NAN,
                            dtype=self._float_dtype,
                            device=self._device)

    def step(self, input_a: torch.Tensor, input_b: torch.Tensor):
        self._increment_pos()

        self._buffer[self._last_pos] = self._compute_accuracy(input_a, input_b)

        # compute Accuracy even if the buffer is not full - use what data there is so far
        # accuracy = self._buffer[0:self._elements_in_buffer].sum() / float(self._elements_in_buffer)
        # self._accuracy_output.copy_(accuracy)
        accuracy_per_flock = self._average_buffer(self._buffer[0:self._elements_in_buffer])
        self._accuracy_output_per_flock.copy_(accuracy_per_flock)
        self._accuracy_output.copy_(accuracy_per_flock.mean())

    @staticmethod
    def _compute_accuracy(input_a: torch.Tensor, input_b: torch.Tensor) -> torch.Tensor:
        diff = (input_a - input_b) ** 2
        flattened = diff.view(diff.shape[0], -1)
        summed = flattened.sum(dim=1)
        mask = summed < SMALL_CONSTANT
        return mask

    @staticmethod
    def _average_buffer(buffer: torch.Tensor) -> torch.Tensor:
        return buffer.float().mean(dim=0)

    def _increment_pos(self):
        self._elements_in_buffer = min(self._elements_in_buffer + 1, self._buffer_size)
        self._last_pos += 1
        if self._last_pos == self._buffer_size:
            self._last_pos = 0

    def _save(self, saver: Saver):
        saver.description['_last_pos'] = self._last_pos
        saver.description['_elements_in_buffer'] = self._elements_in_buffer

    def _load(self, loader: Loader):
        self._last_pos = loader.description['_last_pos']
        self._elements_in_buffer = loader.description['_elements_in_buffer']


class AccuracyInputs(Inputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.input_a = self.create("Input_A")
        self.input_b = self.create("Input_B")


class AccuracyOutputs(MemoryBlocks):
    def __init__(self, owner):
        super().__init__(owner)
        self.accuracy = self.create("Accuracy")
        self.accuracy_per_flock = self.create("Accuracy per flock")

    def prepare_slots(self, unit: AccuracyUnit):
        self.accuracy.tensor = unit._accuracy_output
        self.accuracy_per_flock.tensor = unit._accuracy_output_per_flock


class AccuracyInternals(MemoryBlocks):
    def __init__(self, owner):
        super().__init__(owner)
        self.buffer = self.create("Buffer")

    def prepare_slots(self, unit: AccuracyUnit):
        self.buffer.tensor = unit._buffer


@dataclass
class AccuracyNodeParams:
    buffer_size: int = 1


class AccuracyNode(WorkerNodeWithInternalsBase[AccuracyInputs, AccuracyInternals, AccuracyOutputs]):
    """Computes accuracy of one hot inputs over the flock (dim=0)."""

    _unit: AccuracyUnit
    _params: AccuracyNodeParams
    outputs: AccuracyOutputs
    inputs: AccuracyInputs
    internals: AccuracyInternals

    _seed: int

    def __init__(self, buffer_size: int, name: str = 'Accuracy Node'):
        super().__init__(outputs=AccuracyOutputs(self), inputs=AccuracyInputs(self),
                         memory_blocks=AccuracyInternals(self), name=name)

        self._params = AccuracyNodeParams(buffer_size)
        self._plot_observer = StepLinePlotObserver()
        self._plot_per_flock_observer = StepLinePlotObserver()

    def _create_unit(self, creator: TensorCreator):
        input_shape = self.inputs.input_a.tensor.shape

        return AccuracyUnit(creator, input_shape, self._params.buffer_size)

    def _step(self):
        self._unit.step(self.inputs.input_a.tensor, self.inputs.input_b.tensor)
        self._plot_observer.add_value([self.outputs.accuracy.tensor.item()])
        self._plot_per_flock_observer.add_value(self.outputs.accuracy_per_flock.tensor.tolist())

    @property
    def buffer_size(self) -> int:
        return self._params.buffer_size

    @buffer_size.setter
    def buffer_size(self, value: int):
        validate_positive_int(value)
        self._params.buffer_size = value
        logger.info('Updating the buffer_size, will take effect after the restart')

    def get_properties(self) -> List[ObserverPropertiesItem]:
        return [
            self._prop_builder.auto('Buffer size', type(self).buffer_size)
        ]

    def _get_observables(self) -> Dict[str, Observable]:
        result = super()._get_observables()

        result['Custom.chart'] = self._plot_observer
        result['Custom.chart_per_flock'] = self._plot_per_flock_observer

        return result
