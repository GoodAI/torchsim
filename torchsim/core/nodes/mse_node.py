import logging
from dataclasses import dataclass
from typing import Dict
import numpy as np

import torch
from torchsim.core import FLOAT_NAN
from torchsim.core.eval.series_plotter import plot_multiple_runs
from torchsim.core.graph.slot_container import Inputs, MemoryBlocks
from torchsim.core.graph.unit import Unit
from torchsim.core.graph.worker_node_base import WorkerNodeWithInternalsBase
from torchsim.core.memory.tensor_creator import TensorCreator
from torchsim.core.persistence.loader import Loader
from torchsim.core.persistence.saver import Saver
from torchsim.gui.observables import Observable, MatplotObservable
from torchsim.gui.observer_system import ObserverPropertiesItem
from torchsim.gui.observers.memory_block_observer import CustomTensorObserver
from torchsim.gui.observers.plot_observers import StepLinePlotObserver
from torchsim.gui.validators import *
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


class MseUnit(Unit):
    _buffer_size: int

    _buffer_a: torch.Tensor
    _buffer_b: torch.Tensor
    _buffer_squared_diffs: torch.Tensor

    _mean_square_error_output: torch.Tensor

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

        buffer_shape = [self._buffer_size, *input_shape]

        self._buffer_squared_diffs = self._create_tensor(buffer_shape, creator)
        self._buffer_a = self._create_tensor(buffer_shape, creator)
        self._buffer_b = self._create_tensor(buffer_shape, creator)

        self._mean_square_error_output = self._create_tensor([1], creator)

    def _create_tensor(self, sizes, creator: TensorCreator):
        return creator.full(sizes,
                            fill_value=FLOAT_NAN,
                            dtype=self._float_dtype,
                            device=self._device)

    def step(self, input_a: torch.Tensor, input_b: torch.Tensor):
        self._increment_pos()

        self._buffer_a[self._last_pos] = input_a
        self._buffer_b[self._last_pos] = input_b

        self._compute_mse()

    def _compute_mse(self):
        # compute mse even if the buffer is not full - use what data there is so far
        self._buffer_squared_diffs.copy_((self._buffer_a - self._buffer_b) ** 2)

        n = float(self._elements_in_buffer) * self._buffer_squared_diffs[0].numel()
        self._mean_square_error_output.copy_(self._buffer_squared_diffs[0:self._elements_in_buffer].sum() / n)

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


class MseInputs(Inputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.input_a = self.create("Input_A")
        self.input_b = self.create("Input_B")


class MseOutputs(MemoryBlocks):
    def __init__(self, owner):
        super().__init__(owner)
        self.mean_square_error_output = self.create("MSE")

    def prepare_slots(self, unit: MseUnit):
        self.mean_square_error_output.tensor = unit._mean_square_error_output


class MseInternals(MemoryBlocks):
    def __init__(self, owner):
        super().__init__(owner)
        self.buffer_a = self.create("Buffer_A")
        self.buffer_b = self.create("Buffer_B")
        self.buffer_diffs = self.create("Buffer_diffs")

    def prepare_slots(self, unit: MseUnit):
        self.buffer_a.tensor = unit._buffer_a
        self.buffer_b.tensor = unit._buffer_b
        self.buffer_diffs.tensor = unit._buffer_squared_diffs


@dataclass
class MseNodeParams:
    buffer_size: int = 1


class MseNode(WorkerNodeWithInternalsBase[MseInputs, MseInternals, MseOutputs]):
    """Computes mean squared error form buffered inputs."""

    _unit: MseUnit
    _params: MseNodeParams
    outputs: MseOutputs
    inputs: MseInputs
    internals: MseInternals

    _seed: int

    def __init__(self, buffer_size: int, name: str = 'MSE Node'):
        super().__init__(outputs=MseOutputs(self), inputs=MseInputs(self), memory_blocks=MseInternals(self), name=name)

        self._params = MseNodeParams(buffer_size)
        self._plot_observer = StepLinePlotObserver()

    def _create_unit(self, creator: TensorCreator):
        input_shape = self.inputs.input_a.tensor.shape

        return MseUnit(creator, input_shape, self._params.buffer_size)

    def _step(self):
        self._unit.step(self.inputs.input_a.tensor, self.inputs.input_b.tensor)
        self._plot_observer.add_value(self.outputs.mean_square_error_output.tensor.item())

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

        result['Custom.mse_chart'] = self._plot_observer

        return result
