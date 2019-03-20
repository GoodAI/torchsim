import torch

from torchsim.core.graph.slot_container import MemoryBlocks
from torchsim.core.graph.unit import Unit
from torchsim.core.graph.worker_node_base import WorkerNodeBase
from torchsim.core.memory.tensor_creator import TensorCreator
from typing import List


class FourPoints(Unit):

    _noise_amplitude: float
    bitmap: torch.Tensor
    _current_step: int = 0
    _no_data: int

    def __init__(self, creator: TensorCreator, noise_amplitude=0.2):
        super().__init__(creator.device)
        self._noise_amplitude = noise_amplitude

        self.bitmap = creator.zeros([1, 2], dtype=self._float_dtype, device=self._device)

        self._current_step = 0

        self._data = creator.tensor([[0., 0], [1., 0], [0., 1], [1, 1]], dtype=self._float_dtype, device=self._device)
        self._no_data = self._data.shape[1]

    def step(self, tensors: List[torch.Tensor]):

        self.bitmap.uniform_() * self._noise_amplitude
        self._current_step += 1

        pos = self._current_step % self._no_data
        self.bitmap.copy_(self._data[pos])


class FourPointsOutputs(MemoryBlocks):
    def __init__(self, owner):
        super().__init__(owner)
        self.output = self.create("Output")

    def prepare_slots(self, unit: FourPoints):
        self.output.tensor = unit.bitmap


class FourPointsNode(WorkerNodeBase):
    """A trivial Node which publishes several points."""
    def __init__(self):
        super().__init__(name="FourPointsNode", outputs=FourPointsOutputs(self))

    def _create_unit(self, creator: TensorCreator):
        return FourPoints(creator)

    def _step(self):
        self._unit.step()

