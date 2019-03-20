from typing import List

import torch

from torchsim.core.graph.inverse_pass_packet import InversePassOutputPacket, InversePassInputPacket
from torchsim.core.graph.slot_container import MemoryBlocks, Inputs
from torchsim.core.graph.unit import InvertibleUnit
from torchsim.core.graph.worker_node_base import InvertibleWorkerNodeBase
from torchsim.core.memory.tensor_creator import TensorCreator


class FlattenUnit(InvertibleUnit):
    def __init__(self, creator: TensorCreator, input_shape, start_dim: int, end_dim: int):
        super().__init__(creator.device)

        if start_dim < 0:
            start_dim = len(input_shape) + start_dim
        if end_dim < 0:
            end_dim = len(input_shape) + end_dim

        if not (0 <= start_dim < len(input_shape)):
            raise ValueError(f"start_dim {start_dim} out of input_shape {input_shape}.")
        if not (0 <= end_dim < len(input_shape)):
            raise ValueError(f"end_dim {end_dim} out of input_shape {input_shape}.")

        self.input_shape = input_shape
        self.start_dim = start_dim
        self.end_dim = end_dim

        dim_product = int(torch.prod(torch.tensor(input_shape[start_dim:end_dim + 1])).item())
        self.new_shape = input_shape[:start_dim] + (dim_product,)

        if end_dim + 1 < len(input_shape):
            self.new_shape += input_shape[end_dim + 1:]

        self.output = creator.zeros(tuple(self.new_shape), dtype=self._float_dtype, device=self._device)

    def step(self, data: torch.Tensor):
        self.output.copy_(data.flatten(start_dim=self.start_dim, end_dim=self.end_dim))

    def inverse_projection(self, data: torch.Tensor) -> torch.Tensor:
        return data.view(self.input_shape)


class FlattenNodeInputs(Inputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.input = self.create("Input")


class FlattenNodeOutputs(MemoryBlocks):
    def __init__(self, owner):
        super().__init__(owner)
        self.output = self.create("Output")

    def prepare_slots(self, unit: FlattenUnit):
        self.output.tensor = unit.output


class FlattenNode(InvertibleWorkerNodeBase[FlattenNodeInputs, FlattenNodeOutputs]):
    """Flatten all dimensions by default or specified dimensions. Dimension indexing supports negative ints."""

    _unit: FlattenUnit
    inputs: FlattenNodeInputs
    outputs: FlattenNodeOutputs

    def __init__(self, start_dim: int = 0, end_dim: int = -1, name="Flatten"):
        super().__init__(name=name, inputs=FlattenNodeInputs(self), outputs=FlattenNodeOutputs(self))
        self.start_dim = start_dim
        self.end_dim = end_dim

    def _create_unit(self, creator: TensorCreator) -> FlattenUnit:
        return FlattenUnit(creator, self.inputs.input.tensor.shape, self.start_dim, self.end_dim)

    def _step(self):
        self._unit.step(self.inputs.input.tensor)

    def _inverse_projection(self, data: InversePassOutputPacket) -> List[InversePassInputPacket]:
        projection = self._unit.inverse_projection(data.tensor)
        return [InversePassInputPacket(projection, self.inputs.input)]
