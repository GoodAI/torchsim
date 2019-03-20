from typing import Tuple, List

import torch

from torchsim.core.graph.inverse_pass_packet import InversePassOutputPacket, InversePassInputPacket
from torchsim.core.graph.slot_container import MemoryBlocks, Inputs
from torchsim.core.graph.unit import InvertibleUnit
from torchsim.core.graph.worker_node_base import InvertibleWorkerNodeBase
from torchsim.core.memory.tensor_creator import TensorCreator
from torchsim.core.utils.tensor_utils import change_dim, multi_unsqueeze


class Expand(InvertibleUnit):
    """Expands the input across the specified dim.

     If the current size of the dim at that point is not 1, the input will be unsqueezed first, so the expanded dim is
     effectively inserted at the desired place. If the expanded dim is higher than the current number of dims,
     singular dims are added until there is a dim which can be expanded.
     """

    def __init__(self, creator: TensorCreator, input_shape: Tuple[int], dim: int, desired_size: int):
        super().__init__(creator.device)

        self.dim = dim

        # Three options for expanded missing_dims:
        #   1.) The new dim is out of range, so append it to the end (first appending necessary number of singular
        #  dimensions)
        #   2.) The value at that dim is not 1, so unsqueeze there and then expand the new dim.
        #   3.) The dim is in range and is 1, so we will expand there.

        self.unsqueezed_dims = []

        # add singular dimensions if necessary
        if self.dim >= len(input_shape):
            missing_dims = self._n_missing_dims(len(input_shape), self.dim + 1)
            self.unsqueezed_dims = [i + len(input_shape) for i in range(missing_dims)]
            input_shape += tuple([1] * missing_dims)

        if input_shape[self.dim] != 1:  # option 2
            new_shape = input_shape[:self.dim] + (dim,) + input_shape[self.dim:]
            self.new_shape = change_dim(new_shape, self.dim, desired_size)
            self.unsqueezed_dims.append(self.dim)
        else:  # options 1 and 3
            self.new_shape = change_dim(input_shape, self.dim, desired_size)

        self.output = creator.zeros(self.new_shape, dtype=self._float_dtype, device=self._device)

    def step(self, data: torch.Tensor):
        # Unsqueeze if we need to (option 1 or 2)
        if data.dim() != self.output.dim():
            if self.dim < data.dim():  # option 2
                data = data.unsqueeze(self.dim)
            else:  # option 1
                missing_dims = range(data.dim(), data.dim() + self._n_missing_dims(data.dim(), self.output.dim()))
                data = multi_unsqueeze(data, missing_dims)

        data = data.expand(self.new_shape)
        self.output.copy_(data)

    @staticmethod
    def _n_missing_dims(n_input_dims: int, n_target_dims: int):
        return n_target_dims - n_input_dims

    def inverse_projection(self, data: torch.Tensor) -> torch.Tensor:
        index = [0 if dim == self.dim else slice(None)
                 for dim in range(len(self.output.shape))]

        # The indexing gets rid of the dimension, return the 1 there by unsqueezing.
        original_data = data[index].unsqueeze(self.dim)

        i = 0
        for dim in self.unsqueezed_dims:
            dim -= i
            original_data = original_data.squeeze(dim=dim)
            i += 1

        return original_data


class ExpandInputs(Inputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.input = self.create("Input")


class ExpandOutputs(MemoryBlocks):
    def __init__(self, owner):
        super().__init__(owner)
        self.output = self.create("Output")

    def prepare_slots(self, unit: Expand):
        self.output.tensor = unit.output


class ExpandNode(InvertibleWorkerNodeBase[ExpandInputs, ExpandOutputs]):
    """Expands the input across the specified dim.

    If the current size of the dim at that point is not 1, the input will be unsqueezed first.
    """

    _unit: Expand
    inputs: ExpandInputs
    outputs: ExpandOutputs

    def __init__(self, dim: int, desired_size: int, name="Expand"):
        super().__init__(name=name, inputs=ExpandInputs(self), outputs=ExpandOutputs(self))
        self.dim = dim
        self.desired_size = desired_size

    def _create_unit(self, creator: TensorCreator) -> Expand:
        return Expand(creator, self.inputs.input.tensor.shape, self.dim, self.desired_size)

    def _step(self):
        self._unit.step(self.inputs.input.tensor)

    def _inverse_projection(self, data: InversePassOutputPacket) -> List[InversePassInputPacket]:
        projection = self._unit.inverse_projection(data.tensor.view(self._unit.output.shape))
        return [InversePassInputPacket(projection.view(self.inputs.input.tensor.shape), self.inputs.input)]
