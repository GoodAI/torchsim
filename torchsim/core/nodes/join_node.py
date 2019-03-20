from functools import reduce
from typing import List

import torch

from torchsim.core.graph.inverse_pass_packet import InversePassInputPacket
from torchsim.core.graph.slot_container import Inputs, MemoryBlocks
from torchsim.core.graph.invertible_node import InversePassOutputPacket
from torchsim.core.graph.unit import InvertibleUnit
from torchsim.core.graph.worker_node_base import InvertibleWorkerNodeBase
from torchsim.core.memory.tensor_creator import TensorCreator
from torchsim.core.utils.tensor_utils import check_shape
from torchsim.gui.observer_system import ObserverPropertiesItem


class IncompatibleInputsException(Exception):
    pass


class Join(InvertibleUnit):
    def __init__(self, creator: TensorCreator, input_tensor_shapes, dtype, dim=0, flatten=False):
        """Usage of flatten: flatten input vectors first."""
        super().__init__(creator.device)

        self._flatten = flatten
        self._dim = dim
        self._input_tensor_shapes = input_tensor_shapes

        if not flatten:
            output_dims = list(input_tensor_shapes[0])
            output_dims[dim] = 0

            for input_shape in input_tensor_shapes:
                self._check_shape(input_shape, output_dims, dim)
                output_dims[dim] += input_shape[dim]
        else:
            self._flatten_input_shapes = [reduce(lambda a, b: a * b, x) for x in input_tensor_shapes]
            output_dims = (sum(self._flatten_input_shapes),)

        self.output = creator.zeros(*output_dims, dtype=dtype, device=self._device)

    @staticmethod
    def _check_shape(first, second, join_dim):
        if len(first) != len(second):
            raise IncompatibleInputsException(f"Incompatible number of input dimensions: {first} vs {second}")

        for i, (x, y) in enumerate(zip(first, second)):
            if i != join_dim and x != y:
                raise IncompatibleInputsException(f"Dimensions not matching at non-join dimension {i} ({x} vs {y})")

    def step(self, tensors: List[torch.Tensor]):
        if self._flatten:
            tensors = [x.view(-1) for x in tensors]
        torch.cat(tensors, dim=self._dim, out=self.output)

    def inverse_projection(self, output_tensor: torch.Tensor) -> List[torch.Tensor]:
        if self._flatten:
            split = torch.split(output_tensor, self._flatten_input_shapes)
            return [split.view(shape) for shape, split in zip(self._input_tensor_shapes, split)]
        else:
            check_shape(output_tensor.shape, self.output.shape)

            split_sizes = [shape[self._dim] for shape in self._input_tensor_shapes]
            return torch.split(output_tensor, split_sizes, dim=self._dim)


class JoinInputs(Inputs):
    def __init__(self, owner, n_inputs):
        super().__init__(owner)
        for i in range(n_inputs):
            self.create(f"Input {i}")


class JoinOutputs(MemoryBlocks):
    def __init__(self, owner):
        super().__init__(owner)
        self.output = self.create("Output")

    def prepare_slots(self, unit: Join):
        self.output.tensor = unit.output


class JoinNode(InvertibleWorkerNodeBase[JoinInputs, JoinOutputs]):
    """Joins multiple inputs along a specified dimension.

    The inputs must have the same dimensions except for the joining dimension.

    Args:
        dim: The dimension along which the inputs should be joined.
        n_inputs: The number of inputs to be joined.
        flatten: If True, the output will be viewed with a single dimension.
        name: The name of the node.
    """

    _sum_input_sizes: int = 0
    _unit: Join
    _n_inputs: int

    inputs: JoinInputs
    outputs: JoinOutputs

    def __init__(self, dim=0, n_inputs=2, flatten: bool = False, name: str = "Join"):
        super().__init__(name=name, inputs=JoinInputs(self, n_inputs), outputs=JoinOutputs(self))
        self._dim = dim
        self._n_inputs = n_inputs
        self._flatten = flatten

    def _create_unit(self, creator: TensorCreator) -> Join:
        dtype = self.inputs[0].tensor.dtype

        input_tensor_shapes = [input_slot.tensor.shape for input_slot in self.inputs]

        return Join(creator, dim=self._dim, input_tensor_shapes=input_tensor_shapes, dtype=dtype,
                    flatten=self._flatten)

    def _inverse_projection(self, data: InversePassOutputPacket) -> List[InversePassInputPacket]:
        # Join only has one output, no need to identify what actually came in.
        result_tensors = self._unit.inverse_projection(data.tensor.view(self._unit.output.shape))

        # Create inverse pass units based on the order of the inputs.
        return [InversePassInputPacket(result_tensor.view(input_block.tensor.shape), input_block)
                for result_tensor, input_block in zip(result_tensors, self.inputs)]

    def get_properties(self) -> List[ObserverPropertiesItem]:
        return []

    def _step(self):
        self._unit.step([i.tensor for i in self.inputs])
