import torch
from typing import List

from torchsim.core.graph.inverse_pass_packet import InversePassInputPacket, InversePassOutputPacket
from torchsim.core.graph.slot_container import MemoryBlocks, Inputs
from torchsim.core.graph.node_base import NodeValidationException
from torchsim.core.graph.unit import InvertibleUnit
from torchsim.core.graph.worker_node_base import InvertibleWorkerNodeBase
from torchsim.core.memory.tensor_creator import TensorCreator
from torchsim.core.utils.tensor_utils import check_shape
from torchsim.gui.observer_system import ObserverPropertiesItem


class InvalidParameterException(Exception):
    pass


class Fork(InvertibleUnit):
    """The forking unit.

    This unit splits the tensor held by input_block in the dimension specified by fork_dim, based on the split_sizes.
    The interpret_dims are ignored right now.
    """

    def __init__(self, creator: TensorCreator, dim, input_shape, split_sizes, dtype):
        super().__init__(creator.device)

        self._input_shape = input_shape
        self._split_sizes = split_sizes
        self._dim = dim
        self.output_tensors = []
        self._indices = []

        if -1 in split_sizes:
            raise InvalidParameterException("Automatic split size not supported")

        input_dim_size = input_shape[dim]

        outputs_size = sum(split_sizes)
        if outputs_size > input_dim_size:
            message = f"The combined output size ({outputs_size} is larger than dimension {dim}: {input_dim_size})"
            raise InvalidParameterException(message)

        split_start = 0
        for split in split_sizes:
            output_dims = list(input_shape)
            output_dims[dim] = split
            output_tensor = creator.zeros(output_dims, dtype=dtype, device=self._device)
            self.output_tensors.append(output_tensor)

            # Create the index which will be used for the data splitting.
            index = creator.arange(split_start, split_start + split, dtype=torch.long, device=self._device)

            self._indices.append(index)
            split_start += split

    def _check_shape(self, data: torch.Tensor):
        check_shape(self._input_shape, data.shape)

    def step(self, data: torch.Tensor):
        # There is only one input.
        self._check_shape(data)

        for i, index in enumerate(self._indices):
            torch.index_select(data, dim=self._dim, index=index, out=self.output_tensors[i])

    def inverse_projection(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        if len(tensors) != len(self.output_tensors):
            message = f"Less tensors given ({len(tensors)}) than expected ({len(self._split_sizes)})"
            raise InvalidParameterException(message)

        for expected_size, tensor in zip(self._split_sizes, tensors):
            if tensor.shape[self._dim] != expected_size:
                message = f"Wrong tensor size in the fork dimension ({self._dim})."
                raise InvalidParameterException(message)

            dims = list(tensor.shape)
            dims[self._dim] = self._input_shape[self._dim]
            check_shape(dims, list(self._input_shape))

        return torch.cat(tensors, dim=self._dim)


class ForkInputs(Inputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.input = self.create('Input')


class ForkOutputs(MemoryBlocks):
    def __init__(self, owner, n_outputs):
        super().__init__(owner)
        for i in range(n_outputs):
            self.create(f"Output {i}")

    def prepare_slots(self, unit: Fork):
        for output_block, tensor in zip(self, unit.output_tensors):
            output_block.tensor = tensor


class ForkNode(InvertibleWorkerNodeBase[ForkInputs, ForkOutputs]):
    """Forks the input into N outputs with specified dimensions."""

    _split_sizes = List[int]

    _element_count: int
    _unit: Fork
    _creator: TensorCreator

    inputs: ForkInputs
    outputs: ForkOutputs

    def __init__(self, dim: int, split_sizes: List[int], name="Fork"):
        super().__init__(name=name, inputs=ForkInputs(self), outputs=ForkOutputs(self, len(split_sizes)))

        self._dim = dim
        self._split_sizes = split_sizes

    def _create_unit(self, creator: TensorCreator) -> Fork:
        input_tensor = self.inputs.input.tensor

        # TODO (UC): This is quite ugly, but we need it now for the inverse projection.
        self._creator = creator

        return Fork(creator, self._dim, input_tensor.shape, self._split_sizes, dtype=input_tensor.dtype)

    def validate(self):
        outputs_numel = [output.tensor.shape[self._dim] for output in self.outputs]
        if self.inputs.input.tensor.shape[self._dim] != sum(outputs_numel):
            raise NodeValidationException(f"Input tensor selected dim {self.inputs.input.tensor.shape[self._dim]} "
                                          f"is not corresponding with output dims "
                                          f"{' + '.join(map(str, outputs_numel))} = "
                                          f"{sum(outputs_numel)}.")

    def _inverse_projection(self, data: InversePassOutputPacket) -> List[InversePassInputPacket]:
        output_index = self.outputs.index(data.slot)
        dtype = self.outputs[0].tensor.dtype
        device = self.outputs[0].tensor.device
        all_other_tensors = [self._creator.zeros(output_block.tensor.shape, dtype=dtype, device=device)
                             for output_block in self.outputs if output_block != data.slot]

        all_tensors = all_other_tensors[:output_index] + [data.tensor] + all_other_tensors[output_index:]

        result = self._unit.inverse_projection(all_tensors)

        return [InversePassInputPacket(result, self.inputs.input)]

    def get_properties(self) -> List[ObserverPropertiesItem]:
        return []

    def _step(self):
        self._unit.step(self.inputs.input.tensor)
