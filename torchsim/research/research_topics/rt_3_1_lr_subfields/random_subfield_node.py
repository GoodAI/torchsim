import logging
import numpy as np

from typing import List, Union, Optional

import torch
from torchsim.core.graph.inverse_pass_packet import InversePassOutputPacket, InversePassInputPacket
from torchsim.core.graph.slot_container import MemoryBlocks, Inputs

from torchsim.core.graph.unit import InvertibleUnit
from torchsim.core.graph.worker_node_base import InvertibleWorkerNodeWithInternalsBase
from torchsim.core.memory.tensor_creator import TensorCreator
from torchsim.utils.list_utils import dim_prod
from torchsim.utils.seed_utils import get_rand_generator_or_set_cuda_seed

logger = logging.getLogger(__name__)


class RandomSubfieldForkNodeUnit(InvertibleUnit):
    def __init__(self, input_shape, random: np.random.RandomState, creator: TensorCreator, first_non_expanded_dim,
                 n_outputs, n_samples):
        super().__init__(creator.device)

        self.n_outputs = n_outputs

        input_shape = tuple(input_shape)

        total_size = dim_prod(input_shape[first_non_expanded_dim:])

        filters_sparse = []
        self.counts = torch.zeros(input_shape[first_non_expanded_dim:], dtype=self._float_dtype, device=self._device)
        for _ in range(n_outputs):
            filter_dense = random.choice(total_size, n_samples, replace=False)
            filter_sparse = np.zeros(total_size)
            filter_sparse[filter_dense] = 1
            filter_sparse = torch.tensor(filter_sparse, dtype=self._float_dtype, device=self._device)
            filter_sparse = filter_sparse.view(input_shape[first_non_expanded_dim:])
            self.counts += filter_sparse
            filter_sparse = filter_sparse.expand(input_shape)
            filters_sparse.append(filter_sparse)

        self.filters = filters_sparse

        if (self.counts == 0).any().item() == 1:
            logger.warning("Some subset of receptive field is not included.")

        self.counts = self.counts.expand(input_shape)

        self.output_tensors = \
            [creator.zeros(input_shape, dtype=self._float_dtype, device=self._device) for _ in range(n_outputs)]

    def step(self, tensor: torch.Tensor):
        for i in range(self.n_outputs):
            self.output_tensors[i].copy_(torch.mul(tensor, self.filters[i]))

    def inverse_projection(self, tensor: torch.Tensor, index: int) -> Union[torch.Tensor, List[torch.Tensor]]:
        return torch.div(torch.mul(tensor, self.filters[index]), self.counts)


class RandomSubfieldForkNodeInputs(Inputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.input = self.create("Input")


class RandomSubfieldForkNodeMemoryBlocks(MemoryBlocks):
    def __init__(self, owner, n_outputs):
        super().__init__(owner)
        for i in range(n_outputs):
            self.create(f"Filter {i}")

    def prepare_slots(self, unit: RandomSubfieldForkNodeUnit):
        for output_block, tensor in zip(self, unit.filters):
            output_block.tensor = tensor


class RandomSubfieldForkNodeOutputs(MemoryBlocks):
    def __init__(self, owner, n_outputs):
        super().__init__(owner)
        for i in range(n_outputs):
            self.create(f"Output {i}")

    def prepare_slots(self, unit: RandomSubfieldForkNodeUnit):
        for output_block, tensor in zip(self, unit.output_tensors):
            output_block.tensor = tensor


class RandomSubfieldForkNode(
    InvertibleWorkerNodeWithInternalsBase[
        RandomSubfieldForkNodeInputs, RandomSubfieldForkNodeMemoryBlocks, RandomSubfieldForkNodeOutputs]):
    inputs: RandomSubfieldForkNodeInputs
    outputs: RandomSubfieldForkNodeOutputs
    _unit: RandomSubfieldForkNodeUnit

    _seed: Optional[int]

    @property
    def seed(self) -> Optional[int]:
        return self._seed

    def __init__(self, n_outputs, n_samples, first_non_expanded_dim, seed=0):
        super().__init__(
            name="RandomSubfieldNode",
            inputs=RandomSubfieldForkNodeInputs(self),
            memory_blocks=RandomSubfieldForkNodeMemoryBlocks(self, n_outputs),
            outputs=RandomSubfieldForkNodeOutputs(self, n_outputs),
        )
        self.n_samples = n_samples
        self.n_outputs = n_outputs
        self.first_non_expanded_dim = first_non_expanded_dim
        self._seed = seed

    def _create_unit(self, creator: TensorCreator) -> InvertibleUnit:
        random = get_rand_generator_or_set_cuda_seed(creator.device, self._seed)
        return RandomSubfieldForkNodeUnit(self.inputs.input.tensor.shape, random, creator,
                                          first_non_expanded_dim=self.first_non_expanded_dim,
                                          n_outputs=self.n_outputs,
                                          n_samples=self.n_samples)

    def _step(self):
        self._unit.step(self.inputs.input.tensor)

    def _inverse_projection(self, data: InversePassOutputPacket) -> List[InversePassInputPacket]:
        output_index = self.outputs.index(data.slot)
        result = self._unit.inverse_projection(data.tensor, output_index)
        return [InversePassInputPacket(result, self.inputs.input)]
