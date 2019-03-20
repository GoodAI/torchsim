from torchsim.core.nodes.fork_node import Fork, ForkNode
from torchsim.core.graph.connection import Connector
from torchsim.core.graph.slots import MemoryBlock
from torchsim.core.graph.invertible_node import InversePassOutputPacket
from torchsim.core.memory.tensor_creator import AllocatingCreator
from torchsim.core.utils.tensor_utils import same

from tests.core.nodes.node_unit_test_base import NodeTestBase


class TestFork(NodeTestBase):
    @classmethod
    def setup_class(cls, device: str = 'cuda'):
        super().setup_class(device)
        cls._dim = 1
        cls._split_sizes = [1, 2]

    def _generate_input_tensors(self):
        tensors = [self._creator.full((2, 1, 2), fill_value=i, device=self._creator.device, dtype=self._dtype) for i in
                   range(3)]
        yield [self._creator.cat(tensors, dim=1)]

    def _generate_expected_results(self):
        yield [self._creator.tensor([[[0, 0]],
                                     [[0, 0]]], dtype=self._dtype, device=self._creator.device),
               self._creator.tensor([[[1, 1], [2, 2]],
                                     [[1, 1], [2, 2]]], dtype=self._dtype, device=self._creator.device)]

    def _create_node(self):
        return ForkNode(dim=self._dim, split_sizes=self._split_sizes)


def _test_fork_inverse(dim, output_tensors, creator, dtype, device):
    expected_results = creator.tensor([[1, 2, 3, 4],
                                       [5, 6, 7, 8],
                                       [9, 10, 11, 12],
                                       [13, 14, 15, 16]], dtype=dtype, device=device)

    input_shape = (4, 4)
    split_sizes = [2, 2]

    join_unit = Fork(creator, dim, input_shape, split_sizes, dtype)

    results = join_unit.inverse_projection(output_tensors)

    for expected, result in zip(expected_results, results):
        assert same(expected, result)


def test_fork_inverse_dim_0():
    creator = AllocatingCreator(device='cpu')
    dtype = creator.float32

    dim = 0

    outputs = [creator.tensor([[1, 2, 3, 4],
                               [5, 6, 7, 8]], dtype=dtype, device=creator.device),
               creator.tensor([[9, 10, 11, 12],
                               [13, 14, 15, 16]], dtype=dtype, device=creator.device)]

    _test_fork_inverse(dim, outputs, creator, dtype, creator.device)


def test_fork_inverse_dim_1():
    creator = AllocatingCreator(device='cpu')
    dtype = creator.float32

    dim = 1

    outputs = [creator.tensor([[1, 2], [5, 6], [9, 10], [13, 14]], dtype=dtype, device=creator.device),
               creator.tensor([[3, 4], [7, 8], [11, 12], [15, 16]], dtype=dtype, device=creator.device)]

    _test_fork_inverse(dim, outputs, creator, dtype, creator.device)


def test_fork_node_inverse_0():
    # TODO (Test): add for dim = 1, then refactor.
    creator = AllocatingCreator(device='cpu')
    dtype = creator.float32

    dim = 0

    expected_results = [creator.tensor([[1, 2, 3, 4],
                                        [5, 6, 7, 8],
                                        [0, 0, 0, 0],
                                        [0, 0, 0, 0]], dtype=dtype, device=creator.device)]

    input_memory_block = MemoryBlock()
    input_memory_block.tensor = creator.zeros((4, 4))

    output_tensor = creator.tensor([[1, 2, 3, 4],
                                    [5, 6, 7, 8]], dtype=dtype, device=creator.device)

    fork_node = ForkNode(dim, split_sizes=[2, 2])
    Connector.connect(input_memory_block, fork_node.inputs.input)

    output_inverse_packet = InversePassOutputPacket(output_tensor, fork_node.outputs[0])

    fork_node.allocate_memory_blocks(creator)
    results = fork_node.recursive_inverse_projection_from_output(output_inverse_packet)

    for expected, result in zip(expected_results, results):
        assert same(expected, result.tensor)
