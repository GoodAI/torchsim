from torchsim.core.nodes.join_node import JoinNode, Join
from torchsim.core.graph.connection import Connector
from torchsim.core.graph.slots import MemoryBlock
from torchsim.core.graph.invertible_node import InversePassOutputPacket
from torchsim.core.memory.tensor_creator import AllocatingCreator
from torchsim.core.utils.tensor_utils import same

from tests.core.nodes.node_unit_test_base import NodeTestBase


class TestJoin(NodeTestBase):
    @classmethod
    def setup_class(cls, device: str = 'cuda'):
        super().setup_class(device)
        cls._dim = 1

    def _generate_input_tensors(self):
        yield [
            self._creator.full((2, 1, 3), fill_value=1, device=self._device, dtype=self._dtype),
            self._creator.full((2, 2, 3), fill_value=2, device=self._device, dtype=self._dtype),
            self._creator.full((2, 1, 3), fill_value=3, device=self._device, dtype=self._dtype)
        ]

    def _generate_expected_results(self):
        yield [self._creator.tensor([[[1, 1, 1], [2, 2, 2], [2, 2, 2], [3, 3, 3]],
                                     [[1, 1, 1], [2, 2, 2], [2, 2, 2], [3, 3, 3]]],
                                    dtype=self._dtype, device=self._device)]

    def _create_node(self):
        return JoinNode(dim=self._dim, n_inputs=3)


class TestJoinFlatten(NodeTestBase):
    @classmethod
    def setup_class(cls, device: str = 'cuda'):
        super().setup_class(device)
        cls._flatten = True

    def _generate_input_tensors(self):
        yield [
            self._creator.full((2, 5, 4), fill_value=1, device=self._device, dtype=self._dtype),
            self._creator.full((2, 2, 3), fill_value=2, device=self._device, dtype=self._dtype),
            self._creator.full((3, 1, 3), fill_value=3, device=self._device, dtype=self._dtype)
        ]

    def _generate_expected_results(self):
        yield [self._creator.tensor(([1] * (2 * 5 * 4)) + ([2] * (2 * 2 * 3)) + ([3] * (3 * 1 * 3)),
                                    dtype=self._dtype, device=self._device)]

    def _create_node(self):
        return JoinNode(flatten=self._flatten, n_inputs=3)


def _test_join_inverse(dim, input_shapes, expected_results, creator, dtype, device):
    output_tensor = creator.tensor([[1, 2, 3, 4],
                                    [5, 6, 7, 8],
                                    [9, 10, 11, 12],
                                    [13, 14, 15, 16]], dtype=dtype, device=device)

    join_unit = Join(creator, dim=dim, input_tensor_shapes=input_shapes, dtype=dtype)

    results = join_unit.inverse_projection(output_tensor)

    for expected, result in zip(expected_results, results):
        assert same(expected, result)


def test_join_inverse_dim_0():
    creator = AllocatingCreator(device='cpu')
    dtype = creator.float32

    dim = 0

    input_shapes = [(2, 4), (2, 4)]

    expected_results = [creator.tensor([[1, 2, 3, 4],
                                        [5, 6, 7, 8]], dtype=dtype, device=creator.device),
                        creator.tensor([[9, 10, 11, 12],
                                        [13, 14, 15, 16]], dtype=dtype, device=creator.device)]

    _test_join_inverse(dim, input_shapes, expected_results, creator, dtype, creator.device)


def test_join_inverse_dim_1():
    creator = AllocatingCreator(device='cpu')
    dtype = creator.float32

    dim = 1

    input_shapes = [(4, 2), (4, 2)]

    expected_results = [creator.tensor([[1, 2], [5, 6], [9, 10], [13, 14]], dtype=dtype, device=creator.device),
                        creator.tensor([[3, 4], [7, 8], [11, 12], [15, 16]], dtype=dtype, device=creator.device)]

    _test_join_inverse(dim, input_shapes, expected_results, creator, dtype, creator.device)


def test_join_node_inverse_0():
    # TODO (Test): Make a dim = 1 variant
    # TODO (Test): Then, refactor tests here, maybe something to match the test class above, but for the backward projection.
    creator = AllocatingCreator(device='cpu')
    dtype = creator.float32

    dim = 0

    # The result of the inverse projection should only be one tensor.
    expected_results = [creator.tensor([[1, 2, 3, 4],
                                        [5, 6, 7, 8]], dtype=dtype, device=creator.device),
                        creator.tensor([[9, 10, 11, 12],
                                        [13, 14, 15, 16]], dtype=dtype, device=creator.device)]

    input_memory_blocks = [MemoryBlock(), MemoryBlock()]
    input_memory_blocks[0].tensor = creator.zeros((2, 4))
    input_memory_blocks[1].tensor = creator.zeros((2, 4))

    output_tensor = creator.tensor([[1, 2, 3, 4],
                                    [5, 6, 7, 8],
                                    [9, 10, 11, 12],
                                    [13, 14, 15, 16]], dtype=dtype, device=creator.device)

    join_node = JoinNode(dim, n_inputs=2)
    Connector.connect(input_memory_blocks[0], join_node.inputs[0])
    Connector.connect(input_memory_blocks[1], join_node.inputs[1])

    output_inverse_packet = InversePassOutputPacket(output_tensor, join_node.outputs.output)

    join_node.allocate_memory_blocks(creator)
    results = join_node.recursive_inverse_projection_from_output(output_inverse_packet)

    for expected, result in zip(expected_results, results):
        assert same(expected, result.tensor)


def test_join_node_inverse_flatten():
    device = 'cpu'
    creator = AllocatingCreator(device)
    dtype = creator.float32

    # The result of the inverse projection should only be one tensor.
    expected_results = [creator.tensor([[1, 2, 3, 4],
                                        [5, 6, 7, 8]], dtype=dtype, device=device),
                        creator.tensor([9, 10], dtype=dtype, device=device)]

    input_memory_blocks = [MemoryBlock(), MemoryBlock()]
    input_memory_blocks[0].tensor = creator.zeros((2, 4))
    input_memory_blocks[1].tensor = creator.zeros((2,))

    output_tensor = creator.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=dtype, device=device)

    join_node = JoinNode(flatten=True)
    Connector.connect(input_memory_blocks[0], join_node.inputs[0])
    Connector.connect(input_memory_blocks[1], join_node.inputs[1])

    output_inverse_packet = InversePassOutputPacket(output_tensor, join_node.outputs.output)

    join_node.allocate_memory_blocks(creator)
    results = join_node.recursive_inverse_projection_from_output(output_inverse_packet)

    for expected, result in zip(expected_results, results):
        assert same(expected, result.tensor)
