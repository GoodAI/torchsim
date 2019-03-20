import pytest

import torch
from torchsim.core import get_float
from torchsim.core.graph.connection import Connector
from torchsim.core.graph.slots import MemoryBlock
from torchsim.core.memory.tensor_creator import AllocatingCreator
from torchsim.core.nodes.to_one_hot_node import ToOneHotNode, ToOneHotMode
from tests.testing_utils import measure_time


class TestToOneHotNode:
    @pytest.mark.parametrize('mode, vector, expected_indexes', [
        (ToOneHotMode.RANDOM, [0, 1, -1], [1, 1, 0]),
        (ToOneHotMode.RANDOM, [0, 0, -1], [1, 1, 0]),
        (ToOneHotMode.RANDOM, [1, 0, 1], [1, 0, 1]),
        (ToOneHotMode.RANDOM, [0.5, 0.4, -10], [1, 1, 0]),
        (ToOneHotMode.RANDOM, [0.5, 0.4, 1], [1, 1, 1]),
        (ToOneHotMode.RANDOM, [0.5, 0.5, 1], [1, 1, 1]),
        (ToOneHotMode.RANDOM, [-2, -2, 1], [0, 0, 1]),
        (ToOneHotMode.RANDOM, [0, 0, 1], [0, 0, 1]),
        (ToOneHotMode.RANDOM, [1, 0, 0, 0, 1], [1, 0, 0, 0, 1]),
        (ToOneHotMode.RANDOM, [1, 1], [1, 1]),

        (ToOneHotMode.MAX, [0, 1, -1], [0, 1, 0]),
        (ToOneHotMode.MAX, [0, 1, 1], [0, 1, 1]),
        (ToOneHotMode.MAX, [0, 1, 0], [0, 1, 0]),
        (ToOneHotMode.MAX, [0.1, 1, 0, 0.5], [0, 1, 0, 0]),
        (ToOneHotMode.MAX, [0, 0, 0, 0], [1, 1, 1, 1]),

        (ToOneHotMode.MAX, [
            [0.5, 1, 0],
            [0, 0.5, 1],
        ], [
             [0, 1, 0],
             [0, 0, 1],
         ]),
        (ToOneHotMode.MAX, [
            [0.5, 1, 0.5],
            [0, 0, 0.5],
        ], [
             [0, 1, 0],
             [0, 0, 1],
         ]),
        (ToOneHotMode.RANDOM, [
            [0.5, 1, 0],
            [0.5, 0.5, 1],
        ], [
             [1, 1, 0],
             [1, 1, 1],
         ]),
        (ToOneHotMode.RANDOM, [
            [-0.5, 1, 0],
            [0, 0.5, 1],
        ], [
             [0, 1, 1],
             [0, 1, 1],
         ]),
        (ToOneHotMode.RANDOM, [
            [1, 1, 0.5],
            [0, 0, 0.5],
        ], [
             [1, 1, 1],
             [0, 0, 1],
         ]),

        # over different dims
        (ToOneHotMode.MAX, [
            [[0.5, 1], [0, 0.3], [-5, -6]],
            [[-1, 1], [0, 0.2], [0, 1]]
        ], [
            [[0, 1], [0, 1], [1, 0]],
            [[0, 1], [0, 1], [0, 1]]
         ]),
        (ToOneHotMode.RANDOM, [
            [[0.5, 1], [0, 0.3], [-5, -6]],
            [[-1, 1], [0, 0.2], [0, 1]]
        ], [
             [[1, 1], [0, 1], [1, 0]],
             [[0, 1], [0, 1], [0, 1]]
         ]),

    ])
    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    def test_to_one_hot_test(self, mode, vector, expected_indexes, device):
        mb = MemoryBlock()
        mb.tensor = torch.tensor(vector, device=device, dtype=get_float(device))
        t_expected_indexes = torch.tensor(expected_indexes, device=device, dtype=torch.uint8)

        to_one_hot = ToOneHotNode(mode=mode)
        Connector.connect(mb, to_one_hot.inputs.input)
        to_one_hot.allocate_memory_blocks(AllocatingCreator(device))
        to_one_hot.step()

        last_dim = mb.tensor.shape[-1]
        output_tensor = to_one_hot.outputs.output.tensor.view(-1, last_dim)
        expected_idxs = t_expected_indexes.view(-1, last_dim)

        for output, idxs in zip(output_tensor, expected_idxs):
            assert (output[idxs] == 1).any()
            assert (output[~idxs] == 0).all()

    @pytest.mark.skip("Used for measuring performance of the node.")
    def test_to_one_hot_speed(self, capsys):

        @measure_time(iterations=100, function_repetitions=100)
        def measured_step():
            mb.tensor.copy_(torch.rand(input_shape, dtype=get_float(device), device=device))
            to_one_hot.step()

        input_shape = (150,)
        device = 'cuda'

        vector = torch.zeros(input_shape, dtype=get_float(device), device=device)

        mb = MemoryBlock()
        mb.tensor = torch.tensor(vector, device=device, dtype=get_float(device))

        to_one_hot = ToOneHotNode(mode=ToOneHotMode.RANDOM)
        Connector.connect(mb, to_one_hot.inputs.input)
        to_one_hot.allocate_memory_blocks(AllocatingCreator(device))

        with capsys.disabled():
            measured_step()




