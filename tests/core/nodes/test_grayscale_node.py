import pytest

import torch

from torchsim.core.graph.connection import Connector
from torchsim.core.graph.slots import MemoryBlock
from torchsim.core.memory.tensor_creator import AllocatingCreator
from torchsim.core.nodes.grayscale_node import GrayscaleNode
from torchsim.core.utils.tensor_utils import same


class TestGrayscaleNode:

    @pytest.mark.parametrize('squeeze_channel, tensor_data', [(True, [1, 0, 0.2126, 0.7152, 0.0722]),
                                                              (False, [[1], [0], [0.2126], [0.7152], [0.0722]])])
    def test_grayscale_node(self, squeeze_channel, tensor_data):
        device = 'cpu'
        dtype = torch.float32

        input_tensor = torch.tensor([[1, 1, 1], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], device=device, dtype=dtype)
        expected_tensor = torch.tensor(tensor_data, device=device, dtype=dtype)

        mb0 = MemoryBlock()
        mb0.tensor = input_tensor

        gn = GrayscaleNode(squeeze_channel=squeeze_channel)
        Connector.connect(mb0, gn.inputs.input)
        gn.allocate_memory_blocks(AllocatingCreator(device))
        gn._step()

        assert same(gn.outputs.output.tensor, expected_tensor, 0.000001)  # allow small epsilon differences
