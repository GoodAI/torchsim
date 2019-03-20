import torch

from torchsim.core.graph.connection import Connector
from torchsim.core.graph.slots import MemoryBlock
from torchsim.core.memory.tensor_creator import AllocatingCreator
from torchsim.core.nodes.pass_node import PassNode
from torchsim.core.utils.tensor_utils import same


def test_pass_node():
    device = 'cpu'
    dtype = torch.float32

    input_tensor = torch.tensor([[0, 1, -1], [1, 2, 3]], device=device, dtype=dtype)

    expected_tensor = input_tensor

    mb0 = MemoryBlock()
    mb0.tensor = input_tensor

    pn = PassNode((2, 3))
    pn.allocate_memory_blocks(AllocatingCreator(device))
    Connector.connect(mb0, pn.inputs.input)
    pn._step()

    assert same(pn.outputs.output.tensor, expected_tensor)
