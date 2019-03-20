import torch

from torchsim.core.graph.connection import Connector
from torchsim.core.graph.slots import MemoryBlock
from torchsim.core.memory.tensor_creator import AllocatingCreator
from torchsim.core.nodes.lambda_node import LambdaNode
from torchsim.core.utils.tensor_utils import same


def test_lambda_node():
    device = 'cpu'
    dtype = torch.float32

    input_tensor_0 = torch.tensor([0, 1, -1], device=device, dtype=dtype)
    input_tensor_1 = torch.tensor([1, 1, -1], device=device, dtype=dtype)

    expected_tensor = input_tensor_0 + input_tensor_1

    mb0, mb1 = MemoryBlock(), MemoryBlock()
    mb0.tensor, mb1.tensor = input_tensor_0, input_tensor_1

    def add_f(inputs, outputs):
        outputs[0][:] = 0
        outputs[0].add_(inputs[0])
        outputs[0].add_(inputs[1])

    ln = LambdaNode(add_f, 2, [input_tensor_0.shape])
    ln.allocate_memory_blocks(AllocatingCreator(device=device))
    Connector.connect(mb0, ln.inputs[0])
    Connector.connect(mb1, ln.inputs[1])
    ln._step()

    assert same(ln.outputs[0].tensor, expected_tensor)
