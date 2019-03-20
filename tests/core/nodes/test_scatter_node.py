import pytest
import torch

from torchsim.core import get_float
from torchsim.core.actions import SpaceEngineersActionsDescriptor
from torchsim.core.graph.connection import Connector
from torchsim.core.graph.slots import MemoryBlock
from torchsim.core.memory.tensor_creator import AllocatingCreator
from torchsim.core.nodes.agent_actions_parser_node import AgentActionsParserNode
from torchsim.core.nodes.scatter_node import ScatterNode
from torchsim.core.utils.tensor_utils import same


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
class TestScatterNode:
    @pytest.mark.parametrize('input, mask, output_shape, dimension, expected_result', [
        (
                [[0, 1, 2, 3], [4, 5, 6, 7]],
                [[4, 3, 2, 1], [0, 1, 2, 3]],
                (2, 5),
                1,
                [[0, 3, 2, 1, 0], [4, 5, 6, 7, 0]]
        ),
        (
                [[0, 1, 2, 3]],
                [[0, 0, 0, 0]],
                (3, 4),
                0,
                [[0, 1, 2, 3], [0, 0, 0, 0], [0, 0, 0, 0]]
        ),
        (
                [[[[0, 1, 2, 3]]]],
                [[[[0, 0, 0, 0]]]],
                (1, 1, 3, 4),
                2,
                [[[[0, 1, 2, 3], [0, 0, 0, 0], [0, 0, 0, 0]]]]
        )
    ])
    def test_scatter_node(self, device, input, mask, output_shape, dimension, expected_result):
        float_dtype = get_float(device)
        input_mb = MemoryBlock()
        input_mb.tensor = torch.tensor(input, device=device, dtype=float_dtype)
        expected = torch.tensor(expected_result, device=device, dtype=float_dtype)
        sn = ScatterNode(mapping=mask, output_shape=output_shape, dimension=dimension, device=device)

        Connector.connect(input_mb, sn.inputs.input)
        sn.allocate_memory_blocks(AllocatingCreator(device))
        sn.step()

        assert same(sn.outputs.output.tensor, expected)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_actions_parser_node(device):
    float_dtype = get_float(device)
    sed = SpaceEngineersActionsDescriptor()

    input_mb = MemoryBlock()
    input_mb.tensor = torch.tensor([1, 0.5], device=device, dtype=float_dtype)

    expected = torch.tensor([1, 0, 0.5, 0, 0, 0, 0, 0, 0, 0], device=device, dtype=float_dtype)

    sn = AgentActionsParserNode(sed, ['UP', 'FORWARD'], device=device)

    Connector.connect(input_mb, sn.inputs.input)

    sn.allocate_memory_blocks(AllocatingCreator(device))

    sn.step()

    assert same(sn.outputs.output.tensor, expected)
