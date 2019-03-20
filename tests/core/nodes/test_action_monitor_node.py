import torch
import numpy as np

import pytest

from torchsim.core import get_float
from torchsim.core.actions import SpaceEngineersActionsDescriptor
from torchsim.core.graph.connection import Connector
from torchsim.core.graph.slots import MemoryBlock
from torchsim.core.memory.tensor_creator import AllocatingCreator
from torchsim.core.nodes.actions_monitor_node import ActionMonitorNode
from torchsim.core.utils.tensor_utils import same

descriptor = SpaceEngineersActionsDescriptor()


@pytest.mark.parametrize("action_in, override, override_action", [(torch.zeros(descriptor.ACTION_COUNT), False,
                                                                   np.ones(descriptor.ACTION_COUNT)),
                                                                  (torch.zeros(descriptor.ACTION_COUNT), True,
                                                                   np.ones(descriptor.ACTION_COUNT))])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.slow)])
def test_action_monitor_node(action_in, override, override_action, device):

    float_dtype = get_float(device)

    action_in = action_in.type(float_dtype).to(device)

    node = ActionMonitorNode(descriptor)
    node.allocate_memory_blocks(AllocatingCreator(device=device))

    block = MemoryBlock()
    block.tensor = action_in

    Connector.connect(block, node.inputs.action_in)

    node._override_checked = override
    node._actions_values = override_action
    node.step()

    output = node.outputs.action_out.tensor
    expected_output = torch.tensor(override_action, dtype=float_dtype, device=device) if override else action_in

    assert same(expected_output, output)


