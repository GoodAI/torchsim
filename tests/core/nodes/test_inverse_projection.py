import torch

from torchsim.core.graph import Topology
from torchsim.core.graph.connection import Connector
from torchsim.core.graph.invertible_node import InversePassOutputPacket
from torchsim.core.nodes.fork_node import ForkNode
from torchsim.core.nodes.join_node import JoinNode
from torchsim.core.nodes.constant_node import ConstantNode
from torchsim.core.utils.tensor_utils import same


def test_inverse_fork_join():
    """Checks that if you fork and join, the inverse projection will only have one tensor result."""

    device = 'cpu'
    dtype = torch.float32

    graph = Topology(device)

    source_node = ConstantNode(shape=(2, 4), constant=0)
    fork_node = ForkNode(dim=1, split_sizes=[1, 3])
    join_node = JoinNode(dim=1, n_inputs=2)

    graph.add_node(source_node)
    graph.add_node(fork_node)
    graph.add_node(join_node)

    Connector.connect(source_node.outputs.output, fork_node.inputs.input)
    Connector.connect(fork_node.outputs[0], join_node.inputs[0])
    Connector.connect(fork_node.outputs[1], join_node.inputs[1])

    graph.step()

    output_tensor = torch.rand((2, 4), device=device, dtype=dtype)
    inverse_pass_packet = InversePassOutputPacket(output_tensor, join_node.outputs.output)
    results = join_node.recursive_inverse_projection_from_output(inverse_pass_packet)

    assert 1 == len(results)
    assert same(results[0].tensor, output_tensor)
