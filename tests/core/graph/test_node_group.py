import torch

import pytest

from torchsim.core import get_float
from torchsim.core.graph import Topology, GenericNodeGroup, IdGenerator
from torchsim.core.graph.connection import Connector
from torchsim.core.graph.inverse_pass_packet import InversePassOutputPacket
from torchsim.core.nodes.fork_node import ForkNode
from torchsim.core.nodes.join_node import JoinNode
from torchsim.core.nodes.lambda_node import LambdaNode
from torchsim.core.nodes.constant_node import ConstantNode
from torchsim.core.utils.tensor_utils import same
from tests.core.graph.test_node_ordering import NodeStub


def create_source_node():
    def source(inputs, outputs):
        outputs[0][0] = 1
    return LambdaNode(source, 0, [(1,)])


def create_pass_through_node(name='pass through'):
    def pass_through(inputs, outputs):
        outputs[0].copy_(inputs[0])
    return LambdaNode(pass_through, 1, [(1,)], name=name)


def test_node_id_generation():
    graph = GenericNodeGroup('cpu', 2, 2)

    node_1 = NodeStub()
    node_2 = NodeStub()
    node_3 = NodeStub()

    graph.add_node(node_1)
    graph.add_node(node_2)
    graph.add_node(node_3)

    graph._assign_ids_to_nodes(IdGenerator())

    assert 1 == node_1.id
    assert 2 == node_2.id
    assert 3 == node_3.id


def test_node_group_empty():
    graph = Topology('cpu')

    source_node = create_source_node()
    group_node = graph.create_generic_node_group('group', 1, 1)

    Connector.connect(source_node.outputs[0], group_node.inputs[0])

    graph.add_node(source_node)
    graph.step()


def test_node_group_pass_through():
    graph = Topology('cpu')

    source_node = create_source_node()
    group_node = graph.create_generic_node_group('group', 1, 1)
    destination_node = create_pass_through_node('destination')

    Connector.connect(group_node.inputs[0].output, group_node.outputs[0].input)

    Connector.connect(source_node.outputs[0], group_node.inputs[0])
    Connector.connect(group_node.outputs[0], destination_node.inputs[0])

    graph.add_node(source_node)
    graph.add_node(destination_node)

    graph.step()

    assert 1 == destination_node.outputs[0].tensor[0]


def test_node_group_pass_through_node():
    graph = Topology('cpu')

    source_node = create_source_node()
    destination_node = create_pass_through_node('destination')

    group_node = graph.create_generic_node_group('group', 1, 1)
    pass_through_node = create_pass_through_node()
    group_node.add_node(pass_through_node)

    Connector.connect(group_node.inputs[0].output, pass_through_node.inputs[0])
    Connector.connect(pass_through_node.outputs[0], group_node.outputs[0].input)

    Connector.connect(source_node.outputs[0], group_node.inputs[0])
    Connector.connect(group_node.outputs[0], destination_node.inputs[0])

    graph.add_node(source_node)
    graph.add_node(destination_node)

    graph.step()

    assert 1 == destination_node.outputs[0].tensor[0]


def test_node_group_no_data_on_output():
    graph = Topology('cpu')

    source_node = create_source_node()
    destination_node = create_pass_through_node('destination')

    group_node = graph.create_generic_node_group('group', 1, 1)
    pass_through_node = create_pass_through_node()
    group_node.add_node(pass_through_node)

    Connector.connect(group_node.inputs[0].output, pass_through_node.inputs[0])

    Connector.connect(source_node.outputs[0], group_node.inputs[0])
    Connector.connect(group_node.outputs[0], destination_node.inputs[0])

    graph.add_node(source_node)
    graph.add_node(destination_node)

    # This should fail because the node output does not have anything connected to it from the inside.
    with pytest.raises(TypeError):
        graph.step()


def test_node_group_inner_source():
    graph = Topology('cpu')

    source_node = create_source_node()
    destination_node = create_pass_through_node('destination')

    group_node = graph.create_generic_node_group('group', 1, 1)
    group_node.add_node(source_node)

    Connector.connect(source_node.outputs[0], group_node.outputs[0].input)

    Connector.connect(group_node.outputs[0], destination_node.inputs[0])

    graph.add_node(destination_node)

    graph.step()

    assert 1 == destination_node.outputs[0].tensor[0]


# The following test are clones of test_inverse_projection.test_inverse_fork_join(), but with a node group.
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.slow)])
def test_node_group_inverse_projection_pass_through(device):
    float_dtype = get_float(device)

    graph = Topology(device)

    source_node = ConstantNode(shape=(2, 4), constant=0)
    fork_node = ForkNode(dim=1, split_sizes=[1, 3])
    join_node = JoinNode(dim=1, n_inputs=2)
    group_node = graph.create_generic_node_group('group', 2, 2)

    graph.add_node(source_node)
    graph.add_node(fork_node)
    graph.add_node(join_node)

    Connector.connect(source_node.outputs.output, fork_node.inputs.input)
    Connector.connect(fork_node.outputs[0], group_node.inputs[0])
    Connector.connect(fork_node.outputs[1], group_node.inputs[1])

    Connector.connect(group_node.inputs[0].output, group_node.outputs[0].input)
    Connector.connect(group_node.inputs[1].output, group_node.outputs[1].input)

    Connector.connect(group_node.outputs[0], join_node.inputs[0])
    Connector.connect(group_node.outputs[1], join_node.inputs[1])

    graph.step()

    output_tensor = torch.rand((2, 4), device=device, dtype=float_dtype)
    inverse_pass_packet = InversePassOutputPacket(output_tensor, join_node.outputs.output)
    results = join_node.recursive_inverse_projection_from_output(inverse_pass_packet)

    assert 1 == len(results)
    assert same(results[0].tensor, output_tensor)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.slow)])
def test_node_group_inverse_projection_fork_inside(device):
    float_dtype = get_float(device)

    graph = Topology(device)

    source_node = ConstantNode(shape=(2, 4), constant=0)
    fork_node = ForkNode(dim=1, split_sizes=[1, 3])
    join_node = JoinNode(dim=1, n_inputs=2)
    group_node = graph.create_generic_node_group('group', 1, 2)

    graph.add_node(source_node)
    group_node.add_node(fork_node)
    graph.add_node(join_node)

    Connector.connect(source_node.outputs.output, group_node.inputs[0])

    Connector.connect(group_node.inputs[0].output, fork_node.inputs.input)

    Connector.connect(fork_node.outputs[0], group_node.outputs[0].input)
    Connector.connect(fork_node.outputs[1], group_node.outputs[1].input)

    Connector.connect(group_node.outputs[0], join_node.inputs[0])
    Connector.connect(group_node.outputs[1], join_node.inputs[1])

    graph.step()

    output_tensor = torch.rand((2, 4), device=device, dtype=float_dtype)
    inverse_pass_packet = InversePassOutputPacket(output_tensor, join_node.outputs.output)
    results = join_node.recursive_inverse_projection_from_output(inverse_pass_packet)

    assert 1 == len(results)
    assert same(results[0].tensor, output_tensor)
