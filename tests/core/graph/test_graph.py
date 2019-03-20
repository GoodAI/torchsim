from pytest import raises
from tempfile import TemporaryDirectory

from torchsim.core.exceptions import IllegalStateException
from torchsim.core.graph import Topology, GenericNodeGroup
from torchsim.core.graph.connection import Connector
from torchsim.core.persistence.loader import Loader
from torchsim.core.persistence.saver import Saver
from torchsim.core.utils.tensor_utils import same
from tests.core.graph.node_stub import RandomNodeStub
from tests.core.graph.test_node_ordering import NodeStub


def create_graph():
    graph = Topology('cpu')

    node_1 = RandomNodeStub()
    node_2 = RandomNodeStub()
    node_3 = RandomNodeStub()

    graph.add_node(node_1)
    graph.add_node(node_2)
    graph.add_node(node_3)

    return graph


def test_graph_save_load():
    graph = create_graph()
    graph2 = create_graph()

    graph.step()
    graph2.step()

    graph2.nodes[0].outputs.output.tensor.random_()
    graph2.nodes[1].outputs.output.tensor.random_()
    graph2.nodes[2].outputs.output.tensor.random_()

    with TemporaryDirectory() as folder:
        saver = Saver(folder)
        graph.save(saver)
        saver.save()

        loader = Loader(folder)
        graph2.load(loader)

    for i in range(2):
        assert same(graph.nodes[i].outputs.output.tensor, graph2.nodes[i].outputs.output.tensor)


def test_graph_node_group_ordering():
    graph = Topology('cpu')

    node1 = NodeStub()
    node_group = GenericNodeGroup('group', 1, 1)
    inner_node = NodeStub()
    node2 = NodeStub()

    graph.add_node(node1)
    graph.add_node(node_group)
    graph.add_node(node2)

    node_group.add_node(inner_node)

    Connector.connect(node1.outputs[0], node_group.inputs[0])

    Connector.connect(node_group.inputs[0].output, inner_node.inputs[0])
    Connector.connect(inner_node.outputs[0], node_group.outputs[0].input)

    Connector.connect(node_group.outputs[0], node2.inputs[0])

    Connector.connect(node1.outputs[1], node2.inputs[1])

    graph.order_nodes()
    assert [node1, node_group, node2] == graph._ordered_nodes
    assert [inner_node] == node_group._ordered_nodes


def test_is_initialized():
    graph = Topology('cpu')
    assert not graph.is_initialized()

    graph.add_node(NodeStub())
    assert not graph.is_initialized()

    graph.prepare()
    assert graph.is_initialized()

    with raises(IllegalStateException):
        graph.add_node(NodeStub())
