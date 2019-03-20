import torch

import pytest

from torchsim.core.graph import GenericNodeGroup, IdGenerator
from torchsim.core.graph.connection import Connector
from torchsim.core.graph.node_ordering import order_nodes, IllegalCycleException
from torchsim.core.graph.slot_container import Inputs
from torchsim.core.graph.unit import Unit
from torchsim.core.graph.worker_node_base import WorkerNodeBase
from torchsim.core.memory.tensor_creator import TensorCreator
from torchsim.utils.node_utils import TestMemoryBlocks


class NodeStub(WorkerNodeBase[Inputs, TestMemoryBlocks]):
    def __init__(self):
        super().__init__(inputs=Inputs(self), outputs=TestMemoryBlocks(self))

        self.inputs.input1 = self.inputs.create('input 1')
        self.inputs.input2 = self.inputs.create('input 2')

        self.outputs.output1 = self.outputs.create('output_1')
        self.outputs.output2 = self.outputs.create('output_2')

    def _create_unit(self, creator: TensorCreator) -> Unit:
        pass

    def _fill_memory_blocks(self):
        self.outputs.output1.tensor = torch.tensor([1])
        self.outputs.output2.tensor = torch.tensor([1])

    def _step(self):
        pass


def _create_graph(n_nodes, connections, low_priority=None):
    nodes = [NodeStub() for _ in range(n_nodes)]

    if low_priority is None:
        for ((from_node, from_output), (to_node, to_input)) in connections:
            Connector.connect(nodes[from_node].outputs[from_output], nodes[to_node].inputs[to_input])
    else:
        for (((from_node, from_output), (to_node, to_input)), priority) in zip(connections, low_priority):
            Connector.connect(nodes[from_node].outputs[from_output], nodes[to_node].inputs[to_input], is_backward=priority)

    return nodes


def test_node_ordering_linear_graph():
    nodes = _create_graph(n_nodes=3, connections=[((2, 0), (1, 0)),
                                                  ((1, 0), (0, 0))])

    ordered_nodes = order_nodes(nodes)

    expected_ordered_nodes = [nodes[2], nodes[1], nodes[0]]

    assert expected_ordered_nodes == ordered_nodes
    assert [1, 2, 3] == [node.topological_order for node in ordered_nodes]


def test_node_ordering_fork_graph():
    nodes = _create_graph(n_nodes=3, connections=[((2, 0), (0, 0)),
                                                  ((2, 0), (1, 0))])

    ordered_nodes = order_nodes(nodes)

    assert nodes[2] == ordered_nodes[0]


def test_node_ordering_join_graph():
    nodes = _create_graph(n_nodes=3, connections=[((0, 0), (2, 0)),
                                                  ((1, 0), (2, 1))])

    ordered_nodes = order_nodes(nodes)

    assert nodes[2] == ordered_nodes[-1]


def test_node_ordering_low_priority_mini():
    nodes = _create_graph(n_nodes=3, connections=[((0, 0), (1, 0)),
                                                  ((1, 0), (2, 0)),
                                                  ((2, 0), (1, 1))], low_priority=[False, False, True])

    ordered_nodes = order_nodes(nodes)

    assert nodes[0] == ordered_nodes[0]
    assert nodes[1] == ordered_nodes[1]
    assert nodes[2] == ordered_nodes[2]


def test_node_ordering_low_priority_cycle():
    nodes = _create_graph(n_nodes=3, connections=[((0, 0), (1, 0)),
                                                  ((1, 0), (2, 0)),
                                                  ((2, 0), (0, 0))], low_priority=[False, False, True])

    ordered_nodes = order_nodes(nodes)

    assert nodes[0] == ordered_nodes[0]
    assert nodes[1] == ordered_nodes[1]
    assert nodes[2] == ordered_nodes[2]


def test_node_ordering_full_cycle():
    nodes = _create_graph(n_nodes=3, connections=[((0, 0), (1, 0)),
                                                  ((1, 0), (2, 0)),
                                                  ((2, 0), (0, 0))])

    with pytest.raises(IllegalCycleException):
        order_nodes(nodes)


def tricky_bad_cycle():
    """Test a cyclic topology.

    A topology of  *-l-> * ----> *
                          ^    /
                           \ v
                            *
    Which would beat the cycle checking algorithm if it was run once starting with the leftmost node.
    """
    nodes = _create_graph(n_nodes=4, connections=[((0, 0), (1, 0)),
                                                  ((1, 0), (2, 0)),
                                                  ((2, 0), (3, 0)),
                                                  ((3, 0), (1, 1))], low_priority=[True, False, False, False])

    with pytest.raises(IllegalCycleException):
        order_nodes(nodes)


def test_node_ordering_group():
    """This test checks that the node ordering doesn't care about NodeGroups."""
    node1 = NodeStub()
    node_group = GenericNodeGroup('group', 1, 1)
    inner_node = NodeStub()
    node2 = NodeStub()

    Connector.connect(node1.outputs[0], node_group.inputs[0])

    Connector.connect(node_group.inputs[0].output, inner_node.inputs[0])
    Connector.connect(inner_node.outputs[0], node_group.outputs[0].input)

    Connector.connect(node_group.outputs[0], node2.inputs[0])

    Connector.connect(node1.outputs[1], node2.inputs[1])

    nodes = [node1, node_group, node2]

    ordered_nodes = order_nodes(nodes)
    assert [node1, node_group, node2] == ordered_nodes
