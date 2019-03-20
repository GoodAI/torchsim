from typing import List

from torchsim.core.graph.connection import Connector
from torchsim.core.graph.inverse_pass_packet import InversePassOutputPacket, InversePassInputPacket
from torchsim.core.graph.slot_container import Inputs
from torchsim.core.graph.unit import InvertibleUnit
from torchsim.core.graph.worker_node_base import InvertibleWorkerNodeBase
from torchsim.core.memory.tensor_creator import TensorCreator
from torchsim.utils.node_utils import TestMemoryBlocks


class InfiniteLoopException(Exception):
    pass


class DummyNode(InvertibleWorkerNodeBase[Inputs, TestMemoryBlocks]):

    def __init__(self):
        super().__init__(inputs=Inputs(self), outputs=TestMemoryBlocks(self))

        self.inputs.input1 = self.inputs.create('input 1')
        self.outputs.output1 = self.outputs.create('output 1')

        self.reentry = False

    def _inverse_projection(self, data: InversePassOutputPacket) -> List[InversePassInputPacket]:
        if self.reentry:
            raise InfiniteLoopException
        self.reentry = True

        return [InversePassInputPacket(data, self.inputs.input1)]

    def _step(self):
        pass

    def _create_unit(self, creator: TensorCreator) -> InvertibleUnit:
        pass


def _create_graph(n_nodes, connections, low_priority=None):
    nodes = [DummyNode() for _ in range(n_nodes)]

    if low_priority is None:
        for ((from_node, from_output), (to_node, to_input)) in connections:
            Connector.connect(nodes[from_node].outputs[from_output], nodes[to_node].inputs[to_input])
    else:
        for (((from_node, from_output), (to_node, to_input)), priority) in zip(connections, low_priority):
            Connector.connect(nodes[from_node].outputs[from_output], nodes[to_node].inputs[to_input],
                              is_backward=priority)

    return nodes


def test_inversion_cycles():
    graph = _create_graph(3, connections=[((0, 0), (1, 0)),
                                          ((1, 0), (2, 0)),
                                          ((2, 0), (0, 0))], low_priority=[False, False, True])

    graph[2].recursive_inverse_projection_from_output(InversePassOutputPacket(None, graph[2].outputs.output1))
