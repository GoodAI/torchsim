from torchsim.core.graph.connection import Connector
from torchsim.core.graph.slot_container import Inputs, MemoryBlocks
from torchsim.core.graph.unit import Unit
from torchsim.core.graph.worker_node_base import WorkerNodeBase
from torchsim.core.memory.tensor_creator import TensorCreator


class StubInputs(Inputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.input1 = self.create('input 1')


class StubMemoryBlocks(MemoryBlocks):
    def __init__(self, owner):
        super().__init__(owner)
        self.mb1 = self.create('mb 1')

    def prepare_slots(self, unit: Unit):
        pass


class StubOutputs(MemoryBlocks):
    def __init__(self, owner):
        super().__init__(owner)
        self.output1 = self.create('output 1')

    def prepare_slots(self, unit: Unit):
        pass


class NodeStub(WorkerNodeBase):
    inputs: StubInputs
    memory_blocks: StubMemoryBlocks
    outputs: StubOutputs

    def __init__(self):
        super().__init__(inputs=StubInputs(self), memory_blocks=StubMemoryBlocks(self), outputs=StubOutputs(self))

    def _create_unit(self, creator: TensorCreator) -> Unit:
        pass

    def _fill_memory_blocks(self):
        pass

    def _step(self):
        pass


def _create_graph():
    node1 = NodeStub()
    node2 = NodeStub()

    Connector.connect(node1.outputs.output1, node2.inputs.input1)

    return node1, node2


def test_connecting():
    node1, node2 = _create_graph()

    assert 'output 1' == node1.outputs.output1.name
    assert 'input 1' == node2.inputs.input1.name
    assert 'mb 1' == node2.memory_blocks.mb1.name

    assert 1 == len(node1.outputs.output1.connections)

    connection = node1.outputs.output1.connections[0]
    assert connection == node2.inputs.input1.connection

    assert connection.source == node1.outputs.output1
    assert connection.target == node2.inputs.input1


def test_disconnecting():
    node1, node2 = _create_graph()

    Connector.disconnect_input(node2.inputs.input1)

    assert node2.inputs.input1.connection is None
    assert 0 == len(node1.outputs.output1.connections)
