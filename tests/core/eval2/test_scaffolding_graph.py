import torch
from abc import ABC

from torchsim.core.eval2.scaffolding import TopologyScaffoldingFactory
from torchsim.core.graph import Topology
from torchsim.core.graph.connection import Connector
from torchsim.core.graph.node_group import GenericGroupInputs, GenericGroupOutputs, NodeGroupBase
from torchsim.core.nodes import ConstantNode
from torchsim.core.nodes.lambda_node import LambdaNode


# Define inputs, outputs, and the group ('plug') itself. This is the definition of the interface of the thing that
# needs to get plugged into the experiment.

class NodeGroupStubInputs(GenericGroupInputs['NodeGroupStubBase']):
    """Inputs of the group required by the experiment."""

    def __init__(self, owner):
        super().__init__(owner)
        self.input = self.create('Input')


class NodeGroupStubOutputs(GenericGroupOutputs['NodeGroupStubBase']):
    """Outputs of the group required by the experiment."""

    def __init__(self, owner):
        super().__init__(owner)
        self.output = self.create('Output')


class NodeGroupStubBase(NodeGroupBase[NodeGroupStubInputs, NodeGroupStubOutputs], ABC):
    """The group required by the experiment.

    This should be subclassed by individual implementations."""

    def __init__(self, name: str):
        super().__init__(name, inputs=NodeGroupStubInputs(self), outputs=NodeGroupStubOutputs(self))


# Define the specific group that gets plugged into the experiment. There will be one class like this for each thing
# that you want to test in the experiment - e.g. for every agent, and for every environment.


class NodeGroupStub(NodeGroupStubBase):
    """An implementation of the group required by the experiment.

    This just connects input to output for testing purposes.
    """

    def __init__(self, param: int):
        self.param = param
        super().__init__('stub')
        Connector.connect(self.inputs.input.output, self.outputs.output.input)


# Define the topology of the experiment. The 'scaffold' method gets called with instances of the requested
# nodes which need to be plugged in.


class ScaffoldingGraphStub(Topology):
    def __init__(self, node_group1: NodeGroupStubBase, node_group2: NodeGroupStubBase, device: str = 'cpu'):
        super().__init__(device)
        self.source = ConstantNode((2, 2), constant=42)
        self.node_group1 = node_group1
        self.node_group2 = node_group2
        self.sink = LambdaNode(lambda i, o: torch.add(input=i[0], other=i[1], out=o[0]), n_inputs=2,
                               output_shapes=[(2, 2)])

        self.add_node(self.source)
        self.add_node(self.node_group1)
        self.add_node(self.node_group2)
        self.add_node(self.sink)

        Connector.connect(self.source.outputs.output, self.node_group1.inputs.input)
        Connector.connect(self.source.outputs.output, self.node_group2.inputs.input)
        Connector.connect(self.node_group1.outputs.output, self.sink.inputs[0])
        Connector.connect(self.node_group2.outputs.output, self.sink.inputs[1])


def test_scaffolding():
    parameters = [
        {'node_group1': {'param': 42}, 'node_group2': {'param': 1337}},
        {'node_group1': {'param': 41}, 'node_group2': {'param': 1336}, 'device': 'cuda'}
    ]

    scaffolding = TopologyScaffoldingFactory(ScaffoldingGraphStub, node_group1=NodeGroupStub, node_group2=NodeGroupStub)

    for params in parameters:
        graph = scaffolding.create_topology(**params)
        # The param should not be inferred here, it's actually correct that PyCharm complains if we allow inspections.
        # It's just used for testing purposes.
        # noinspection PyUnresolvedReferences
        assert graph.node_group1.param == params['node_group1']['param']
        # noinspection PyUnresolvedReferences
        assert graph.node_group2.param == params['node_group2']['param']
