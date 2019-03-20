from typing import Generator, List, Any

import torch
from torchsim.core.graph.connection import Connector
from torchsim.core.graph.node_base import NodeBase
from torchsim.core.graph.node_group import SimpleGroupInputs, SimpleGroupOutputs, NodeGroupWithInternalsBase
from torchsim.core.nodes import JoinNode
from torchsim.core.nodes.periodic_update_node_group import PeriodicUpdateNodeGroup
from tests.core.nodes.node_unit_test_base import NodeTestBase


class ExamplePeriodicUpdateGroup(PeriodicUpdateNodeGroup):
    inputs: SimpleGroupInputs
    outputs: SimpleGroupOutputs

    def __init__(self, name: str, update_period: int):
        super().__init__(name, update_period, inputs=SimpleGroupInputs(self, n_inputs=1),
                         outputs=SimpleGroupOutputs(self, n_outputs=1))

        join_node = JoinNode(n_inputs=1)
        self.add_node(join_node)
        Connector.connect(self.inputs[0].output, join_node.inputs[0])
        Connector.connect(join_node.outputs.output, self.outputs[0].input)
        self.order_nodes()


class TestPeriodicUpdateNodeGroup(NodeTestBase):
    UPDATE_PERIOD: int = 4
    N_STEPS: int = UPDATE_PERIOD * 3

    def _generate_input_tensors(self) -> Generator[List[torch.Tensor], None, None]:
        for step in range(self.N_STEPS):
            yield [torch.tensor([step], device='cuda')]

    def _generate_expected_results(self) -> Generator[List[Any], None, None]:
        for step in range(self.N_STEPS):
            yield [torch.tensor([self.UPDATE_PERIOD * (step // self.UPDATE_PERIOD)], device='cuda')]

    def _create_node(self) -> NodeBase:
        return ExamplePeriodicUpdateGroup("Something", self.UPDATE_PERIOD)
