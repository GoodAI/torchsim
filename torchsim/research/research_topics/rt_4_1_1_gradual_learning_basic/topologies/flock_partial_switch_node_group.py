import logging
from dataclasses import dataclass

from torchsim.core.graph.connection import Connector
from torchsim.core.graph.node_group import NodeGroupBase, GroupInputs, GroupOutputs
from torchsim.core.nodes import UnsqueezeNode, ScatterNode, ForkNode, SwitchNode, JoinNode

logger = logging.getLogger(__name__)


@dataclass
class FlockPartialSwitchNodeGroupParams:
    flock_size: int
    split_idx: int  # number of items in the first part


class FlockPartialSwitchInputs(GroupInputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.input_1 = self.create("Input 1")
        self.input_2 = self.create("Input 2")


class FlockPartialSwitchOutputs(GroupOutputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.output = self.create("Output")


class FlockPartialSwitchNodeGroup(NodeGroupBase[FlockPartialSwitchInputs, FlockPartialSwitchOutputs]):
    def __init__(self,
                 params: FlockPartialSwitchNodeGroupParams,
                 name: str = "FlockPartialSwitchNodeGroup"):
        super().__init__(name, inputs=FlockPartialSwitchInputs(self), outputs=FlockPartialSwitchOutputs(self))
        dim = 0
        n_fork_1 = ForkNode(dim, [params.split_idx, params.flock_size - params.split_idx])
        n_fork_2 = ForkNode(dim, [params.split_idx, params.flock_size - params.split_idx])
        n_switch = SwitchNode(n_inputs=2)
        n_join = JoinNode(dim, n_inputs=2)
        self.add_node(n_fork_1)
        self.add_node(n_fork_2)
        self.add_node(n_switch)
        self.add_node(n_join)

        Connector.connect(self.inputs.input_1.output, n_fork_1.inputs.input)
        Connector.connect(self.inputs.input_2.output, n_fork_2.inputs.input)
        Connector.connect(n_fork_1.outputs[0], n_switch.inputs[0])
        Connector.connect(n_fork_2.outputs[0], n_switch.inputs[1])
        Connector.connect(n_switch.outputs.output, n_join.inputs[0])
        Connector.connect(n_fork_2.outputs[1], n_join.inputs[1])
        Connector.connect(n_join.outputs.output, self.outputs.output.input)

        self._n_switch = n_switch

    def set_active_input_index(self, dataset_id):
        self._n_switch.active_input_index = dataset_id
