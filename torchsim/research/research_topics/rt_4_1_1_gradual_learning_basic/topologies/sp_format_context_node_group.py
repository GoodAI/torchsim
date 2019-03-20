import logging
from torchsim.core.graph.connection import Connector
from torchsim.core.graph.node_group import NodeGroupBase, GroupInputs, GroupOutputs
from torchsim.core.nodes import UnsqueezeNode, ScatterNode

logger = logging.getLogger(__name__)


class SPFormatContextInputs(GroupInputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.input = self.create("Input")


class SPFormatContextOutputs(GroupOutputs):
    def __init__(self, owner):
        super().__init__(owner)
        self.output = self.create("Output")


class SPFormatContextNodeGroup(NodeGroupBase[SPFormatContextInputs, SPFormatContextOutputs]):
    def __init__(self,
                 input_data_size: int,
                 flock_size: int,
                 name: str = "SPFormatContextNodeGroup",):
        super().__init__(name, inputs=SPFormatContextInputs(self), outputs=SPFormatContextOutputs(self))

        scatter_mapping = [[[[0]*input_data_size]]] * flock_size
        gw_context_unsqueeze_1 = UnsqueezeNode(1, name="Context format scatter unsqueeze 1")
        gw_context_unsqueeze_2 = UnsqueezeNode(1, name="Context format scatter unsqueeze 2")

        gw_context_node = ScatterNode(mapping=scatter_mapping, output_shape=(flock_size, 1, 3, input_data_size),
                                      dimension=2, name="Context format scatter")

        self.add_node(gw_context_unsqueeze_1)
        self.add_node(gw_context_unsqueeze_2)
        self.add_node(gw_context_node)

        Connector.connect(self.inputs.input.output, gw_context_unsqueeze_1.inputs.input)
        Connector.connect(gw_context_unsqueeze_1.outputs.output, gw_context_unsqueeze_2.inputs.input)
        Connector.connect(gw_context_unsqueeze_2.outputs.output, gw_context_node.inputs.input)
        Connector.connect(gw_context_node.outputs.output, self.outputs.output.input)
