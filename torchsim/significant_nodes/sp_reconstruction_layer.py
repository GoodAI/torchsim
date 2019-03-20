from typing import Optional

from torchsim.core.graph.connection import Connector
from torchsim.core.graph.node_group import NodeGroupBase
from torchsim.core.models.expert_params import ExpertParams
from torchsim.core.nodes import UnsqueezeNode, JoinNode, SpatialPoolerFlockNode, ForkNode
from torchsim.significant_nodes.reconstruction_interface import ClassificationInputs, ClassificationOutputs


class SpReconstructionLayer(NodeGroupBase[ClassificationInputs, ClassificationOutputs]):

    sp_node: SpatialPoolerFlockNode

    def __init__(self,
                 input_data_size: int,
                 labels_size: int,
                 sp_params: Optional[ExpertParams]=None,
                 name: str = "",
                 seed: Optional[int]=None):
        super().__init__("SpReconstructionLayer", inputs=ClassificationInputs(self),
                         outputs=ClassificationOutputs(self))

        join_node = JoinNode(n_inputs=2, flatten=True, name=name + " Join")
        self.add_node(join_node)
        self.join_node = join_node
        Connector.connect(self.inputs.data.output, join_node.inputs[0])
        Connector.connect(self.inputs.label.output, join_node.inputs[1])

        unsqueeze_node = UnsqueezeNode(0)
        self.add_node(unsqueeze_node)
        Connector.connect(join_node.outputs.output, unsqueeze_node.inputs.input)

        if sp_params is None:
            sp_params = ExpertParams()

        sp_node = SpatialPoolerFlockNode(sp_params, name=name + " SP Expert", seed=seed)
        self.add_node(sp_node)
        self.sp_node = sp_node
        Connector.connect(unsqueeze_node.outputs.output, sp_node.inputs.sp.data_input)

        fork_node = ForkNode(1, [input_data_size, labels_size], name=name + " Fork")
        self.add_node(fork_node)
        self.fork_node = fork_node
        Connector.connect(sp_node.outputs.sp.current_reconstructed_input, fork_node.inputs.input)

        Connector.connect(fork_node.outputs[1], self.outputs.label.input)

    def switch_learning(self, on):
        self.sp_node.switch_learning(on)


