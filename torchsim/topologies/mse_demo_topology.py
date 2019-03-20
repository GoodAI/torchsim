from torchsim.core.graph.connection import Connector
from torchsim.core.nodes.random_number_node import RandomNumberNode
from torchsim.core.graph import Topology
from torchsim.core.nodes.mse_node import MseNode


class MseDemoTopology(Topology):

    def __init__(self):
        super().__init__(device='cuda')

        bound = 30
        buffer_size = 100

        random_number_node_a = RandomNumberNode(upper_bound=bound, seed=None)
        random_number_node_b = RandomNumberNode(upper_bound=bound, seed=None)

        mse_node = MseNode(buffer_size)

        self.add_node(random_number_node_a)
        self.add_node(random_number_node_b)
        self.add_node(mse_node)

        Connector.connect(
            random_number_node_a.outputs.one_hot_output,
            mse_node.inputs.input_a
        )

        Connector.connect(
            random_number_node_b.outputs.one_hot_output,
            mse_node.inputs.input_b
        )
