from torchsim.core.nodes.random_number_node import RandomNumberNode
from torchsim.core.graph import Topology


class RandomNumberTopology(Topology):

    def __init__(self):
        super().__init__(device='cuda')

        self.random_number_node = RandomNumberNode(upper_bound=10, seed=None)
        self.add_node(self.random_number_node)

