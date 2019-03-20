from torchsim.core.nodes import SimpleBouncingBallNode
from torchsim.core.graph import Topology
from torchsim.core.nodes.simple_bouncing_ball_node import SimpleBouncingBallNodeParams


class BouncingBallTopology(Topology):

    def __init__(self):
        super().__init__(device='cpu')

        # Just the Bouncing ball node
        params = SimpleBouncingBallNodeParams()
        self.add_node(SimpleBouncingBallNode(params=params))

