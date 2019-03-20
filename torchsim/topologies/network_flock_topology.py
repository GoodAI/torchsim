from torchsim.core.graph import Topology
from torchsim.core.graph.connection import Connector
from torchsim.core.nodes import RandomNoiseParams, RandomNoiseNode
from torchsim.core.nodes.flock_networks.neural_network_flock import NeuralNetworkFlockParams
from torchsim.core.nodes.network_flock_node import NetworkFlockNode, NetworkFlockNodeParams
from torchsim.core.nodes.simple_bouncing_ball_node import SimpleBouncingBallNodeParams, SimpleBouncingBallNode


class NetworkFlockTopology(Topology):
    """Contains the NetworkFlockNode"""

    def __init__(self):
        super().__init__("cuda")

        # world
        params = SimpleBouncingBallNodeParams()
        params.sx = 20
        params.sy = 30
        params.ball_radius = 5

        self.world = SimpleBouncingBallNode(params=params)
        self.add_node(self.world)

        flock_size = 3

        # network
        network_params = NeuralNetworkFlockParams()
        network_params.learning_rate = 0.001
        network_params.mini_batch_size = 50

        #node
        node_params = NetworkFlockNodeParams()
        node_params.flock_size = flock_size
        node_params.batch_size = 190
        node_params.buffer_size = 200

        self.network_flock_node = NetworkFlockNode(node_params=node_params, network_params=network_params)
        self.add_node(self.network_flock_node)

        # weights
        weights_params = RandomNoiseParams()
        weights_params.shape = (flock_size, )

        self.noise_node = RandomNoiseNode(weights_params)
        self.add_node(self.noise_node)

        Connector.connect(self.world.outputs.bitmap, self.network_flock_node.inputs.input_data)
        Connector.connect(self.world.outputs.bitmap, self.network_flock_node.inputs.target_data)
        Connector.connect(self.noise_node.outputs.output, self.network_flock_node.inputs.learning_coefficients)



