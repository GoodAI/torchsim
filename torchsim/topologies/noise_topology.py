from torchsim.core.nodes.dataset_mnist_node import DatasetMNISTParams, DatasetMNISTNode
from torchsim.core.nodes.random_noise_node import RandomNoiseNode, RandomNoiseParams
from torchsim.core.graph import Topology
from torchsim.core.graph.connection import Connector


class RandomNoiseTopology(Topology):
    _mnist_params: DatasetMNISTParams = DatasetMNISTParams()
    _noise_params: RandomNoiseParams = RandomNoiseParams()

    def __init__(self):
        super().__init__(device='cuda')

        mnist_node = DatasetMNISTNode(params=self._mnist_params)
        self.add_node(mnist_node)

        noise_node = RandomNoiseNode()
        self.add_node(noise_node)

        Connector.connect(mnist_node.outputs.data, noise_node.inputs.input)


class RandomNoiseOnlyTopology(Topology):
    _noise_params: RandomNoiseParams = RandomNoiseParams((32, 32), distribution='Normal', amplitude=.3)

    def __init__(self):
        super().__init__(device='cpu')

        noise_node = RandomNoiseNode(params=self._noise_params)
        self.add_node(noise_node)
