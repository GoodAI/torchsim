from torchsim.core.nodes.dataset_mnist_node import DatasetMNISTParams, DatasetMNISTNode
from torchsim.core.graph import Topology


class MnistTopology(Topology):

    _params: DatasetMNISTParams = DatasetMNISTParams()

    def __init__(self):
        super().__init__(device='cpu')

        node = DatasetMNISTNode(params=self._params)
        self.add_node(node)
