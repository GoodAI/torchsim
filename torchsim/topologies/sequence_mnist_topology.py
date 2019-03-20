from torchsim.core.nodes import DatasetMNISTParams, DatasetSequenceMNISTNodeParams, DatasetSequenceMNISTNode
from torchsim.core.graph import Topology


class SequenceMnistTopology(Topology):

    _seq_params: DatasetSequenceMNISTNodeParams = DatasetSequenceMNISTNodeParams([[1, 2, 3]])
    _params: DatasetMNISTParams = DatasetMNISTParams()

    def __init__(self):
        super().__init__('cpu')

        node = DatasetSequenceMNISTNode(params=self._params, seq_params=self._seq_params)
        self.add_node(node)
