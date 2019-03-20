import torch

from torchsim.core.graph import Topology
from torchsim.core.graph.connection import Connector
from torchsim.core.nodes import DatasetMNISTParams, DatasetMNISTNode
from torchsim.core.nodes import RandomNoiseParams, RandomNoiseNode
from torchsim.core.nodes import SwitchNode


class SwitchTopology(Topology):
    """Topology for testing the SwitchNode.

    The topology connects the MNIST label (1 or 2, to pick input), MNIST digits, and noise as inputs to the
    SwitchNode. The node's output will be a MNIST '1' digit or noise depending on the input label.
    """
    _mnist_params: DatasetMNISTParams = DatasetMNISTParams(class_filter=[1, 2], one_hot_labels=False)
    _noise_params: RandomNoiseParams = RandomNoiseParams(torch.Size((28, 28)), distribution='Normal', amplitude=.3)

    def __init__(self):
        super().__init__(device='cuda')

        noise_node = RandomNoiseNode(params=self._noise_params)
        self.add_node(noise_node)

        mnist_node = DatasetMNISTNode(params=self._mnist_params)
        self.add_node(mnist_node)

        switch_node = SwitchNode(n_inputs=2, get_index_from_input=True)
        self.add_node(switch_node)

        Connector.connect(mnist_node.outputs.label, switch_node.inputs.switch_signal)
        Connector.connect(mnist_node.outputs.data, switch_node.inputs[0])
        Connector.connect(noise_node.outputs.output, switch_node.inputs[1])

