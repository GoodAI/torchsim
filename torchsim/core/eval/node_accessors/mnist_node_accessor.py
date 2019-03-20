import torch

from torchsim.core.nodes.dataset_mnist_node import DatasetMNISTNode


class MnistNodeAccessor:
    """Adaptor for the MNIST Node allowing access to the basic measurable values."""

    @staticmethod
    def get_data(node: DatasetMNISTNode) -> torch.Tensor:
        return node.outputs.data.tensor.clone()

    @staticmethod
    def get_label_id(node: DatasetMNISTNode) -> int:
        """Get the label id of the current bitmap."""
        return node._unit.label_tensor.to('cpu').item()
