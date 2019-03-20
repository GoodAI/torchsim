# import torch
# from torchsim.core import FLOAT_TYPE_CPU
#
# from torchsim.core.nodes.dataset_mnist_node import DatasetMNISTNode
#

# Not used anywhere and a lot of noise, commented out.

# class SeNodeAccessor:
#     """Accessor for the SE Node allowing access to the basic measurable values."""
#     @staticmethod
#     def get_data(node: DatasetMNISTNode) -> torch.Tensor:
#         return node.get_data().to('cpu').type(FLOAT_TYPE_CPU)
#
#     # for the beginning, we can just convert XY coords to unique int and use this one (I'll just improve this tommorrow)
#     @staticmethod
#     def get_label_id(node: DatasetMNISTNode) -> int:
#         """Get the label id of the current bitmap."""
#         return node.get_current_label_id()
#
#     @staticmethod
#     def reset(node: DatasetMNISTNode, seed: int):
#         print('reset is called on the SE!')
#         node.reset(seed)