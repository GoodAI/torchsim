import torch

from torchsim.core.nodes.random_number_node import RandomNumberNode


class RandomNumberNodeAccessor:

    @staticmethod
    def get_output_id(node: RandomNumberNode) -> int:
        return node._unit._current_value.item()

    @staticmethod
    def get_output_tensor(node: RandomNumberNode) -> torch.Tensor:
        return node.outputs.one_hot_output.tensor
