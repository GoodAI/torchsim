import torch

from torchsim.core.nodes import SpatialPoolerFlockNode
from torchsim.core.nodes.expert_node import ExpertFlockNode


class FlockNodeAccessor:

    # TODO unit tests

    @staticmethod
    def get_sp_output_tensor(node: ExpertFlockNode) -> torch.Tensor:
        """Return output of the SP for entire flock. Tensor should be cloned somewhere so that it can be used later."""
        return node.memory_blocks.sp.forward_clusters.tensor

    @staticmethod
    def get_sp_output_id(node: ExpertFlockNode) -> int:
        assert node.params.flock_size == 1

        if isinstance(node, SpatialPoolerFlockNode):
            tensor = node.outputs.sp.forward_clusters.tensor
        else:
            tensor = node.memory_blocks.sp.forward_clusters.tensor

        max_id = torch.argmax(tensor)

        return max_id.to('cpu').data.item()

    @staticmethod
    def get_sp_deltas(node: ExpertFlockNode) -> torch.Tensor:
        return node.memory_blocks.sp.cluster_center_deltas.tensor

    @staticmethod
    def get_sp_boosting_durations(node: ExpertFlockNode) -> torch.Tensor:
        return node.memory_blocks.sp.cluster_boosting_durations.tensor

    @staticmethod
    def get_sp_output_size(node: ExpertFlockNode) -> int:
        """Return number of elements in the SP output - for entire flock together!"""
        return node.params.flock_size * node.params.n_cluster_centers
        # return node.memory_blocks.sp.forward_clusters.numel()

