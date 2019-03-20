import torch

from torchsim.core.nodes.spatial_pooler_node import SpatialPoolerFlockNode


class SpatialPoolerFlockNodeAccessor:
    """Adaptor for the SpatialPoolerFlockNode allowing access to the basic measurable values."""

    @staticmethod
    def get_output_id(node: SpatialPoolerFlockNode) -> int:
        """Get argmax of the output of the spatial pooler.

        Args:
            node: target node

        Returns:
            Scalar from the range <0, sp_size).
        """
        assert node.params.flock_size == 1
        max_id = torch.argmax(node.outputs.sp.forward_clusters.tensor)
        return max_id.to('cpu').data.item()

    @staticmethod
    def get_output_tensor(node: SpatialPoolerFlockNode) -> torch.Tensor:
        """
        Args:
            node:
        Returns: tensor containing the output of the SP
        """
        return node.outputs.sp.forward_clusters.tensor

    @staticmethod
    def get_reconstruction(node: SpatialPoolerFlockNode) -> torch.Tensor:
        return node.outputs.sp.current_reconstructed_input.tensor

    @staticmethod
    def get_sp_deltas(node: SpatialPoolerFlockNode) -> torch.Tensor:
        return node.memory_blocks.sp.cluster_center_deltas.tensor

    @staticmethod
    def get_sp_boosting_durations(node: SpatialPoolerFlockNode) -> torch.Tensor:
        return node.memory_blocks.sp.cluster_boosting_durations.tensor
