import torch
import logging

from typing import List

from torchsim.core import FLOAT_NAN
from torchsim.core.graph.hierarchical_observable_node import HierarchicalObservableNode
from torchsim.core.graph.inverse_pass_packet import InversePassInputPacket
from torchsim.core.memory.tensor_creator import TensorSurrogate

logger = logging.getLogger(__name__)


def get_inverse_projections_for_all_clusters(node: HierarchicalObservableNode, expert_no: int) -> List[List[torch.Tensor]]:
    torch.cuda.synchronize()
    _invalid_state_tensor = [[torch.full((1, 1), float('nan'))]]

    # First get the inverse projection.
    projected_values = node.projected_values

    if projected_values is None:
        return _invalid_state_tensor

    if type(projected_values) is TensorSurrogate:
        return _invalid_state_tensor

    all_projections = []
    for i in range(projected_values.shape[1]):
        projected_values_of_expert = torch.zeros_like(projected_values[:, i])
        projected_values_of_expert[expert_no].copy_(projected_values[expert_no, i])

        packet = InversePassInputPacket(projected_values_of_expert, node.projection_input)
        projections = node.recursive_inverse_projection_from_input(packet)
        all_projections.append(projections)

    if len(all_projections) == 0:
        logger.warning('No projections were created.')
        return _invalid_state_tensor

    # The projections are (cluster_center, projection_type).
    # We need (projection_type, cluster_center).
    grouped_projections = []
    for i in range(len(all_projections[0])):
        same_input_projections = [cc_projections[i].tensor for cc_projections in all_projections]
        grouped_projections.append(same_input_projections)

    return grouped_projections

def replace_cluster_ids_with_projections(source: torch.Tensor, projections: torch.Tensor) -> torch.Tensor:
    """Add projections to a tensor.

    Args:
        source: Long tensor of arbitrary dims [*s_dims] - source data, values are treated as index to projections
                and must be less than n_of_projections. Negative values are masked out and replaced with zeros in the result.
        projections: Float tensor of dims [n_of_projections, *p_dims] - p_dims are arbitrary

    Returns:
        Tensor of dims [*s_dims, *p_dims] - projection dims are added to the end of tensor
    """
    # Determine mask of invalid values
    mask = source < 0
    safe_index = source.clone()
    safe_index[mask] = 0

    result = torch.index_select(projections, 0, safe_index.view(-1))

    result_shape = [*source.shape, *projections.shape[1:]]
    result = result.view(result_shape)
    # Clear results of invalid values
    result[mask] = FLOAT_NAN

    return result
