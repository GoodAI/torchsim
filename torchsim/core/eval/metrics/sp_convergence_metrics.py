import torch

from torchsim.core.utils.tensor_utils import average_abs_values_to_float


def average_sp_delta(cluster_center_deltas: torch.Tensor) -> float:
    """Should converge towards 0.

    Args:
        cluster_center_deltas: [flock_size, n_cluster_centers, input_size]

    Returns:
        Average delta in one dimension (sum of deltas divided by flock_size, n_cluster_centers, input_size).
    """
    return average_abs_values_to_float(cluster_center_deltas)


def average_boosting_duration(cluster_boosting_durations: torch.Tensor) -> float:
    """Should be 0 in case nothing is boosted.

    Args:
        cluster_boosting_durations: [flock_size, n_cluster_centers]: how many steps is each cluster boosted

    Returns:
        Sum of cluster_boosting_durations divided by total num of cluster centers in the flock.
    """
    return average_abs_values_to_float(cluster_boosting_durations)


def num_boosted_clusters(cluster_boosting_durations: torch.Tensor) -> float:
    """Returns number of boosted clusters (boosting_duration > 0) divided by total num. of clusters in the flock.
    Args:
        cluster_boosting_durations: tensor in the flock
    Returns: value between 0 (no boosting) and 1 (all clusters boosted)
    """

    nonzeros = float(cluster_boosting_durations.nonzero().size(0))
    return nonzeros / cluster_boosting_durations.numel()
