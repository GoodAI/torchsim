import torch


def cluster_agreement(cluster_ids_1: torch.Tensor, cluster_ids_2: torch.Tensor) -> float:
    """Given two series of cluster assignments, returns the fraction of identical assignments.

    This is a measure of representation stability. A series of data points get assigned to a corresponding series
    of clusters. We then allow the expert to continue learning. Then we classify the same points again and compute
    the fraction of points that get assigned to the same clusters.
    """
    assert cluster_ids_1.dtype == torch.long
    assert cluster_ids_2.dtype == torch.long
    assert cluster_ids_1.numel() == cluster_ids_2.numel()
    assert cluster_ids_1.dim() == cluster_ids_2.dim() == 1

    series_length = cluster_ids_1.numel()
    n_same_cluster = cluster_ids_1.eq(cluster_ids_2).sum().item()
    return n_same_cluster / series_length

