import pytest
import torch

from torchsim.core.eval.metrics.cluster_agreement import cluster_agreement


def test_cluster_agreement():
    cluster_ids_1 = torch.tensor([0, 0, 1, 1, 0, 3, 5, 3, 2, 9])
    cluster_ids_2 = torch.tensor([0, 23, 33, 1, 3, 3, 0, 3, 9, 0])
    assert cluster_agreement(cluster_ids_1, cluster_ids_2) == .4
