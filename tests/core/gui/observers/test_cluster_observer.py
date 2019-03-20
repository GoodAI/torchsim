import pytest

import torch
from abc import ABC

from torchsim.gui.observers.cluster_observer import ClusterUtils, PcaTransformer
from torchsim.core.utils.tensor_utils import same


class TestClusterUtils(ABC):

    def test_cluster_centers_similarities(self):
        cluster_centers_count = 4
        sequences = torch.tensor([
            [0, 1, 0],  # 5
            [0, 1, 2],  # 2
            [0, 1, 3],  # 1
            [0, 2, 1],  # 1
        ])
        seq_occurrences = torch.tensor([5, 2, 1, 1], dtype=torch.float)
        expected = torch.tensor([
            [0, 8, 1, 0],
            [5, 0, 2, 1],
            [0, 1, 0, 0],
            [0, 0, 0, 0]
        ], dtype=torch.float) / 8

        result = ClusterUtils.compute_similarities(cluster_centers_count, sequences, seq_occurrences)
        assert same(expected, result, eps=1e-4)

    def test_cluster_centers_similarities_single_cluster(self):
        cluster_centers_count = 1
        sequences = torch.tensor([[0, 0, 0, 0, 0]], dtype=torch.long)
        seq_occurrences = torch.tensor([1.5], dtype=torch.float)
        expected = torch.tensor([
            [1],
        ], dtype=torch.float)

        result = ClusterUtils.compute_similarities(cluster_centers_count, sequences, seq_occurrences)
        assert same(expected, result, eps=1e-4)

    def test_cluster_centers_similarities_empty(self):
        cluster_centers_count = 2
        sequences = torch.tensor([
            [-1, -1, -1],  # 0
        ])
        seq_occurrences = torch.tensor([0], dtype=torch.float)
        expected = torch.tensor([
            [0, 0],
            [0, 0],
        ], dtype=torch.float)

        result = ClusterUtils.compute_similarities(cluster_centers_count, sequences, seq_occurrences)
        assert same(expected, result, eps=1e-4)

    def test_cluster_centers_similarities_orderless(self):
        sequences = torch.tensor([
            [0, 1, 0],  # 5
            [0, 1, 2],  # 2
            [0, 1, 3],  # 1
            [0, 2, 1],  # 1
        ])
        seq_occurrences = torch.tensor([5, 2, 1, 1], dtype=torch.float)
        expected = torch.tensor([
            [9, 9, 3, 1],
            [9, 9, 3, 1],
            [3, 3, 3, 0],
            [1, 1, 0, 1]
        ], dtype=torch.float) / 9

        result = ClusterUtils.compute_similarities_orderless(4, sequences, seq_occurrences)
        assert same(expected, result, eps=0.001)

    def test_cluster_centers_similarities_orderless_single_cluster(self):
        cluster_centers_count = 1
        sequences = torch.tensor([[0, 0, 0, 0, 0]], dtype=torch.long)
        seq_occurrences = torch.tensor([1.5], dtype=torch.float)
        expected = torch.tensor([
            [1],
        ], dtype=torch.float)

        result = ClusterUtils.compute_similarities(cluster_centers_count, sequences, seq_occurrences)
        assert same(expected, result, eps=1e-4)

    def test_cluster_centers_similarities_orderless_empty(self):
        sequences = torch.tensor([
            [-1, -1, -1],  # 0
        ])
        seq_occurrences = torch.tensor([0], dtype=torch.float)
        expected = torch.tensor([
            [0, 0],
            [0, 0],
        ], dtype=torch.float)

        result = ClusterUtils.compute_similarities_orderless(2, sequences, seq_occurrences)
        assert same(expected, result, eps=0.001)


class TestPcaTransformer:
    def pca_transformer(self, data_dims: int):
        transformer = PcaTransformer()
        transformer._means = torch.zeros([data_dims]).float()
        transformer._std_devs = torch.ones([data_dims]).float()
        return transformer

    @pytest.mark.parametrize('data, n_dims, expected_result', [
        ([[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]], 3, [[1, 2, 3], [10, 20, 30]]),
        ([[1, 2, 3], [4, 5, 6]], 3, [[1, 2, 3], [4, 5, 6]]),
        ([[1, 2], [10, 20]], 3, [[1, 2, 0], [10, 20, 0]]),
        ([[1], [10]], 3, [[1, 0, 0], [10, 0, 0]]),
    ])
    def test_project(self, data, n_dims, expected_result):
        data_tensor = torch.tensor(data).float()
        data_dim = data_tensor.shape[-1]
        transformer = self.pca_transformer(data_dim)
        projection_matrix = torch.zeros((data_dim, min(data_dim, 3)))
        for i in range(min(data_dim, 3)):
            projection_matrix[i, i] = 1

        transformer._projection_matrix = projection_matrix.float()
        result = transformer.project(data_tensor, n_dims)
        assert same(torch.Tensor(expected_result).float(), result)
