import torch

import pytest

from torchsim.core import get_float
from torchsim.core.models.temporal_pooler.tp_output_projection import TPOutputProjection
from torchsim.core.utils.tensor_utils import same


class TestTPOutputProjection:
    @pytest.mark.parametrize('seq_length, seq_lookahead, expected_result', [
        (3, 1, [1, 2, 1]),
        (4, 0, [1, 2, 3, 4]),
        (4, 1, [1, 2, 3, 2]),
        (4, 2, [2, 3, 2, 1]),
        (4, 3, [4, 3, 2, 1]),
        (5, 0, [1, 2, 3, 4, 5]),
        (5, 1, [1, 2, 3, 4, 3]),
        (5, 2, [1, 2, 3, 2, 1]),
        (5, 3, [3, 4, 3, 2, 1]),
        (5, 4, [5, 4, 3, 2, 1])
    ])
    def test_generate_prob_scaling(self, seq_length, seq_lookahead, expected_result):
        scaling = TPOutputProjection._generate_prob_scaling(seq_length, seq_lookahead)
        assert expected_result == scaling

    @pytest.mark.parametrize('device', ['cpu'])
    @pytest.mark.parametrize('seqs, likelihoods, expected_output_projection', [
        (
                [
                    [[0, 1, 2],
                     [0, 1, 3]],
                    [[0, 1, 2],
                     [1, 3, 2]],
                ],
                [[1.0, 0.0], [0.5, 0.5]],
                [[0.25, 0.5, 0.25, 0.0], [0.125, 0.375, 0.250, 0.250]],
        ),
        (
                [
                    [[0, 1, 0]],
                    [[0, 1, 2]],
                    [[0, 1, 3]],
                    [[0, 2, 0]],
                    [[0, 2, 1]],
                ], [[1.], [1.], [1.], [1.], [1.]],
                [
                    [0.5, 0.5, 0, 0],
                    [0.25, 0.5, 0.25, 0.0],
                    [0.25, 0.5, 0.0, 0.25],
                    [0.5, 0, 0.5, 0],
                    [0.25, 0.25, 0.5, 0.0],
                ]
        )
    ])
    def test_compute_output_projection(self, device, seqs, likelihoods, expected_output_projection):
        seqs_tensor = torch.tensor(seqs, device=device, dtype=torch.int64)
        likelihoods_tensor = torch.tensor(likelihoods, device=device, dtype=get_float(device))
        expected_output_projection_tensor = torch.tensor(expected_output_projection, device=device,
                                                         dtype=get_float(device))
        flock_size, n_frequent_seqs, seq_length = seqs_tensor.shape
        n_cluster_centers = 4
        seq_lookahead = 1

        output_projection = TPOutputProjection(flock_size, n_frequent_seqs, n_cluster_centers, seq_length,
                                               seq_lookahead, device)

        projection_outputs = torch.zeros((flock_size, n_cluster_centers), device=device, dtype=get_float(device))

        output_projection.compute_output_projection(
            seqs_tensor,
            likelihoods_tensor,
            projection_outputs
        )

        assert same(expected_output_projection_tensor, projection_outputs)

    @pytest.mark.parametrize('device', ['cpu'])
    @pytest.mark.parametrize('seqs, expected_output_projection', [
        (
                [[
                    [0, 1, 0],
                    [0, 1, 2],
                    [0, 1, 3],
                    [0, 2, 0],
                    [0, 2, 1],
                ]],
                [[
                    [0.5, 0.5, 0, 0],
                    [0.25, 0.5, 0.25, 0.0],
                    [0.25, 0.5, 0.0, 0.25],
                    [0.5, 0, 0.5, 0],
                    [0.25, 0.25, 0.5, 0.0],
                ]]
        )
    ])
    def test_compute_output_projection_per_sequence(self, device, seqs, expected_output_projection):
        seqs_tensor = torch.tensor(seqs, device=device, dtype=torch.int64)
        expected_output_projection_tensor = torch.tensor(expected_output_projection, device=device,
                                                         dtype=get_float(device))
        flock_size, n_frequent_seqs, seq_length = seqs_tensor.shape
        n_cluster_centers = 4
        seq_lookahead = 1

        output_projection = TPOutputProjection(flock_size, n_frequent_seqs, n_cluster_centers, seq_length,
                                               seq_lookahead, device)

        projection_outputs = torch.zeros((flock_size, n_frequent_seqs, n_cluster_centers), device=device,
                                         dtype=get_float(device))

        output_projection.compute_output_projection_per_sequence(
            seqs_tensor,
            projection_outputs
        )

        assert same(expected_output_projection_tensor, projection_outputs)

    @pytest.mark.parametrize('device', ['cpu'])
    @pytest.mark.parametrize('item, sequences, expected_result', [
        ([[0.25, 0.5, 0.25, 0.0]],
         [[
             [0.5, 0.5, 0, 0],
             [0.25, 0.5, 0.25, 0.0],
             [0.25, 0.5, 0.0, 0.25],
             [0.5, 0, 0.5, 0],
             [0.25, 0.25, 0.5, 0.0]
         ]],
         [[1 - 0.125, 1, 1 - 0.125, 1 - 0.25, 1 - 0.125]]
         ),
        ([
             [0.0],
             [1.0]
         ],
         [
             [
                 [0.1],
                 [0.2],
                 [0.3],
             ],
             [
                 [0.4],
                 [0.5],
                 [0.6],
             ],
         ],
         [
             [0.9, 0.8, 0.7],
             [0.4, 0.5, 0.6],
         ]
        )
    ])
    def test_compute_similarity(self, device, item, sequences, expected_result):
        def t(data):
            return torch.tensor(data, device=device, dtype=get_float(device))

        result = TPOutputProjection.compute_similarity(t(item), t(sequences))
        assert same(t(expected_result), result, eps=1e-3)
