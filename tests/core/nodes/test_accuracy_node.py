import torch

import pytest

from torchsim.core import get_float, SMALL_CONSTANT
from torchsim.core.nodes.accuracy_node import AccuracyNode, AccuracyUnit
from torchsim.core.utils.tensor_utils import same


class TestAccuracyNode:

    @pytest.mark.parametrize('input_a, input_b, expected_result', [
        (
                [[0, 1], [0, 0]],
                [[0, 1], [0, 0]],
                [1, 1]
        ),
        (
                [[0, 0], [0, 0]],
                [[0, 1], [0, 0]],
                [0, 1]
        ),
        (
                [[0, 0], [1, 2]],
                [[0, 1], [0, 0]],
                [0, 0]
        ),
        (
                [[0, 0], [1, 1], [3, 3], [4, 4]],
                [[5, 5], [1, 1], [3, 3], [4, 4]],
                [0, 1, 1, 1]
        ),
        (
                [[[0, 0], [1, 1]], [[2, 2], [3, 3]]],
                [[[0, 0], [1, 1]], [[1, 2], [3, 3]]],
                [1, 0]
        ),
    ])
    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    def test_compute_accuracy(self, device, input_a, input_b, expected_result):
        float_type = get_float(device)
        t_a = torch.tensor(input_a, dtype=float_type, device=device)
        t_b = torch.tensor(input_b, dtype=float_type, device=device)
        t_expected = torch.tensor(expected_result, dtype=torch.uint8, device=device)
        # assert expected_result - AccuracyUnit._compute_accuracy(t_a, t_b) < SMALL_CONSTANT
        result = AccuracyUnit._compute_accuracy(t_a, t_b)
        assert same(t_expected, result)

    @pytest.mark.parametrize('buffer, expected_result', [
        (
                [
                    [0, 1],
                    [0, 0]
                ],
                [0, 0.5]
        ),
        (
                [
                    [0, 1, 1],
                    [0, 1, 0],
                    [0, 1, 1],
                    [0, 0, 0]
                ],
                [0, 0.75, 0.5]
        ),
    ])
    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    def test_average_buffer(self, device, buffer, expected_result):
        float_type = get_float(device)
        t_buffer = torch.tensor(buffer, dtype=torch.uint8, device=device)
        t_expected = torch.tensor(expected_result, dtype=float_type, device=device)
        result = AccuracyUnit._average_buffer(t_buffer)
        assert same(t_expected, result, eps=SMALL_CONSTANT)
