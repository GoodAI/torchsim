import torch

import pytest

from torchsim.core import FLOAT_NAN
from torchsim.core.utils.inverse_projection_utils import replace_cluster_ids_with_projections
from torchsim.core.utils.tensor_utils import same


class TestInverseProjectionUtils:
    @pytest.mark.parametrize('source, projections, expected_result', [
        ([[0, 1], [2, -1]],
         [[1, 2], [3, 4], [5, 6]],
         [
             [[1, 2], [3, 4]],
             [[5, 6], [FLOAT_NAN, FLOAT_NAN]],
          ]
         )
    ])
    @pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.slow)])
    def test_replace_cluster_ids_with_projections(self, source, projections, expected_result, device):
        t_source = torch.Tensor(source).long().to(device)
        t_projections = torch.Tensor(projections).float().to(device)
        t_expected = torch.Tensor(expected_result).to(device)
        t_result = replace_cluster_ids_with_projections(t_source, t_projections)
        assert same(t_expected, t_result)
