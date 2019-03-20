import torch

import pytest
from pytest import raises

from torchsim.core.exceptions import FailedValidationException
from torchsim.core.models.receptive_field.grid import Grids, Stride
from torchsim.utils.param_utils import Size2D
from torchsim.core.utils.tensor_utils import same

nan = -1


class TestGrids:
    def test_invalid_params(self):
        with raises(FailedValidationException):
            image_grid_dims = Size2D(16, 16)
            parent_rf_dims = Size2D(3, 4)
            Grids(image_grid_dims, parent_rf_dims)

    @pytest.mark.parametrize('child_dims, parent_dims, stride, expected_result', [
        (Size2D(4, 12), Size2D(2, 3), None, [(0, 0), (0, 1), (0, 2), (0, 3),
                                             (1, 0), (1, 1), (1, 2), (1, 3)]),
        (Size2D(4, 12), Size2D(2, 3), Stride(1, 3), [(0, 0), (0, 1), (0, 2), (0, 3),
                                                     (1, 0), (1, 1), (1, 2), (1, 3),
                                                     (2, 0), (2, 1), (2, 2), (2, 3)]),
        (Size2D(8, 9), Size2D(2, 3), None, [(0, 0), (0, 1), (0, 2),
                                            (1, 0), (1, 1), (1, 2),
                                            (2, 0), (2, 1), (2, 2),
                                            (3, 0), (3, 1), (3, 2)]),
        (Size2D(4, 3), Size2D(2, 2), Stride(1, 1), [(0, 0), (0, 1),
                                                    (1, 0), (1, 1),
                                                    (2, 0), (2, 1),
                                                    ]),
    ])
    def test_gen_parent_receptive_fields(self, child_dims, parent_dims, stride, expected_result):
        grids = Grids(child_dims, parent_dims, stride)
        result = list(grids.gen_parent_receptive_fields())
        assert expected_result == result

    @pytest.mark.parametrize('grids, expected_result, expected_shape', [
        (Grids(child_grid_dims=Size2D(1, 5), parent_rf_dims=Size2D(1, 3), parent_rf_stride=Stride(1, 1)),
         [
             [
                 [[[nan, nan], [nan, nan], [0, 0]]],
                 [[[nan, nan], [0, 0], [0, 1]]],
                 [[[0, 0], [0, 1], [0, 2]]],
                 [[[0, 1], [0, 2], [nan, nan]]],
                 [[[0, 2], [nan, nan], [nan, nan]]]
             ]
         ], [1, 5, 1, 3, 2]),
        (Grids(child_grid_dims=Size2D(1, 5), parent_rf_dims=Size2D(1, 3), parent_rf_stride=Stride(1, 2)),
         [
             [
                 [[[nan, nan], [nan, nan], [0, 0]]],
                 [[[nan, nan], [0, 0], [nan, nan]]],
                 [[[0, 0], [nan, nan], [0, 1]]],
                 [[[nan, nan], [0, 1], [nan, nan]]],
                 [[[0, 1], [nan, nan], [nan, nan]]]
             ]
         ], [1, 5, 1, 3, 2]),
        (Grids(child_grid_dims=Size2D(3, 1), parent_rf_dims=Size2D(2, 1), parent_rf_stride=Stride(1, 1)),
         [
             [[[[nan, nan]], [[0, 0]]]],
             [[[[0, 0]], [[1, 0]]]],
             [[[[1, 0]], [[nan, nan]]]],
         ], [3, 1, 2, 1, 2]),
        (Grids(child_grid_dims=Size2D(4, 4), parent_rf_dims=Size2D(2, 2), parent_rf_stride=Stride(1, 2)),
         [
             [[[[nan, nan], [nan, nan]], [[nan, nan], [0, 0]]],
              [[[nan, nan], [nan, nan]], [[0, 0], [nan, nan]]],
              [[[nan, nan], [nan, nan]], [[nan, nan], [0, 1]]],
              [[[nan, nan], [nan, nan]], [[0, 1], [nan, nan]]]],

             [[[[nan, nan], [0, 0]], [[nan, nan], [1, 0]]],
              [[[0, 0], [nan, nan]], [[1, 0], [nan, nan]]],
              [[[nan, nan], [0, 1]], [[nan, nan], [1, 1]]],
              [[[0, 1], [nan, nan]], [[1, 1], [nan, nan]]]],

             [[[[nan, nan], [1, 0]], [[nan, nan], [2, 0]]],
              [[[1, 0], [nan, nan]], [[2, 0], [nan, nan]]],
              [[[nan, nan], [1, 1]], [[nan, nan], [2, 1]]],
              [[[1, 1], [nan, nan]], [[2, 1], [nan, nan]]]],

             [[[[nan, nan], [2, 0]], [[nan, nan], [nan, nan]]],
              [[[2, 0], [nan, nan]], [[nan, nan], [nan, nan]]],
              [[[nan, nan], [2, 1]], [[nan, nan], [nan, nan]]],
              [[[2, 1], [nan, nan]], [[nan, nan], [nan, nan]]]],

         ], [4, 4, 2, 2, 2])
    ])
    def test_gen_positioned_parent_child_map(self, grids, expected_result, expected_shape):
        result = grids.gen_positioned_parent_child_map()
        expected_tensor = torch.tensor(expected_result).long()
        assert expected_shape == list(expected_tensor.shape), "Check expected integrity"
        assert expected_shape == list(result.shape), "Check result shape"
        assert same(expected_tensor, result)
