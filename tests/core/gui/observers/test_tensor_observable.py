import math

import numpy as np
import pytest

import torch
from abc import ABC

from torchsim.core.global_settings import GlobalSettings
from torchsim.gui.observers.tensor_observable import TensorObservable, TensorViewProjection
from torchsim.core.utils.tensor_utils import same


class MyTensorObservable(TensorObservable):

    def get_tensor(self) -> torch.Tensor:
        pass


def gr(value: float):
    return [value, value, value]


def g(value: float):
    return [0.0, value, 0.0]


def r(value: float):
    return [value, 0.0, 0.0]


def pad():
    return [0.3, 0.3, 0.3]


def nan():
    return float('nan')


class TestTensorObservable(ABC):
    instance: MyTensorObservable

    def setup_method(self):
        self.instance = MyTensorObservable()

    def test_scale(self):
        data = torch.Tensor([[1, 2], [3, 4]])
        result = self.instance._scale_tensor(data, 2).numpy()
        assert (4, 4) == result.shape
        np.testing.assert_array_equal(np.array([
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 4, 4],
            [3, 3, 4, 4],
        ]), result)

    def test_scale_2(self):
        data = torch.Tensor([[1], [3]])
        result = self.instance._scale_tensor(data, 3).numpy()
        assert (6, 3) == result.shape
        np.testing.assert_array_equal(np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
        ]), result)

    @pytest.mark.parametrize("scale, dims, minimal_size, expected_size", [
        (1, [2, 5, 3], 3, 2),
        (1, [2, 5, 3], 5, 3),
        (1, [2, 5, 3], 20, 10),
        (1, [5, 2, 3], 20, 10),
        (2, [2, 5, 3], 5, 3),
    ])
    def test_update_scale_to_respect_minimum_size(self, scale, dims, minimal_size, expected_size):
        data = torch.zeros(dims)
        self.instance._scale = scale
        GlobalSettings.instance().observer_memory_block_minimal_size = minimal_size
        self.instance._update_scale_to_respect_minimum_size(data)
        assert expected_size == self.instance._scale


class TestTensorViewProjection(ABC):
    instance: TensorViewProjection

    def setup_method(self):
        self.instance = TensorViewProjection(False)

    @pytest.mark.parametrize("t_input, minimum, maximum, expected_result", [
        (torch.Tensor([[-2.0, -1.0, -0.5, 0], [2.0, 1.0, 0.5, 0]]), 0, 1,
         torch.Tensor([[r(1.0), r(1.0), r(0.5), gr(0.0)], [g(1.0), g(1.0), g(0.5), gr(0.0)]])
         ),
        (torch.Tensor([[-2.0, -1.0, -0.5, 0], [2.0, 1.0, 0.5, 0]]), 0.5, 1,
         torch.Tensor([[r(1.0), r(1.0), r(0.0), gr(0.0)], [g(1.0), g(1.0), g(0.0), gr(0.0)]])
         ),
        (torch.Tensor([[-2.0, -1.0, -0.5, -0.2], [2.0, 1.0, 0.5, 0.2]]), 0.5, 1.5,
         torch.Tensor([[r(1.0), r(0.5), r(0.0), gr(0.0)], [g(1.0), g(0.5), g(0.0), gr(0.0)]])
         ),
    ])
    def test_colorize(self, t_input, minimum, maximum, expected_result):
        result = self.instance._colorize(t_input, minimum, maximum)
        assert same(expected_result, result, eps=0.001)

    def test_colorize_big_tensor(self):
        """Check for a tensor coloring problem. Note: torch was crashing hard, before the fix."""
        w, h = 10000, 100
        data = torch.full((h, w), 1.0)
        result = self.instance._colorize(data, minimum=0, maximum=4).numpy()
        assert (h, w, 3) == result.shape
        expected = np.repeat(np.array([[[0, .25, 0]]]), h, axis=0)
        expected = np.repeat(expected, w, axis=1)
        np.testing.assert_array_equal(expected, result)

    def test_colorize_inf_nan(self):
        data = torch.Tensor([[float('NaN'), float('Inf'), -float('Inf')]])
        result = self.instance._colorize(data, minimum=-2, maximum=4).numpy()
        assert (1, 3, 3) == result.shape
        np.testing.assert_array_equal(np.array([
            [
                [0, 0, 1],
                [0, 1, 1],
                [1, 0, 1],
            ],
        ]), result)

    @pytest.mark.parametrize("t_input_size, t_shape, items_per_row, result", [
        (6, [1, 6], 1,
         torch.Tensor([[
             g(0.01),
             g(0.02),
             g(0.03),
             g(0.04),
             g(0.05),
             g(0.06),
         ]])),
        (6, [2, 3], 1,
         torch.Tensor([
             [
                 g(0.01), g(0.02), g(0.03)
             ], [
                 g(0.04), g(0.05), g(0.06)
             ]])),
        (12, [2, 3], 2,
         torch.Tensor([
             [
                 g(0.01), g(0.02), g(0.03), g(0.07), g(0.08), g(0.09)
             ], [
                 g(0.04), g(0.05), g(0.06), g(0.10), g(0.11), g(0.12)
             ]])),
        (12, [3, 2], 2,
         torch.Tensor([
             [
                 g(0.01), g(0.02), g(0.07), g(0.08)
             ], [
                 g(0.03), g(0.04), g(0.09), g(0.10)
             ], [
                 g(0.05), g(0.06), g(0.11), g(0.12)
             ]])),
        (10, [2, 3], 2,
         torch.Tensor([
             [
                 g(0.01), g(0.02), g(0.03), g(0.07), g(0.08), g(0.09)
             ], [
                 g(0.04), g(0.05), g(0.06), g(0.10), pad(), pad()
             ]])),
        (9, [2, 2], 2,
         torch.Tensor([
             [
                 g(0.01), g(0.02), g(0.05), g(0.06),
             ], [
                 g(0.03), g(0.04), g(0.07), g(0.08),
             ], [
                 g(0.09), pad(), pad(), pad()
             ], [
                 pad(), pad(), pad(), pad()
             ]])),
    ])
    def test_transform_tensor_tiling_padding(self, t_input_size, t_shape, items_per_row, result):
        t_input = torch.arange(1, t_input_size + 1).float() / 100
        # Add dimensions - there should be no effect
        t_input = t_input.expand((1, 1, 1, t_input_size))
        t_input = t_input.transpose(3, 1)

        view_projection = TensorViewProjection(False)
        view_projection._shape = t_shape
        view_projection._items_per_row = items_per_row
        tensor, _ = view_projection.transform_tensor(t_input, False)
        assert same(result, tensor, eps=0.001)

    @pytest.mark.parametrize("tensor_dims, shape, is_rgb, result_dims", [
        ([3, 4, 5], [], False, [4, 5]),
        ([4, 5], [], False, [4, 5]),
        ([5], [], False, [1, 1]),  # Linear tensor should have tile 1x1
        ([5, 6, 3], [], True, [5, 6]),
        ([8, 5, 6, 3], [], True, [5, 6]),
        ([8, 5, 6, 1], [], True, [5, 6]),
        ([5, 3], [], True, [1, 1]),  # RGB Linear tensor should have tile 1x1
        ([3], [], True, [1, 1]),  # RGB Linear tensor should have tile 1x1
        ([4, 5], [3], False, [4, 5]),  # Incomplete shape has no effect
        ([4, 5], [2, 3], False, [2, 3]),
        ([4, 5], [2, 3], True, [2, 3]),
        ([4, 5], [1, 2, 3], True, [2, 3]),
    ])
    def test_compute_tile_dimensions(self, tensor_dims, shape, is_rgb, result_dims):
        view_projection = TensorViewProjection(False)
        view_projection._shape = shape
        height, width = view_projection._compute_tile_dimensions(tensor_dims, is_rgb)
        assert result_dims == [height, width]

    @pytest.mark.parametrize("tensor_dims, shape, is_rgb, result_dims", [
        ([3, 4, 5], [], False, [1, 5]),
        ([4, 5], [], False, [1, 5]),
        ([5], [], False, [1, 1]),  # Linear tensor should have tile 1x1
        ([5, 6, 3], [], True, [1, 6]),
        ([8, 5, 6, 3], [], True, [1, 6]),
        ([5, 3], [], True, [1, 1]),  # RGB Linear tensor should have tile 1x1
        ([3], [], True, [1, 1]),  # RGB Linear tensor should have tile 1x1
        ([4, 5], [3], False, [1, 5]),  # Incomplete shape has no effect
        ([4, 5], [2, 3], False, [2, 3]),
        ([4, 5], [2, 3], True, [2, 3]),
        ([4, 5], [1, 2, 3], True, [2, 3]),
    ])
    def test_compute_tile_dimensions_buffer(self, tensor_dims, shape, is_rgb, result_dims):
        view_projection = TensorViewProjection(True)
        view_projection._shape = shape
        height, width = view_projection._compute_tile_dimensions(tensor_dims, is_rgb)
        assert result_dims == [height, width]

    @pytest.mark.parametrize("t_input_size, t_shape, items_per_row, result", [
        (6, [1, 6], 1,
         torch.Tensor([[
             g(0.01),
             g(0.02),
             g(0.03),
             g(0.04),
             g(0.05),
             g(0.06),
         ]])),
        (6, [2, 3], 1,
         torch.Tensor([
             [
                 g(0.01), g(0.02), g(0.03)
             ], [
                 g(0.04), g(0.05), g(0.06)
             ]])),
        (12, [2, 3], 2,
         torch.Tensor([
             [
                 g(0.01), g(0.02), g(0.03), g(0.07), g(0.08), g(0.09)
             ], [
                 g(0.04), g(0.05), g(0.06), g(0.10), g(0.11), g(0.12)
             ]])),
        (12, [3, 2], 2,
         torch.Tensor([
             [
                 g(0.01), g(0.02), g(0.07), g(0.08)
             ], [
                 g(0.03), g(0.04), g(0.09), g(0.10)
             ], [
                 g(0.05), g(0.06), g(0.11), g(0.12)
             ]])),
        (10, [2, 3], 2,
         torch.Tensor([
             [
                 g(0.01), g(0.02), g(0.03), g(0.07), g(0.08), g(0.09)
             ], [
                 g(0.04), g(0.05), g(0.06), g(0.10), pad(), pad()
             ]])),
        (9, [2, 2], 2,
         torch.Tensor([
             [
                 g(0.01), g(0.02), g(0.05), g(0.06),
             ], [
                 g(0.03), g(0.04), g(0.07), g(0.08),
             ], [
                 g(0.09), pad(), pad(), pad()
             ], [
                 pad(), pad(), pad(), pad()
             ]])),
    ])
    def test_transform_tensor_tiling_padding_rgb(self, t_input_size, t_shape, items_per_row, result):
        t_input = torch.arange(1, t_input_size + 1).float() / 100
        # Add dimensions - there should be no effect
        t_input = torch.tensor(g(1)).expand((t_input_size, 3)).mul(t_input.unsqueeze(1))

        view_projection = TensorViewProjection(False)
        view_projection._shape = t_shape
        view_projection._items_per_row = items_per_row
        tensor, _ = view_projection.transform_tensor(t_input, True)
        assert same(result, tensor, eps=0.001)

    @pytest.mark.parametrize("t_input_size, t_shape, items_per_row, result", [
        (6, [2, 3], 1,
         torch.Tensor([
             [
                 g(0.01), g(0.02), g(0.03)
             ], [
                 g(0.04), g(0.05), g(0.06)
             ]])),
    ])
    def test_transform_tensor_one_channel_rgb(self, t_input_size, t_shape, items_per_row, result):
        t_input = torch.arange(1, t_input_size + 1).float() / 100
        t_input = t_input.view([*t_shape, 1])
        view_projection = TensorViewProjection(False)
        # view_projection._shape = t_shape
        view_projection._items_per_row = items_per_row
        tensor, _ = view_projection.transform_tensor(t_input, True)
        assert same(result, tensor, eps=0.001)

    @pytest.mark.parametrize("t_input_size, t_shape, items_per_row, result", [
        (12, [2, 3], 2,
         torch.Tensor([
             [
                 0.01, 0.02, 0.03, 0.07, 0.08, 0.09
             ], [
                 0.04, 0.05, 0.06, 0.10, 0.11, 0.12
             ]])),
        (12, [3, 2], 2,
         torch.Tensor([
             [
                 0.01, 0.02, 0.07, 0.08
             ], [
                 0.03, 0.04, 0.09, 0.10
             ], [
                 0.05, 0.06, 0.11, 0.12
             ]])),
        (9, [2, 2], 2,
         torch.Tensor([
             [
                 0.01, 0.02, 0.05, 0.06,
             ], [
                 0.03, 0.04, 0.07, 0.08,
             ], [
                 0.09, nan(), nan(), nan()
             ], [
                 nan(), nan(), nan(), nan()
             ]])),

    ])
    def test_value_at(self, t_input_size, t_shape, items_per_row, result):
        t_input = torch.arange(1, t_input_size + 1).float() / 100
        # Add dimensions - there should be no effect
        t_input = t_input.expand((1, 1, 1, t_input_size))
        t_input = t_input.transpose(3, 1)

        view_projection = TensorViewProjection(False)
        view_projection._shape = t_shape
        view_projection._items_per_row = items_per_row

        height, width = result.size()
        for y in range(height):
            for x in range(width):
                value = view_projection.value_at(t_input, x, y)
                expected = float(result[y, x])
                if math.isnan(expected):
                    assert math.isnan(value), f'Value at {x}, {y}'
                else:
                    assert expected == value, f'Value at {x}, {y}'

        # tensor, _ = view_projection.transform_tensor(t_input, False)
        # assert same(result, tensor, eps=0.001)

    @pytest.mark.parametrize("t_input, p_min, p_max, expected_result", [
        (torch.Tensor([[gr(0.5), gr(1.0), gr(1.5), gr(2.0)]]),
         0, 1,
         torch.Tensor([[gr(0.5), gr(1.0), gr(1.0), gr(1.0)]])
         ),
        (torch.Tensor([[gr(-4.0), gr(-1.0), gr(0), gr(2.0)]]),
         -2, 0,
         torch.Tensor([[gr(0.0), gr(0.5), gr(1.0), gr(1.0)]])
         ),
        (torch.Tensor([[gr(-4.0), gr(-1.0), gr(0), gr(2.0), gr(6.0), gr(8.0)]]),
         -2, 6,
         torch.Tensor([[gr(0), gr(0.125), gr(0.25), gr(0.5), gr(1.0), gr(1.0)]])
         )
    ])
    def test_rgb_transform(self, t_input, p_min, p_max, expected_result):
        result = TensorViewProjection._rgb_transform(t_input, p_min, p_max)
        assert same(expected_result, result, eps=0.001)
