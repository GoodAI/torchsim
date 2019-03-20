import pytest

import torch

from torchsim.core import get_float
from torchsim.core.models.receptive_field.grid import Grids, Stride
from torchsim.utils.param_utils import Size2D
from torchsim.core.models.receptive_field.mapping import Mapping


class TestMapping:
    @pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.slow)])
    def test_expert_expert_mapping(self, device):
        float_dtype = get_float(device)
        child_output_size = 1000
        number_of_child_rfs_x = number_of_child_rfs_y = 10
        parent_rf_size_x = parent_rf_size_y = 2

        grids = Grids(
            Size2D(number_of_child_rfs_y, number_of_child_rfs_x),
            parent_rf_dims=Size2D(parent_rf_size_y, parent_rf_size_x),
            flatten_output_grid_dimensions=True)
        mapping = Mapping.from_child_expert_output(grids, device, child_output_size)

        children_output = torch.zeros(grids.child_grid_height, grids.child_grid_width, child_output_size,
                                      dtype=float_dtype,
                                      device=device)
        children_output[0, 2, 0] = 1
        children_output[number_of_child_rfs_y - parent_rf_size_y - 1, 0, 0] = 1

        parents_input = mapping.map(children_output)

        eyxc_view = parents_input.view(grids.n_parent_rfs, parent_rf_size_y, parent_rf_size_x, -1)
        assert eyxc_view[1, 0, 0, 0] == 1
        assert eyxc_view[grids.n_parent_rfs - 2 * grids.n_parent_rfs_x, parent_rf_size_y - 1, 0, 0] == 1

    @pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.slow)])
    def test_image_expert_mapping(self, device):
        float_dtype = get_float(device)
        n_channels = 3
        image_grid_size_x = 16
        image_grid_size_y = 8
        image_grid_dims = Size2D(image_grid_size_y, image_grid_size_x)
        parent_rf_size_x = parent_rf_size_y = 4
        parent_rf_dims = Size2D(parent_rf_size_y, parent_rf_size_x)

        grids = Grids(image_grid_dims, parent_rf_dims, flatten_output_grid_dimensions=True)
        mapping = Mapping.from_sensory_input(grids, device, n_channels)

        input_image = torch.zeros(image_grid_size_y, image_grid_size_x, n_channels, dtype=float_dtype, device=device)
        input_image[0, parent_rf_size_x, 0] = 1
        input_image[1, parent_rf_size_x + 2, 0] = 2
        input_image[parent_rf_size_y, parent_rf_size_x, 0] = 3

        expert_input = mapping.map(input_image)

        assert expert_input[1, 0, 0, 0] == 1
        assert expert_input[1, 1, 2, 0] == 2
        assert expert_input[image_grid_size_x // parent_rf_size_x + 1, 0, 0, 0] == 3

    @pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.slow)])
    def test_overlapping_rfs(self, device):
        float_dtype = get_float(device)
        x = 3
        y = 4
        input_image_1 = torch.arange(0, x * y, dtype=float_dtype, device=device).view(y, x)
        input_image_2 = torch.rand_like(input_image_1, dtype=float_dtype, device=device)
        grids = Grids(Size2D(y, x), parent_rf_dims=Size2D(2, 2), parent_rf_stride=Stride(1, 1),
                      flatten_output_grid_dimensions=True)
        mapping = Mapping.from_default_input(grids, chunk_size=1, device=device)

        expert_input = mapping.map(input_image_1)
        assert expert_input.equal(torch.tensor(
            [[0, 1, 3, 4],
             [1, 2, 4, 5],
             [3, 4, 6, 7],
             [4, 5, 7, 8],
             [6, 7, 9, 10],
             [7, 8, 10, 11]],
            dtype=float_dtype,
            device=device
        ).view(6, 2, 2, 1))

        back_projection_1 = mapping.inverse_map(expert_input)
        assert input_image_1.equal(back_projection_1)
        back_projection_2 = mapping.inverse_map(mapping.map(input_image_2))
        assert input_image_2.equal(back_projection_2)

    @pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.slow)])
    def test_occupancy(self, device):
        float_dtype = get_float(device)
        grids = Grids(Size2D(4, 3), parent_rf_dims=Size2D(2, 2), parent_rf_stride=Stride(1, 1),
                      flatten_output_grid_dimensions=True)
        mapping = Mapping.from_default_input(grids, device, chunk_size=1)
        assert mapping._occupancies.equal(torch.tensor(
            [[1, 2, 1],
             [2, 4, 2],
             [2, 4, 2],
             [1, 2, 1]], dtype=float_dtype, device=device).view(-1))

        mapping = Mapping.from_default_input(grids, device, chunk_size=2)
        assert mapping._occupancies.size() == torch.Size([4 * 3 * 2])
