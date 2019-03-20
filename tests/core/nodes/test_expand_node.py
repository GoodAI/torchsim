import torch
import pytest

from torchsim.core.memory.tensor_creator import AllocatingCreator
from torchsim.core.nodes.expand_node import Expand
from torchsim.core.utils.tensor_utils import same


@pytest.mark.parametrize("input_shape, dim, desired_size, expected_output_shape",
                         [((4, 5, 6), 0, 3, (3, 4, 5, 6)),
                          ((4, 5, 6), 1, 10, (4, 10, 5, 6)),
                          ((4, 5, 6), 3, 6, (4, 5, 6, 6)),
                          ((1, 4, 5), 0, 5, (5, 4, 5)),
                          ((5, 1, 2, 1), 3, 6, (5, 1, 2, 6)),
                          ((2, 1, 3), 6, 7, (2, 1, 3, 1, 1, 1, 7))])
def test_expansion(input_shape, dim, desired_size, expected_output_shape):
    input_data = torch.zeros(input_shape)
    expected_output = torch.zeros(expected_output_shape)

    expand_unit = Expand(AllocatingCreator(device='cpu'), input_shape, dim=dim, desired_size=desired_size)
    expand_unit.step(input_data)

    assert same(expected_output, expand_unit.output)


@pytest.mark.parametrize("expected_input_shape, dim, desired_size, output_shape",
                         [((4, 5, 6), 0, 3, (3, 4, 5, 6)),
                          ((4, 5, 6), 1, 10, (4, 10, 5, 6)),
                          ((4, 5, 6), 3, 6, (4, 5, 6, 6)),
                          ((1, 4, 5), 0, 5, (5, 4, 5)),
                          ((5, 1, 2, 1), 3, 6, (5, 1, 2, 6)),
                          ((2, 1, 3), 6, 7, (2, 1, 3, 1, 1, 1, 7))])
def test_inverse_projection(expected_input_shape, dim, desired_size, output_shape):
    output_data = torch.zeros(output_shape)
    expected_input = torch.zeros(expected_input_shape)

    expand_unit = Expand(AllocatingCreator(device='cpu'), expected_input_shape, dim=dim, desired_size=desired_size)
    # expand_unit.step(input_data)

    projection = expand_unit.inverse_projection(output_data)

    assert same(expected_input, projection)
